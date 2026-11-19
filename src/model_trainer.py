"""model_trainer.py — Stage 4: compare models, study imbalance handling, calibrate, pick the final model.

What happens here (all on the TRAINING split only — the test set is never touched):
  1. GridSearchCV (stratified 5-fold, PR-AUC) for every candidate family.
  2. Selection: "one-standard-error" rule -> simplest model whose CV score is within 1 SE of the best.
  3. Imbalance ablation: none vs class_weight vs SMOTE for every tuned family.
  4. Calibration check (Brier score) -> keep CalibratedClassifierCV only if it helps.
  5. Out-of-fold probabilities of the final model (used to choose the decision threshold).
"""
from __future__ import annotations

import copy
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor
from src.feature_engineering import ChurnFeatureEngineer
from src.utils import abs_path, ensure_dirs, get_logger

try:  # optional dependencies (training only)
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover
    SMOTE = ImbPipeline = None

warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, config: dict, fast: bool = False):
        self.config = config
        self.fast = fast
        self.mcfg = config["model"]
        self.logger = get_logger("ModelTrainer", config)
        self.rs = self.mcfg["random_state"]
        self.cv = StratifiedKFold(n_splits=3 if fast else self.mcfg["cv_folds"], shuffle=True, random_state=self.rs)
        ensure_dirs("artifacts")

    # ───────────────────────── building blocks ─────────────────────────
    def _classifier(self, name: str, class_weight=None, pos_weight=None):
        rs = self.rs
        if name == "LogisticRegression":
            return LogisticRegression(max_iter=3000, random_state=rs, class_weight=class_weight)
        if name == "RandomForest":
            return RandomForestClassifier(random_state=rs, n_jobs=1, class_weight=class_weight)
        if name == "HistGradientBoosting":
            return HistGradientBoostingClassifier(random_state=rs, class_weight=class_weight)
        if name == "XGBoost":
            return XGBClassifier(random_state=rs, n_jobs=1, eval_metric="logloss", verbosity=0,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=pos_weight or 1.0)
        raise KeyError(name)

    def available_candidates(self) -> list[str]:
        names = list(self.mcfg["candidates"].keys())
        if XGBClassifier is None and "XGBoost" in names:
            self.logger.warning("xgboost not installed -> skipping XGBoost candidate")
            names.remove("XGBoost")
        return names

    def make_pipeline(self, clf, sampler=None):
        feat = self.config["features"]
        steps = [("fe", ChurnFeatureEngineer(feat["add_engineered"], feat.get("drop_columns"))),
                 ("pre", build_preprocessor(self.config))]
        if sampler is not None:
            steps.append(("smote", sampler))
            return ImbPipeline(steps + [("clf", clf)])
        return Pipeline(steps + [("clf", clf)])

    # ───────────────────────── step 1: grid search ─────────────────────────
    def tune_all(self, X, y) -> dict:
        key = "params_fast" if self.fast else "params"
        results = {}
        for name in self.available_candidates():
            grid = {f"clf__{k}": v for k, v in self.mcfg["candidates"][name][key].items()}
            gs = GridSearchCV(self.make_pipeline(self._classifier(name)), grid, cv=self.cv, n_jobs=1,
                              scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
                              refit=self.mcfg["selection_metric"], return_train_score=False)
            gs.fit(X, y)
            i = gs.best_index_
            r = gs.cv_results_
            results[name] = {
                "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                "pr_auc_mean": float(r["mean_test_average_precision"][i]),
                "pr_auc_std": float(r["std_test_average_precision"][i]),
                "roc_auc_mean": float(r["mean_test_roc_auc"][i]),
                "roc_auc_std": float(r["std_test_roc_auc"][i]),
                "n_combinations": int(len(r["params"])),
            }
            self.logger.info(f"{name:22s} PR-AUC {results[name]['pr_auc_mean']:.4f}±{results[name]['pr_auc_std']:.4f} "
                             f"| ROC-AUC {results[name]['roc_auc_mean']:.4f} | {results[name]['best_params']}")
        return results

    # ───────────────────────── step 2: selection ─────────────────────────
    def select_model(self, results: dict) -> tuple[str, str]:
        best = max(results, key=lambda k: results[k]["pr_auc_mean"])
        if self.mcfg["selection_rule"] != "one_se":
            return best, f"highest mean CV PR-AUC ({results[best]['pr_auc_mean']:.4f})"
        se = results[best]["pr_auc_std"] / np.sqrt(self.cv.get_n_splits())
        floor = results[best]["pr_auc_mean"] - se
        ok = [n for n in results if results[n]["pr_auc_mean"] >= floor]
        rank = self.mcfg["complexity_rank"]
        chosen = min(ok, key=lambda n: (rank.get(n, 9), -results[n]["pr_auc_mean"]))
        why = (f"one-standard-error rule: best was {best} ({results[best]['pr_auc_mean']:.4f}); floor = best - 1 SE = {floor:.4f}; "
               f"candidates within the floor = {ok}; simplest of them = {chosen}")
        self.logger.info(why)
        return chosen, why

    # ───────────────────────── step 3: imbalance ablation ─────────────────────────
    def imbalance_ablation(self, X, y, results: dict) -> list[dict]:
        pos_w = float((y == 0).sum() / (y == 1).sum())
        scoring = ["roc_auc", "average_precision", "recall", "precision", "f1"]
        rows = []
        for name, res in results.items():
            strategies = {"none": dict(), "class_weight": dict(class_weight="balanced", pos_weight=pos_w)}
            if SMOTE is not None:
                strategies["SMOTE"] = dict(sampler=True)
            for strat, kw in strategies.items():
                if strat == "class_weight" and name == "XGBoost":
                    clf = self._classifier(name, pos_weight=pos_w)
                else:
                    clf = self._classifier(name, class_weight=kw.get("class_weight"))
                clf.set_params(**res["best_params"])
                sampler = SMOTE(random_state=self.rs) if kw.get("sampler") else None
                cvres = cross_validate(self.make_pipeline(clf, sampler), X, y, cv=self.cv, scoring=scoring, n_jobs=1)
                row = {"model": name, "strategy": strat}
                for m in scoring:
                    row[m] = round(float(cvres[f"test_{m}"].mean()), 4)
                rows.append(row)
                self.logger.info(f"ablation {name:22s} {strat:12s} PR-AUC {row['average_precision']:.4f} "
                                 f"recall@0.5 {row['recall']:.3f} precision@0.5 {row['precision']:.3f} F1 {row['f1']:.3f}")
        return rows

    # ───────────────────────── step 4: calibration ─────────────────────────
    def build_final(self, name: str, params: dict, X, y):
        base = self.make_pipeline(self._classifier(name))
        base.set_params(**{f"clf__{k}": v for k, v in params.items()})
        calibrated = CalibratedClassifierCV(copy.deepcopy(base), method="sigmoid", cv=self.cv)
        oof_raw = cross_val_predict(base, X, y, cv=self.cv, method="predict_proba")[:, 1]
        oof_cal = cross_val_predict(calibrated, X, y, cv=self.cv, method="predict_proba")[:, 1]
        b_raw, b_cal = brier_score_loss(y, oof_raw), brier_score_loss(y, oof_cal)
        mode = self.mcfg["calibrate"]
        use_cal = (mode is True) or (mode == "auto" and b_cal < b_raw - 0.0005)
        self.logger.info(f"Brier score raw={b_raw:.4f} calibrated={b_cal:.4f} -> calibration {'KEPT' if use_cal else 'not needed'}")
        final = calibrated if use_cal else base
        final.fit(X, y)
        return final, (oof_cal if use_cal else oof_raw), {
            "brier_raw": round(float(b_raw), 5), "brier_calibrated": round(float(b_cal), 5), "calibration_used": bool(use_cal)}

    # ───────────────────────── run ─────────────────────────
    def run(self, X_train, y_train):
        self.logger.info("=== Model Training Started ===")
        self.logger.info(f"Training rows={len(X_train)} | CV folds={self.cv.get_n_splits()} | fast={self.fast}")
        results = self.tune_all(X_train, y_train)
        chosen, why = self.select_model(results)
        self.logger.info(f">> Selected model: {chosen}")
        ablation = self.imbalance_ablation(X_train, y_train, results)
        final, oof, calib = self.build_final(chosen, results[chosen]["best_params"], X_train, y_train)
        joblib.dump(final, abs_path(self.mcfg["model_path"]))
        self.logger.info(f"Model saved -> {self.mcfg['model_path']}")
        info = {"model_name": chosen, "selection_reason": why, "best_params": results[chosen]["best_params"],
                "comparison": results, "imbalance_ablation": ablation, "calibration": calib}
        return final, oof, info


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    from src.data_preprocessing import DataPreprocessing
    from src.utils import load_config
    cfg = load_config()
    Xtr, Xte, ytr, yte = DataPreprocessing(cfg).run(DataIngestion(cfg).run())
    _, _, info = ModelTrainer(cfg, fast=True).run(Xtr, ytr)
    print(info["model_name"])
