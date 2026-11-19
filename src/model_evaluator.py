"""model_evaluator.py — Stage 6: evaluate ONCE on the untouched test set, with a frozen threshold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve, matthews_corrcoef)

from src.threshold_optimizer import net_benefit
from src.utils import PALETTE, abs_path, ensure_dirs, get_logger, save_json, set_plot_style


def metrics_at(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    return {"threshold": round(float(thr), 3),
            "accuracy": round(accuracy_score(y, pred), 4), "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4), "recall": round(recall_score(y, pred), 4),
            "f1_score": round(f1_score(y, pred), 4), "mcc": round(matthews_corrcoef(y, pred), 4)}


def confusion(y, p, thr) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int)).ravel()
    return {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}


def expected_calibration_error(y, p, bins=10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def lift_table(y, p, bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = pd.qcut(d.index, bins, labels=range(1, bins + 1))
    g = d.groupby("decile", observed=True).agg(customers=("y", "size"), churners=("y", "sum"), avg_prob=("p", "mean"))
    g["churn_rate"] = g["churners"] / g["customers"]
    g["lift"] = g["churn_rate"] / d["y"].mean()
    g["cum_churners_captured"] = g["churners"].cumsum() / d["y"].sum()
    return g.reset_index()


class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("ModelEvaluator", config)
        self.fig_dir = abs_path(config["reports"]["figures_dir"])
        ensure_dirs(self.fig_dir)
        self.plt = set_plot_style()

    # ---------- bootstrap CI ----------
    def bootstrap_ci(self, y, p, thr, n) -> dict:
        rng = np.random.default_rng(self.config["model"]["random_state"])
        y = np.asarray(y); keys = ["roc_auc", "pr_auc", "recall", "precision", "f1_score"]
        vals = {k: [] for k in keys}
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            pred = (p[i] >= thr).astype(int)
            vals["roc_auc"].append(roc_auc_score(y[i], p[i])); vals["pr_auc"].append(average_precision_score(y[i], p[i]))
            vals["recall"].append(recall_score(y[i], pred)); vals["precision"].append(precision_score(y[i], pred, zero_division=0))
            vals["f1_score"].append(f1_score(y[i], pred))
        return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in vals.items()}

    # ---------- fairness ----------
    def fairness_audit(self, X_test, y, p, thr) -> list[dict]:
        rows = []
        for col in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
            if col not in X_test.columns:
                continue
            for val, idx in X_test.groupby(col).groups.items():
                m = X_test.index.get_indexer(idx)
                yy, pp = np.asarray(y)[m], p[m]
                pred = (pp >= thr).astype(int)
                rows.append({"attribute": col, "group": str(val), "n": int(len(m)), "actual_churn_rate": round(float(yy.mean()), 3),
                             "flag_rate": round(float(pred.mean()), 3), "recall": round(float(recall_score(yy, pred)), 3),
                             "precision": round(float(precision_score(yy, pred, zero_division=0)), 3),
                             "roc_auc": round(float(roc_auc_score(yy, pp)), 3)})
        return rows

    # ---------- plots ----------
    def _save(self, fig, name):
        fig.savefig(self.fig_dir / name); self.plt.close(fig)

    def plot_roc_pr(self, y, p, m):
        plt = self.plt
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(5, 4.3))
        ax.plot(fpr, tpr, lw=2.2, color=PALETTE["stay"], label=f"Final model (AUC = {m['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["grey"], label="Random guess")
        ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve — test set"); ax.legend(loc="lower right")
        self._save(fig, "roc_curve.png")
        pr, rc, _ = precision_recall_curve(y, p)
        fig, ax = plt.subplots(figsize=(5, 4.3))
        ax.plot(rc, pr, lw=2.2, color=PALETTE["churn"], label=f"Final model (PR-AUC = {m['pr_auc']:.3f})")
        ax.axhline(np.mean(y), ls="--", color=PALETTE["grey"], label=f"Baseline = churn rate ({np.mean(y):.2f})")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall curve — test set"); ax.legend()
        self._save(fig, "pr_curve.png")

    def plot_confusions(self, y, p, thr_opt):
        plt = self.plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, thr, title in zip(axes, [0.5, thr_opt], ["Default threshold 0.50", f"Cost-optimal threshold {thr_opt:.2f}"]):
            cm = confusion_matrix(y, (p >= thr).astype(int))
            ax.imshow(cm, cmap="Blues"); ax.grid(False)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=15, fontweight="bold",
                            color="white" if cm[i, j] > cm.max() / 2 else "#1e293b")
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"],
                   xlabel="Predicted", ylabel="Actual", title=title)
        fig.suptitle("Confusion matrices — test set", fontweight="bold"); fig.tight_layout()
        self._save(fig, "confusion_matrices.png")

    def plot_calibration(self, y, p):
        plt = self.plt
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 4.3))
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["grey"], label="Perfectly calibrated")
        ax.plot(mean, frac, "o-", color=PALETTE["stay"], lw=2, label="Final model")
        ax.set(xlabel="Predicted churn probability", ylabel="Observed churn rate", title="Calibration curve — test set"); ax.legend()
        self._save(fig, "calibration_curve.png")

    def plot_threshold(self, curve, thr_opt, thr_f1):
        plt = self.plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.plot(curve.threshold, curve.precision, label="Precision", color=PALETTE["stay"])
        a1.plot(curve.threshold, curve.recall, label="Recall", color=PALETTE["churn"])
        a1.plot(curve.threshold, curve.f1, label="F1", color=PALETTE["good"])
        a1.axvline(thr_opt, ls="--", color="k", label=f"Cost-optimal = {thr_opt:.2f}")
        a1.set(xlabel="Decision threshold", title="Precision / recall trade-off (out-of-fold)"); a1.legend(loc="center right", fontsize=8)
        sm = curve.net_benefit.rolling(5, center=True, min_periods=1).mean()
        a2.plot(curve.threshold, curve.net_benefit, alpha=.35, color=PALETTE["grey"], label="Raw")
        a2.plot(curve.threshold, sm, lw=2.2, color=PALETTE["good"], label="Smoothed")
        a2.axvline(thr_opt, ls="--", color="k"); a2.axvline(0.5, ls=":", color=PALETTE["churn"], label="Default 0.50")
        a2.set(xlabel="Decision threshold", ylabel="Net benefit ($, training set)", title="Business value vs threshold"); a2.legend()
        fig.tight_layout(); self._save(fig, "threshold_analysis.png")

    def plot_lift(self, lt):
        plt = self.plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.bar(lt.decile.astype(int), lt.churn_rate * 100, color=PALETTE["stay"])
        a1.axhline(lt.churners.sum() / lt.customers.sum() * 100, ls="--", color=PALETTE["churn"], label="Overall churn rate")
        a1.set(xlabel="Risk decile (1 = highest predicted risk)", ylabel="Actual churn rate (%)", title="Churn rate by risk decile"); a1.legend()
        x = np.r_[0, lt.decile.astype(int) * 10]; yv = np.r_[0, lt.cum_churners_captured * 100]
        a2.plot(x, yv, "o-", color=PALETTE["good"], lw=2, label="Model"); a2.plot([0, 100], [0, 100], "--", color=PALETTE["grey"], label="Random targeting")
        a2.set(xlabel="% of customers contacted (highest risk first)", ylabel="% of churners captured", title="Cumulative gains"); a2.legend()
        fig.tight_layout(); self._save(fig, "lift_gains.png")

    def plot_comparison(self, comparison, ablation, final):
        plt = self.plt
        names = list(comparison); x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(7, 4))
        pr = [comparison[n]["pr_auc_mean"] for n in names]; er = [comparison[n]["pr_auc_std"] for n in names]
        cols = [PALETTE["good"] if n == final else PALETTE["stay"] for n in names]
        ax.bar(x, pr, yerr=er, color=cols, capsize=5); ax.set_xticks(x); ax.set_xticklabels(names, rotation=12)
        ax.set_ylim(min(pr) - 0.04, max(pr) + 0.03); ax.set(ylabel="CV PR-AUC (mean ± std)", title="Model comparison (5-fold CV, training data)")
        for i, v in enumerate(pr): ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
        self._save(fig, "model_comparison.png")
        ab = pd.DataFrame(ablation)
        if ab.empty: return
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric, ttl in zip(axes, ["average_precision", "recall"], ["PR-AUC (ranking quality)", "Recall @ threshold 0.5"]):
            piv = ab.pivot(index="model", columns="strategy", values=metric)
            piv.plot.bar(ax=ax, rot=12, width=.8, legend=False); ax.set(title=ttl, xlabel="")
            if metric == "average_precision": ax.set_ylim(piv.min().min() - .02, piv.max().max() + .015)
        axes[1].legend(title="strategy", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
        fig.suptitle("Imbalance handling: same ranking, different recall", fontweight="bold"); fig.tight_layout()
        self._save(fig, "imbalance_ablation.png")

    def plot_importance(self, imp: dict):
        plt = self.plt
        items = list(imp.items())[:12][::-1]
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.barh([k for k, _ in items], [v for _, v in items], color=PALETTE["stay"])
        ax.set(xlabel="Drop in ROC-AUC when feature is shuffled", title="Permutation importance — test set"); 
        self._save(fig, "permutation_importance.png")

    # ---------- main ----------
    def run(self, model, X_test, y_test, model_info: dict, threshold_info: dict, curve, calib_info: dict, oof_proba=None):
        self.logger.info("=== Model Evaluation Started (test set used once) ===")
        y = np.asarray(y_test); p = model.predict_proba(X_test)[:, 1]
        thr = threshold_info["cost_optimal_threshold"]
        m_def, m_opt = metrics_at(y, p, 0.5), metrics_at(y, p, thr)
        roc, pr = float(roc_auc_score(y, p)), float(average_precision_score(y, p))
        base_metrics = {"roc_auc": round(roc, 4), "pr_auc": round(pr, 4), "brier": round(float(brier_score_loss(y, p)), 4),
                        "ece": round(expected_calibration_error(y, p), 4)}
        ci = self.bootstrap_ci(y, p, thr, self.config["model"]["bootstrap_samples"])
        lt = lift_table(y, p)

        self.logger.info("Computing permutation importance on raw features ...")
        pi = permutation_importance(model, X_test.drop(columns=["customerID"], errors="ignore"), y, scoring="roc_auc",
                                    n_repeats=8, random_state=self.config["model"]["random_state"], n_jobs=1)
        cols = X_test.drop(columns=["customerID"], errors="ignore").columns
        imp = dict(sorted(zip(cols, pi.importances_mean.round(5).tolist()), key=lambda kv: -kv[1]))
        monthly = X_test["MonthlyCharges"].astype(float).values
        biz = self.config["business"]
        strategies = {
            "Contact nobody": 0.0,
            "Contact everyone": net_benefit(y, np.ones_like(p), monthly, 0.0, biz),
            f"Model @ default 0.50": net_benefit(y, p, monthly, 0.5, biz),
            f"Model @ cost-optimal {thr:.2f}": net_benefit(y, p, monthly, thr, biz)}

        report = {
            "model_name": model_info["model_name"], "selection_reason": model_info["selection_reason"],
            "best_params": model_info["best_params"], "threshold": thr,
            "test_metrics": {**m_opt, **base_metrics},                      # headline = cost-optimal threshold
            "test_metrics_default_threshold": {**m_def, **base_metrics},
            "confusion_matrix": confusion(y, p, thr), "confusion_matrix_default": confusion(y, p, 0.5),
            "bootstrap_95ci_at_optimal_threshold": ci,
            "top_features": dict(list(imp.items())[:10]), "permutation_importance": imp,
            "model_comparison": model_info["comparison"], "imbalance_ablation": model_info["imbalance_ablation"],
            "calibration": calib_info, "threshold_analysis": threshold_info,
            "business_test_net_benefit": {k: round(v, 2) for k, v in strategies.items()},
            "business_assumptions": biz, "lift_table": lt.round(4).to_dict("records"),
            "fairness_audit": self.fairness_audit(X_test, y, p, thr),
            "classification_report": classification_report(y, (p >= thr).astype(int), target_names=["No Churn", "Churn"]),
            "classification_report_default": classification_report(y, (p >= 0.5).astype(int), target_names=["No Churn", "Churn"]),
            "n_test": int(len(y)), "test_churn_rate": round(float(y.mean()), 4),
        }
        self.plot_roc_pr(y, p, {"roc_auc": roc, "pr_auc": pr}); self.plot_confusions(y, p, thr)
        self.plot_calibration(y, p); self.plot_threshold(curve, thr, threshold_info["f1_optimal_threshold"])
        self.plot_lift(lt); self.plot_comparison(model_info["comparison"], model_info["imbalance_ablation"], model_info["model_name"])
        self.plot_importance(imp)
        save_json(report, self.config["model"]["report_path"])
        self.logger.info(f"TEST @0.50  : recall={m_def['recall']} precision={m_def['precision']} F1={m_def['f1_score']}")
        self.logger.info(f"TEST @{thr:.2f}  : recall={m_opt['recall']} precision={m_opt['precision']} F1={m_opt['f1_score']} | ROC-AUC={roc:.4f} PR-AUC={pr:.4f}")
        return report, p
