"""
run_pipeline.py — Runs the whole ML pipeline end-to-end.

    python run_pipeline.py            # full run (tuning grids from config.yaml, ~10-15 min on 1 CPU)
    python run_pipeline.py --fast     # quick run with small grids (~1-2 min) for demos / CI

Stages: 1 ingestion → 2 validation → EDA → 3 split → 4 training (compare, imbalance, calibrate)
        → 5 threshold optimisation → 6 evaluation (test, once) → 7 SHAP → 8 business impact → save metadata
"""
import argparse
import shutil
import datetime as dt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from src.business_impact import BusinessImpact
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.data_validation import DataValidation
from src.eda import EDA
from src.explainability import Explainer
from src.feature_engineering import clean_total_charges
from src.model_evaluator import ModelEvaluator
from src.model_trainer import ModelTrainer
from src.predictor import REQUIRED
from src.threshold_optimizer import ThresholdOptimizer
from src.utils import abs_path, environment_info, file_sha256, get_logger, load_config, save_json


def build_metadata(config, report, model_info, thr_info, X_train, df):
    allowed = config["validation"]["allowed_values"]
    ref = X_train[REQUIRED].copy()
    ref["TotalCharges"] = clean_total_charges(X_train)
    ref.sample(min(config["explainability"]["reference_rows_for_app"] * 5, len(ref)), random_state=3).to_csv(
        abs_path(config["model"]["artifacts_dir"]) / "reference_sample.csv", index=False)
    meta = {
        "project": config["project"], "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "environment": environment_info(), "data_sha256": file_sha256(abs_path(config["data"]["raw_path"])),
        "model_name": report["model_name"], "best_params": report["best_params"],
        "threshold": report["threshold"], "high_risk_probability": config["business"]["high_risk_probability"],
        "theoretical_threshold": thr_info["theoretical_threshold"],
        "calibration": report["calibration"], "test_metrics": report["test_metrics"],
        "schema": {"required": REQUIRED, "optional": ["TotalCharges"],
                   "options": {k: v for k, v in allowed.items() if k not in ("Churn", "gender")} | {"SeniorCitizen": [0, 1]},
                   "numeric_ranges": {"tenure": [0, 120], "MonthlyCharges": [0, 200], "TotalCharges": [0, 20000]},
                   "population_medians": {c: float(pd.to_numeric(X_train[c]).median()) for c in ("tenure", "MonthlyCharges")}},
    }
    save_json(meta, config["model"]["metadata_path"])


def run_pipeline(fast: bool = False, config_path: str = "config.yaml"):
    config = load_config(config_path)
    log = get_logger("Pipeline", config)
    log.info("=" * 64); log.info(f"  CUSTOMER CHURN PREDICTION — PIPELINE STARTED (fast={fast})"); log.info("=" * 64)

    log.info("[STAGE 1] Data ingestion");   df = DataIngestion(config).run()
    log.info("[STAGE 2] Data validation");  DataValidation(config).run(df)
    log.info("[EDA] Exploratory analysis"); EDA(config).run(df)
    log.info("[STAGE 3] Preprocessing / split"); X_train, X_test, y_train, y_test = DataPreprocessing(config).run(df)
    log.info("[STAGE 4] Model training");   model, oof, info = ModelTrainer(config, fast=fast).run(X_train, y_train)
    log.info("[STAGE 5] Threshold optimisation")
    thr_info, curve = ThresholdOptimizer(config).run(y_train, oof, X_train["MonthlyCharges"].astype(float).values)
    log.info("[STAGE 6] Evaluation on held-out test set")
    report, proba = ModelEvaluator(config).run(model, X_test, y_test, info, thr_info, curve, info["calibration"])
    log.info("[STAGE 7] Explainability (SHAP)"); Explainer(config).run(model, X_train, X_test)
    log.info("[STAGE 8] Business impact");  BusinessImpact(config).run(df, X_test, proba, report["threshold"], report)
    build_metadata(config, report, info, thr_info, X_train, df)
    shutil.copytree(abs_path(config['reports']['figures_dir']), abs_path('static/figures'), dirs_exist_ok=True)

    m, d = report["test_metrics"], report["test_metrics_default_threshold"]
    log.info("=" * 64); log.info("  PIPELINE COMPLETE")
    log.info(f"  Final model      : {report['model_name']}  (calibrated: {info['calibration']['calibration_used']})")
    log.info(f"  ROC-AUC / PR-AUC : {m['roc_auc']} / {m['pr_auc']}")
    log.info(f"  @0.50  threshold : recall {d['recall']}  precision {d['precision']}  F1 {d['f1_score']}")
    log.info(f"  @{report['threshold']:.2f} (cost-opt): recall {m['recall']}  precision {m['precision']}  F1 {m['f1_score']}")
    log.info("=" * 64)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small grids / 3-fold CV for a quick run")
    run_pipeline(ap.parse_args().fast)
