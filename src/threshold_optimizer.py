"""threshold_optimizer.py — Stage 5: choose the decision threshold from BUSINESS economics, not 0.5.

Economics of one retention campaign
    contact a customer      -> pay  offer_cost
    that customer churns    -> with prob. success_rate we keep  MonthlyCharges x horizon  of revenue
    net benefit(customer)   =  y * success_rate * MonthlyCharges * horizon  -  offer_cost
Flag a customer when  p * success_rate * V  >  offer_cost   i.e.  p > offer_cost / (success_rate * V)
(the "theoretical threshold").  We also search empirically on OUT-OF-FOLD predictions of the
training set and smooth the curve so we do not chase noise.  The test set is never used here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

from src.utils import get_logger


def net_benefit(y_true, proba, monthly, threshold, biz: dict) -> float:
    flag = np.asarray(proba) >= threshold
    value = np.asarray(monthly) * biz["value_horizon_months"] * biz["offer_success_rate"]
    gain = np.where(np.asarray(y_true) == 1, value, 0.0) - biz["retention_offer_cost"]
    return float(gain[flag].sum())


def benefit_curve(y_true, proba, monthly, biz, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2) if grid is None else grid
    y = np.asarray(y_true)
    rows = []
    for t in grid:
        pred = (np.asarray(proba) >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        rows.append({"threshold": float(t), "flagged": int(pred.sum()), "flagged_pct": pred.mean(),
                     "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                     "f1": fbeta_score(y, pred, beta=1, zero_division=0),
                     "f2": fbeta_score(y, pred, beta=2, zero_division=0),
                     "net_benefit": net_benefit(y, proba, monthly, t, biz)})
    return pd.DataFrame(rows)


class ThresholdOptimizer:
    def __init__(self, config: dict):
        self.config = config
        self.biz = config["business"]
        self.logger = get_logger("ThresholdOptimizer", config)

    def theoretical_threshold(self, avg_monthly: float) -> float:
        v = avg_monthly * self.biz["value_horizon_months"] * self.biz["offer_success_rate"]
        return float(min(1.0, self.biz["retention_offer_cost"] / v))

    def sensitivity(self, y, proba, monthly) -> list[dict]:
        out = []
        for c in self.biz["sensitivity"]["offer_cost"]:
            for s in self.biz["sensitivity"]["success_rate"]:
                b = {**self.biz, "retention_offer_cost": c, "offer_success_rate": s}
                curve = benefit_curve(y, proba, monthly, b)
                sm = curve["net_benefit"].rolling(5, center=True, min_periods=1).mean()
                i = int(sm.idxmax())
                out.append({"offer_cost": c, "success_rate": s, "best_threshold": float(curve.loc[i, "threshold"]),
                            "net_benefit_per_1000": float(curve.loc[i, "net_benefit"] / len(y) * 1000)})
        return out

    def run(self, y_train, oof_proba, monthly_train) -> dict:
        self.logger.info("=== Threshold Optimization (out-of-fold, training data only) ===")
        curve = benefit_curve(y_train, oof_proba, monthly_train, self.biz)
        smooth = curve["net_benefit"].rolling(5, center=True, min_periods=1).mean()
        best = float(curve.loc[int(smooth.idxmax()), "threshold"])
        theo = self.theoretical_threshold(float(np.mean(monthly_train)))
        res = {
            "cost_optimal_threshold": best,
            "theoretical_threshold": round(theo, 3),
            "f1_optimal_threshold": float(curve.loc[curve["f1"].idxmax(), "threshold"]),
            "f2_optimal_threshold": float(curve.loc[curve["f2"].idxmax(), "threshold"]),
            "default_threshold": 0.5,
            "net_benefit_at_default_per_1000": float(curve.loc[curve["threshold"] == 0.5, "net_benefit"].iloc[0] / len(y_train) * 1000),
            "net_benefit_at_optimal_per_1000": float(curve.loc[curve["threshold"] == best, "net_benefit"].iloc[0] / len(y_train) * 1000),
            "sensitivity": self.sensitivity(y_train, oof_proba, monthly_train),
        }
        self.logger.info(f"cost-optimal={best:.2f} theoretical={theo:.2f} F1-opt={res['f1_optimal_threshold']:.2f} "
                         f"F2-opt={res['f2_optimal_threshold']:.2f}")
        return res, curve
