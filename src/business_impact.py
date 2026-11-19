"""business_impact.py — Stage 8: turn probabilities into decisions (segments, risk tiers, at-risk list)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import ChurnFeatureEngineer
from src.utils import PALETTE, abs_path, get_logger, save_json, set_plot_style


def risk_tier(p: float, thr: float, high: float) -> str:
    return "High" if p >= high else ("Medium" if p >= thr else "Low")


class BusinessImpact:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("BusinessImpact", config)
        self.plt = set_plot_style()
        self.fig_dir = abs_path(config["reports"]["figures_dir"])

    # ---------- segmentation ----------
    def segment(self, df_raw: pd.DataFrame) -> dict:
        cfg = self.config["segmentation"]
        fe = ChurnFeatureEngineer(True, None).transform(df_raw)
        fe["Churn"] = (df_raw["Churn"] == "Yes").astype(int).values
        Z = StandardScaler().fit_transform(fe[cfg["features"]])
        scores = {}
        for k in cfg["k_range"]:
            km = KMeans(k, n_init=10, random_state=self.config["model"]["random_state"]).fit(Z)
            scores[k] = float(silhouette_score(Z, km.labels_, sample_size=3000, random_state=1))
        k = max(scores, key=scores.get)
        km = KMeans(k, n_init=10, random_state=self.config["model"]["random_state"]).fit(Z)
        fe["segment"] = km.labels_
        prof = fe.groupby("segment").agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"), avg_tenure=("tenure", "mean"),
                                         avg_monthly=("MonthlyCharges", "mean"), avg_addons=("num_addon_services", "mean"),
                                         pct_month_to_month=("is_month_to_month", "mean")).round(3)
        order = prof["churn_rate"].sort_values(ascending=False).index.tolist()
        rename = {old: f"S{i + 1}" for i, old in enumerate(order)}         # S1 = highest churn
        prof = prof.rename(index=rename).sort_index(); fe["segment"] = fe["segment"].map(rename)
        plt = self.plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
        for s, g in fe.sample(2500, random_state=2).groupby("segment"):
            a1.scatter(g["tenure"], g["MonthlyCharges"], s=9, alpha=.55, label=s)
        a1.set(xlabel="Tenure (months)", ylabel="Monthly charges ($)", title=f"Customer segments (k={k})"); a1.legend(markerscale=2)
        a2.bar(prof.index, prof["churn_rate"] * 100, color=PALETTE["stay"]); a2.axhline(fe.Churn.mean() * 100, ls="--", color=PALETTE["churn"], label="Overall")
        a2.set(ylabel="Churn rate (%)", title="Churn rate by segment"); a2.legend()
        for i, v in enumerate(prof["churn_rate"]): a2.text(i, v * 100 + 1, f"{v:.0%}", ha="center")
        fig.tight_layout(); fig.savefig(self.fig_dir / "customer_segments.png"); plt.close(fig)
        out = {"k": int(k), "silhouette_by_k": scores, "profile": prof.reset_index().to_dict("records")}
        self.logger.info(f"KMeans k={k} (silhouette {scores[k]:.3f})\n{prof}")
        return out

    # ---------- at-risk customers ----------
    def at_risk_list(self, X_test, proba, thr) -> pd.DataFrame:
        high = self.config["business"]["high_risk_probability"]
        d = X_test[["customerID", "tenure", "Contract", "InternetService", "PaymentMethod", "MonthlyCharges"]].copy()
        d["churn_probability"] = np.round(proba, 4)
        d["risk_tier"] = [risk_tier(p, thr, high) for p in proba]
        d["monthly_revenue_at_risk"] = np.where(d["risk_tier"] != "Low", d["MonthlyCharges"], 0.0)
        return d.sort_values("churn_probability", ascending=False)

    def run(self, df_raw, X_test, proba, thr, report) -> dict:
        self.logger.info("=== Business Impact Analysis ===")
        seg = self.segment(df_raw)
        lst = self.at_risk_list(X_test, proba, thr)
        lst.to_csv(abs_path(self.config["reports"]["reports_dir"]) / "at_risk_customers.csv", index=False)
        tiers = lst["risk_tier"].value_counts().to_dict()
        summary = {"segments": seg, "risk_tier_counts_test": tiers,
                   "monthly_revenue_flagged": round(float(lst["monthly_revenue_at_risk"].sum()), 2)}
        save_json(summary, f"{self.config['reports']['reports_dir']}/business_impact.json")
        self.plot_business(report)
        return summary

    def plot_business(self, report):
        plt = self.plt
        b = report["business_test_net_benefit"]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        names = list(b); vals = list(b.values())
        ax.bar(range(len(names)), vals, color=[PALETTE["grey"], PALETTE["warn"], PALETTE["churn"], PALETTE["good"]][:len(names)])
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(" @ ", "\n@ ") for n in names], fontsize=8.5)
        for i, v in enumerate(vals): ax.text(i, v + max(vals) * .02, f"${v:,.0f}", ha="center", fontweight="bold")
        ax.set(ylabel="Net benefit on 1,409 test customers ($)", title="Retention campaign value by targeting strategy")
        fig.savefig(self.fig_dir / "business_value.png"); plt.close(fig)
