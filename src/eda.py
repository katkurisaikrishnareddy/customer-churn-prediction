"""eda.py — Exploratory data analysis: figures + a JSON of headline insights for the report/slides."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import ChurnFeatureEngineer
from src.utils import PALETTE, abs_path, get_logger, save_json, set_plot_style


class EDA:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("EDA", config)
        self.plt = set_plot_style()
        self.fig_dir = abs_path(config["reports"]["figures_dir"])

    def _bar_rate(self, ax, df, col, title, order=None):
        g = df.groupby(col)["Churn_bin"].agg(["mean", "size"])
        if order: g = g.loc[order]
        bars = ax.bar(g.index.astype(str), g["mean"] * 100, color=PALETTE["stay"])
        ax.axhline(df["Churn_bin"].mean() * 100, ls="--", color=PALETTE["churn"], lw=1.2)
        for b, (m, n) in zip(bars, g.values): ax.text(b.get_x() + b.get_width() / 2, m * 100 + 1, f"{m:.0%}", ha="center", fontsize=9)
        ax.set(title=title, ylabel="Churn rate (%)"); ax.tick_params(axis="x", labelsize=8, rotation=10)
        return g["mean"].to_dict()

    def run(self, df_raw: pd.DataFrame) -> dict:
        plt = self.plt
        self.logger.info("=== Exploratory Data Analysis ===")
        df = ChurnFeatureEngineer(True, None).transform(df_raw)
        df["Churn_bin"] = (df_raw["Churn"] == "Yes").astype(int).values
        ins = {"rows": len(df), "overall_churn_rate": round(df.Churn_bin.mean(), 4)}

        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        vc = df_raw["Churn"].value_counts(); ax[0].pie(vc, labels=["No (stay)", "Yes (churn)"], autopct="%1.1f%%", colors=[PALETTE["stay"], PALETTE["churn"]], startangle=90)
        ax[0].set_title("Target distribution"); ax[0].grid(False)
        ax[1].bar(["Stay", "Churn"], vc.values, color=[PALETTE["stay"], PALETTE["churn"]]); ax[1].set_title("Class counts (imbalance ≈ 2.8 : 1)")
        fig.tight_layout(); fig.savefig(self.fig_dir / "eda_target.png"); plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        ins["by_contract"] = self._bar_rate(axes[0, 0], df, "Contract", "Churn rate by contract")
        ins["by_internet"] = self._bar_rate(axes[0, 1], df, "InternetService", "Churn rate by internet service")
        ins["by_payment"] = self._bar_rate(axes[0, 2], df, "PaymentMethod", "Churn rate by payment method")
        ins["by_tenure_bucket"] = self._bar_rate(axes[1, 0], df, "tenure_bucket", "Churn rate by tenure", ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"])
        ins["by_techsupport"] = self._bar_rate(axes[1, 1], df, "TechSupport", "Churn rate by tech support")
        ins["by_addons"] = self._bar_rate(axes[1, 2], df, "num_addon_services", "Churn rate by # add-on services")
        fig.suptitle("What drives churn? (dashed line = overall 26.5%)", fontweight="bold"); fig.tight_layout()
        fig.savefig(self.fig_dir / "eda_churn_drivers.png"); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
        for ax, col, ttl in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"], ["Tenure (months)", "Monthly charges ($)", "Total charges ($)"]):
            for v, c, l in [(0, PALETTE["stay"], "Stay"), (1, PALETTE["churn"], "Churn")]:
                ax.hist(df.loc[df.Churn_bin == v, col], bins=30, alpha=.6, color=c, label=l, density=True)
            ax.set(title=ttl, ylabel="Density"); ax.legend()
        fig.suptitle("Numeric feature distributions by outcome", fontweight="bold"); fig.tight_layout()
        fig.savefig(self.fig_dir / "eda_numeric.png"); plt.close(fig)

        num = ["tenure", "MonthlyCharges", "TotalCharges", "num_addon_services", "is_autopay", "is_month_to_month",
               "has_family", "internet_no_protection", "price_change_ratio", "SeniorCitizen", "Churn_bin"]
        corr = df[num].corr()
        fig, ax = plt.subplots(figsize=(8.5, 7)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
        ax.set_xticks(range(len(num))); ax.set_yticks(range(len(num))); ax.set_xticklabels(num, rotation=60, ha="right", fontsize=8); ax.set_yticklabels(num, fontsize=8)
        for i in range(len(num)):
            for j in range(len(num)): ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, shrink=.8); ax.set_title("Correlation matrix (numeric & engineered features)")
        fig.savefig(self.fig_dir / "eda_correlation.png"); plt.close(fig)
        ins["corr_with_churn"] = corr["Churn_bin"].drop("Churn_bin").round(3).sort_values().to_dict()
        ins["gender_churn"] = df_raw.assign(c=df.Churn_bin.values).groupby("gender")["c"].mean().round(4).to_dict()
        ins["month_to_month_share"] = round(float((df.Contract == "Month-to-month").mean()), 3)
        save_json(ins, f"{self.config['reports']['reports_dir']}/eda_insights.json")
        self.logger.info("EDA figures saved")
        return ins
