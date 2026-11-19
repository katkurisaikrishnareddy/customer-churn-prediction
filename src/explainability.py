"""explainability.py — Stage 7: global SHAP explanation of the final model type (training-time only)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.utils import abs_path, get_logger, save_json, set_plot_style


class Explainer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("Explainability", config)
        self.plt = set_plot_style()

    @staticmethod
    def _unwrap(model):
        """Return a plain fitted Pipeline (first fold of the calibrated ensemble if wrapped)."""
        if hasattr(model, "calibrated_classifiers_"):
            return model.calibrated_classifiers_[0].estimator
        return model

    def run(self, model, X_train, X_test) -> dict:
        try:
            import shap
        except Exception:
            self.logger.warning("shap not installed -> skipping SHAP (pip install shap)"); return {}
        pipe = self._unwrap(model)
        fe, pre, clf = pipe.named_steps["fe"], pipe.named_steps["pre"], pipe.named_steps["clf"]
        n = self.config["explainability"]["shap_sample_size"]
        bg = pre.transform(fe.transform(X_train.sample(min(300, len(X_train)), random_state=1)))
        smp = X_test.sample(min(n, len(X_test)), random_state=1)
        Xs = pd.DataFrame(pre.transform(fe.transform(smp)), columns=pre.get_feature_names_out())
        try:
            if hasattr(clf, "coef_"):
                sv = shap.LinearExplainer(clf, pd.DataFrame(bg, columns=Xs.columns)).shap_values(Xs)
            else:
                sv = shap.TreeExplainer(clf).shap_values(Xs)
            sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv)
        except Exception as e:
            self.logger.warning(f"SHAP failed ({e}); skipping"); return {}
        plt = self.plt
        fig_dir = abs_path(self.config["reports"]["figures_dir"])
        plt.figure(figsize=(8, 6)); shap.summary_plot(sv, Xs, show=False, max_display=15, plot_size=None)
        plt.title("SHAP summary — what pushes churn risk up or down"); plt.savefig(fig_dir / "shap_summary.png"); plt.close()
        plt.figure(figsize=(7, 5)); shap.summary_plot(sv, Xs, plot_type="bar", show=False, max_display=15, plot_size=None)
        plt.title("Mean |SHAP| — global feature importance"); plt.savefig(fig_dir / "shap_bar.png"); plt.close()
        imp = pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values(ascending=False)
        out = {k: round(float(v), 4) for k, v in imp.head(15).items()}
        save_json(out, f"{self.config['reports']['reports_dir']}/shap_importance.json")
        self.logger.info(f"SHAP done. Top drivers: {list(out)[:5]}")
        return out
