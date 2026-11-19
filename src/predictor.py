"""predictor.py — ONE inference layer shared by the web form, the JSON API and CSV batch scoring.

Responsibilities
  * validate + normalise raw customer records (clear error messages, no silent fall-backs)
  * predict calibrated churn probability, risk tier and decision (probability >= business threshold)
  * explain each prediction: "factor-effect" = how much the churn probability changes versus
    replacing that factor with values drawn from real customers (model-agnostic, works for any model)
  * suggest retention actions from a transparent rule table
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
REQUIRED = ["SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService",
            *ADDONS, "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges"]
OPTIONAL = ["TotalCharges"]
NUMERIC = {"tenure": (0, 120), "MonthlyCharges": (0, 200), "TotalCharges": (0, 20000)}

# factor groups for the per-customer explanation (fields changed together to stay realistic)
FACTOR_GROUPS = {
    "Contract type": ["Contract"],
    "Tenure (time as customer)": ["tenure", "TotalCharges"],
    "Monthly charges": ["MonthlyCharges", "TotalCharges"],
    "Internet service type": ["InternetService"],
    "Online security & tech support": ["OnlineSecurity", "TechSupport"],
    "Online backup & device protection": ["OnlineBackup", "DeviceProtection"],
    "Streaming services": ["StreamingTV", "StreamingMovies"],
    "Phone service": ["PhoneService", "MultipleLines"],
    "Payment method": ["PaymentMethod"],
    "Paperless billing": ["PaperlessBilling"],
    "Family status (partner/dependents)": ["Partner", "Dependents"],
    "Senior citizen": ["SeniorCitizen"],
}
NON_ACTIONABLE = {"Senior citizen", "Family status (partner/dependents)"}


class ValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised consistency rules between related fields."""
    df = df.copy()
    no_net = df["InternetService"] == "No"
    for c in ADDONS:
        df.loc[no_net, c] = "No internet service"
        df.loc[~no_net & (df[c] == "No internet service"), c] = "No"
    no_phone = df["PhoneService"] == "No"
    df.loc[no_phone, "MultipleLines"] = "No phone service"
    df.loc[~no_phone & (df["MultipleLines"] == "No phone service"), "MultipleLines"] = "No"
    return df


class ChurnPredictor:
    def __init__(self, artifacts_dir: str = "artifacts"):
        a = Path(artifacts_dir)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = joblib.load(a / "churn_model.joblib")
        with open(a / "model_metadata.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.threshold = float(self.meta["threshold"])
        self.high = float(self.meta["high_risk_probability"])
        self.options = self.meta["schema"]["options"]
        self.reference = pd.read_csv(a / "reference_sample.csv")
        self.rng_seed = 7

    # ───────────────────────── validation ─────────────────────────
    def clean_record(self, rec: dict) -> tuple[dict, list[str]]:
        errors, notes, out = [], [], {}
        rec = {k: (v.strip() if isinstance(v, str) else v) for k, v in dict(rec).items()}
        for f in REQUIRED:
            if rec.get(f) in (None, ""):
                errors.append(f"Missing required field '{f}'")
        if errors:
            raise ValidationError(errors)
        sc = str(rec["SeniorCitizen"]).lower()
        if sc in ("1", "yes", "true", "1.0"): out["SeniorCitizen"] = 1
        elif sc in ("0", "no", "false", "0.0"): out["SeniorCitizen"] = 0
        else: errors.append("SeniorCitizen must be 0/1 (or Yes/No)")
        for f in ("tenure", "MonthlyCharges"):
            try:
                v = float(rec[f]); lo, hi = NUMERIC[f]
                if not (lo <= v <= hi) or np.isnan(v): errors.append(f"{f} must be between {lo} and {hi} (got {rec[f]})")
                out[f] = v
            except (TypeError, ValueError):
                errors.append(f"{f} must be a number (got '{rec[f]}')")
        for f, opts in self.options.items():
            if f in REQUIRED and f != "SeniorCitizen":
                v = rec[f]
                match = next((o for o in opts if str(o).lower() == str(v).lower()), None)
                if match is None: errors.append(f"{f} must be one of {opts} (got '{v}')")
                out[f] = match
        if errors:
            raise ValidationError(errors)
        tc = rec.get("TotalCharges")
        if tc in (None, ""):
            out["TotalCharges"] = round(out["tenure"] * out["MonthlyCharges"], 2)
            notes.append("TotalCharges not supplied -> estimated as tenure x MonthlyCharges")
        else:
            try:
                out["TotalCharges"] = float(tc)
                lo, hi = NUMERIC["TotalCharges"]
                if not (lo <= out["TotalCharges"] <= hi): raise ValidationError([f"TotalCharges must be between {lo} and {hi}"])
            except (TypeError, ValueError):
                raise ValidationError([f"TotalCharges must be a number (got '{tc}')"])
        fixed = _normalize_frame(pd.DataFrame([out])).iloc[0].to_dict()
        for k in ("InternetService", "PhoneService"):
            pass
        if any(fixed[c] != out[c] for c in ADDONS + ["MultipleLines"]):
            notes.append("Add-on fields were adjusted to be consistent with Internet/Phone service")
        return fixed, notes

    # ───────────────────────── prediction ─────────────────────────
    def tier(self, p: float) -> str:
        return "High" if p >= self.high else ("Medium" if p >= self.threshold else "Low")

    def predict_proba_frame(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df)[:, 1]

    def predict_one(self, rec: dict, explain: bool = True) -> dict:
        clean, notes = self.clean_record(rec)
        p = float(self.predict_proba_frame(pd.DataFrame([clean]))[0])
        res = {"churn_probability": round(p * 100, 1), "probability": round(p, 4), "risk_level": self.tier(p) + " Risk",
               "tier": self.tier(p), "prediction": int(p >= self.threshold),
               "label": "Likely to churn — retention action recommended" if p >= self.threshold else "Likely to stay",
               "threshold": self.threshold, "notes": notes, "input": clean}
        if explain:
            drivers = self.explain(clean, p)
            res["drivers_up"] = [d for d in drivers if d["impact_pp"] > 0.5][:4]
            res["drivers_down"] = [d for d in drivers[::-1] if d["impact_pp"] < -0.5][:3]
            res["actions"] = self.recommend(clean, res["drivers_up"], res["tier"])
        return res

    def predict_batch(self, df: pd.DataFrame):
        good_idx, rows, bad = [], [], []
        for i, rec in enumerate(df.to_dict("records")):
            try:
                clean, _ = self.clean_record(rec); rows.append(clean); good_idx.append(i)
            except ValidationError as e:
                bad.append({"row": i + 2, "errors": e.errors})   # +2 = spreadsheet row incl. header
        out = df.copy(); out["churn_probability"] = np.nan; out["risk_tier"] = ""; out["flag_for_retention"] = ""
        if rows:
            p = self.predict_proba_frame(pd.DataFrame(rows))
            out.loc[good_idx, "churn_probability"] = np.round(p, 4)
            out.loc[good_idx, "risk_tier"] = [self.tier(x) for x in p]
            out.loc[good_idx, "flag_for_retention"] = np.where(p >= self.threshold, "Yes", "No")
        return out, bad

    # ───────────────────────── explanation ─────────────────────────
    def explain(self, clean: dict, p: float) -> list[dict]:
        ref = self.reference
        has_net = clean["InternetService"] != "No"
        results = []
        for factor, cols in FACTOR_GROUPS.items():
            pool = ref
            if factor in ("Online security & tech support", "Online backup & device protection", "Streaming services"):
                if not has_net: continue
                pool = ref[ref["InternetService"] != "No"]
            if factor == "Phone service" or factor == "Internet service type":
                pool = ref
            samp = pool.sample(min(60, len(pool)), random_state=self.rng_seed).reset_index(drop=True)
            cf = pd.DataFrame([clean] * len(samp))
            if factor == "Tenure (time as customer)":
                cf["tenure"] = samp["tenure"]; cf["TotalCharges"] = (samp["tenure"] * cf["MonthlyCharges"]).round(2)
            elif factor == "Monthly charges":
                cf["MonthlyCharges"] = samp["MonthlyCharges"]; cf["TotalCharges"] = (cf["tenure"] * samp["MonthlyCharges"]).round(2)
            else:
                for c in cols: cf[c] = samp[c]
            cf = _normalize_frame(cf)
            q = self.predict_proba_frame(cf).mean()
            results.append({"factor": factor, "value": self._describe(factor, clean), "impact_pp": round((p - q) * 100, 1)})
        return sorted(results, key=lambda d: -d["impact_pp"])

    @staticmethod
    def _describe(factor: str, c: dict) -> str:
        if factor == "Contract type": return c["Contract"]
        if factor.startswith("Tenure"): return f"{int(c['tenure'])} months"
        if factor == "Monthly charges": return f"${c['MonthlyCharges']:.2f}/month"
        if factor == "Internet service type": return c["InternetService"]
        if factor == "Online security & tech support": return f"Security: {c['OnlineSecurity']}, Tech support: {c['TechSupport']}"
        if factor == "Online backup & device protection": return f"Backup: {c['OnlineBackup']}, Protection: {c['DeviceProtection']}"
        if factor == "Streaming services": return f"TV: {c['StreamingTV']}, Movies: {c['StreamingMovies']}"
        if factor == "Phone service": return f"Phone: {c['PhoneService']}, Multiple lines: {c['MultipleLines']}"
        if factor == "Payment method": return c["PaymentMethod"]
        if factor == "Paperless billing": return c["PaperlessBilling"]
        if factor.startswith("Family"): return f"Partner: {c['Partner']}, Dependents: {c['Dependents']}"
        return "Yes" if c["SeniorCitizen"] else "No"

    @staticmethod
    def recommend(c: dict, drivers_up: list[dict], tier: str) -> list[str]:
        if tier == "Low":
            return ["No retention offer needed — keep the normal customer-care journey."]
        top = {d["factor"] for d in drivers_up if d["factor"] not in NON_ACTIONABLE}
        acts = []
        if "Contract type" in top and c["Contract"] == "Month-to-month":
            acts.append("Offer a 1- or 2-year contract with a loyalty discount (month-to-month customers churn most).")
        if "Tenure (time as customer)" in top and c["tenure"] < 12:
            acts.append("Start a 90-day onboarding check-in call — early-life customers are the most fragile.")
        if "Monthly charges" in top or ("Internet service type" in top and c["InternetService"] == "Fiber optic"):
            acts.append("Review plan fit: offer a price-lock or bundle to reduce the monthly bill; check fiber service quality.")
        if "Online security & tech support" in top:
            acts.append("Offer a free 3-month Tech Support / Online Security trial (customers without them churn more).")
        if "Online backup & device protection" in top:
            acts.append("Offer a discounted device-protection / backup bundle to deepen the relationship.")
        if "Payment method" in top and c["PaymentMethod"] == "Electronic check":
            acts.append("Give a small bill credit for switching to automatic payment.")
        return acts or ["Proactive courtesy call to understand satisfaction before the customer leaves."]
