"""
feature_engineering.py — Stateless cleaning + feature engineering for the churn model.

Why a transformer class?
    Everything here lives INSIDE the sklearn Pipeline, so exactly the same code runs
    during training, cross-validation, batch scoring and the live web app. That removes
    "training/serving skew" (the #1 silent bug in deployed ML systems).

The transformer is stateless (nothing is learned from data), so it cannot leak
information from the test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
PROTECTION_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]

RAW_CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                   "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                   "PaperlessBilling", "PaymentMethod"]
RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

ENGINEERED_NUMERIC = ["num_addon_services", "num_protection_services", "avg_monthly_billed",
                      "price_change_ratio", "has_streaming", "has_internet",
                      "internet_no_protection", "is_autopay", "is_month_to_month", "has_family"]
ENGINEERED_CATEGORICAL = ["tenure_bucket"]

TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]


def clean_total_charges(df: pd.DataFrame) -> pd.Series:
    """TotalCharges arrives as text with blanks for brand-new customers (tenure == 0).

    Correct treatment: a customer with 0 months of tenure has been billed 0 so far.
    (The first version of this project imputed the *median*, which gave 11 new
    customers a fake lifetime spend of ~$1,400.)  Any other missing value is estimated
    as MonthlyCharges x tenure.
    """
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    total = total.where(~((tenure == 0) & total.isna()), 0.0)
    return total.fillna(monthly * tenure)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleans raw telco columns and (optionally) adds business-driven features.

    Parameters
    ----------
    add_features : add the engineered columns (set False for the ablation study).
    drop_columns : raw columns to remove (e.g. ``["gender"]`` — protected attribute with
                   no predictive signal).
    """

    def __init__(self, add_features: bool = True, drop_columns=None):
        self.add_features = add_features
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in ("customerID", "Churn"):
            if col in df.columns:
                df = df.drop(columns=col)

        for col in ("SeniorCitizen", "tenure", "MonthlyCharges"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["TotalCharges"] = clean_total_charges(df)

        if self.add_features:
            addon_yes = pd.concat([(df[c] == "Yes") for c in ADDON_SERVICES], axis=1)
            prot_yes = pd.concat([(df[c] == "Yes") for c in PROTECTION_SERVICES], axis=1)
            df["num_addon_services"] = addon_yes.sum(axis=1).astype(float)
            df["num_protection_services"] = prot_yes.sum(axis=1).astype(float)
            df["has_streaming"] = ((df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")).astype(float)
            df["has_internet"] = (df["InternetService"] != "No").astype(float)
            df["internet_no_protection"] = (
                (df["InternetService"] != "No")
                & (df["OnlineSecurity"] != "Yes")
                & (df["TechSupport"] != "Yes")
            ).astype(float)
            df["is_autopay"] = df["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(float)
            df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(float)
            df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(float)
            avg_billed = df["TotalCharges"] / df["tenure"].clip(lower=1)
            df["avg_monthly_billed"] = avg_billed
            # >1 means the customer now pays more than their historical average (a price rise)
            df["price_change_ratio"] = (df["MonthlyCharges"] / avg_billed.replace(0, np.nan)).fillna(1.0).clip(0, 5)
            df["tenure_bucket"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)

        if self.drop_columns:
            df = df.drop(columns=[c for c in self.drop_columns if c in df.columns])
        return df


def get_feature_lists(add_features: bool = True, drop_columns=None, drop_total_charges: bool = False):
    """Return (numeric_columns, categorical_columns) AFTER ChurnFeatureEngineer."""
    drop = set(drop_columns or [])
    if drop_total_charges:
        drop.add("TotalCharges")
    num = list(RAW_NUMERIC)
    cat = list(RAW_CATEGORICAL)
    if add_features:
        num += ENGINEERED_NUMERIC
        cat += ENGINEERED_CATEGORICAL
    return [c for c in num if c not in drop], [c for c in cat if c not in drop]
