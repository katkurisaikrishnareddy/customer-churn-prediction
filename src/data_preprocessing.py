"""data_preprocessing.py — Stage 3: clean target, split FIRST, build the preprocessing recipe.

KEY FIX vs. v1:  the original code fitted the StandardScaler on the *entire* dataset and only
then split into train/test. The test set's mean/std therefore leaked into training (data
leakage). Here nothing is fitted until after the split, and the scaler / encoder live inside
the sklearn Pipeline so they are re-fitted on the training portion of every CV fold.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import get_feature_lists
from src.utils import abs_path, ensure_dirs, get_logger, load_config


def build_preprocessor(config: dict) -> ColumnTransformer:
    feat = config["features"]
    num, cat = get_feature_lists(feat["add_engineered"], feat.get("drop_columns"))
    return ColumnTransformer(
        [("num", StandardScaler(), num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        verbose_feature_names_out=False,
    )


class DataPreprocessing:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("DataPreprocessing", config)
        self.target = config["target_column"]
        ensure_dirs(config["data"]["processed_dir"])

    def encode_target(self, df: pd.DataFrame) -> pd.Series:
        y = df[self.target].map({"Yes": 1, "No": 0})
        if y.isna().any():
            raise ValueError("Target contains labels other than Yes/No")
        self.logger.info(f"Target encoded Yes=1/No=0 | churn rate = {y.mean():.2%}")
        return y.astype(int)

    def split_data(self, df: pd.DataFrame):
        cfg = self.config["model"]
        y = self.encode_target(df)
        X = df.drop(columns=[self.target])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y)
        self.logger.info(f"Stratified split | train={len(X_train)} test={len(X_test)} "
                         f"| churn train={y_train.mean():.2%} test={y_test.mean():.2%}")
        return X_train, X_test, y_train, y_test

    def run(self, df: pd.DataFrame):
        self.logger.info("=== Data Preprocessing Started ===")
        X_train, X_test, y_train, y_test = self.split_data(df)
        out = abs_path(self.config["data"]["processed_dir"])
        X_train.assign(Churn=y_train).to_csv(out / "train.csv", index=False)
        X_test.assign(Churn=y_test).to_csv(out / "test.csv", index=False)
        self.logger.info(f"Saved train/test splits to {out}  (raw columns kept; scaling/encoding happens inside the Pipeline)")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    from src.data_validation import DataValidation
    cfg = load_config()
    d = DataIngestion(cfg).run()
    DataValidation(cfg).run(d)
    Xtr, *_ = DataPreprocessing(cfg).run(d)
    print(Xtr.shape)
