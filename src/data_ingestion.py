"""data_ingestion.py — Stage 1: load the raw dataset (local copy first, download as fallback)."""
from __future__ import annotations

import pandas as pd

from src.utils import abs_path, ensure_dirs, file_sha256, get_logger, load_config


class DataIngestion:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("DataIngestion", config)
        ensure_dirs("data", "artifacts", "logs", "reports/figures")
        self.raw_path = abs_path(config["data"]["raw_path"])

    def download_data(self) -> pd.DataFrame:
        url = self.config["data"]["url"]
        self.logger.info(f"Downloading dataset from: {url}")
        try:
            df = pd.read_csv(url)
        except Exception as e:  # network down, URL moved, ...
            self.logger.error(f"Data download failed: {e}")
            raise
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.raw_path, index=False)
        self.logger.info(f"Saved to {self.raw_path} | shape={df.shape}")
        return df

    def load_data(self) -> pd.DataFrame:
        if self.raw_path.exists():
            self.logger.info(f"Loading local dataset: {self.raw_path}")
            return pd.read_csv(self.raw_path)
        self.logger.info("No local copy found -> downloading")
        return self.download_data()

    def run(self) -> pd.DataFrame:
        self.logger.info("=== Data Ingestion Started ===")
        df = self.load_data()
        expected = self.config["data"].get("expected_rows")
        if expected and len(df) != expected:
            self.logger.warning(f"Row count {len(df)} differs from expected {expected}")
        self.logger.info(f"Ingestion complete | rows={df.shape[0]} cols={df.shape[1]} "
                         f"| sha256={file_sha256(self.raw_path)[:12]}...")
        return df


if __name__ == "__main__":
    print(DataIngestion(load_config()).run().head())
