"""data_validation.py — Stage 2: automated data-quality gate.

Hard ERRORS stop the pipeline (bad data must never reach training).
WARNINGS are recorded in reports/data_quality_report.json for the project report.
"""
from __future__ import annotations

import pandas as pd

from src.utils import get_logger, load_config, save_json


class DataValidation:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("DataValidation", config)
        self.cfg = config["validation"]
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[dict] = []

    def _record(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})
        (self.logger.info if passed else self.logger.warning)(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    def check_required_columns(self, df):
        missing = [c for c in self.cfg["required_columns"] if c not in df.columns]
        if missing:
            self.errors.append(f"Missing required columns: {missing}")
        self._record("required columns", not missing, f"missing={missing}" if missing else f"all {len(self.cfg['required_columns'])} present")

    def check_null_percentage(self, df):
        mx = self.cfg["max_null_percent"]
        pct = (df.isnull().mean() * 100)
        high = pct[pct > mx].to_dict()
        if high:
            self.warnings.append(f"Columns above {mx}% nulls: {high}")
        self._record("null percentage", not high, f"threshold={mx}%")

    def check_hidden_missing(self, df):
        """Blank strings hide inside text columns (TotalCharges has 11 of them)."""
        blanks = {}
        for col in df.columns:
            s = df[col]
            if not pd.api.types.is_numeric_dtype(s):
                n = int((s.astype(str).str.strip() == "").sum())
                if n:
                    blanks[col] = n
        if blanks:
            self.warnings.append(f"Blank-string values found (treated as missing): {blanks}")
        self._record("hidden blank values", not blanks, f"{blanks}" if blanks else "none")
        return blanks

    def check_target(self, df):
        t = self.config["target_column"]
        if t not in df.columns:
            self.errors.append(f"Target '{t}' missing")
            self._record("target column", False)
            return
        dist = df[t].value_counts(normalize=True).round(4).to_dict()
        bad = set(df[t].dropna().unique()) - set(self.cfg["allowed_values"][t])
        if bad:
            self.errors.append(f"Unexpected target labels: {bad}")
        if min(dist.values()) < 0.10:
            self.warnings.append(f"Severe class imbalance: {dist}")
        self._record("target column", not bad, f"distribution={dist}")

    def check_allowed_values(self, df):
        problems = {}
        for col, allowed in self.cfg["allowed_values"].items():
            if col in df.columns:
                bad = set(df[col].dropna().astype(str).unique()) - set(map(str, allowed))
                if bad:
                    problems[col] = sorted(bad)
        if problems:
            self.errors.append(f"Unexpected category values: {problems}")
        self._record("category values", not problems, f"{problems}" if problems else "all within expected sets")

    def check_numeric_ranges(self, df):
        problems = {}
        for col, (lo, hi) in self.cfg["numeric_ranges"].items():
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce")
                out = int(((v < lo) | (v > hi)).sum())
                if out:
                    problems[col] = out
        if problems:
            self.errors.append(f"Values out of range: {problems}")
        self._record("numeric ranges", not problems, f"{problems}" if problems else "ok")

    def check_total_charges(self, df):
        if "TotalCharges" not in df.columns or "tenure" not in df.columns:
            return   # missing columns are already reported by check_required_columns
        v = pd.to_numeric(df["TotalCharges"], errors="coerce")
        n_bad = int(v.isna().sum())
        explained = int(((v.isna()) & (pd.to_numeric(df["tenure"], errors="coerce") == 0)).sum())
        if n_bad:
            self.warnings.append(f"{n_bad} non-numeric TotalCharges ({explained} belong to tenure=0 customers -> set to 0)")
        self._record("TotalCharges numeric", n_bad == 0, f"{n_bad} bad, {explained} explained by tenure=0")

    def check_duplicates(self, df):
        idc = self.config["id_column"]
        dup_rows = int(df.duplicated().sum())
        dup_ids = int(df[idc].duplicated().sum()) if idc in df.columns else 0
        if dup_rows or dup_ids:
            self.warnings.append(f"duplicates: rows={dup_rows}, ids={dup_ids}")
        self._record("duplicates", dup_rows == 0 and dup_ids == 0, f"rows={dup_rows} ids={dup_ids}")

    def run(self, df: pd.DataFrame) -> bool:
        self.logger.info("=== Data Validation Started ===")
        self.check_required_columns(df)
        self.check_null_percentage(df)
        blanks = self.check_hidden_missing(df)
        self.check_target(df)
        self.check_allowed_values(df)
        self.check_numeric_ranges(df)
        self.check_total_charges(df)
        self.check_duplicates(df)

        report = {"rows": int(len(df)), "columns": int(df.shape[1]), "checks": self.checks,
                  "errors": self.errors, "warnings": self.warnings, "hidden_blanks": blanks,
                  "passed": not self.errors}
        save_json(report, f"{self.config['reports']['reports_dir']}/data_quality_report.json")
        self.logger.info(f"Validation summary: {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        if self.errors:
            raise ValueError("Data validation failed: " + " | ".join(self.errors))
        self.logger.info("=== Data Validation Passed ===")
        return True


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    cfg = load_config()
    DataValidation(cfg).run(DataIngestion(cfg).run())
