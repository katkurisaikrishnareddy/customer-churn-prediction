"""utils.py — config loading, logging and small shared helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str, config: dict | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = (config or {}).get("logging", {}).get("level", "INFO")
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if config:
        log_file = config.get("logging", {}).get("log_file", "logs/pipeline.log")
        log_path = ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_dirs(*dirs):
    for d in dirs:
        p = Path(d)
        if not p.is_absolute():
            p = ROOT / p
        p.mkdir(parents=True, exist_ok=True)


def abs_path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path) -> None:
    path = abs_path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(abs_path(str(path)), "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def environment_info() -> dict:
    import sklearn, pandas, numpy
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "pandas": pandas.__version__, "numpy": numpy.__version__}


def set_plot_style():
    """One consistent look for every figure in the project."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 10,
        "axes.prop_cycle": matplotlib.cycler(color=["#4f46e5", "#ef4444", "#10b981", "#f59e0b", "#06b6d4", "#8b5cf6"]),
    })
    return plt


PALETTE = {"churn": "#ef4444", "stay": "#4f46e5", "good": "#10b981", "warn": "#f59e0b", "grey": "#94a3b8"}
