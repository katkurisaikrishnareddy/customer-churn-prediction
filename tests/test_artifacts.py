import json
from src.utils import abs_path


def test_report_has_expected_keys_and_sane_values(config):
    r = json.load(open(abs_path(config["model"]["report_path"])))
    for k in ("model_name", "test_metrics", "confusion_matrix", "threshold_analysis", "bootstrap_95ci_at_optimal_threshold", "fairness_audit"):
        assert k in r
    m = r["test_metrics"]
    assert 0.80 < m["roc_auc"] < 0.90 and 0 < m["recall"] <= 1
    cm = r["confusion_matrix"]
    assert sum(cm.values()) == r["n_test"]


def test_metadata_threshold_matches_report(config):
    meta = json.load(open(abs_path(config["model"]["metadata_path"])))
    rep = json.load(open(abs_path(config["model"]["report_path"])))
    assert meta["threshold"] == rep["threshold"]
