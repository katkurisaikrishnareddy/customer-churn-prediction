import numpy as np
from src.threshold_optimizer import ThresholdOptimizer, benefit_curve, net_benefit

BIZ = {"retention_offer_cost": 50.0, "offer_success_rate": 0.3, "value_horizon_months": 12, "sensitivity": {"offer_cost": [50], "success_rate": [0.3]}}


def test_contact_nobody_is_zero():
    y = np.array([1, 0, 1]); p = np.array([.9, .1, .8])
    assert net_benefit(y, p, np.array([70.] * 3), 1.01, BIZ) == 0.0


def test_net_benefit_arithmetic():
    # one churner (value 100*12*0.3=360) flagged -> 360-50 ; one non-churner flagged -> -50
    assert net_benefit(np.array([1, 0]), np.array([.9, .9]), np.array([100., 100.]), 0.5, BIZ) == (360 - 50) + (-50)


def test_theoretical_threshold_formula(config):
    opt = ThresholdOptimizer({**config, "business": BIZ})
    assert abs(opt.theoretical_threshold(65.0) - 50 / (65 * 12 * 0.3)) < 1e-9


def test_optimizer_prefers_low_threshold_when_misses_are_costly():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.27).astype(int)
    p = np.clip(0.25 + 0.4 * y + rng.normal(0, 0.18, 3000), 0.01, 0.99)
    curve = benefit_curve(y, p, np.full(3000, 65.0), BIZ)
    best = curve.loc[curve.net_benefit.idxmax(), "threshold"]
    assert best < 0.5
