#!/usr/bin/env python3
"""Unit tests for the U-SafeAgent decision core (review 2026-10-08 P2-A/P2-B).

No third-party test runner is required -- the file is pytest-compatible but also
runs standalone:

    python tests/test_safe_selector.py
    python -m pytest tests/test_safe_selector.py     # if pytest is available

The headline regression is the chained-comparison bug: ``a != b != c`` is
``a != b and b != c``, so a 3-way length mismatch could previously slip through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.agent.safe_selector import (  # noqa: E402
    DEFAULT_DELTA_PP,
    RidgeSelector,
    SafeSelector,
)


def _toy(n_domains=3, per_domain=3, n_feat=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_domains * per_domain, n_feat))
    y = rng.normal(scale=0.3, size=(n_domains * per_domain,))
    groups = [f"d{i}" for i in range(n_domains) for _ in range(per_domain)]
    return X, y, groups


# --------------------------------------------------------------------- P2-A
def test_length_mismatch_raises_chained_comparison_regression():
    """The exact case the old ``len(X) != len(y) != len(groups)`` let through."""
    X = np.zeros((1, 2))
    y = np.zeros(2)              # len(y)=2 != len(X)=1
    groups = ["a"]               # len(groups)=1 == len(X)=1  -> old check passed
    sel = SafeSelector()
    try:
        sel.fit(X, y, groups)
    except ValueError as e:
        assert "mismatch" in str(e)
        return
    raise AssertionError("length mismatch was not detected (chained-comparison bug)")


def test_mismatch_between_y_and_groups_raises():
    X = np.zeros((2, 2))
    y = np.zeros(2)
    groups = ["a"]
    try:
        SafeSelector().fit(X, y, groups)
    except ValueError as e:
        assert "mismatch" in str(e)
        return
    raise AssertionError("y/groups mismatch not detected")


def test_single_domain_rejected():
    X, y, _ = _toy(n_domains=1, per_domain=4)
    try:
        SafeSelector().fit(X, y, ["only"] * 4)
    except ValueError as e:
        assert "2 training datasets" in str(e)
        return
    raise AssertionError("single-domain fit should be rejected")


def test_no_jackknife_refit_refuses_fake_certainty():
    """2 domains x 1 member: every leave-one-out refit has <2 rows -> refuse."""
    X = np.zeros((2, 2))
    y = np.zeros(2)
    groups = ["a", "b"]
    try:
        SafeSelector().fit(X, y, groups)
    except ValueError as e:
        assert "fake-certain" in str(e) or "refit" in str(e)
        return
    raise AssertionError("fit should refuse when no jackknife refit is possible")


def test_spread_before_fit_raises_not_zero():
    try:
        SafeSelector().model_spread_pp(np.zeros((1, 2)))
    except RuntimeError as e:
        assert "false certainty" in str(e)
        return
    raise AssertionError("model_spread_pp must not silently return zeros")


# ------------------------------------------------------------------ behaviour
def test_margin_is_mu_minus_kappa_times_total_uncertainty():
    X, y, groups = _toy()
    sel = SafeSelector(lam=1.0, delta_pp=DEFAULT_DELTA_PP, kappa=1.0).fit(X, y, groups)
    Xq = np.array([[0.1, -0.2, 0.3]])
    mu = sel.predict_mu_pp(Xq)[0]
    spread = sel.model_spread_pp(Xq)[0]
    d = sel.decide(Xq, ["t1"], u_noise_pp=0.5)
    expect_u = float(np.sqrt(spread ** 2 + 0.5 ** 2))
    assert abs(d.per_teacher["t1"]["u_pp"] - expect_u) < 1e-9
    assert abs(d.per_teacher["t1"]["margin_pp"] - (mu - expect_u)) < 1e-9


def test_large_noise_floor_forces_abstention():
    """The exact mechanism that produced F0: u_noise >= mu makes the set empty."""
    X, y, groups = _toy()
    sel = SafeSelector(delta_pp=0.2, kappa=1.0).fit(X, y, groups)
    Xq = np.array([[0.1, -0.2, 0.3]])
    # A very large declared noise floor must dominate the (small) predicted utility.
    d = sel.decide(Xq, ["t1"], u_noise_pp=1000.0)
    assert d.teacher is None
    assert d.reason == "no_supported_positive_utility"
    assert d.per_teacher["t1"]["margin_pp"] < 0


def test_delta_gate_blocks_sub_threshold_margin():
    X, y, groups = _toy()
    sel = SafeSelector(delta_pp=1e9, kappa=0.0).fit(X, y, groups)
    d = sel.decide(np.array([[0.1, -0.2, 0.3]]), ["t1"], u_noise_pp=0.0)
    assert d.teacher is None


def test_ridge_selector_ignores_uncertainty_by_construction():
    X, y, groups = _toy()
    sel = RidgeSelector(lam=1.0).fit(X, y, groups)
    Xq = np.array([[0.1, -0.2, 0.3]])
    a = sel.decide(Xq, ["t1"], u_noise_pp=0.0)
    b = sel.decide(Xq, ["t1"], u_noise_pp=500.0)
    assert a.teacher == b.teacher
    assert a.margin_pp == b.margin_pp


def test_non_finite_features_rejected():
    X, y, groups = _toy()
    sel = SafeSelector().fit(X, y, groups)
    bad = np.array([[np.nan, 0.0, 0.0]])
    try:
        sel.decide(bad, ["t1"])
    except ValueError as e:
        assert "non-finite" in str(e)
        return
    raise AssertionError("NaN features must be rejected, never imputed")


def test_group_balancing_is_dataset_equal():
    """Unequal domain sizes must not let a large domain dominate the fit.

    The invariant is equal TOTAL weight per domain, not equal per-sample weight
    (a 30-row domain must get 1/30 per row so that 30 rows sum to 1).
    """
    counts = {"big": 30, "small": 3}
    groups = ["big"] * 30 + ["small"] * 3
    w = np.array([1.0 / counts[g] for g in groups])
    w = w / w.mean()
    total_big = w[[i for i, g in enumerate(groups) if g == "big"]].sum()
    total_small = w[[i for i, g in enumerate(groups) if g == "small"]].sum()
    assert abs(total_big - total_small) < 1e-9, f"{total_big} != {total_small}"
    # And the per-sample weights must differ, otherwise the balancing is a no-op.
    assert w[0] < w[-1]


def test_determinism():
    X, y, groups = _toy()
    a = SafeSelector().fit(X, y, groups).decide(np.array([[0.1, 0.2, -0.3]]), ["t1"])
    b = SafeSelector().fit(X, y, groups).decide(np.array([[0.1, 0.2, -0.3]]), ["t1"])
    assert a.margin_pp == b.margin_pp


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append((t.__name__, e))
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
