#!/usr/bin/env python3
"""Unit tests for the S-PCG ladder (review §5.1) and DCA gate invariance.

Guarantees under test:
  * ``moe128`` (S0/legacy) is *unchanged*: same param count as the frozen C1
    anchor and the same gate arithmetic as the original implementation;
  * S1/S2 stay exchange-symmetric (T1/T2 swap error < 1e-6);
  * S1 and S2 differ in their gate *inputs* and the difference is real;
  * every mode stays under the 5M deploy budget.

    python tests/test_dca_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.a2net import A2Net_LWGANet_L0  # noqa: E402
from models.decoder.deployable_change_adapter import DeployableChangeAdapter  # noqa: E402

C0_PARAMS = 2_913_094
C1_PARAMS = 3_280_274
CAP = 5_000_000
SWAP_ATOL = 1e-6


def _small_dca(gate_mode, in_d=8, width=16):
    torch.manual_seed(0)
    return DeployableChangeAdapter(in_d, width=width, gate_mode=gate_mode).eval()


def test_legacy_gate_arithmetic_is_preserved():
    """Reproduce the original formula by hand and compare against the module."""
    dca = _small_dca("legacy")
    f = [torch.randn(2, 8, 4, 4) for _ in range(4)]
    with torch.no_grad():
        gap = torch.cat([x.mean(dim=(2, 3)) for x in f], dim=1)
        expect = dca.gate(gap).view(-1, 4, 3).softmax(dim=2)
        got = dca._gate_weights(tuple(f), None)
    assert torch.equal(expect, got), "legacy gate formula changed"


def test_legacy_ignores_the_pair_argument():
    dca = _small_dca("legacy")
    f = [torch.randn(1, 8, 4, 4) for _ in range(4)]
    pair = ([torch.randn(1, 8, 4, 4) for _ in range(4)] for _ in (0,))
    a, b = [torch.randn(1, 8, 4, 4) for _ in range(4)], [torch.randn(1, 8, 4, 4) for _ in range(4)]
    with torch.no_grad():
        w_none = dca._gate_weights(tuple(f), None)
        w_pair = dca._gate_weights(tuple(f), (a, b))
    assert torch.equal(w_none, w_pair)


def test_new_modes_require_the_pair():
    for mode in ("stats", "sympcg"):
        dca = _small_dca(mode)
        f = [torch.randn(1, 8, 4, 4) for _ in range(4)]
        try:
            dca._gate_weights(tuple(f), None)
        except ValueError as e:
            assert "pre-fusion pair" in str(e)
            continue
        raise AssertionError(f"{mode} must refuse to run without (a_s, b_s)")


def test_gate_is_exchange_symmetric_by_construction():
    """Swapping T1/T2 must leave the gate weights exactly unchanged for S1/S2."""
    for mode in ("stats", "sympcg"):
        dca = _small_dca(mode)
        a = [torch.randn(2, 8, 4, 4) for _ in range(4)]
        b = [torch.randn(2, 8, 4, 4) for _ in range(4)]
        c = [torch.randn(2, 8, 4, 4) for _ in range(4)]
        with torch.no_grad():
            w_ab = dca._gate_weights(tuple(c), (a, b))
            w_ba = dca._gate_weights(tuple(c), (b, a))
        assert torch.equal(w_ab, w_ba), f"{mode} gate is not exchange-symmetric"


def test_s2_uses_more_information_than_s1():
    """S2 sees m_s and v_s, so its gate input dim (and weights) must differ."""
    s1, s2 = _small_dca("stats"), _small_dca("sympcg")
    a = [torch.randn(1, 8, 4, 4) for _ in range(4)]
    b = [torch.randn(1, 8, 4, 4) for _ in range(4)]
    c = [torch.randn(1, 8, 4, 4) for _ in range(4)]
    with torch.no_grad():
        w1 = s1._gate_weights(tuple(c), (a, b))
        w2 = s2._gate_weights(tuple(c), (a, b))
    assert s1.gate_s[0][0].in_features == 8
    assert s2.gate_s[0][0].in_features == 24
    assert w1.shape == w2.shape == (1, 4, 3)
    # rows must be proper softmax distributions
    for w in (w1, w2):
        assert torch.allclose(w.sum(dim=2), torch.ones(1, 4), atol=1e-6)


def test_s2_gate_changes_when_only_the_common_context_changes():
    """If only m_s moves (a and b shift together), S2 must react and S1 must not."""
    s1, s2 = _small_dca("stats"), _small_dca("sympcg")
    a = [torch.randn(1, 8, 6, 6) for _ in range(4)]
    b = [torch.randn(1, 8, 6, 6) for _ in range(4)]
    shift = [torch.randn(1, 8, 6, 6) * 3.0 for _ in range(4)]
    a2 = [x + s for x, s in zip(a, shift)]
    b2 = [x + s for x, s in zip(b, shift)]     # d_s is IDENTICAL, m_s changes
    c = [torch.randn(1, 8, 6, 6) for _ in range(4)]
    with torch.no_grad():
        w1a = s1._gate_weights(tuple(c), (a, b))
        w1b = s1._gate_weights(tuple(c), (a2, b2))
        w2a = s2._gate_weights(tuple(c), (a, b))
        w2b = s2._gate_weights(tuple(c), (a2, b2))
    assert torch.allclose(w1a, w1b, atol=1e-6), "S1 must depend on d_s only"
    assert not torch.allclose(w2a, w2b, atol=1e-4), "S2 must react to the common context"


# ------------------------------------------------------------------ full model
def _model(dca_mode):
    torch.manual_seed(0)
    m = A2Net_LWGANet_L0(pretrained=False, dca_mode=dca_mode).eval()
    return m


def test_frozen_param_counts_for_c0_c1():
    assert sum(p.numel() for p in _model("none").parameters()) == C0_PARAMS
    assert sum(p.numel() for p in _model("moe128").parameters()) == C1_PARAMS


def test_all_modes_under_deploy_budget():
    counts = {}
    for mode in ("none", "moe128", "moe128_stats", "moe128_sympcg"):
        n = sum(p.numel() for p in _model(mode).parameters())
        counts[mode] = n
        assert n < CAP, f"{mode} uses {n:,} params"
    # S1/S2 must be close to C1: the plan forbids "adding a fourth expert".
    assert counts["moe128_sympcg"] - counts["moe128"] < 50_000, counts
    print("    param counts:", counts)


def test_swap_invariance_all_modes():
    torch.manual_seed(0)
    for mode in ("none", "moe128", "moe128_stats", "moe128_sympcg"):
        m = _model(mode)
        pre, post = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            ab = m(pre, post)
            ba = m(post, pre)
            err = max((x - y).abs().max().item() for x, y in zip(ab, ba))
        assert err < SWAP_ATOL, f"{mode} swap error {err:.3e}"


def test_gradients_reach_the_pair_only_through_the_gate():
    """For S1/S2 the gate must be trainable (it is the only new learnable head)."""
    for gate_mode, dca_mode in (("stats", "moe128_stats"), ("sympcg", "moe128_sympcg")):
        m = _model(dca_mode)
        pre, post = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        out = m(pre, post)[0].sum()
        out.backward()
        g = [p.grad for p in m.dca.gate_s.parameters()]
        assert any(x is not None and float(x.abs().sum()) > 0 for x in g), gate_mode


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
