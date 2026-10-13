#!/usr/bin/env python3
"""Unit tests for PSFB (fourth-round main mechanism).

Covers the report's P0 list (§7.2) plus the two review findings:
  * P/R initialisation must be TEACHER-INDEPENDENT (review finding #1), otherwise
    cross-teacher effect comparisons are confounded by differing R;
  * the collapse indicator ||H_T - H_0|| must be exposed (review finding #2),
    because L_feat trains R's zero-input branch and can silently destroy the U path;
  * the shadow decoder pass must not write the main path's BatchNorm buffers;
  * the main prediction must not depend on the bridge at all.

    python tests/test_privileged_spatial_bridge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.privileged_spatial_bridge import (  # noqa: E402
    PrivilegedSpatialBridge,
    bridge_init_fingerprint,
    feature_transfer_loss,
    shadow_decoder_forward,
)

STU, B, H, W = 16, 4, 8, 8
TH, TW = 5, 5


def _batch(seed=0, teach_ch=12):
    torch.manual_seed(seed)
    c3 = torch.randn(B, STU, H, W, requires_grad=True)
    ev = torch.rand(B, teach_ch, TH, TW)
    ctx = torch.randn(B, teach_ch, TH, TW)
    return c3, ev, ctx


def _bridge(mode="real", teach_ch=12, d=6):
    return PrivilegedSpatialBridge(STU, teach_ch, d=d, mode=mode)


# ------------------------------------------------------- review finding #1
def test_R_initialisation_is_teacher_independent():
    """Different teachers (different channel counts) must share the SAME R."""
    a = _bridge(teach_ch=12)
    b = _bridge(teach_ch=40)          # a different teacher -> P has a different shape
    assert a.P[0].weight.shape != b.P[0].weight.shape
    assert bridge_init_fingerprint(a) == bridge_init_fingerprint(b), (
        "R must not depend on the teacher's channel count, or cross-teacher "
        "comparisons are confounded by initialisation noise")
    assert torch.equal(a.R[0].weight, b.R[0].weight)


def test_R_initialisation_does_not_consume_global_rng():
    """Building a bridge must not shift the global RNG stream (student init safety)."""
    torch.manual_seed(1234)
    before = torch.randn(3)
    torch.manual_seed(1234)
    _bridge(teach_ch=12)
    after = torch.randn(3)
    assert torch.equal(before, after), "bridge construction must be RNG-neutral"


# ------------------------------------------------------- review finding #2
def test_collapse_indicator_is_exposed_and_zero_in_zero_mode():
    c3, ev, ctx = _batch()
    z = _bridge("zero")(c3, ev, ctx)
    assert z.stats["ht_h0_gap"] == 0.0, "with U=0 the two views must coincide exactly"
    assert z.stats["u_norm"] == 0.0
    r = _bridge("real")(c3, ev, ctx)
    assert r.stats["ht_h0_gap"] > 0.0, "REAL must separate the two views"
    assert r.stats["ht_c3_gap"] > 0.0


def test_real_zero_shuffle_share_every_parameter():
    a, b, c = _bridge("real"), _bridge("zero"), _bridge("shuffle")
    pa = sum(p.numel() for p in a.parameters())
    assert pa == sum(p.numel() for p in b.parameters()) == sum(p.numel() for p in c.parameters())
    for (na, xa), (_, xb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(xa, xb), f"{na} differs between arms"


def test_modes_differ_only_in_u():
    c3, ev, ctx = _batch()
    with torch.no_grad():
        ur = _bridge("real")(c3, ev, ctx).u
        uz = _bridge("zero")(c3, ev, ctx).u
        us = _bridge("shuffle")(c3, ev, ctx).u
    assert float(uz.abs().sum()) == 0.0
    assert not torch.equal(ur, us), "shuffle must actually permute"
    assert torch.allclose(ur.sort(dim=0).values, us.sort(dim=0).values), "magnitudes preserved"


def test_single_sample_shuffle_is_a_noop():
    torch.manual_seed(0)
    c3 = torch.randn(1, STU, H, W)
    ev, ctx = torch.rand(1, 12, TH, TW), torch.randn(1, 12, TH, TW)
    out = _bridge("shuffle")(c3, ev, ctx)
    assert torch.isfinite(out.h_t).all()


# ------------------------------------------------------------------ gradients
def test_gradients_reach_c3_and_both_adapters():
    c3, ev, ctx = _batch()
    br = _bridge("real")
    out = br(c3, ev, ctx)
    loss = feature_transfer_loss(out.h_0, out.h_t) + out.h_t.pow(2).mean()
    loss.backward()
    assert c3.grad is not None and float(c3.grad.abs().sum()) > 0
    assert float(br.R[0].weight.grad.abs().sum()) > 0
    assert float(br.P[0].weight.grad.abs().sum()) > 0


def test_feature_loss_leaves_the_U_slice_of_R_ungradiented():
    """Documents review finding #2: L_feat only trains R's zero-input branch."""
    c3, ev, ctx = _batch()
    br = _bridge("real")
    out = br(c3, ev, ctx)
    feature_transfer_loss(out.h_0, out.h_t).backward()
    g = br.R[0].weight.grad                      # [stu_ch, stu_ch+d, 1, 1]
    z_slice = g[:, :STU].abs().sum()
    u_slice = g[:, STU:].abs().sum()
    assert float(z_slice) > 0, "the Z branch must receive gradient"
    assert float(u_slice) == 0.0, (
        "the U slice gets no gradient from L_feat -- this is why ||H_T-H_0|| "
        "collapsing must be treated as a failure")


def test_teacher_evidence_is_a_constant():
    c3, ev, ctx = _batch()
    ev.requires_grad_(False)
    br = _bridge("real")
    out = br(c3, ev, ctx)
    (out.h_t.pow(2).mean() + feature_transfer_loss(out.h_0, out.h_t)).backward()
    assert ev.grad is None


def test_feature_loss_kinds_and_validation():
    a = torch.randn(2, 4, 3, 3)
    assert float(feature_transfer_loss(a, a, "cosine")) < 1e-5
    assert float(feature_transfer_loss(a, a, "smoothl1")) < 1e-5
    try:
        feature_transfer_loss(a, a, "bogus")
    except ValueError:
        return
    raise AssertionError("unknown kind must be rejected")


# ------------------------------------------------------- shadow decoder / BN
class _BNDecoder(nn.Module):
    def __init__(self, ch=STU):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch)
        self.conv = nn.Conv2d(ch, ch, 1)

    def forward(self, c2, c3, c4, c5):
        return (self.conv(self.bn(c3)),)


def test_shadow_forward_does_not_touch_main_bn_buffers():
    torch.manual_seed(0)
    dec = _BNDecoder().train()
    c2 = torch.randn(2, STU, 4, 4)
    c3 = torch.randn(2, STU, H, W)
    c4 = torch.randn(2, STU, 2, 2)
    c5 = torch.randn(2, STU, 1, 1)

    before = {k: v.clone() for k, v in dec.named_buffers()}
    _ = shadow_decoder_forward(dec, c2, c3, c4, c5)
    for k, v in dec.named_buffers():
        assert torch.equal(v, before[k]), f"shadow pass wrote main buffer {k}"
    assert int(dec.bn.num_batches_tracked) == int(before["bn.num_batches_tracked"])


def test_shadow_forward_still_gives_gradients_to_shared_weights():
    torch.manual_seed(0)
    dec = _BNDecoder().train()
    c2 = torch.randn(2, STU, 4, 4)
    c3 = torch.randn(2, STU, H, W)
    c4 = torch.randn(2, STU, 2, 2)
    c5 = torch.randn(2, STU, 1, 1)
    out = shadow_decoder_forward(dec, c2, c3, c4, c5)
    out[0].pow(2).mean().backward()
    assert dec.conv.weight.grad is not None and float(dec.conv.weight.grad.abs().sum()) > 0
    assert dec.bn.weight.grad is not None and float(dec.bn.weight.grad.abs().sum()) > 0


def test_shadow_forward_actually_consumes_the_substituted_c3():
    torch.manual_seed(0)
    dec = _BNDecoder().eval()          # eval: BN uses running stats, deterministic
    c2 = torch.randn(1, STU, 4, 4)
    c4 = torch.randn(1, STU, 2, 2)
    c5 = torch.randn(1, STU, 1, 1)
    a = shadow_decoder_forward(dec, c2, torch.zeros(1, STU, H, W), c4, c5)[0]
    b = shadow_decoder_forward(dec, c2, torch.ones(1, STU, H, W), c4, c5)[0]
    assert not torch.allclose(a, b), "shadow forward must depend on the c3 it is given"


def test_invalid_mode_rejected():
    try:
        PrivilegedSpatialBridge(STU, 12, mode="bogus")
    except ValueError as e:
        assert "unknown mode" in str(e)
        return
    raise AssertionError("invalid mode must be rejected")


def test_determinism():
    c3, ev, ctx = _batch(seed=7)
    a = _bridge("real")(c3, ev, ctx).h_t
    b = _bridge("real")(c3, ev, ctx).h_t
    assert torch.equal(a, b)


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
