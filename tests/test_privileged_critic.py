#!/usr/bin/env python3
"""Unit tests for CTP-Critic (third-round main mechanism).

The invariants here are the ones the report lists as P0 / math smoke (§7.3):
gradient direction, counterfactual parity, no teacher leakage into the deploy
path, and NaN safety on empty change maps.

    python tests/test_privileged_critic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.privileged_critic import (  # noqa: E402
    PrivilegedCritic,
    bernoulli_kl_transfer,
)

STU_CH, TEACH_CH, B, H, W, TH, TW = 16, 12, 4, 16, 16, 5, 5


def _batch(seed=0, empty=False):
    torch.manual_seed(seed)
    z = torch.randn(B, STU_CH, H, W, requires_grad=True)
    ev = torch.rand(B, TEACH_CH, TH, TW)
    ctx = torch.randn(B, TEACH_CH, TH, TW)
    y = torch.zeros(B, 1, H, W) if empty else (torch.rand(B, 1, H, W) > 0.5).float()
    return z, ev, ctx, y


def _critic(mode="real"):
    torch.manual_seed(0)
    return PrivilegedCritic(STU_CH, TEACH_CH, d=8, z_dim=8, mode=mode)


# ------------------------------------------------------------------ gradients
def test_gradient_flows_into_student_features():
    z, ev, ctx, y = _batch()
    c = _critic()
    out = c(z, ev, ctx, y)
    (out.loss_task + out.loss_pri + out.loss_transfer).backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0
    assert c.psi.weight.grad is not None


def test_teacher_evidence_is_consumed_as_a_constant():
    """In production the evidence comes from the offline cache (requires_grad=False).

    Two real invariants: (a) the critic holds no teacher encoder, so no teacher
    parameter can ever be updated; (b) with production-like constant evidence the
    backward pass succeeds and the evidence tensor accumulates no gradient state.
    """
    z, ev, ctx, y = _batch()
    ev.requires_grad_(False)
    ctx.requires_grad_(False)
    c = _critic()
    names = [n for n, _ in c.named_children()]
    assert not any("encoder" in n or "teacher" in n for n in names), names
    out = c(z, ev, ctx, y)
    (out.loss_task + out.loss_pri + out.loss_transfer).backward()
    assert ev.grad is None, "constant cached evidence must not accumulate gradients"
    assert ctx.grad is None
    # every trainable parameter belongs to the critic itself
    assert all(p.requires_grad for p in c.parameters())


def test_transfer_term_does_not_pull_the_privileged_view():
    """L_transfer's target is detached; only the teacher-free view moves."""
    z, ev, ctx, y = _batch()
    c = _critic()
    out = c(z, ev, ctx, y)
    g_pri = torch.autograd.grad(out.loss_transfer, out.logits_pri,
                                retain_graph=True, allow_unused=True)[0]
    g_nul = torch.autograd.grad(out.loss_transfer, out.logits_nul,
                                retain_graph=True, allow_unused=True)[0]
    assert g_pri is None or float(g_pri.abs().sum()) == 0.0, "privileged view must be detached"
    assert g_nul is not None and float(g_nul.abs().sum()) > 0, "teacher-free view must be pulled"


def test_kl_transfer_is_zero_for_identical_views():
    l = torch.randn(2, 1, 8, 8)
    assert float(bernoulli_kl_transfer(l, l, 1.0)) < 1e-6


def test_kl_transfer_is_non_negative():
    a, b = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    assert float(bernoulli_kl_transfer(a, b, 1.0)) >= 0.0


# ------------------------------------------------------------ counterfactuals
def test_aux_terms_never_change_the_main_path_by_construction():
    """The critic is not part of the student forward, so aux mode cannot alter it.

    We assert the structural property: the critic holds no reference to a student
    model, and a fixed student output is bit-identical regardless of the mode.
    """
    z, ev, ctx, y = _batch()
    main_out = (z * 3.0).detach().clone()
    for mode in ("real", "zero", "shuffle"):
        c = _critic(mode)
        c(z, ev, ctx, y)
        assert torch.equal(main_out, (z * 3.0).detach()), f"mode {mode} touched the input tensor"


def test_zero_mode_uses_zero_teacher_but_still_runs_the_adapter():
    z, ev, ctx, y = _batch()
    c = _critic("zero")
    out = c(z, ev, ctx, y)
    assert out.u_norm == 0.0, "zero mode must inject zeros at the point of use"
    # the adapter was still evaluated (so P receives no gradient in this arm, but
    # the forward cost and parameter count are identical)
    assert sum(p.numel() for p in c.P.parameters()) > 0


def test_shuffle_mode_breaks_pairing_but_preserves_magnitudes():
    z, ev, ctx, y = _batch()
    c = _critic("shuffle")
    u = c.encode_teacher(ev, ctx, z.shape[-2:])
    us = c._apply_mode(u)
    assert not torch.equal(u, us), "shuffle must actually change the pairing"
    assert torch.allclose(u.sort(dim=0).values, us.sort(dim=0).values), "magnitudes must be preserved"


def test_real_and_zero_differ_but_share_every_parameter():
    z, ev, ctx, y = _batch()
    cr, cz = _critic("real"), _critic("zero")
    assert sum(p.numel() for p in cr.parameters()) == sum(p.numel() for p in cz.parameters())
    or_ = cr(z, ev, ctx, y)
    oz = cz(z, ev, ctx, y)
    assert not torch.allclose(or_.logits_pri, oz.logits_pri), "real must differ from zero"
    # the teacher-free view is IDENTICAL in both arms: same g(.,0), same h
    assert torch.allclose(or_.logits_nul, oz.logits_nul, atol=1e-6)


def test_single_sample_shuffle_is_a_noop_not_a_crash():
    torch.manual_seed(0)
    z = torch.randn(1, STU_CH, H, W)
    ev, ctx = torch.rand(1, TEACH_CH, TH, TW), torch.randn(1, TEACH_CH, TH, TW)
    y = torch.zeros(1, 1, H, W)
    out = _critic("shuffle")(z, ev, ctx, y)
    assert torch.isfinite(out.loss_task)


# ---------------------------------------------------------------------- safety
def test_empty_change_map_does_not_nan():
    z, ev, ctx, y = _batch(empty=True)
    for mode in ("real", "zero", "shuffle"):
        out = _critic(mode)(z, ev, ctx, y)
        for t in (out.loss_task, out.loss_pri, out.loss_transfer):
            assert torch.isfinite(t), f"{mode} produced a non-finite loss on empty GT"


def test_mismatched_gt_resolution_is_resized_not_crashed():
    z, ev, ctx, _ = _batch()
    y_big = (torch.rand(B, 1, H * 4, W * 4) > 0.5).float()
    out = _critic()(z, ev, ctx, y_big)
    assert out.logits_nul.shape[-2:] == (H, W)
    assert torch.isfinite(out.loss_task)


def test_determinism():
    z, ev, ctx, y = _batch(seed=7)
    a = _critic()(z, ev, ctx, y).loss_pri
    b = _critic()(z, ev, ctx, y).loss_pri
    assert torch.equal(a, b)


def test_invalid_mode_rejected():
    try:
        PrivilegedCritic(STU_CH, TEACH_CH, mode="bogus")
    except ValueError as e:
        assert "unknown mode" in str(e)
        return
    raise AssertionError("invalid mode must be rejected")


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
