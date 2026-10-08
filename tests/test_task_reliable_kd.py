#!/usr/bin/env python3
"""Unit tests for GT-Conditioned Reliable Task Distillation (J1 mechanism).

These must pass BEFORE any 40K J1 run is launched: the whole point of J1 is a
clean single-variable contrast, so region/​target/​capacity mistakes would silently
invalidate the verdict.

    python tests/test_task_reliable_kd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.task_reliable_kd import (  # noqa: E402
    TaskReliableAuxHead,
    TaskReliableConfig,
    aux_task_loss,
    boundary_band,
    build_regions_256,
    compute_q,
)

CFG = TaskReliableConfig(
    teacher_package="dinov3_lvd", dataset="SYSU",
    z_median=0.2, z_q90=0.6, t_plus=0.6, t_minus=0.4,
    boundary_r=2, native_hw=(16, 16), n_samples=100,
)


def _square_gt(b=2, size=64, half=16, fill=1.0):
    """A filled square in the middle of a [b,1,size,size] tensor."""
    y = torch.zeros(b, 1, size, size)
    c = size // 2
    y[:, :, c - half:c + half, c - half:c + half] = fill
    return y


# ------------------------------------------------------------------ primitives
def test_compute_q_is_monotone_and_bounded():
    z = torch.linspace(0.0, 1.0, 101).reshape(1, 1, 1, 101)
    q = compute_q(z, CFG)
    assert float(q.min()) > 0.0 and float(q.max()) < 1.0
    d = (q[..., 1:] - q[..., :-1])
    assert bool((d >= 0).all()), "q must be non-decreasing in z"
    # q(m_T) == 0.5 exactly, by construction
    at_median = compute_q(torch.tensor([[[[CFG.z_median]]]]), CFG)
    assert abs(float(at_median) - 0.5) < 1e-6


def test_boundary_band_traces_the_gt_edge_only():
    y = _square_gt(size=64, half=16)
    band = boundary_band(y, 2)
    assert band.shape == y.shape
    assert float(band.max()) == 1.0
    # interior and far background must be untouched
    assert float(band[0, 0, 32, 32]) == 0.0
    assert float(band[0, 0, 2, 2]) == 0.0
    # the edge itself must be marked
    assert float(band[0, 0, 32, 16]) == 1.0
    # r=0 disables the band entirely
    assert float(boundary_band(y, 0).sum()) == 0.0


def test_regions_exclude_the_boundary_and_split_on_teacher_agreement():
    y = _square_gt(size=64)
    # teacher confidently right everywhere -> q high inside, low outside
    q = torch.where(y > 0.5, torch.full_like(y, 0.9), torch.full_like(y, 0.1))
    reg = build_regions_256(y, q, CFG)
    assert float(reg["g_pos"][0, 0, 32, 32]) == 1.0     # interior changed, kept
    assert float(reg["g_pos"][0, 0, 32, 16]) == 0.0     # changed edge, excluded
    assert float(reg["g_neg"][0, 0, 2, 2]) == 1.0       # far unchanged, kept

    # teacher confidently WRONG -> nothing survives the agreement test
    q_bad = torch.where(y > 0.5, torch.full_like(y, 0.1), torch.full_like(y, 0.9))
    reg_bad = build_regions_256(y, q_bad, CFG)
    assert float(reg_bad["g_pos"].sum()) == 0.0
    assert float(reg_bad["g_neg"].sum()) == 0.0


# ---------------------------------------------------------------------- losses
def test_aux_head_has_exactly_65_params():
    head = TaskReliableAuxHead(64, scale_index=0)
    assert sum(p.numel() for p in head.parameters()) == 65


def test_empty_region_returns_none_not_a_fabricated_loss():
    # no changed pixels, and the teacher says "changed" everywhere -> G is empty
    y = torch.zeros(2, 1, 64, 64)
    q_conf = torch.full((2, 1, 64, 64), 1.0)
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    loss, stats = aux_task_loss("gate", logits, y, CFG, confidence=q_conf)
    assert loss is None, "an empty region must skip the aux term, never invent one"
    assert stats["skipped"] is True


def test_gate_and_taskkd_see_the_identical_region():
    """The core J1 control: same pixels, only the target differs."""
    torch.manual_seed(0)
    y = _square_gt(b=2, size=64, half=10)
    q = torch.where(y > 0.5, torch.full_like(y, 0.85), torch.full_like(y, 0.15))
    # make it realistic: a few teacher disagreements
    q[:, :, 5:10, 5:10] = 0.9
    logits = torch.randn(2, 1, 16, 16)
    _, s_gate = aux_task_loss("gate", logits, y, CFG, confidence=q)
    _, s_kd = aux_task_loss("taskkd", logits, y, CFG, confidence=q)
    for k in ("n_pos", "n_neg", "g_pos_frac", "g_neg_frac", "band_frac"):
        assert abs(s_gate[k] - s_kd[k]) < 1e-9, f"region differs on {k}"


def test_gt_arm_uses_all_pixels_and_no_teacher():
    y = _square_gt(b=1, size=64, half=10)
    logits = torch.zeros(1, 1, 16, 16)
    loss, stats = aux_task_loss("gt", logits, y, None, None)
    assert loss is not None
    assert stats["region"] == "all_pixels"
    # pooled pos/neg must reproduce the true change ratio of this GT
    frac = stats["n_pos"] / (stats["n_pos"] + stats["n_neg"])
    assert abs(frac - float(y.mean())) < 1e-6, (frac, float(y.mean()))
    assert float(y.mean()) < 0.15, "fixture should be background-dominated like real CD"


def test_gradients_reach_the_aux_head_and_the_student_scale():
    y = _square_gt(b=1, size=64, half=12)
    q = torch.where(y > 0.5, torch.full_like(y, 0.9), torch.full_like(y, 0.1))
    change = [torch.randn(1, 64, 64, 64, requires_grad=True)]
    head = TaskReliableAuxHead(64, scale_index=0)
    logits = head(change)
    loss, _ = aux_task_loss("taskkd", logits, y, CFG, confidence=q)
    assert loss is not None
    loss.backward()
    assert head.proj.weight.grad is not None and float(head.proj.weight.grad.abs().sum()) > 0
    # the student-side feature must receive gradient too (no detach on the student)
    assert change[0].grad is not None and float(change[0].grad.abs().sum()) > 0


def test_teacher_side_is_never_differentiated():
    """q_T is a frozen target: no gradient may flow into the confidence tensor."""
    y = _square_gt(b=1, size=64, half=12)
    conf = torch.where(y > 0.5, torch.full_like(y, 0.9), torch.full_like(y, 0.1))
    conf = conf.detach().requires_grad_(True)
    logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
    loss, _ = aux_task_loss("taskkd", logits, y, CFG, confidence=conf)
    loss.backward()
    assert conf.grad is None or float(conf.grad.abs().sum()) == 0.0


def test_case_insensitive_to_annotation_geometry():
    """A 32x32 GT upsampled from a 16x16 teacher map must still pool coherently."""
    y = _square_gt(b=1, size=64, half=8)
    z = torch.rand(1, 1, 16, 16) * 0.4 + 0.3
    logits = torch.zeros(1, 1, 16, 16)
    loss, stats = aux_task_loss("taskkd", logits, y, CFG, confidence=z)
    assert loss is not None and stats["n_pos"] > 0


def test_non_square_aux_grid_is_supported():
    y = _square_gt(b=1, size=64, half=8)
    q = torch.where(y > 0.5, torch.full_like(y, 0.9), torch.full_like(y, 0.1))
    logits = torch.zeros(1, 1, 32, 16)
    loss, _ = aux_task_loss("gate", logits, y, CFG,
                            confidence=torch.nn.functional.interpolate(q, size=(32, 16)))
    assert loss is not None


def test_invalid_task_name_is_rejected():
    try:
        aux_task_loss("bogus", torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 64, 64))
    except ValueError as e:
        assert "unknown aux task" in str(e)
        return
    raise AssertionError("unknown aux task must be rejected")


def test_determinism():
    torch.manual_seed(1)
    y = _square_gt(b=2, size=64, half=10)
    q = torch.where(y > 0.5, torch.full_like(y, 0.9), torch.full_like(y, 0.1))
    logits = torch.randn(2, 1, 16, 16)
    a, _ = aux_task_loss("taskkd", logits, y, CFG, confidence=q)
    b, _ = aux_task_loss("taskkd", logits, y, CFG, confidence=q)
    assert abs(float(a) - float(b)) < 1e-12


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
