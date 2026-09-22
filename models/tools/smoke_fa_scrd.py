#!/usr/bin/env python3
"""Synthetic smoke test for FA-SCRD.

Verifies, without reading a dataset:
1. A2Net keeps 2,913,094 params (joint_bn False/True) and swap/deploy invariance.
2. symmetric_relation_kd and compute_failure_weight are finite and well-shaped.
3. The DINOv3 teacher loads, extracts 16x16 mid/deep features, and produces a
   change logit (only when --weight_path is given).

Usage:
    python -m models.tools.smoke_fa_scrd --device cuda --gpu_id 0 \
        --weight_path /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import A2Net_LWGANet_L0  # noqa: E402
from models.distill import FASCRDTeacher, compute_failure_weight, symmetric_relation_kd  # noqa: E402

EXPECTED_PARAMS = 2_913_094
ATOL = 1e-6


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_student(device):
    for jb in (False, True):
        model = A2Net_LWGANet_L0(pretrained=False, temporal_calibration_mode="none", joint_temporal_bn=jb).to(device)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        require(params == EXPECTED_PARAMS, f"joint_bn={jb} params {params:,} != {EXPECTED_PARAMS:,}")

        x1 = torch.randn(1, 3, 256, 256, device=device)
        x2 = torch.randn_like(x1)
        with torch.no_grad():
            out_ab = model(x1, x2)
            out_ba = model(x2, x1)
            swap = max((a - b).abs().max().item() for a, b in zip(out_ab, out_ba))
            require(swap < ATOL, f"joint_bn={jb} swap error {swap:.2e}")
            _, change = model(x1, x2, return_change_features=True)
            require(len(change) == 4 and change[2].shape[-1] == 16, "change feature c4 should be 1/16 res")
            model.switch_to_deploy()
            after = model(x1, x2)
            deploy = max((a - b).abs().max().item() for a, b in zip(out_ab, after))
            require(deploy < ATOL, f"joint_bn={jb} deploy error {deploy:.2e}")
        print(f"[PASS] student joint_bn={jb} params={params:,} swap={swap:.2e} deploy={deploy:.2e}")
        del model


def test_distill(device):
    s = torch.randn(2, 64, 16, 16, device=device)
    t = torch.randn(2, 768, 16, 16, device=device)
    w = torch.rand(2, 1, 16, 16, device=device)
    loss = symmetric_relation_kd(s, t, w)
    require(bool(torch.isfinite(loss)), "relation_kd not finite")

    ps = torch.sigmoid(torch.randn(2, 1, 16, 16, device=device))
    pt = torch.sigmoid(torch.randn(2, 1, 16, 16, device=device))
    y = (torch.rand(2, 1, 16, 16, device=device) > 0.9).float()
    weight = compute_failure_weight(ps, pt, y)
    require(bool(torch.isfinite(weight).all()), "failure_weight not finite")
    require(float(weight.max()) <= 1.0 + 1e-6, "failure_weight out of range")
    print(f"[PASS] relation_kd loss={float(loss):.6f} failure_weight range=[{float(weight.min()):.4f},{float(weight.max()):.4f}]")


def test_teacher(device, weight_path):
    teacher = FASCRDTeacher(weight_path).to(device)
    trainable = teacher.trainable_param_count
    require(trainable < 5_000_000, f"teacher trainable params {trainable:,} too large")
    teacher.eval()
    x = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        mid, deep = teacher.extract_features(x)
        logit = teacher.forward_change_logit(x, x)
    require(mid.shape == (1, 768, 16, 16), f"mid shape {tuple(mid.shape)}")
    require(deep.shape == (1, 768, 16, 16), f"deep shape {tuple(deep.shape)}")
    require(logit.shape == (1, 1, 256, 256), f"logit shape {tuple(logit.shape)}")
    print(f"[PASS] teacher trainable={trainable:,} mid/deep={tuple(mid.shape)} logit={tuple(logit.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--weight_path", default=None)
    args = parser.parse_args()

    if args.device == "cuda":
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu")

    test_student(device)
    test_distill(device)
    if args.weight_path:
        test_teacher(device, args.weight_path)
    else:
        print("[skip] teacher test (no --weight_path)")
    print("=" * 60)
    print("FA-SCRD smoke: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
