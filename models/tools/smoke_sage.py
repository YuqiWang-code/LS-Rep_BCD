#!/usr/bin/env python3
"""Synthetic smoke test for SAGE-CD (G0/G1/G2 Safe-DINO).

Verifies (no dataset needed):
1. A2Net keeps 2,913,094 params and swap/deploy invariance (joint_bn False/True).
2. AuxSemanticHead produces a full-res logit and propagates gradient to c4.
3. semantic_kd_loss is finite; gradient_budget_lambda bounds the KD weight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import A2Net_LWGANet_L0  # noqa: E402
from models.distill.sage_heads import AuxSemanticHead  # noqa: E402
from models.distill.sage_kd import gradient_budget_lambda, semantic_kd_loss  # noqa: E402

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
            require(swap < ATOL, f"swap error {swap:.2e}")
            _, change = model(x1, x2, return_change_features=True)
            require(change[2].shape[-1] == 16, "c4 should be 1/16 res")
            model.switch_to_deploy()
            after = model(x1, x2)
            deploy = max((a - b).abs().max().item() for a, b in zip(out_ab, after))
            require(deploy < ATOL, f"deploy error {deploy:.2e}")
        print(f"[PASS] student joint_bn={jb} params={params:,} swap={swap:.2e} deploy={deploy:.2e}")
        del model


def test_aux_and_kd(device):
    head = AuxSemanticHead().to(device)
    c4 = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
    z = head(c4)
    require(z.shape == (2, 1, 256, 256), f"aux head shape {tuple(z.shape)}")
    loss = semantic_kd_loss(z, torch.randn(2, 1, 256, 256, device=device))
    require(bool(torch.isfinite(loss)), "kd loss not finite")
    loss.backward()
    require(c4.grad is not None and bool(torch.isfinite(c4.grad).all()), "no gradient to c4")

    c4b = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
    main = (c4b ** 2).mean()
    # kd must depend on c4b (via the aux head) so the gradient budget is meaningful
    kd = semantic_kd_loss(head(c4b), torch.randn(2, 1, 256, 256, device=device))
    lam = gradient_budget_lambda(main, kd, c4b, rho=0.25, lambda_max=1.0)
    require(0.0 <= float(lam) <= 1.0, f"lambda out of range {float(lam)}")
    print(f"[PASS] aux head {tuple(z.shape)} kd_loss={float(loss):.4f} lambda={float(lam):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda":
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu")
    test_student(device)
    test_aux_and_kd(device)
    print("=" * 60)
    print("SAGE-CD smoke: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
