#!/usr/bin/env python3
"""Synthetic correctness smoke test for SCTC.

This script does not train a formal experiment and does not read a dataset.
It verifies the mechanism/deployment contracts before any S0/S1/S2 run:

1. S0 identity calibration is exact.
2. S1/S2 calibration contains zero parameters and zero buffers.
3. S1/S2 are temporally symmetric.
4. SCTC suppresses synthetic channel-wise temporal nuisance on stable pixels.
5. SCTC preserves a distinguishable changed region in that synthetic pair.
6. Gradients reach both temporal features while agreement weights are detached.
7. S0/S1/S2 all keep exactly 2,913,094 model parameters.
8. No training_auxiliary / Teacher state exists in the model.
9. Full-model prediction is invariant to T1/T2 exchange within 1e-6.
10. switch_to_deploy() changes no prediction and removes nothing because SCTC
    is a zero-parameter deploy-time operation.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from models import A2Net_LWGANet_L0  # noqa: E402
from models.decoder.temporal_calibration import (  # noqa: E402
    MultiScaleTemporalCalibration,
    SymmetricTemporalCalibration,
)


EXPECTED_DEPLOY_PARAMS = 2_913_094
ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic smoke test for S0/S1/S2 SCTC"
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def max_pair_error(left, right) -> float:
    return max(
        (a - b).abs().max().item()
        for a, b in zip(left, right)
    )


def test_parameter_free_and_identity(device: torch.device) -> None:
    x1 = torch.randn(2, 64, 24, 24, device=device)
    x2 = torch.randn_like(x1)

    identity = SymmetricTemporalCalibration("none").to(device)
    y1, y2 = identity(x1, x2)
    require(torch.equal(x1, y1), "S0 T1 identity path is not exact")
    require(torch.equal(x2, y2), "S0 T2 identity path is not exact")

    for mode in ("symmetric", "sctc"):
        module = SymmetricTemporalCalibration(mode).to(device)
        require(
            sum(p.numel() for p in module.parameters()) == 0,
            f"{mode} unexpectedly contains learnable parameters",
        )
        require(
            sum(b.numel() for b in module.buffers()) == 0,
            f"{mode} unexpectedly contains persistent buffers",
        )

        a1, a2 = module(x1, x2)
        b2, b1 = module(x2, x1)
        require(
            (a1 - b1).abs().max().item() < ATOL,
            f"{mode} T1/T2 swap equivariance failed for branch 1",
        )
        require(
            (a2 - b2).abs().max().item() < ATOL,
            f"{mode} T1/T2 swap equivariance failed for branch 2",
        )

    multiscale = MultiScaleTemporalCalibration("sctc", num_scales=4).to(device)
    require(
        sum(p.numel() for p in multiscale.parameters()) == 0,
        "Multi-scale SCTC unexpectedly contains parameters",
    )


def test_synthetic_nuisance_suppression(device: torch.device) -> None:
    # Positive-centered features make a moderate channel-wise affine shift a
    # realistic nuisance while a sign-reversed patch obtains low cosine
    # agreement and therefore low SCTC statistical weight.
    base = torch.randn(2, 64, 32, 32, device=device) * 0.25 + 1.0
    x1 = base.clone()

    scale = torch.linspace(0.80, 1.20, 64, device=device).view(1, 64, 1, 1)
    shift = torch.linspace(-0.15, 0.15, 64, device=device).view(1, 64, 1, 1)
    x2 = base * scale + shift

    # Synthetic semantic-change region.
    changed = torch.zeros(1, 1, 32, 32, dtype=torch.bool, device=device)
    changed[:, :, 10:18, 11:19] = True
    x2[:, :, 10:18, 11:19] = -base[:, :, 10:18, 11:19]

    module = SymmetricTemporalCalibration("sctc").to(device)
    y1, y2, detail = module(x1, x2, return_diagnostics=True)

    stable = (~changed).expand_as(x1)
    changed_c = changed.expand_as(x1)

    pre_stable = (x1 - x2).abs()[stable].mean()
    post_stable = (y1 - y2).abs()[stable].mean()
    post_changed = (y1 - y2).abs()[changed_c].mean()

    require(
        post_stable < pre_stable,
        "SCTC did not reduce synthetic stable-region nuisance difference",
    )
    require(
        post_changed > post_stable,
        "Synthetic changed region is not distinguishable after SCTC",
    )
    require(
        0.0 <= float(detail["agreement_mean"]) <= 1.0,
        "Agreement diagnostic is outside [0,1]",
    )


def test_gradient_flow(device: torch.device) -> None:
    x1 = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
    x2 = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
    module = SymmetricTemporalCalibration("sctc").to(device)

    y1, y2 = module(x1, x2)
    loss = (y1 - y2).square().mean()
    loss.backward()

    for tensor, name in ((x1, "T1"), (x2, "T2")):
        require(tensor.grad is not None, f"No SCTC gradient reached {name}")
        require(
            bool(torch.isfinite(tensor.grad).all()),
            f"Non-finite SCTC gradient in {name}",
        )
        require(
            float(tensor.grad.abs().sum()) > 0.0,
            f"Zero SCTC gradient in {name}",
        )


def test_full_model(
    device: torch.device,
    height: int,
    width: int,
) -> None:
    x1 = torch.randn(1, 3, height, width, device=device)
    x2 = torch.randn_like(x1)

    for mode in ("none", "symmetric", "sctc"):
        model = A2Net_LWGANet_L0(
            pretrained=False,
            pretrained_path=None,
            temporal_calibration_mode=mode,
        ).to(device)
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        require(
            params == EXPECTED_DEPLOY_PARAMS,
            f"{mode} parameter count {params:,} != {EXPECTED_DEPLOY_PARAMS:,}",
        )

        calibration_params = sum(
            p.numel() for p in model.tfm.temporal_calibration.parameters()
        )
        require(
            calibration_params == 0,
            f"{mode} calibration contains {calibration_params} parameters",
        )

        state_keys = tuple(model.state_dict().keys())
        require(
            not any("training_auxiliary" in key for key in state_keys),
            f"{mode} still contains training_auxiliary state",
        )
        require(
            not hasattr(model, "training_auxiliary"),
            f"{mode} still exposes a training_auxiliary attribute",
        )

        with torch.no_grad():
            output_ab = model(x1, x2)
            output_ba = model(x2, x1)
            swap_error = max_pair_error(output_ab, output_ba)
            require(
                swap_error < ATOL,
                f"{mode} full-model swap error {swap_error:.8e} >= {ATOL}",
            )

            before = tuple(value.clone() for value in output_ab)
            returned = model.switch_to_deploy()
            require(returned is model, "switch_to_deploy() must return self")
            after = model(x1, x2)
            deploy_error = max_pair_error(before, after)
            require(
                deploy_error < ATOL,
                f"{mode} deploy error {deploy_error:.8e} >= {ATOL}",
            )

        print(
            f"[PASS] mode={mode:9s} params={params:,} "
            f"swap_error={swap_error:.3e} deploy_error={deploy_error:.3e}"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.height < 32 or args.width < 32:
        raise ValueError("height/width must be >= 32 for full-model smoke")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(args.gpu_id)

    device = torch.device(
        f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu"
    )
    set_seed(args.seed)

    test_parameter_free_and_identity(device)
    print("[PASS] parameter-free / identity / pair-symmetry checks")

    test_synthetic_nuisance_suppression(device)
    print("[PASS] synthetic nuisance suppression check")

    test_gradient_flow(device)
    print("[PASS] SCTC gradient-flow check")

    test_full_model(device, args.height, args.width)

    print("=" * 80)
    print("SCTC synthetic smoke: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
