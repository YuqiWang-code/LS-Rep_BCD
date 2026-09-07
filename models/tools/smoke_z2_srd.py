#!/usr/bin/env python3
"""Functional acceptance test for Run3 Z2-SRD N0-N4."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import A2Net_LWGANet_L0
from models.distill.sam_hsd import (
    ExchangeInvariantStructuralEvidence,
    RelationalStructureBuilder,
    Z2PairProjector,
    prepare_relational_pack,
)
from models.tools.smoke_sam_hsd import batch_pack, raw_pack


RECIPES = {
    "N0": ({"pair_mode": "group", "use_even": True, "use_odd": True}, 2_921_930),
    "N1": ({"pair_mode": "group", "use_even": True, "use_odd": False}, 2_921_882),
    "N2": ({"pair_mode": "mixed", "use_even": True, "use_odd": False}, 2_921_370),
    "N3": ({"pair_mode": "invariant", "use_even": True, "use_odd": False}, 2_921_370),
    "N4": ({"pair_mode": "group", "use_even": False, "use_odd": True}, 2_921_862),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Z2-SRD Run3 smoke test")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--pretrained_path",
        default=str(PROJECT_ROOT / "pre-trained_weights" / "lwganet_l0_e299.pth"),
    )
    return parser.parse_args()


def assert_close(left, right, message, atol=1e-6):
    error = (left.float() - right.float()).abs().max().item()
    if error > atol:
        raise AssertionError(f"{message}: max_error={error:.8e}")


def nonzero_gradient(parameters):
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.detach().abs().sum()) > 0
        for parameter in parameters
    )


def evidence_acceptance(size, batch_size, device):
    prepared = prepare_relational_pack(raw_pack(size))
    teacher_pack = batch_pack(prepared, batch_size, device)
    structure = RelationalStructureBuilder().to(device)(teacher_pack)
    swapped_structure = {"t1": structure["t2"], "t2": structure["t1"]}

    builder = ExchangeInvariantStructuralEvidence().to(device)
    evidence = builder(structure)
    swapped = builder(swapped_structure)
    assert_close(
        evidence["structural_code"], swapped["structural_code"],
        "even teacher is not exchange invariant",
    )
    assert_close(
        evidence["signed_structural_code"],
        -swapped["signed_structural_code"],
        "odd teacher is not exchange anti-equivariant", atol=2e-3,
    )
    for key in ("trust_change", "trust_stable", "uncertainty"):
        assert_close(evidence[key], swapped[key], f"symmetric evidence/{key}")

    mixed_builder = ExchangeInvariantStructuralEvidence(
        directional_restore=True,
    ).to(device)
    mixed = mixed_builder(structure)
    mixed_swapped = mixed_builder(swapped_structure)
    assert_close(
        mixed["structural_code"][:, :3]
        + mixed_swapped["structural_code"][:, :3],
        torch.ones_like(mixed["structural_code"][:, :3]),
        "mixed signed control did not complement under exchange", atol=2e-3,
    )
    return teacher_pack


def projector_acceptance(device):
    projector = Z2PairProjector(16).to(device)
    left = torch.randn(2, 16, 11, 13, device=device)
    right = torch.randn_like(left)
    even, odd = projector(left, right)
    swapped_even, swapped_odd = projector(right, left)
    assert_close(even, swapped_even, "Z2 even latent is not invariant")
    assert_close(odd, -swapped_odd, "Z2 odd latent is not anti-equivariant")


def build_recipe(experiment, checkpoint):
    config, _ = RECIPES[experiment]
    return A2Net_LWGANet_L0(
        pretrained=checkpoint.is_file(),
        pretrained_path=str(checkpoint) if checkpoint.is_file() else None,
        auxiliary_mode="z2_srd",
        z2_srd_cfg={
            **config,
            "spatial_adapter": False,
            "odd_weight": 1.0,
            "boundary_tolerance": 2,
            "trust_gamma": 1.0,
        },
    )


def recipe_acceptance(args, teacher_pack, device):
    size = args.image_size
    first = torch.randn(args.batch_size, 3, size, size, device=device)
    second = torch.randn_like(first)
    target = torch.zeros(args.batch_size, 1, size, size, device=device)
    target[..., size // 2:4 * size // 5, size // 2:4 * size // 5] = 1.0
    checkpoint = Path(args.pretrained_path)
    observed = {}

    for experiment, (_, expected_params) in RECIPES.items():
        model = build_recipe(experiment, checkpoint).to(device).train()
        # Isolate representational symmetry from stochastic BN state here;
        # real train-mode state symmetry is checked separately below.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        train_params = sum(parameter.numel() for parameter in model.parameters())
        if train_params != expected_params:
            raise AssertionError(
                f"{experiment} train params {train_params:,} != {expected_params:,}"
            )

        with torch.no_grad():
            without_auxiliary, _ = model(
                first, second, target=target, teacher_pack=teacher_pack,
                compute_auxiliary=False,
            )
            with_auxiliary, _ = model(
                first, second, target=target, teacher_pack=teacher_pack,
                compute_auxiliary=True,
            )
        for left, right in zip(without_auxiliary, with_auxiliary):
            assert_close(left, right, f"{experiment} auxiliary changed main output", atol=0.0)

        model.zero_grad(set_to_none=True)
        predictions, auxiliary = model(
            first, second, target=target, teacher_pack=teacher_pack,
            compute_auxiliary=True,
        )
        details = auxiliary["z2_srd"]
        if not torch.isfinite(details["total"]):
            raise AssertionError(f"{experiment} auxiliary loss is non-finite")
        details["total"].backward()
        if not nonzero_gradient(model.backbone.parameters()):
            raise AssertionError(f"{experiment} auxiliary did not reach backbone")
        if not nonzero_gradient(model.training_auxiliary.parameters()):
            raise AssertionError(f"{experiment} auxiliary parameters have no gradient")

        even = float(details["z2_even"].detach())
        odd = float(details["z2_odd"].detach())
        if experiment == "N0" and not (even > 0 and odd > 0):
            raise AssertionError("N0 did not activate both separated branches")
        if experiment == "N1" and not (even > 0 and odd == 0):
            raise AssertionError("N1 is not an even-only control")
        if experiment in {"N2", "N3"} and odd != 0:
            raise AssertionError(f"{experiment} unexpectedly activated the odd head")
        if experiment == "N4" and not (even == 0 and odd > 0):
            raise AssertionError("N4 is not an odd-only control")

        swapped_pack = {"t1": teacher_pack["t2"], "t2": teacher_pack["t1"]}
        with torch.no_grad():
            swapped_predictions, swapped_auxiliary = model(
                second, first, target=target, teacher_pack=swapped_pack,
                compute_auxiliary=True,
            )
        for original, exchanged in zip(predictions, swapped_predictions):
            assert_close(
                original, exchanged,
                f"{experiment} main prediction is not exchange invariant",
                atol=5e-6,
            )
        if experiment != "N2":
            assert_close(
                details["total"], swapped_auxiliary["z2_srd"]["total"],
                f"{experiment} auxiliary loss is not exchange invariant", atol=2e-6,
            )

        model.eval()
        with torch.no_grad():
            before = model(first[:1], second[:1])
        model.switch_to_deploy().eval()
        with torch.no_grad():
            after = model(first[:1], second[:1])
        deploy_error = max(
            (left - right).abs().max().item()
            for left, right in zip(before, after)
        )
        deploy_params = sum(parameter.numel() for parameter in model.parameters())
        if deploy_params != 2_913_094 or deploy_error >= 1e-6:
            raise AssertionError(
                f"{experiment} deploy invariant failed: params={deploy_params:,}, "
                f"error={deploy_error:.8e}"
            )
        observed[experiment] = train_params
        del model
    return observed


def train_state_acceptance(args, teacher_pack, device):
    """Check that the Run3 joint Siamese pass removes BN order leakage."""
    size = args.image_size
    batch = args.batch_size
    first = torch.randn(batch, 3, size, size, device=device)
    second = torch.randn_like(first)
    target = torch.zeros(batch, 1, size, size, device=device)
    checkpoint = Path(args.pretrained_path)
    original = build_recipe("N0", checkpoint).to(device).train()
    exchanged = copy.deepcopy(original).train()
    swapped_pack = {"t1": teacher_pack["t2"], "t2": teacher_pack["t1"]}

    original.zero_grad(set_to_none=True)
    output_ab, auxiliary_ab = original(
        first, second, target=target, teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )
    auxiliary_ab["z2_srd"]["total"].backward()
    exchanged.zero_grad(set_to_none=True)
    output_ba, auxiliary_ba = exchanged(
        second, first, target=target, teacher_pack=swapped_pack,
        compute_auxiliary=True,
    )
    auxiliary_ba["z2_srd"]["total"].backward()

    # The group projector and eval/deploy graph are checked at strict
    # tolerances above. CUDA BN/Conv reductions can legitimately accumulate
    # in a different order after the 2B batch is permuted, especially on
    # Blackwell. This tolerance applies only to the train-state diagnostic.
    state_atol = 1e-2 if device.type == "cuda" else 1e-3
    gradient_atol = 2e-2 if device.type == "cuda" else 2e-3
    for left, right in zip(output_ab, output_ba):
        assert_close(left, right, "train-mode main swap state", atol=state_atol)
    assert_close(
        auxiliary_ab["z2_srd"]["total"], auxiliary_ba["z2_srd"]["total"],
        "train-mode auxiliary swap state", atol=state_atol,
    )
    right_buffers = dict(exchanged.named_buffers())
    for name, left in original.named_buffers():
        right = right_buffers[name]
        if torch.is_floating_point(left):
            assert_close(left, right, f"train-mode buffer/{name}", atol=state_atol)
        elif not torch.equal(left, right):
            raise AssertionError(f"train-mode integer buffer differs: {name}")

    right_parameters = dict(exchanged.named_parameters())
    for name, left in original.named_parameters():
        right = right_parameters[name]
        if left.grad is None and right.grad is None:
            continue
        if left.grad is None or right.grad is None:
            raise AssertionError(f"train-mode gradient presence differs: {name}")
        assert_close(
            left.grad, right.grad, f"train-mode gradient/{name}",
            atol=gradient_atol,
        )


def main():
    args = parse_args()
    if args.batch_size < 1 or args.image_size < 64:
        raise ValueError("batch_size must be positive and image_size at least 64")
    if args.cpu:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Smoke checks representation semantics, not reduced-precision speed.
        # Disabling TF32 narrows Blackwell's batch-permutation error envelope.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2333)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2333)
    teacher_pack = evidence_acceptance(args.image_size, args.batch_size, device)
    projector_acceptance(device)
    params = recipe_acceptance(args, teacher_pack, device)
    train_state_acceptance(args, teacher_pack, device)
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    print(
        "[OK] Z2 teacher/projector/controls/grad/train-state/deploy; "
        f"params={params}; deploy=2,913,094; peak={peak:.2f}GiB"
    )
    print("Z2-SRD RUN3 SMOKE PASSED")


if __name__ == "__main__":
    main()
