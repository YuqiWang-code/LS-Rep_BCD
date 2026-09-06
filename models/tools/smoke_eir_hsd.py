#!/usr/bin/env python3
"""Functional acceptance test for EIR-HSD Run2 R0-R8."""

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
    prepare_relational_pack,
)
from models.distill.sam_hsd.losses import (
    residual_correction_loss,
    structural_code_loss,
)
from models.tools.smoke_sam_hsd import batch_pack, raw_pack


RECIPES = {
    "R0": ("none", None, 2_913_094),
    "R1": ("sam_hsd", {
        "mode": "full", "evidence_mode": "unsigned", "use_scgr": True,
        "robust_filter": True, "boundary_band": 7,
    }, 2_920_791),
    "R2": ("eir_hsd", {
        "mode": "encoder", "spatial_adapter": False,
        "use_correction": False, "fixed_fusion": False,
        "directional_restore": False,
    }, 2_921_370),
    "R3": ("eir_hsd", {
        "mode": "encoder", "spatial_adapter": True,
        "use_correction": False, "fixed_fusion": False,
        "directional_restore": False,
    }, 2_922_970),
    "R4": ("eir_hsd", {
        "mode": "decoder", "spatial_adapter": False,
        "use_correction": True, "fixed_fusion": False,
        "directional_restore": False,
    }, 2_914_330),
    "R5": ("eir_hsd", {
        "mode": "full", "spatial_adapter": True,
        "use_correction": True, "fixed_fusion": False,
        "directional_restore": False,
    }, 2_924_206),
    "R6": ("eir_hsd", {
        "mode": "full", "spatial_adapter": True,
        "use_correction": True, "fixed_fusion": True,
        "directional_restore": False,
    }, 2_924_206),
    "R7": ("eir_hsd", {
        "mode": "full", "spatial_adapter": True,
        "use_correction": False, "fixed_fusion": False,
        "directional_restore": False,
    }, 2_924_206),
    "R8": ("eir_hsd", {
        "mode": "full", "spatial_adapter": True,
        "use_correction": True, "fixed_fusion": False,
        "directional_restore": True,
    }, 2_924_206),
}


def parse_args():
    parser = argparse.ArgumentParser(description="EIR-HSD Run2 smoke test")
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
        raise AssertionError(f"{message}: max_error={error}")


def nonzero_gradient(parameters):
    return any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.detach().abs().sum()) > 0
        for parameter in parameters
    )


def evidence_acceptance(size, batch_size, device):
    prepared = prepare_relational_pack(raw_pack(size))
    relabeled = prepare_relational_pack(raw_pack(size, relabel=True))
    for phase in ("t1", "t2"):
        for key in (
            "occupancy", "affinity_r1", "affinity_r2", "affinity_r4",
            "interior_depth", "object_scale", "compactness",
        ):
            if not torch.equal(prepared[phase][key], relabeled[phase][key]):
                raise AssertionError(f"Instance relabeling changed {phase}/{key}")

    teacher_pack = batch_pack(prepared, batch_size, device)
    structure = RelationalStructureBuilder().to(device)(teacher_pack)
    builder = ExchangeInvariantStructuralEvidence().to(device)
    evidence = builder(structure)
    swapped = {"t1": structure["t2"], "t2": structure["t1"]}
    exchanged = builder(swapped)
    for key in (
        "structural_code", "boundary_residual", "local_residual",
        "geometry_residual", "stable_consensus", "trust_change",
        "trust_stable", "uncertainty",
    ):
        assert_close(evidence[key], exchanged[key], f"exchange invariance/{key}")

    directional = ExchangeInvariantStructuralEvidence(
        directional_restore=True,
    ).to(device)(structure)
    directional_swapped = ExchangeInvariantStructuralEvidence(
        directional_restore=True,
    ).to(device)(swapped)
    assert_close(
        directional["structural_code"][:, :3]
        + directional_swapped["structural_code"][:, :3],
        torch.ones_like(directional["structural_code"][:, :3]),
        "R8 signed channels did not reverse under exchange", atol=2e-3,
    )
    assert_close(
        directional["structural_code"][:, 3:],
        directional_swapped["structural_code"][:, 3:],
        "R8 stable channel is not invariant",
    )

    uncovered = copy.deepcopy(structure)
    for phase in ("t1", "t2"):
        uncovered[phase] = {
            key: value.clone() for key, value in uncovered[phase].items()
        }
        uncovered[phase]["quality"].zero_()
    abstained = builder(uncovered)
    if float(abstained["trust_change"].abs().sum()) != 0.0 \
            or float(abstained["trust_stable"].abs().sum()) != 0.0:
        raise AssertionError("Missing SAM coverage did not abstain")
    prediction = torch.rand(batch_size, 4, size // 4, size // 4, device=device)
    zero_loss, _ = structural_code_loss(
        prediction, abstained["structural_code"],
        abstained["trust_change"], abstained["trust_stable"],
    )
    if float(zero_loss) != 0.0:
        raise AssertionError("Zero-trust structural loss is nonzero")
    return teacher_pack


def correction_acceptance(device):
    size = 8
    target = torch.zeros(1, 1, size, size, device=device)
    target[..., :4, :] = 1.0
    logits = torch.full_like(target, 2.0, requires_grad=True)
    with torch.no_grad():
        logits[..., :4, :] = -2.0
    prediction = torch.sigmoid(logits)
    evidence = {
        "change_strength": target.clone(),
        "trust_change": target.clone(),
        "stable_consensus": 1.0 - target,
        "trust_stable": 1.0 - target,
    }
    result = residual_correction_loss(
        (prediction,) * 4, target, evidence,
    )
    result["total"].backward()
    if float(logits.grad[..., :4, :].mean()) >= 0:
        raise AssertionError("FN correction does not increase change probability")
    if float(logits.grad[..., 4:, :].mean()) <= 0:
        raise AssertionError("FP correction does not suppress change probability")
    if float(result["fn_ratio"]) != 0.5 or float(result["fp_ratio"]) != 0.5:
        raise AssertionError("Student-error correction masks are incorrect")


def build_recipe(experiment, checkpoint):
    mode, config, _ = RECIPES[experiment]
    kwargs = {
        "pretrained": checkpoint.is_file(),
        "pretrained_path": str(checkpoint) if checkpoint.is_file() else None,
        "auxiliary_mode": mode,
    }
    if mode == "sam_hsd":
        kwargs["sam_hsd_cfg"] = config
    elif mode == "eir_hsd":
        kwargs["eir_hsd_cfg"] = config
    return A2Net_LWGANet_L0(**kwargs)


def recipe_acceptance(args, teacher_pack, device):
    size = args.image_size
    first = torch.randn(args.batch_size, 3, size, size, device=device)
    second = torch.randn_like(first)
    target = torch.zeros(args.batch_size, 1, size, size, device=device)
    target[..., size // 2:4 * size // 5, size // 2:4 * size // 5] = 1.0
    checkpoint = Path(args.pretrained_path)
    observed = {}

    for experiment, (mode, _, expected_params) in RECIPES.items():
        model = build_recipe(experiment, checkpoint).to(device).train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        train_params = sum(parameter.numel() for parameter in model.parameters())
        if train_params != expected_params:
            raise AssertionError(
                f"{experiment} train params {train_params:,} != {expected_params:,}"
            )

        with torch.no_grad():
            without_aux, _ = model(
                first, second, target=target, teacher_pack=teacher_pack,
                compute_auxiliary=False,
            )
            with_aux, _ = model(
                first, second, target=target, teacher_pack=teacher_pack,
                compute_auxiliary=mode != "none",
            )
        error = max(
            (left - right).abs().max().item()
            for left, right in zip(without_aux, with_aux)
        )
        if error != 0.0:
            raise AssertionError(f"{experiment} auxiliary changed main output: {error}")

        model.zero_grad(set_to_none=True)
        train_predictions, auxiliary = model(
            first, second, target=target, teacher_pack=teacher_pack,
            compute_auxiliary=mode != "none",
        )
        if mode != "none":
            key = "sam_hsd" if mode == "sam_hsd" else "eir_hsd"
            details = auxiliary[key]
            if not torch.isfinite(details["total"]):
                raise AssertionError(f"{experiment} auxiliary loss is non-finite")
            details["total"].backward(retain_graph=experiment == "R7")
            if not nonzero_gradient(model.backbone.parameters()):
                raise AssertionError(f"{experiment} auxiliary did not reach backbone")
            if mode == "eir_hsd" and not nonzero_gradient(
                model.training_auxiliary.parameters()
            ):
                raise AssertionError(f"{experiment} EIR parameters have no gradient")
            if experiment == "R5":
                assert_close(
                    details["total"], details["encoder"] + details["decoder"],
                    "R5 is not independent normalized fusion",
                )
            if experiment == "R6":
                assert_close(
                    details["total"],
                    0.5 * details["encoder"] + 0.5 * details["decoder"],
                    "R6 is not fixed 0.5/0.5 fusion",
                )
            if experiment == "R7" and float(details["correction"]) != 0.0:
                raise AssertionError("R7 residual correction is enabled")
            if experiment == "R7":
                direct_prediction_grads = torch.autograd.grad(
                    details["decoder"], train_predictions,
                    retain_graph=True, allow_unused=True,
                )
                if any(
                    gradient is not None and float(gradient.abs().sum()) > 0
                    for gradient in direct_prediction_grads
                ):
                    raise AssertionError(
                        "R7 structure/relation directly supervises main probabilities"
                    )
            if experiment == "R5":
                swapped_pack = {
                    "t1": teacher_pack["t2"], "t2": teacher_pack["t1"],
                }
                with torch.no_grad():
                    swapped_predictions, swapped_auxiliary = model(
                        second, first, target=target,
                        teacher_pack=swapped_pack, compute_auxiliary=True,
                    )
                for original, swapped_prediction in zip(
                    train_predictions, swapped_predictions,
                ):
                    assert_close(
                        original, swapped_prediction,
                        "R5 main prediction is not exchange invariant",
                    )
                assert_close(
                    details["total"], swapped_auxiliary["eir_hsd"]["total"],
                    "R5 auxiliary is not exchange invariant",
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
                f"{experiment} deploy invariant: {train_params}->{deploy_params}, "
                f"error={deploy_error}"
            )
        observed[experiment] = train_params
        del model
    return observed


def main():
    args = parse_args()
    if args.batch_size < 1 or args.image_size < 64:
        raise ValueError("batch_size must be positive and image_size at least 64")
    if args.cpu:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    teacher_pack = evidence_acceptance(
        args.image_size, args.batch_size, device,
    )
    correction_acceptance(device)
    params = recipe_acceptance(args, teacher_pack, device)
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    print(
        "[OK] EIR code/exchange/abstention/permutation/correction/R0-R8/"
        "aux-grad/deploy; "
        f"params={params}; deploy=2,913,094; peak={peak:.2f}GiB"
    )
    print("EIR-HSD RUN2 SMOKE PASSED")


if __name__ == "__main__":
    main()
