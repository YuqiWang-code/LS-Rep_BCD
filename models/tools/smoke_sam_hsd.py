#!/usr/bin/env python3
"""Functional acceptance test for corrected SAM-HSD Run1."""

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
    DirectionalTemporalEvidence,
    RelationalStructureBuilder,
    prepare_relational_pack,
)
from models.distill.sam_hsd.losses import scgr_loss


def parse_args():
    parser = argparse.ArgumentParser(description="SAM-HSD Run1 smoke test")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--pretrained_path",
        default=str(PROJECT_ROOT / "pre-trained_weights" / "lwganet_l0_e299.pth"),
    )
    return parser.parse_args()


def raw_pack(size, relabel=False):
    pack = {}
    for phase in ("t1", "t2"):
        instance = torch.zeros(1, size, size, dtype=torch.int32)
        first_id = (11 if phase == "t1" else 21) if relabel else 1
        instance[:, size // 5:size // 2, size // 5:size // 2] = first_id
        if phase == "t2":
            instance[:, size // 2:4 * size // 5, size // 2:4 * size // 5] = \
                37 if relabel else 3
        mask = instance > 0
        dilated = torch.nn.functional.max_pool2d(mask.float(), 3, 1, 1)
        eroded = -torch.nn.functional.max_pool2d(-mask.float(), 3, 1, 1)
        boundary = (dilated - eroded).clamp(0, 1).to(torch.float16)
        quality = mask.to(torch.float16) * 0.9
        pack[phase] = {
            "instance_id": instance, "boundary": boundary, "quality": quality,
        }
    return pack


def batch_pack(pack, batch_size, device):
    return {
        phase: {
            key: value.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            for key, value in structure.items()
        }
        for phase, structure in pack.items()
    }


def assert_close(left, right, message, atol=1e-6):
    error = (left.float() - right.float()).abs().max().item()
    if error > atol:
        raise AssertionError(f"{message}: max_error={error}")


def evidence_acceptance(prepared, batch_size, device, size):
    teacher_pack = batch_pack(prepared, batch_size, device)
    structure = RelationalStructureBuilder().to(device)(teacher_pack)
    builder = DirectionalTemporalEvidence(
        mode="directional", robust_filter=True,
    ).to(device)
    evidence = builder(structure)

    # A newly appearing T2-only object must retain evidence although Q1=0.
    region = (..., slice(size // 2, 4 * size // 5), slice(size // 2, 4 * size // 5))
    if float(evidence["plus"][region].mean()) <= 0:
        raise AssertionError("One-sided appearance was suppressed by missing T1 quality")
    if float(evidence["plus_conf"][region].mean()) < 0.8:
        raise AssertionError("Appearance confidence does not use Q2")

    # Exchange equivariance: E+ <-> E-, while delta/stable stay invariant.
    swapped = {"t1": structure["t2"], "t2": structure["t1"]}
    exchanged = builder(swapped)
    assert_close(evidence["plus"], exchanged["minus"], "exchange E+ -> E-")
    assert_close(evidence["minus"], exchanged["plus"], "exchange E- -> E+")
    assert_close(evidence["delta"], exchanged["delta"], "exchange delta invariance")
    assert_close(evidence["stable"], exchanged["stable"], "exchange stable invariance")
    for index in range(4):
        assert_close(
            evidence["stage_targets"][index]["plus"],
            exchanged["stage_targets"][index]["minus"],
            f"stage {index + 1} exchange equivariance",
        )

    # A one-pixel boundary displacement is matched by tolerance=2, but not by
    # the H8 no-robustification builder.
    tolerance_structure = copy.deepcopy(structure)
    tolerance_structure["t2"] = {
        key: value.clone() for key, value in structure["t1"].items()
    }
    tolerance_structure["t2"]["boundary"] = torch.roll(
        structure["t1"]["boundary"], shifts=1, dims=-1,
    )
    tolerance_structure["t1"]["quality"] = torch.ones_like(
        tolerance_structure["t1"]["quality"],
    )
    tolerance_structure["t2"]["quality"] = torch.ones_like(
        tolerance_structure["t2"]["quality"],
    )
    robust = builder(tolerance_structure)["boundary_delta"].mean()
    unrobust = DirectionalTemporalEvidence(
        mode="directional", robust_filter=False,
    ).to(device)(tolerance_structure)["boundary_delta"].mean()
    if float(robust) >= float(unrobust):
        raise AssertionError("Spatial boundary tolerance did not reduce registration noise")

    for mode in ("boundary_scalar", "unsigned"):
        variant = DirectionalTemporalEvidence(
            mode=mode, robust_filter=True,
        ).to(device)(structure)
        for key in ("plus", "minus", "stable", "delta", "uncertainty"):
            value = variant[key]
            if not torch.isfinite(value).all() or float(value.min()) < 0 or float(value.max()) > 1:
                raise AssertionError(f"Invalid {mode}/{key} evidence")
    return teacher_pack


def nonzero_gradient(parameters):
    return any(
        parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0
        for parameter in parameters
    )


def scgr_confidence_acceptance(device):
    prediction = torch.full((1, 1, 4, 4), 0.5, device=device)
    unchanged = torch.zeros_like(prediction)
    change = torch.ones_like(prediction)
    low_quality_change = {
        "delta": torch.ones_like(prediction),
        "change_conf": torch.zeros_like(prediction),
        "stable": torch.zeros_like(prediction),
        "stable_conf": torch.zeros_like(prediction),
    }
    _, stats = scgr_loss(prediction, unchanged, low_quality_change)
    if float(stats["hard_negative_ratio"]) != 0.0:
        raise AssertionError("Low-quality SAM delta became a hard negative")
    _, stats = scgr_loss(prediction, change, low_quality_change)
    if float(stats["hard_positive_ratio"]) != 0.0:
        raise AssertionError("Missing SAM coverage became a hard positive")
    confident_stable = {
        **low_quality_change,
        "stable": torch.ones_like(prediction),
        "stable_conf": torch.ones_like(prediction),
    }
    _, stats = scgr_loss(prediction, change, confident_stable)
    if float(stats["hard_positive_ratio"]) != 1.0:
        raise AssertionError("Confident stable-vs-change conflict was not routed")


def main():
    args = parse_args()
    if args.batch_size < 1 or args.image_size < 32:
        raise ValueError("batch_size must be positive and image_size at least 32")
    if args.cpu:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")

    prepared = prepare_relational_pack(raw_pack(args.image_size))
    relabeled = prepare_relational_pack(raw_pack(args.image_size, relabel=True))
    for phase in ("t1", "t2"):
        for key in (
            "occupancy", "affinity_r1", "affinity_r2", "affinity_r4",
            "interior_depth", "object_scale", "compactness",
        ):
            if not torch.equal(prepared[phase][key], relabeled[phase][key]):
                raise AssertionError(f"Instance-label permutation changed {phase}/{key}")
    teacher_pack = evidence_acceptance(
        prepared, args.batch_size, device, args.image_size,
    )
    scgr_confidence_acceptance(device)

    checkpoint = Path(args.pretrained_path)
    model = A2Net_LWGANet_L0(
        pretrained=checkpoint.is_file(),
        pretrained_path=str(checkpoint) if checkpoint.is_file() else None,
        auxiliary_mode="sam_hsd",
        sam_hsd_cfg={
            "mode": "full", "evidence_mode": "directional",
            "use_scgr": True, "robust_filter": True, "boundary_band": 7,
        },
    ).to(device)
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    first = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device,
    )
    second = torch.randn_like(first)
    target = torch.zeros(
        args.batch_size, 1, args.image_size, args.image_size, device=device,
    )
    target[..., args.image_size // 2:4 * args.image_size // 5,
           args.image_size // 2:4 * args.image_size // 5] = 1.0

    with torch.no_grad():
        output_without_aux, _ = model(
            first, second, target=target, teacher_pack=teacher_pack,
            compute_auxiliary=False,
        )
        output_with_aux, _ = model(
            first, second, target=target, teacher_pack=teacher_pack,
            compute_auxiliary=True,
        )
    aux_toggle_error = max(
        (left - right).abs().max().item()
        for left, right in zip(output_without_aux, output_with_aux)
    )
    if aux_toggle_error != 0.0:
        raise AssertionError(f"Auxiliary changed the main output: {aux_toggle_error}")

    model.zero_grad(set_to_none=True)
    _, auxiliary = model(
        first, second, target=target, teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )
    # Auxiliary-only backward proves the teacher path itself reaches both the
    # encoder and decoder; the main BCE/Dice cannot mask a disconnected loss.
    auxiliary["sam_hsd"]["total"].backward()
    if not nonzero_gradient(model.backbone.parameters()):
        raise AssertionError("Auxiliary-only loss did not reach LWGANet")
    if not nonzero_gradient(model.decoder.parameters()):
        raise AssertionError("Decoder-HSD did not reach decoder parameters")
    if not nonzero_gradient(model.training_auxiliary.encoder.probes.parameters()):
        raise AssertionError("Encoder probes did not receive gradients")
    if not nonzero_gradient(model.training_auxiliary.encoder.direction_head.parameters()):
        raise AssertionError("Directional head did not receive gradients")

    model.eval()
    with torch.no_grad():
        before = model(first[:1], second[:1])
    train_params = sum(parameter.numel() for parameter in model.parameters())
    model.switch_to_deploy().eval()
    with torch.no_grad():
        after = model(first[:1], second[:1])
    deploy_error = max(
        (left - right).abs().max().item() for left, right in zip(before, after)
    )
    deploy_params = sum(parameter.numel() for parameter in model.parameters())
    if train_params != 2_920_791 or deploy_params != 2_913_094 or deploy_error >= 1e-6:
        raise AssertionError(
            f"Deployment invariant failed: {train_params}->{deploy_params}, "
            f"error={deploy_error}"
        )
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    print(
        "[OK] direction-quality/spatial-tolerance/stage hierarchy/exchange equivariance/"
        "feature+prediction Decoder-HSD/SCGR/aux-only gradients/deploy; "
        f"params={train_params:,}->{deploy_params:,}; "
        f"aux_toggle_error={aux_toggle_error}; deploy_error={deploy_error}; "
        f"peak={peak:.2f}GiB"
    )
    print("SAM-HSD RUN1 SMOKE PASSED")


if __name__ == "__main__":
    main()
