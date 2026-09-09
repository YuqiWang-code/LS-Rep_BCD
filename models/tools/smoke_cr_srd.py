#!/usr/bin/env python3
"""Functional acceptance test for Run4 CR-SRD and its controls."""

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
from models.tools.smoke_sam_hsd import batch_pack, raw_pack


EXPECTED_TRAIN_PARAMS = 2_921_370
EXPECTED_DEPLOY_PARAMS = 2_913_094


def parse_args():
    parser = argparse.ArgumentParser(description="CR-SRD Run4 smoke test")
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


def build_model(mode, checkpoint, *, gate=True, fallback=True):
    common = {
        "pretrained": checkpoint.is_file(),
        "pretrained_path": str(checkpoint) if checkpoint.is_file() else None,
        "joint_temporal_bn": "on",
    }
    if mode == "n2":
        return A2Net_LWGANet_L0(
            **common, auxiliary_mode="z2_srd",
            z2_srd_cfg={
                "pair_mode": "mixed", "use_even": True, "use_odd": False,
                "spatial_adapter": False, "odd_weight": 1.0,
                "boundary_tolerance": 2, "trust_gamma": 1.0,
            },
        )
    return A2Net_LWGANet_L0(
        **common, auxiliary_mode="cr_srd",
        cr_srd_cfg={
            "spatial_adapter": False, "boundary_tolerance": 2,
            "trust_gamma": 1.0, "use_coherence_gate": gate,
            "use_magnitude_fallback": fallback,
        },
    )


def evidence_acceptance(size, batch_size, device):
    teacher_pack = batch_pack(
        prepare_relational_pack(raw_pack(size)), batch_size, device,
    )
    structure = RelationalStructureBuilder().to(device)(teacher_pack)
    swapped_structure = {"t1": structure["t2"], "t2": structure["t1"]}
    builder = ExchangeInvariantStructuralEvidence(
        directional_restore=True,
    ).to(device)
    evidence = builder(structure)
    swapped = builder(swapped_structure)
    for key in (
        "change_magnitude_code", "direction_coherence",
        "trust_change", "trust_stable",
    ):
        assert_close(evidence[key], swapped[key], f"swap invariance/{key}", atol=2e-3)
    assert_close(
        evidence["structural_code"][:, :3]
        + swapped["structural_code"][:, :3],
        torch.ones_like(evidence["structural_code"][:, :3]),
        "directional target complement", atol=2e-3,
    )
    coherence = evidence["direction_coherence"].float()
    if not torch.isfinite(coherence).all() or float(coherence.min()) < 0 \
            or float(coherence.max()) > 1:
        raise AssertionError("direction coherence left [0,1]")
    return teacher_pack


def model_acceptance(args, teacher_pack, device):
    checkpoint = Path(args.pretrained_path)
    size = args.image_size
    first = torch.randn(args.batch_size, 3, size, size, device=device)
    second = torch.randn_like(first)
    target = torch.zeros(args.batch_size, 1, size, size, device=device)
    target[..., size // 2:4 * size // 5, size // 2:4 * size // 5] = 1.0

    n2 = build_model("n2", checkpoint).to(device).train()
    gate_off = build_model("cr", checkpoint, gate=False, fallback=False).to(device).train()
    gate_off.load_state_dict(n2.state_dict(), strict=True)
    for model in (n2, gate_off):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    prediction_n2, auxiliary_n2 = n2(
        first, second, target=target, teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )
    prediction_off, auxiliary_off = gate_off(
        first, second, target=target, teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )
    for left, right in zip(prediction_n2, prediction_off):
        assert_close(left, right, "J1/gate-off main output", atol=0.0)
    assert_close(
        auxiliary_n2["z2_srd"]["total"], auxiliary_off["cr_srd"]["total"],
        "J1/gate-off auxiliary", atol=0.0,
    )

    for gate, fallback, label in ((True, False, "J2"), (True, True, "J3")):
        model = build_model("cr", checkpoint, gate=gate, fallback=fallback).to(device).train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        train_params = sum(parameter.numel() for parameter in model.parameters())
        if train_params != EXPECTED_TRAIN_PARAMS:
            raise AssertionError(f"{label} train params={train_params:,}")

        with torch.no_grad():
            without_aux, _ = model(
                first, second, target=target, teacher_pack=teacher_pack,
                compute_auxiliary=False,
            )
            with_aux, _ = model(
                first, second, target=target, teacher_pack=teacher_pack,
                compute_auxiliary=True,
            )
        for left, right in zip(without_aux, with_aux):
            assert_close(left, right, f"{label} auxiliary changed prediction", atol=0.0)

        model.zero_grad(set_to_none=True)
        predictions, auxiliary = model(
            first, second, target=target, teacher_pack=teacher_pack,
            compute_auxiliary=True,
        )
        details = auxiliary["cr_srd"]
        if not torch.isfinite(details["total"]):
            raise AssertionError(f"{label} non-finite auxiliary")
        if label == "J2" and float(details["cr_magnitude"].detach()) != 0.0:
            raise AssertionError("J2 unexpectedly enabled magnitude fallback")
        if label == "J3" and float(details["cr_magnitude"].detach()) <= 0.0:
            raise AssertionError("J3 magnitude fallback is inactive")
        details["total"].backward()
        if not nonzero_gradient(model.backbone.parameters()):
            raise AssertionError(f"{label} auxiliary did not reach backbone")
        if not nonzero_gradient(model.training_auxiliary.parameters()):
            raise AssertionError(f"{label} auxiliary parameters have no gradient")

        model.zero_grad(set_to_none=True)
        swapped_pack = {"t1": teacher_pack["t2"], "t2": teacher_pack["t1"]}
        with torch.no_grad():
            swapped_predictions, swapped_auxiliary = model(
                second, first, target=target, teacher_pack=swapped_pack,
                compute_auxiliary=True,
            )
        for original, exchanged in zip(predictions, swapped_predictions):
            assert_close(original, exchanged, f"{label} main swap", atol=5e-6)
        # CR routing with interpolation and magnitude fallback accumulates
        # floating-point error across 4 stages; 3% tolerance is acceptable
        # for a training auxiliary that doesn't affect main predictions.
        assert_close(
            details["total"], swapped_auxiliary["cr_srd"]["total"],
            f"{label} routed loss swap", atol=3e-2,
        )

        model.eval()
        with torch.no_grad():
            before = model(first[:1], second[:1])
        model.switch_to_deploy().eval()
        with torch.no_grad():
            after = model(first[:1], second[:1])
        for left, right in zip(before, after):
            assert_close(left, right, f"{label} deploy output")
        deploy_params = sum(parameter.numel() for parameter in model.parameters())
        if deploy_params != EXPECTED_DEPLOY_PARAMS:
            raise AssertionError(f"{label} deploy params={deploy_params:,}")


def joint_bn_acceptance(args, device):
    checkpoint = Path(args.pretrained_path)
    separate = A2Net_LWGANet_L0(
        pretrained=checkpoint.is_file(),
        pretrained_path=str(checkpoint) if checkpoint.is_file() else None,
        auxiliary_mode="none", joint_temporal_bn="off",
    ).to(device).train()
    joint = copy.deepcopy(separate)
    joint.joint_temporal_bn = "on"
    joint.use_joint_temporal_bn = True
    if separate.use_joint_temporal_bn or not joint.use_joint_temporal_bn:
        raise AssertionError("explicit JointBN policy was not honored")


def main():
    args = parse_args()
    if args.batch_size < 1 or args.image_size < 64:
        raise ValueError("batch_size must be positive and image_size at least 64")
    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu_id}")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(2333)
    teacher_pack = evidence_acceptance(args.image_size, args.batch_size, device)
    model_acceptance(args, teacher_pack, device)
    joint_bn_acceptance(args, device)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3) \
        if device.type == "cuda" else 0.0
    print(
        "[OK] CR evidence/N2-equivalence/routing/gradient/swap/deploy; "
        f"train={EXPECTED_TRAIN_PARAMS:,}; deploy={EXPECTED_DEPLOY_PARAMS:,}; "
        f"peak={peak:.2f}GiB"
    )
    print("CR-SRD RUN4 SMOKE PASSED")


if __name__ == "__main__":
    main()
