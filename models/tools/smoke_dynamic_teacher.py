#!/usr/bin/env python3
"""
Synthetic smoke test for RDT-CD reciprocal dynamic teachers.

This script is intentionally a SMALL correctness test.  It does not train a
formal experiment and it does not read/write the real Teacher Cache.

It verifies the P0 contracts required by the project:

1. dynamic_teacher forward is finite;
2. auxiliary ON/OFF does not change the main prediction;
3. Student loss does not update fast-teacher parameters;
4. Teacher fitting loss does not backpropagate into the Student;
5. EMA target teachers are frozen and actually evolve after a fast-teacher step;
6. dynamic-teacher state is present in model.state_dict();
7. switch_to_deploy() removes every training-only teacher module;
8. deploy prediction is unchanged (< 1e-6);
9. deployed parameter count is exactly 2,913,094.

Run from the project root, for example:

    python models/tools/smoke_dynamic_teacher.py --device cuda --gpu_id 0

CPU is also supported:

    python models/tools/smoke_dynamic_teacher.py --device cpu

This smoke uses a synthetic SAMStruct/OVCDistill-shaped teacher_pack.  A
separate real-cache dry run should be executed after all modified project files
have been installed.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import A2Net_LWGANet_L0  # noqa: E402


EXPECTED_DEPLOY_PARAMS = 2_913_094
PREDICTION_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic P0 smoke test for RDT-CD dynamic teachers"
    )

    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2333)

    parser.add_argument("--teacher_hidden", type=int, default=24)
    parser.add_argument("--teacher_lr", type=float, default=1e-3)
    parser.add_argument("--teacher_ema", type=float, default=0.99)
    parser.add_argument("--max_logit_delta", type=float, default=2.0)
    parser.add_argument("--kd_lambda", type=float, default=0.06)

    parser.add_argument(
        "--no-cache-conditioning",
        action="store_true",
        help=(
            "Smoke the capacity-matched no-cache dynamic-teacher control. "
            "The default tests the full cache-conditioned path using a "
            "synthetic SAM/OV teacher_pack."
        ),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    if args.height != 256 or args.width != 256:
        raise ValueError(
            "The project deploy contract is defined for 256x256 inputs"
        )

    if args.teacher_hidden <= 0:
        raise ValueError("--teacher_hidden must be positive")

    if args.teacher_lr <= 0:
        raise ValueError("--teacher_lr must be positive")

    if not 0.0 <= args.teacher_ema < 1.0:
        raise ValueError("--teacher_ema must satisfy 0 <= ema < 1")

    if args.max_logit_delta <= 0:
        raise ValueError("--max_logit_delta must be positive")

    if args.kd_lambda < 0:
        raise ValueError("--kd_lambda must be non-negative")

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but CUDA is not available"
            )

        if not 0 <= args.gpu_id < torch.cuda.device_count():
            raise ValueError(
                f"Invalid --gpu_id={args.gpu_id}; "
                f"visible CUDA device count={torch.cuda.device_count()}"
            )

        torch.cuda.set_device(args.gpu_id)
        return torch.device("cuda", args.gpu_id)

    return torch.device("cpu")


def parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def grad_norm(
    parameters: Iterable[torch.nn.Parameter],
) -> float:
    square_sum = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach().float()

        square_sum += (
            gradient.square()
            .sum()
            .item()
        )

    return math.sqrt(square_sum)


def parameter_rms_difference(
    before: Tuple[torch.Tensor, ...],
    after_parameters: Iterable[torch.nn.Parameter],
) -> float:
    square_sum = 0.0
    count = 0

    after = tuple(
        parameter.detach()
        for parameter in after_parameters
    )

    if len(before) != len(after):
        raise AssertionError(
            "Parameter snapshot length changed unexpectedly"
        )

    for old, new in zip(before, after):
        delta = (
            new.float()
            - old.float()
        )

        square_sum += (
            delta.square()
            .sum()
            .item()
        )

        count += delta.numel()

    return math.sqrt(
        square_sum / max(count, 1)
    )


def snapshot_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().clone()
        for parameter in parameters
    )


def max_prediction_difference(
    left,
    right,
) -> float:
    if len(left) != len(right):
        raise AssertionError(
            "Prediction tuple lengths do not match"
        )

    maximum = 0.0

    for x, y in zip(left, right):
        if x.shape != y.shape:
            raise AssertionError(
                f"Prediction shapes differ: {x.shape} vs {y.shape}"
            )

        maximum = max(
            maximum,
            (
                x.detach().float()
                - y.detach().float()
            )
            .abs()
            .max()
            .item(),
        )

    return maximum


def assert_finite_tensor(
    name: str,
    value: torch.Tensor,
) -> None:
    if not torch.is_tensor(value):
        raise AssertionError(
            f"{name} is not a tensor"
        )

    if not torch.isfinite(value).all():
        raise AssertionError(
            f"{name} contains NaN/Inf"
        )


# ---------------------------------------------------------------------------
# Synthetic CD batch and synthetic Teacher Cache
# ---------------------------------------------------------------------------

def make_synthetic_target(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a deterministic binary change mask containing several geometric
    regions.  The foreground is neither empty nor dominant.
    """
    target = torch.zeros(
        batch_size,
        1,
        height,
        width,
        device=device,
        dtype=torch.float32,
    )

    for batch_index in range(batch_size):
        offset = 7 * batch_index

        target[
            batch_index,
            0,
            36 + offset:92 + offset,
            42:106,
        ] = 1.0

        target[
            batch_index,
            0,
            132:178,
            148 - offset:210 - offset,
        ] = 1.0

        target[
            batch_index,
            0,
            202:218,
            62 + offset:90 + offset,
        ] = 1.0

    return target


def make_instance_grid(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    block: int = 32,
) -> torch.Tensor:
    """
    Build non-zero SAM-like opaque instance IDs.

    Every block contains many pixels, so leave-one-out instance transport has
    valid source support.
    """
    yy = torch.arange(
        height,
        device=device,
    ).view(height, 1)

    xx = torch.arange(
        width,
        device=device,
    ).view(1, width)

    columns = math.ceil(
        width / block
    )

    ids = (
        (yy // block) * columns
        + (xx // block)
        + 1
    ).to(torch.int32)

    return (
        ids
        .view(1, 1, height, width)
        .expand(batch_size, -1, -1, -1)
        .clone()
    )


def make_boundary_from_ids(
    ids: torch.Tensor,
) -> torch.Tensor:
    """
    Cheap 4-neighbour instance boundary map in the same raster geometry.
    """
    boundary = torch.zeros_like(
        ids,
        dtype=torch.float32,
    )

    boundary[:, :, 1:, :] = torch.maximum(
        boundary[:, :, 1:, :],
        (
            ids[:, :, 1:, :]
            != ids[:, :, :-1, :]
        ).float(),
    )

    boundary[:, :, :-1, :] = torch.maximum(
        boundary[:, :, :-1, :],
        (
            ids[:, :, 1:, :]
            != ids[:, :, :-1, :]
        ).float(),
    )

    boundary[:, :, :, 1:] = torch.maximum(
        boundary[:, :, :, 1:],
        (
            ids[:, :, :, 1:]
            != ids[:, :, :, :-1]
        ).float(),
    )

    boundary[:, :, :, :-1] = torch.maximum(
        boundary[:, :, :, :-1],
        (
            ids[:, :, :, 1:]
            != ids[:, :, :, :-1]
        ).float(),
    )

    return boundary


def make_synthetic_teacher_pack(
    target: torch.Tensor,
) -> Dict:
    """
    Construct the exact fields used by build_task_proposals():

        SAM:
            t1/t2 instance_id
            t1/t2 boundary
            t1/t2 quality

        OV:
            soft_change l1/l2
            confidence l1/l2

    Relation fields are deliberately absent because current task_space.py does
    not consume them.
    """
    batch_size, _, height, width = (
        target.shape
    )

    device = target.device

    ids_t1 = make_instance_grid(
        batch_size,
        height,
        width,
        device,
        block=32,
    )

    # Make T2 partitions different while preserving valid opaque IDs.
    ids_t2 = torch.roll(
        ids_t1,
        shifts=(16, 16),
        dims=(-2, -1),
    )

    boundary_t1 = make_boundary_from_ids(
        ids_t1
    )

    boundary_t2 = make_boundary_from_ids(
        ids_t2
    )

    quality_t1 = torch.full(
        target.shape,
        0.90,
        device=device,
        dtype=torch.float32,
    )

    quality_t2 = torch.full(
        target.shape,
        0.85,
        device=device,
        dtype=torch.float32,
    )

    # Keep boundary locations slightly less reliable.
    quality_t1 = quality_t1 * (
        1.0 - 0.25 * boundary_t1
    )

    quality_t2 = quality_t2 * (
        1.0 - 0.25 * boundary_t2
    )

    # OV-like multi-scale soft change.
    ov_l1 = F.interpolate(
        target,
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
    )

    ov_l2 = F.interpolate(
        target,
        size=(16, 16),
        mode="bilinear",
        align_corners=False,
    )

    # Make them soft rather than exact GT copies.
    ov_l1 = (
        0.80 * ov_l1
        + 0.10
    ).clamp(0.0, 1.0)

    ov_l2 = (
        0.70 * ov_l2
        + 0.15
    ).clamp(0.0, 1.0)

    confidence_l1 = torch.full_like(
        ov_l1,
        0.88,
    )

    confidence_l2 = torch.full_like(
        ov_l2,
        0.78,
    )

    return {
        "sam": {
            "t1": {
                "instance_id": ids_t1,
                "boundary": boundary_t1,
                "quality": quality_t1,
            },
            "t2": {
                "instance_id": ids_t2,
                "boundary": boundary_t2,
                "quality": quality_t2,
            },
        },
        "ov": {
            "soft_change": {
                "l1": ov_l1,
                "l2": ov_l2,
            },
            "confidence": {
                "l1": confidence_l1,
                "l2": confidence_l2,
            },
        },
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_dynamic_model(
    args: argparse.Namespace,
    device: torch.device,
) -> A2Net_LWGANet_L0:
    routing_cfg = {
        "mechanism": "dynamic_teacher",
        "hidden": args.teacher_hidden,
        "ema": args.teacher_ema,
        "max_logit_delta": args.max_logit_delta,
        "boundary_radius": 2,
        "small_area": 64,
        "policy": "advantage",
        "teacher": "both",
        "difficulty": True,
        "cache_conditioning": (
            not args.no_cache_conditioning
        ),
    }

    model = A2Net_LWGANet_L0(
        pretrained=False,
        pretrained_path=None,
        auxiliary_mode="direction_c",
        routing_cfg=routing_cfg,
    )

    return model.to(device)


# ---------------------------------------------------------------------------
# Individual smoke checks
# ---------------------------------------------------------------------------

def check_state_registration(
    model: A2Net_LWGANet_L0,
) -> None:
    state_keys = tuple(
        model.state_dict().keys()
    )

    fast_present = any(
        key.startswith(
            "training_auxiliary.fast."
        )
        for key in state_keys
    )

    target_present = any(
        key.startswith(
            "training_auxiliary.target."
        )
        for key in state_keys
    )

    if not fast_present:
        raise AssertionError(
            "Fast dynamic-teacher parameters are absent from model.state_dict()"
        )

    if not target_present:
        raise AssertionError(
            "EMA target-teacher parameters are absent from model.state_dict()"
        )

    if any(
        parameter.requires_grad
        for parameter
        in model.training_auxiliary.target_teacher_parameters()
    ):
        raise AssertionError(
            "EMA target-teacher parameters must all have requires_grad=False"
        )

    print(
        "[PASS] state registration: "
        "fast + EMA target teachers are checkpoint-visible"
    )


@torch.no_grad()
def check_auxiliary_prediction_invariance(
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    teacher_pack,
    seed: int,
) -> float:
    """
    Compare the unchanged main path with auxiliary disabled/enabled.

    Model stays in train mode because auxiliary computation is intentionally
    training-only.  Torch RNG is reset before each forward so stochastic
    backbone operations, if any, receive identical random streams.
    """
    model.train()

    torch.manual_seed(seed)

    if image_a.is_cuda:
        torch.cuda.manual_seed_all(seed)

    predictions_off, _ = model(
        image_a,
        image_b,
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=False,
    )

    torch.manual_seed(seed)

    if image_a.is_cuda:
        torch.cuda.manual_seed_all(seed)

    predictions_on, auxiliary = model(
        image_a,
        image_b,
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )

    if "direction_c" not in auxiliary:
        raise AssertionError(
            "Dynamic auxiliary output is missing"
        )

    maximum = max_prediction_difference(
        predictions_off,
        predictions_on,
    )

    if maximum >= PREDICTION_ATOL:
        raise AssertionError(
            "Auxiliary ON/OFF changed main prediction: "
            f"max_abs={maximum:.9e}"
        )

    print(
        "[PASS] auxiliary ON/OFF prediction invariance: "
        f"max_abs={maximum:.9e}"
    )

    return maximum


def check_forward_and_gradient_isolation(
    args: argparse.Namespace,
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    teacher_pack,
) -> Dict[str, float]:
    """
    Verify the two disjoint optimization graphs.

    A. Student objective:
           BCE(main prediction, GT)
           + kd_lambda * teacher->student KD

       must give:
           student grad > 0
           fast-teacher grad == 0

    B. Teacher objective:
           teacher_total

       must give:
           fast-teacher grad > 0
           student grad == 0
    """
    model.train()

    model.zero_grad(
        set_to_none=True
    )

    predictions, auxiliary = model(
        image_a,
        image_b,
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )

    dynamic = auxiliary[
        "direction_c"
    ]

    required = (
        "total",
        "teacher_total",
        "student_brier",
        "dynamic_teacher_brier",
        "dynamic_teacher_gain",
        "dynamic_shift",
        "effective_mass",
        "effective_per_error_mass",
    )

    for key in required:
        if key not in dynamic:
            raise AssertionError(
                f"Missing dynamic-teacher output key: {key}"
            )

        assert_finite_tensor(
            key,
            dynamic[key],
        )

    main_loss = (
        F.binary_cross_entropy(
            predictions[0],
            target,
        )
    )

    student_total = (
        main_loss
        + args.kd_lambda
        * dynamic["total"]
    )

    teacher_total = dynamic[
        "teacher_total"
    ]

    assert_finite_tensor(
        "main_loss",
        main_loss,
    )

    assert_finite_tensor(
        "student_total",
        student_total,
    )

    assert_finite_tensor(
        "teacher_total",
        teacher_total,
    )

    # Explicit parameter partitions.
    student_parameters = tuple(
        parameter
        for name, parameter
        in model.named_parameters()
        if (
            parameter.requires_grad
            and not name.startswith(
                "training_auxiliary."
            )
        )
    )

    fast_teacher_parameters = tuple(
        model.training_auxiliary
        .teacher_parameters()
    )

    student_ids = {
        id(parameter)
        for parameter
        in student_parameters
    }

    teacher_ids = {
        id(parameter)
        for parameter
        in fast_teacher_parameters
    }

    if student_ids & teacher_ids:
        raise AssertionError(
            "Student and fast-teacher parameter sets overlap"
        )

    # ---------------------------------------------------------------
    # A. Student backward
    # ---------------------------------------------------------------
    model.zero_grad(
        set_to_none=True
    )

    student_total.backward()

    student_grad_after_student_loss = (
        grad_norm(
            student_parameters
        )
    )

    teacher_grad_after_student_loss = (
        grad_norm(
            fast_teacher_parameters
        )
    )

    if not (
        student_grad_after_student_loss
        > 0.0
    ):
        raise AssertionError(
            "Student objective produced zero Student gradient"
        )

    if teacher_grad_after_student_loss != 0.0:
        raise AssertionError(
            "Student objective leaked gradient into fast teachers: "
            f"{teacher_grad_after_student_loss:.9e}"
        )

    print(
        "[PASS] Student backward isolation: "
        f"student_grad={student_grad_after_student_loss:.6e}, "
        f"teacher_grad={teacher_grad_after_student_loss:.6e}"
    )

    # ---------------------------------------------------------------
    # B. Teacher backward
    #
    # teacher_total belongs to a disconnected graph because Student
    # feature/prediction are detached inside dynamic_teacher.py.
    # ---------------------------------------------------------------
    model.zero_grad(
        set_to_none=True
    )

    teacher_total.backward()

    student_grad_after_teacher_loss = (
        grad_norm(
            student_parameters
        )
    )

    teacher_grad_after_teacher_loss = (
        grad_norm(
            fast_teacher_parameters
        )
    )

    if student_grad_after_teacher_loss != 0.0:
        raise AssertionError(
            "Teacher objective leaked gradient into Student: "
            f"{student_grad_after_teacher_loss:.9e}"
        )

    if not (
        teacher_grad_after_teacher_loss
        > 0.0
    ):
        raise AssertionError(
            "Teacher objective produced zero fast-teacher gradient"
        )

    print(
        "[PASS] Teacher backward isolation: "
        f"student_grad={student_grad_after_teacher_loss:.6e}, "
        f"teacher_grad={teacher_grad_after_teacher_loss:.6e}"
    )

    return {
        "main_loss": float(
            main_loss.detach()
        ),
        "student_total": float(
            student_total.detach()
        ),
        "teacher_total": float(
            teacher_total.detach()
        ),
        "student_grad": (
            student_grad_after_student_loss
        ),
        "teacher_grad": (
            teacher_grad_after_teacher_loss
        ),
        "student_brier": float(
            dynamic["student_brier"]
            .detach()
        ),
        "effective_mass": float(
            dynamic["effective_mass"]
            .detach()
        ),
    }


def check_teacher_optimizer_and_ema(
    args: argparse.Namespace,
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    teacher_pack,
) -> Dict[str, float]:
    """
    Execute one actual fast-teacher optimization step and one EMA update.

    Then verify:
        fast teacher changed;
        EMA target changed;
        update_ema() reports non-zero evolution;
        target/fast state remains finite.
    """
    model.train()

    fast_parameters = tuple(
        model.training_auxiliary
        .teacher_parameters()
    )

    target_parameters = tuple(
        model.training_auxiliary
        .target_teacher_parameters()
    )

    teacher_optimizer = torch.optim.Adam(
        fast_parameters,
        lr=args.teacher_lr,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    fast_before = snapshot_parameters(
        fast_parameters
    )

    target_before = snapshot_parameters(
        target_parameters
    )

    teacher_optimizer.zero_grad(
        set_to_none=True
    )

    _, auxiliary = model(
        image_a,
        image_b,
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )

    teacher_loss = auxiliary[
        "direction_c"
    ]["teacher_total"]

    teacher_loss.backward()

    teacher_gradient = grad_norm(
        fast_parameters
    )

    if not teacher_gradient > 0.0:
        raise AssertionError(
            "Teacher optimizer smoke received zero teacher gradient"
        )

    teacher_optimizer.step()

    fast_parameter_update = (
        parameter_rms_difference(
            fast_before,
            fast_parameters,
        )
    )

    if not fast_parameter_update > 0.0:
        raise AssertionError(
            "Fast-teacher parameters did not change after optimizer.step()"
        )

    ema_reported_update = (
        model.training_auxiliary
        .update_ema()
    )

    target_parameter_update = (
        parameter_rms_difference(
            target_before,
            target_parameters,
        )
    )

    teacher_target_gap = (
        model.training_auxiliary
        .teacher_target_gap()
    )

    if not ema_reported_update > 0.0:
        raise AssertionError(
            "update_ema() reported zero teacher evolution"
        )

    if not target_parameter_update > 0.0:
        raise AssertionError(
            "EMA target-teacher parameters did not change"
        )

    if not math.isfinite(
        teacher_target_gap
    ):
        raise AssertionError(
            "teacher_target_gap is NaN/Inf"
        )

    print(
        "[PASS] dynamic teacher update: "
        f"grad={teacher_gradient:.6e}, "
        f"fast_update_rms={fast_parameter_update:.6e}, "
        f"ema_update_rms={ema_reported_update:.6e}, "
        f"target_update_rms={target_parameter_update:.6e}, "
        f"fast_target_gap={teacher_target_gap:.6e}"
    )

    # One forward AFTER the teacher evolution.  At initialization the
    # residual expert is identity-initialized, so this second forward is the
    # meaningful check that the target teacher has begun to move away from
    # the Student.
    with torch.no_grad():
        _, auxiliary_after = model(
            image_a,
            image_b,
            target=target,
            teacher_pack=teacher_pack,
            compute_auxiliary=True,
        )

    dynamic_after = auxiliary_after[
        "direction_c"
    ]

    dynamic_shift = (
        dynamic_after[
            "dynamic_shift"
        ]
        .detach()
        .float()
        .mean()
        .item()
    )

    dynamic_gain = (
        dynamic_after[
            "dynamic_teacher_gain"
        ]
        .detach()
        .float()
        .mean()
        .item()
    )

    effective_mass = float(
        dynamic_after[
            "effective_mass"
        ].detach()
    )

    error_normalized_mass = float(
        dynamic_after[
            "effective_per_error_mass"
        ].detach()
    )

    if not math.isfinite(
        dynamic_shift
    ):
        raise AssertionError(
            "dynamic_shift is NaN/Inf"
        )

    if dynamic_shift <= 0.0:
        raise AssertionError(
            "EMA teacher updated but dynamic proposal still has zero shift"
        )

    print(
        "[PASS] post-update dynamic signal: "
        f"dynamic_shift={dynamic_shift:.6e}, "
        f"dynamic_gain={dynamic_gain:.6e}, "
        f"effective_mass={effective_mass:.6e}, "
        f"effective/error_mass={error_normalized_mass:.6e}"
    )

    return {
        "teacher_gradient": teacher_gradient,
        "fast_update_rms": fast_parameter_update,
        "ema_update_rms": ema_reported_update,
        "target_update_rms": target_parameter_update,
        "teacher_target_gap": teacher_target_gap,
        "dynamic_shift": dynamic_shift,
        "dynamic_gain": dynamic_gain,
        "effective_mass_after": effective_mass,
        "effective_per_error_mass_after": error_normalized_mass,
    }


@torch.no_grad()
def check_deploy_contract(
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
) -> Dict[str, float]:
    """
    Verify physical auxiliary deletion and deploy prediction equivalence.
    """
    model.eval()

    prediction_before = model(
        image_a,
        image_b,
        compute_auxiliary=False,
    )

    training_parameter_count = (
        parameter_count(model)
    )

    if not hasattr(
        model,
        "training_auxiliary",
    ):
        raise AssertionError(
            "training_auxiliary disappeared before switch_to_deploy()"
        )

    model.switch_to_deploy()

    if hasattr(
        model,
        "training_auxiliary",
    ):
        raise AssertionError(
            "switch_to_deploy() did not delete training_auxiliary"
        )

    prediction_after = model(
        image_a,
        image_b,
        compute_auxiliary=False,
    )

    deploy_difference = (
        max_prediction_difference(
            prediction_before,
            prediction_after,
        )
    )

    if deploy_difference >= PREDICTION_ATOL:
        raise AssertionError(
            "Deploy conversion changed main prediction: "
            f"max_abs={deploy_difference:.9e}"
        )

    deploy_parameter_count = (
        parameter_count(model)
    )

    if (
        deploy_parameter_count
        != EXPECTED_DEPLOY_PARAMS
    ):
        raise AssertionError(
            "Unexpected deployed parameter count: "
            f"expected={EXPECTED_DEPLOY_PARAMS:,}, "
            f"actual={deploy_parameter_count:,}"
        )

    state_keys = tuple(
        model.state_dict().keys()
    )

    leaked_auxiliary_keys = [
        key
        for key in state_keys
        if key.startswith(
            "training_auxiliary."
        )
    ]

    if leaked_auxiliary_keys:
        raise AssertionError(
            "Training auxiliary keys remain after deploy conversion: "
            + ", ".join(
                leaked_auxiliary_keys[:5]
            )
        )

    print(
        "[PASS] deploy contract: "
        f"training_params={training_parameter_count:,}, "
        f"deploy_params={deploy_parameter_count:,}, "
        f"prediction_max_abs={deploy_difference:.9e}"
    )

    return {
        "training_parameters": float(
            training_parameter_count
        ),
        "deploy_parameters": float(
            deploy_parameter_count
        ),
        "deploy_max_abs": deploy_difference,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    device = get_device(
        args
    )

    print("=" * 78)
    print("RDT-CD synthetic smoke")
    print("=" * 78)
    print(f"project_root        : {PROJECT_ROOT}")
    print(f"device              : {device}")
    print(f"torch               : {torch.__version__}")
    print(f"seed                : {args.seed}")
    print(f"batch_size          : {args.batch_size}")
    print(
        "cache_conditioning  : "
        f"{not args.no_cache_conditioning}"
    )

    if device.type == "cuda":
        print(
            "gpu                 : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print("-" * 78)

    model = build_dynamic_model(
        args,
        device,
    )

    if not model.use_training_auxiliary:
        raise AssertionError(
            "Dynamic training auxiliary was not constructed"
        )

    if (
        model.training_mechanism
        != "dynamic_teacher"
    ):
        raise AssertionError(
            "A2Net did not select mechanism='dynamic_teacher'"
        )

    expected_cache_requirement = (
        not args.no_cache_conditioning
    )

    if (
        model.training_auxiliary_requires_cache
        != expected_cache_requirement
    ):
        raise AssertionError(
            "A2Net cache requirement does not match dynamic-teacher mode"
        )

    check_state_registration(
        model
    )

    image_a = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=device,
    )

    image_b = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=device,
    )

    target = make_synthetic_target(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        device=device,
    )

    if args.no_cache_conditioning:
        teacher_pack = None
    else:
        teacher_pack = (
            make_synthetic_teacher_pack(
                target
            )
        )

    # 1. Auxiliary must not perturb main prediction.
    check_auxiliary_prediction_invariance(
        model=model,
        image_a=image_a,
        image_b=image_b,
        target=target,
        teacher_pack=teacher_pack,
        seed=args.seed + 991,
    )

    # 2. Student and teacher computational graphs must be disjoint.
    gradient_stats = (
        check_forward_and_gradient_isolation(
            args=args,
            model=model,
            image_a=image_a,
            image_b=image_b,
            target=target,
            teacher_pack=teacher_pack,
        )
    )

    # Clear all previous smoke gradients before the real one-step teacher
    # update.
    model.zero_grad(
        set_to_none=True
    )

    # 3. Fast teacher -> EMA target teacher must actually evolve.
    dynamic_stats = (
        check_teacher_optimizer_and_ema(
            args=args,
            model=model,
            image_a=image_a,
            image_b=image_b,
            target=target,
            teacher_pack=teacher_pack,
        )
    )

    model.zero_grad(
        set_to_none=True
    )

    # 4. Deployment must delete all auxiliary state and preserve predictions.
    deploy_stats = (
        check_deploy_contract(
            model=model,
            image_a=image_a,
            image_b=image_b,
        )
    )

    print("-" * 78)
    print("Key smoke values")
    print("-" * 78)

    print(
        f"main_loss                    : "
        f"{gradient_stats['main_loss']:.6f}"
    )
    print(
        f"teacher_total                : "
        f"{gradient_stats['teacher_total']:.6f}"
    )
    print(
        f"student_gradient_norm         : "
        f"{gradient_stats['student_grad']:.6e}"
    )
    print(
        f"teacher_gradient_norm         : "
        f"{dynamic_stats['teacher_gradient']:.6e}"
    )
    print(
        f"teacher_ema_update_norm       : "
        f"{dynamic_stats['ema_update_rms']:.6e}"
    )
    print(
        f"teacher_target_gap            : "
        f"{dynamic_stats['teacher_target_gap']:.6e}"
    )
    print(
        f"dynamic_shift                 : "
        f"{dynamic_stats['dynamic_shift']:.6e}"
    )
    print(
        f"dynamic_teacher_gain          : "
        f"{dynamic_stats['dynamic_gain']:.6e}"
    )
    print(
        f"effective_mass                : "
        f"{dynamic_stats['effective_mass_after']:.6e}"
    )
    print(
        f"effective_per_error_mass      : "
        f"{dynamic_stats['effective_per_error_mass_after']:.6e}"
    )
    print(
        f"deploy_parameters             : "
        f"{int(deploy_stats['deploy_parameters']):,}"
    )
    print(
        f"deploy_prediction_max_abs     : "
        f"{deploy_stats['deploy_max_abs']:.9e}"
    )

    print("=" * 78)
    print("ALL RDT-CD SYNTHETIC SMOKE CHECKS PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
