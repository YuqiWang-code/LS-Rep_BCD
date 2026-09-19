#!/usr/bin/env python3
"""
Synthetic smoke test for Run3 BT-SAM-RDT.

This script is a correctness test only.

It does NOT:
    - train a formal experiment;
    - read a real dataset;
    - modify Teacher Cache;
    - write checkpoints;
    - report publishable accuracy.

It verifies the P0 contracts of Run3:

A. BT-SAM structural logic
    1. Same object geometry with different opaque SAM IDs -> no change.
    2. T1 object disappears -> high structural-change prior.
    3. New T2 object -> high structural-change prior.
    4. Expansion:
           stable overlap core -> structural prior ~= 0
           newly occupied outer ring -> structural prior ~= 1
           C12 > C21
    5. Shrinkage:
           remaining overlap core -> structural prior ~= 0
           disappeared outer ring -> structural prior ~= 1
           C21 > C12
    6. Lower SAM quality reduces structural reliability.
    7. Pair-match confidence is finite and bounded.

B. SAM + OV fusion
    8. Agreement keeps high fused reliability.
    9. Strong SAM/OV disagreement suppresses fused reliability.
   10. R3A SAM-only path reduces exactly to the SAM prior.

C. GT safety audit
   11. Better teacher proposal is admitted.
   12. Worse teacher proposal is rejected.

D. Dynamic teacher
   13. Run3 uses one Fast Teacher and one frozen EMA Target Teacher.
   14. ResidualTeacherExpert input contract is C+5.
   15. Dynamic Target Teacher starts from Student identity.
   16. Student objective does not update Fast Teacher.
   17. Teacher fitting objective does not update Student.
   18. Fast Teacher changes after optimization.
   19. EMA Target Teacher changes only after update_ema().
   20. Dynamic teacher state is checkpoint-visible.

E. Deployment
   21. Auxiliary ON/OFF does not change main predictions.
   22. switch_to_deploy() physically removes the auxiliary.
   23. Deploy prediction max error < 1e-6.
   24. Deploy parameter count == 2,913,094.

Run from project root:

    python models/tools/smoke_dynamic_teacher.py --device cuda --gpu_id 0

CPU verification:

    python models/tools/smoke_dynamic_teacher.py --device cpu

A separate real-cache aligned-replay dry run is still required after this
synthetic smoke passes.
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
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from models import A2Net_LWGANet_L0  # noqa: E402
from models.distill.task_space import (  # noqa: E402
    build_bitemporal_structural_prior,
    build_foundation_prior,
    fuse_foundation_priors,
    task_space_audit,
)


EXPECTED_DEPLOY_PARAMS = 2_913_094
PREDICTION_ATOL = 1e-6
STRUCTURE_ATOL = 1e-6


# ============================================================================
# Arguments / environment
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic P0 smoke test for Run3 BT-SAM-RDT"
    )

    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2333,
    )

    parser.add_argument(
        "--teacher_hidden",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--teacher_lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--teacher_ema",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--max_logit_delta",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--kd_lambda",
        type=float,
        default=0.06,
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    if (
        args.height != 256
        or args.width != 256
    ):
        raise ValueError(
            "Project deploy contract is defined for 256x256 inputs"
        )

    if args.teacher_hidden <= 0:
        raise ValueError("--teacher_hidden must be positive")

    if args.teacher_lr <= 0:
        raise ValueError("--teacher_lr must be positive")

    if not (
        0.0
        <= args.teacher_ema
        < 1.0
    ):
        raise ValueError(
            "--teacher_ema must satisfy 0 <= ema < 1"
        )

    if args.max_logit_delta <= 0:
        raise ValueError("--max_logit_delta must be positive")

    if args.kd_lambda < 0:
        raise ValueError("--kd_lambda must be non-negative")

    return args


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(
    args: argparse.Namespace,
) -> torch.device:
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but CUDA is unavailable"
            )

        if not (
            0
            <= args.gpu_id
            < torch.cuda.device_count()
        ):
            raise ValueError(
                f"Invalid GPU {args.gpu_id}; "
                f"visible count={torch.cuda.device_count()}"
            )

        torch.cuda.set_device(args.gpu_id)

        return torch.device(
            "cuda",
            args.gpu_id,
        )

    return torch.device("cpu")


# ============================================================================
# Generic helpers
# ============================================================================


def parameter_count(
    module: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def snapshot_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().clone()
        for parameter in parameters
    )


def parameter_rms_difference(
    before: Tuple[torch.Tensor, ...],
    after_parameters: Iterable[torch.nn.Parameter],
) -> float:
    after = tuple(
        parameter.detach()
        for parameter in after_parameters
    )

    if len(before) != len(after):
        raise AssertionError(
            "Parameter snapshot length changed"
        )

    square_sum = 0.0
    count = 0

    for old, new in zip(before, after):
        delta = new.float() - old.float()

        square_sum += float(
            delta.square().sum()
        )

        count += delta.numel()

    return math.sqrt(
        square_sum
        / max(count, 1)
    )


def grad_norm(
    parameters: Iterable[torch.nn.Parameter],
) -> float:
    square_sum = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = (
            parameter
            .grad
            .detach()
            .float()
        )

        square_sum += float(
            gradient.square().sum()
        )

    return math.sqrt(square_sum)


def max_prediction_difference(
    left,
    right,
) -> float:
    if len(left) != len(right):
        raise AssertionError(
            "Prediction tuple lengths differ"
        )

    maximum = 0.0

    for x, y in zip(left, right):
        if x.shape != y.shape:
            raise AssertionError(
                f"Prediction shape mismatch: "
                f"{x.shape} vs {y.shape}"
            )

        difference = (
            x.detach().float()
            - y.detach().float()
        ).abs().max().item()

        maximum = max(
            maximum,
            difference,
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


def assert_probability_range(
    name: str,
    value: torch.Tensor,
) -> None:
    assert_finite_tensor(
        name,
        value,
    )

    minimum = float(
        value.min()
    )

    maximum = float(
        value.max()
    )

    if (
        minimum < -STRUCTURE_ATOL
        or maximum > 1.0 + STRUCTURE_ATOL
    ):
        raise AssertionError(
            f"{name} is outside [0,1]: "
            f"min={minimum:.6f}, max={maximum:.6f}"
        )


# ============================================================================
# Synthetic SAM structures
# ============================================================================


def make_boundary_from_ids(
    ids: torch.Tensor,
) -> torch.Tensor:
    """4-neighbour boundary map for synthetic opaque instances."""
    boundary = torch.zeros_like(
        ids,
        dtype=torch.float32,
    )

    vertical = (
        ids[:, :, 1:, :]
        != ids[:, :, :-1, :]
    ).float()

    boundary[:, :, 1:, :] = torch.maximum(
        boundary[:, :, 1:, :],
        vertical,
    )

    boundary[:, :, :-1, :] = torch.maximum(
        boundary[:, :, :-1, :],
        vertical,
    )

    horizontal = (
        ids[:, :, :, 1:]
        != ids[:, :, :, :-1]
    ).float()

    boundary[:, :, :, 1:] = torch.maximum(
        boundary[:, :, :, 1:],
        horizontal,
    )

    boundary[:, :, :, :-1] = torch.maximum(
        boundary[:, :, :, :-1],
        horizontal,
    )

    return boundary


def make_sam_pack(
    ids_t1: torch.Tensor,
    ids_t2: torch.Tensor,
    quality_t1: float = 1.0,
    quality_t2: float = 1.0,
    use_boundaries: bool = False,
) -> Dict:
    if ids_t1.shape != ids_t2.shape:
        raise ValueError(
            "Synthetic T1/T2 IDs must have equal shapes"
        )

    if use_boundaries:
        boundary_t1 = make_boundary_from_ids(
            ids_t1
        )

        boundary_t2 = make_boundary_from_ids(
            ids_t2
        )

    else:
        boundary_t1 = torch.zeros_like(
            ids_t1,
            dtype=torch.float32,
        )

        boundary_t2 = torch.zeros_like(
            ids_t2,
            dtype=torch.float32,
        )

    q1 = torch.full(
        ids_t1.shape,
        float(quality_t1),
        device=ids_t1.device,
        dtype=torch.float32,
    )

    q2 = torch.full(
        ids_t2.shape,
        float(quality_t2),
        device=ids_t2.device,
        dtype=torch.float32,
    )

    return {
        "t1": {
            "instance_id": ids_t1.long(),
            "boundary": boundary_t1,
            "quality": q1,
        },
        "t2": {
            "instance_id": ids_t2.long(),
            "boundary": boundary_t2,
            "quality": q2,
        },
    }


# ============================================================================
# Direct BT-SAM tests
# ============================================================================


@torch.no_grad()
def check_bt_sam_identical_geometry(
    device: torch.device,
) -> None:
    """
    Same geometry with completely different opaque instance IDs must match.
    """
    ids1 = torch.zeros(
        1,
        1,
        16,
        16,
        device=device,
        dtype=torch.long,
    )

    ids2 = torch.zeros_like(ids1)

    ids1[:, :, 4:12, 5:13] = 11

    # Deliberately unrelated opaque ID.
    ids2[:, :, 4:12, 5:13] = 907

    result = build_bitemporal_structural_prior(
        make_sam_pack(
            ids1,
            ids2,
        )
    )

    prior = result["prior"]
    object_mask = ids1 > 0

    maximum = (
        prior[
            object_mask
        ]
        .abs()
        .max()
        .item()
    )

    if maximum > STRUCTURE_ATOL:
        raise AssertionError(
            "Identical geometry with different SAM IDs "
            f"produced change={maximum:.6e}"
        )

    pair_iou = float(
        result[
            "pair_match_iou"
        ].mean()
    )

    if pair_iou < 1.0 - STRUCTURE_ATOL:
        raise AssertionError(
            "Identical geometry did not produce IoU=1"
        )

    assert_probability_range(
        "pair_match_confidence",
        result[
            "pair_match_confidence"
        ],
    )

    print(
        "[PASS] BT-SAM opaque-ID invariance"
    )


@torch.no_grad()
def check_bt_sam_disappearance(
    device: torch.device,
) -> None:
    ids1 = torch.zeros(
        1,
        1,
        16,
        16,
        device=device,
        dtype=torch.long,
    )

    ids2 = torch.zeros_like(ids1)

    ids1[:, :, 4:12, 4:12] = 31

    result = build_bitemporal_structural_prior(
        make_sam_pack(
            ids1,
            ids2,
        )
    )

    mask = ids1 > 0

    mean_change = (
        result[
            "prior"
        ][
            mask
        ]
        .mean()
        .item()
    )

    if mean_change < 1.0 - STRUCTURE_ATOL:
        raise AssertionError(
            "Disappearing T1 object did not produce high change prior: "
            f"{mean_change:.6f}"
        )

    print(
        "[PASS] BT-SAM disappearance detection"
    )


@torch.no_grad()
def check_bt_sam_appearance(
    device: torch.device,
) -> None:
    ids1 = torch.zeros(
        1,
        1,
        16,
        16,
        device=device,
        dtype=torch.long,
    )

    ids2 = torch.zeros_like(ids1)

    ids2[:, :, 3:11, 6:14] = 77

    result = build_bitemporal_structural_prior(
        make_sam_pack(
            ids1,
            ids2,
        )
    )

    mask = ids2 > 0

    mean_change = (
        result[
            "prior"
        ][
            mask
        ]
        .mean()
        .item()
    )

    if mean_change < 1.0 - STRUCTURE_ATOL:
        raise AssertionError(
            "New T2 object did not produce high change prior: "
            f"{mean_change:.6f}"
        )

    print(
        "[PASS] BT-SAM appearance detection"
    )


@torch.no_grad()
def check_bt_sam_expansion_localization(
    device: torch.device,
) -> None:
    """
    Expansion contract:

        T1 object is contained in larger T2 object.

        overlap core:
            structural prior == 0

        T2-only ring:
            structural prior == 1

        C12 > C21
    """
    ids1 = torch.zeros(
        1,
        1,
        16,
        16,
        device=device,
        dtype=torch.long,
    )

    ids2 = torch.zeros_like(ids1)

    ids1[:, :, 6:10, 6:10] = 5
    ids2[:, :, 4:12, 4:12] = 1005

    result = build_bitemporal_structural_prior(
        make_sam_pack(
            ids1,
            ids2,
        )
    )

    prior = result["prior"]

    core = (
        (ids1 > 0)
        & (ids2 > 0)
    )

    new_ring = (
        (ids1 == 0)
        & (ids2 > 0)
    )

    core_max = float(
        prior[
            core
        ].abs().max()
    )

    ring_min = float(
        prior[
            new_ring
        ].min()
    )

    cov12 = float(
        result[
            "pair_match_cov12"
        ].mean()
    )

    cov21 = float(
        result[
            "pair_match_cov21"
        ].mean()
    )

    if core_max > STRUCTURE_ATOL:
        raise AssertionError(
            "Expansion stable overlap core was incorrectly marked changed: "
            f"max={core_max:.6f}"
        )

    if ring_min < 1.0 - STRUCTURE_ATOL:
        raise AssertionError(
            "Expansion new outer ring was not fully localized as change: "
            f"min={ring_min:.6f}"
        )

    if not (
        cov12
        > cov21
    ):
        raise AssertionError(
            "Expansion should produce C12 > C21, "
            f"got C12={cov12:.6f}, C21={cov21:.6f}"
        )

    if cov12 < 1.0 - STRUCTURE_ATOL:
        raise AssertionError(
            "Contained T1 object should have C12≈1, "
            f"got {cov12:.6f}"
        )

    print(
        "[PASS] BT-SAM expansion localization: "
        f"core={core_max:.3f}, "
        f"ring={ring_min:.3f}, "
        f"C12={cov12:.3f}, "
        f"C21={cov21:.3f}"
    )


@torch.no_grad()
def check_bt_sam_shrinkage_localization(
    device: torch.device,
) -> None:
    """
    Shrinkage contract:

        T2 object is contained in larger T1 object.

        remaining overlap core:
            structural prior == 0

        T1-only disappeared ring:
            structural prior == 1

        C21 > C12
    """
    ids1 = torch.zeros(
        1,
        1,
        16,
        16,
        device=device,
        dtype=torch.long,
    )

    ids2 = torch.zeros_like(ids1)

    ids1[:, :, 4:12, 4:12] = 23
    ids2[:, :, 6:10, 6:10] = 9001

    result = build_bitemporal_structural_prior(
        make_sam_pack(
            ids1,
            ids2,
        )
    )

    prior = result["prior"]

    core = (
        (ids1 > 0)
        & (ids2 > 0)
    )

    disappeared_ring = (
        (ids1 > 0)
        & (ids2 == 0)
    )

    core_max = float(
        prior[
            core
        ].abs().max()
    )

    ring_min = float(
        prior[
            disappeared_ring
        ].min()
    )

    cov12 = float(
        result[
            "pair_match_cov12"
        ].mean()
    )

    cov21 = float(
        result[
            "pair_match_cov21"
        ].mean()
    )

    if core_max > STRUCTURE_ATOL:
        raise AssertionError(
            "Shrinkage stable overlap core was incorrectly marked changed: "
            f"max={core_max:.6f}"
        )

    if ring_min < 1.0 - STRUCTURE_ATOL:
        raise AssertionError(
            "Shrinkage disappeared outer ring was not fully localized as change: "
            f"min={ring_min:.6f}"
        )

    if not (
        cov21
        > cov12
    ):
        raise AssertionError(
            "Shrinkage should produce C21 > C12, "
            f"got C12={cov12:.6f}, C21={cov21:.6f}"
        )

    if cov21 < 1.0 - STRUCTURE_ATOL:
        raise AssertionError(
            "Contained T2 object should have C21≈1, "
            f"got {cov21:.6f}"
        )

    print(
        "[PASS] BT-SAM shrinkage localization: "
        f"core={core_max:.3f}, "
        f"ring={ring_min:.3f}, "
        f"C12={cov12:.3f}, "
        f"C21={cov21:.3f}"
    )


@torch.no_grad()
def check_sam_quality_reliability(
    device: torch.device,
) -> None:
    ids1 = torch.zeros(
        1,
        1,
        16,
        16,
        device=device,
        dtype=torch.long,
    )

    ids2 = torch.zeros_like(ids1)

    ids1[:, :, 4:12, 4:12] = 1
    ids2[:, :, 4:12, 4:12] = 2

    high = build_bitemporal_structural_prior(
        make_sam_pack(
            ids1,
            ids2,
            quality_t1=1.0,
            quality_t2=1.0,
        )
    )

    low = build_bitemporal_structural_prior(
        make_sam_pack(
            ids1,
            ids2,
            quality_t1=0.1,
            quality_t2=0.1,
        )
    )

    mask = ids1 > 0

    high_r = float(
        high[
            "reliability"
        ][
            mask
        ].mean()
    )

    low_r = float(
        low[
            "reliability"
        ][
            mask
        ].mean()
    )

    if not (
        low_r
        < high_r
    ):
        raise AssertionError(
            "Lower SAM quality did not reduce reliability: "
            f"high={high_r:.6f}, low={low_r:.6f}"
        )

    print(
        "[PASS] SAM quality modulates structural reliability"
    )


# ============================================================================
# Fusion / audit tests
# ============================================================================


@torch.no_grad()
def check_fusion_conflict(
    device: torch.device,
) -> None:
    ones = torch.ones(
        1,
        1,
        8,
        8,
        device=device,
    )

    zeros = torch.zeros_like(
        ones
    )

    agreed = fuse_foundation_priors(
        sam_prior=ones,
        sam_reliability=ones,
        ov_prior=ones,
        ov_reliability=ones,
    )

    conflict = fuse_foundation_priors(
        sam_prior=ones,
        sam_reliability=ones,
        ov_prior=zeros,
        ov_reliability=ones,
    )

    agreed_r = float(
        agreed[
            "reliability"
        ].mean()
    )

    conflict_r = float(
        conflict[
            "reliability"
        ].mean()
    )

    if not (
        agreed_r
        > conflict_r
    ):
        raise AssertionError(
            "SAM/OV disagreement did not reduce fused reliability"
        )

    if conflict_r > STRUCTURE_ATOL:
        raise AssertionError(
            "Maximal SAM/OV disagreement should suppress reliability, "
            f"got {conflict_r:.6e}"
        )

    print(
        "[PASS] SAM/OV conflict suppresses fused reliability"
    )


@torch.no_grad()
def check_task_space_audit(
    device: torch.device,
) -> None:
    # Pixel 0:
    #   Student poor, teacher better -> admit.
    #
    # Pixel 1:
    #   Student good, teacher worse -> reject.
    prediction = torch.tensor(
        [
            [
                [
                    [0.8, 0.1]
                ]
            ]
        ],
        device=device,
        dtype=torch.float32,
    )

    target = torch.tensor(
        [
            [
                [
                    [0.0, 0.0]
                ]
            ]
        ],
        device=device,
        dtype=torch.float32,
    )

    proposal = torch.tensor(
        [
            [
                [
                    [0.2, 0.9]
                ]
            ]
        ],
        device=device,
        dtype=torch.float32,
    )

    reliability = torch.ones_like(
        prediction
    )

    audit = task_space_audit(
        prediction=prediction,
        target=target,
        proposal=proposal,
        reliability=reliability,
        policy="advantage",
    )

    accepted = audit[
        "accepted_map"
    ]

    if not bool(
        accepted[
            0,
            0,
            0,
            0,
        ]
    ):
        raise AssertionError(
            "Better teacher proposal was rejected"
        )

    if bool(
        accepted[
            0,
            0,
            0,
            1,
        ]
    ):
        raise AssertionError(
            "Worse teacher proposal was admitted"
        )

    print(
        "[PASS] GT positive-Brier-gain audit"
    )


# ============================================================================
# Synthetic full Run3 pack
# ============================================================================


def make_synthetic_target(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    target = torch.zeros(
        batch_size,
        1,
        height,
        width,
        device=device,
        dtype=torch.float32,
    )

    for index in range(
        batch_size
    ):
        offset = 5 * index

        target[
            index,
            0,
            40 + offset:100 + offset,
            45:110,
        ] = 1.0

        target[
            index,
            0,
            135:185,
            145 - offset:210 - offset,
        ] = 1.0

        target[
            index,
            0,
            205:220,
            65:95,
        ] = 1.0

    return target


def make_instance_grid(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    block: int = 32,
) -> torch.Tensor:
    yy = torch.arange(
        height,
        device=device,
    ).view(
        height,
        1,
    )

    xx = torch.arange(
        width,
        device=device,
    ).view(
        1,
        width,
    )

    columns = math.ceil(
        width
        / block
    )

    ids = (
        (
            yy
            // block
        )
        * columns
        + (
            xx
            // block
        )
        + 1
    ).long()

    return (
        ids
        .view(
            1,
            1,
            height,
            width,
        )
        .expand(
            batch_size,
            -1,
            -1,
            -1,
        )
        .clone()
    )


def make_synthetic_teacher_pack(
    target: torch.Tensor,
) -> Dict:
    batch_size, _, height, width = target.shape
    device = target.device

    ids_t1 = make_instance_grid(
        batch_size,
        height,
        width,
        device,
        block=32,
    )

    # Same geometry, opaque labels deliberately changed.
    ids_t2 = (
        ids_t1
        + 10_000
    )

    # Introduce one structural segmentation difference.
    ids_t2[
        :,
        :,
        160:224,
        32:96,
    ] = 50_000

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
        0.88,
        device=device,
        dtype=torch.float32,
    )

    quality_t1 *= (
        1.0
        - 0.20
        * boundary_t1
    )

    quality_t2 *= (
        1.0
        - 0.20
        * boundary_t2
    )

    # Synthetic OV fixture only.
    # Formal training never derives Teacher Cache from GT.
    ov_l1 = F.interpolate(
        target,
        size=(
            32,
            32,
        ),
        mode="bilinear",
        align_corners=False,
    )

    ov_l2 = F.interpolate(
        target,
        size=(
            16,
            16,
        ),
        mode="bilinear",
        align_corners=False,
    )

    ov_l1 = (
        0.75
        * ov_l1
        + 0.10
    ).clamp(
        0.0,
        1.0,
    )

    ov_l2 = (
        0.65
        * ov_l2
        + 0.15
    ).clamp(
        0.0,
        1.0,
    )

    confidence_l1 = torch.full_like(
        ov_l1,
        0.85,
    )

    confidence_l2 = torch.full_like(
        ov_l2,
        0.75,
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


@torch.no_grad()
def check_foundation_prior_paths(
    teacher_pack: Dict,
    output_size,
) -> None:
    full = build_foundation_prior(
        teacher_pack=teacher_pack,
        output_size=output_size,
        use_ov=True,
    )

    sam_only = build_foundation_prior(
        teacher_pack=teacher_pack,
        output_size=output_size,
        use_ov=False,
    )

    required = (
        "sam_prior",
        "sam_reliability",
        "ov_prior",
        "ov_reliability",
        "fused_prior",
        "fused_reliability",
        "sam_ov_conflict",
        "pair_match_ratio",
        "pair_match_iou",
        "pair_match_cov12",
        "pair_match_cov21",
        "pair_match_confidence",
    )

    for key in required:
        if key not in full:
            raise AssertionError(
                f"Missing foundation-prior field: {key}"
            )

        assert_finite_tensor(
            key,
            full[key],
        )

    for key in (
        "sam_prior",
        "sam_reliability",
        "ov_prior",
        "ov_reliability",
        "fused_prior",
        "fused_reliability",
        "sam_ov_conflict",
        "pair_match_ratio",
        "pair_match_iou",
        "pair_match_cov12",
        "pair_match_cov21",
        "pair_match_confidence",
    ):
        assert_probability_range(
            key,
            full[
                key
            ],
        )

    difference = (
        sam_only[
            "fused_prior"
        ]
        - sam_only[
            "sam_prior"
        ]
    ).abs().max().item()

    reliability_difference = (
        sam_only[
            "fused_reliability"
        ]
        - sam_only[
            "sam_reliability"
        ]
    ).abs().max().item()

    if difference > STRUCTURE_ATOL:
        raise AssertionError(
            "R3A use_ov=False did not reduce exactly to SAM prior"
        )

    if reliability_difference > STRUCTURE_ATOL:
        raise AssertionError(
            "R3A use_ov=False did not reduce exactly to SAM reliability"
        )

    print(
        "[PASS] R3 / R3A foundation-prior paths"
    )


# ============================================================================
# Full model construction
# ============================================================================


def build_run3_model(
    args: argparse.Namespace,
    device: torch.device,
) -> A2Net_LWGANet_L0:
    model = A2Net_LWGANet_L0(
        pretrained=False,
        pretrained_path=None,
        auxiliary_mode="bt_sam_rdt",
        auxiliary_cfg={
            "hidden": args.teacher_hidden,
            "ema": args.teacher_ema,
            "max_logit_delta": args.max_logit_delta,
            "boundary_radius": 2,
            "small_area": 64,
            "policy": "advantage",
            "difficulty": True,
            "use_ov": True,
        },
    )

    return model.to(device)


# ============================================================================
# Dynamic teacher architecture / state
# ============================================================================


def check_state_registration(
    model: A2Net_LWGANet_L0,
) -> None:
    if not hasattr(
        model,
        "training_auxiliary",
    ):
        raise AssertionError(
            "Run3 model has no training_auxiliary"
        )

    auxiliary = model.training_auxiliary

    if not hasattr(
        auxiliary,
        "fast",
    ):
        raise AssertionError(
            "Run3 auxiliary has no Fast Teacher"
        )

    if not hasattr(
        auxiliary,
        "target",
    ):
        raise AssertionError(
            "Run3 auxiliary has no EMA Target Teacher"
        )

    if isinstance(
        auxiliary.fast,
        torch.nn.ModuleList,
    ):
        raise AssertionError(
            "Run3 Fast Teacher unexpectedly remains a multi-teacher ModuleList"
        )

    if isinstance(
        auxiliary.target,
        torch.nn.ModuleList,
    ):
        raise AssertionError(
            "Run3 EMA Target Teacher unexpectedly remains a multi-teacher ModuleList"
        )

    state_keys = tuple(
        model
        .state_dict()
        .keys()
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
            "Fast Teacher parameters missing from model.state_dict()"
        )

    if not target_present:
        raise AssertionError(
            "EMA Target Teacher parameters missing from model.state_dict()"
        )

    if any(
        parameter.requires_grad
        for parameter in auxiliary.target_teacher_parameters()
    ):
        raise AssertionError(
            "EMA Target Teacher must be completely frozen"
        )

    first_conv = auxiliary.fast.net[0]

    expected_channels = (
        model.mid_d
        + 5
    )

    if first_conv.in_channels != expected_channels:
        raise AssertionError(
            "ResidualTeacherExpert input contract mismatch: "
            f"expected={expected_channels}, "
            f"actual={first_conv.in_channels}"
        )

    print(
        "[PASS] single Fast/EMA teacher registration and C+5 input contract"
    )


# ============================================================================
# Auxiliary/main-path invariance
# ============================================================================


@torch.no_grad()
def check_auxiliary_prediction_invariance(
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    teacher_pack,
) -> float:
    """
    Keep every child in eval mode and only make the root execute its training
    auxiliary branch. This eliminates stochastic main-path differences.
    """
    model.eval()

    # Root only enters the training-return branch.
    model.training = True

    predictions_off, _ = model(
        image_a,
        image_b,
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=False,
    )

    predictions_on, auxiliary = model(
        image_a,
        image_b,
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )

    model.training = False

    if "bt_sam_rdt" not in auxiliary:
        raise AssertionError(
            "Run3 auxiliary output key 'bt_sam_rdt' is missing"
        )

    maximum = max_prediction_difference(
        predictions_off,
        predictions_on,
    )

    if maximum != 0.0:
        raise AssertionError(
            "Auxiliary ON/OFF changed main prediction: "
            f"max_abs={maximum:.9e}"
        )

    print(
        "[PASS] auxiliary ON/OFF main-prediction invariance: "
        f"max_abs={maximum:.9e}"
    )

    return maximum


# ============================================================================
# Gradient isolation
# ============================================================================


def check_forward_and_gradient_isolation(
    args: argparse.Namespace,
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    teacher_pack,
) -> Dict[str, float]:
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

    detail = auxiliary[
        "bt_sam_rdt"
    ]

    required = (
        "total",
        "teacher_total",
        "student_brier",
        "sam_prior_brier",
        "ov_prior_brier",
        "fusion_prior_brier",
        "dynamic_teacher_brier",
        "dynamic_teacher_gain",
        "pair_match_ratio",
        "pair_match_iou",
        "pair_match_cov12",
        "pair_match_cov21",
        "sam_ov_conflict",
        "effective_mass",
        "effective_per_error_mass",
        "target_residual_magnitude",
    )

    for key in required:
        if key not in detail:
            raise AssertionError(
                f"Missing Run3 output: {key}"
            )

        assert_finite_tensor(
            key,
            detail[
                key
            ],
        )

    # Identity-initialized EMA Target Teacher.
    initial_shift = float(
        detail[
            "dynamic_shift"
        ]
    )

    initial_residual = float(
        detail[
            "target_residual_magnitude"
        ]
    )

    if initial_shift >= PREDICTION_ATOL:
        raise AssertionError(
            "Target Teacher is not identity initialized: "
            f"dynamic_shift={initial_shift:.9e}"
        )

    if initial_residual >= PREDICTION_ATOL:
        raise AssertionError(
            "Initial Target Teacher residual is non-zero: "
            f"{initial_residual:.9e}"
        )

    student_parameters = tuple(
        parameter
        for name, parameter in model.named_parameters()
        if (
            parameter.requires_grad
            and not name.startswith(
                "training_auxiliary."
            )
        )
    )

    fast_teacher_parameters = tuple(
        model
        .training_auxiliary
        .teacher_parameters()
    )

    target_teacher_parameters = tuple(
        model
        .training_auxiliary
        .target_teacher_parameters()
    )

    if (
        {
            id(parameter)
            for parameter in student_parameters
        }
        & {
            id(parameter)
            for parameter in fast_teacher_parameters
        }
    ):
        raise AssertionError(
            "Student and Fast Teacher parameter sets overlap"
        )

    main_loss = F.binary_cross_entropy(
        predictions[0],
        target,
    )

    student_total = (
        main_loss
        + args.kd_lambda
        * detail[
            "total"
        ]
    )

    teacher_total = detail[
        "teacher_total"
    ]

    assert_finite_tensor(
        "student_total",
        student_total,
    )

    assert_finite_tensor(
        "teacher_total",
        teacher_total,
    )

    # ------------------------------------------------------------------
    # A. Student objective
    # ------------------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    student_total.backward()

    student_gradient = grad_norm(
        student_parameters
    )

    fast_gradient_from_student = grad_norm(
        fast_teacher_parameters
    )

    target_gradient_from_student = grad_norm(
        target_teacher_parameters
    )

    if not (
        student_gradient > 0.0
    ):
        raise AssertionError(
            "Student objective produced zero Student gradient"
        )

    if fast_gradient_from_student != 0.0:
        raise AssertionError(
            "Student loss leaked gradient into Fast Teacher: "
            f"{fast_gradient_from_student:.9e}"
        )

    if target_gradient_from_student != 0.0:
        raise AssertionError(
            "Student loss leaked gradient into EMA Target Teacher"
        )

    print(
        "[PASS] Student backward isolation: "
        f"student={student_gradient:.6e}, "
        f"fast={fast_gradient_from_student:.6e}"
    )

    # ------------------------------------------------------------------
    # B. Fast Teacher objective
    #
    # The Fast Teacher graph is disconnected from Student graph by detach.
    # ------------------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    teacher_total.backward()

    student_gradient_from_teacher = grad_norm(
        student_parameters
    )

    fast_teacher_gradient = grad_norm(
        fast_teacher_parameters
    )

    target_gradient_from_teacher = grad_norm(
        target_teacher_parameters
    )

    if student_gradient_from_teacher != 0.0:
        raise AssertionError(
            "Teacher loss leaked gradient into Student: "
            f"{student_gradient_from_teacher:.9e}"
        )

    if not (
        fast_teacher_gradient > 0.0
    ):
        raise AssertionError(
            "Teacher loss produced zero Fast Teacher gradient"
        )

    if target_gradient_from_teacher != 0.0:
        raise AssertionError(
            "Teacher loss leaked gradient into EMA Target Teacher"
        )

    print(
        "[PASS] Teacher backward isolation: "
        f"student={student_gradient_from_teacher:.6e}, "
        f"fast={fast_teacher_gradient:.6e}"
    )

    return {
        "student_gradient": student_gradient,
        "fast_teacher_gradient": fast_teacher_gradient,
        "initial_dynamic_shift": initial_shift,
    }


# ============================================================================
# Fast Teacher optimization + EMA
# ============================================================================


def check_teacher_optimizer_and_ema(
    args: argparse.Namespace,
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    teacher_pack,
) -> Dict[str, float]:
    model.train()

    fast_parameters = tuple(
        model
        .training_auxiliary
        .teacher_parameters()
    )

    target_parameters = tuple(
        model
        .training_auxiliary
        .target_teacher_parameters()
    )

    optimizer = torch.optim.Adam(
        fast_parameters,
        lr=args.teacher_lr,
        betas=(
            0.9,
            0.99,
        ),
        eps=1e-8,
    )

    fast_before = snapshot_parameters(
        fast_parameters
    )

    target_before = snapshot_parameters(
        target_parameters
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    model.zero_grad(
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
        "bt_sam_rdt"
    ][
        "teacher_total"
    ]

    teacher_loss.backward()

    teacher_gradient = grad_norm(
        fast_parameters
    )

    if not (
        teacher_gradient > 0.0
    ):
        raise AssertionError(
            "Fast Teacher optimization received zero gradient"
        )

    optimizer.step()

    fast_update = parameter_rms_difference(
        fast_before,
        fast_parameters,
    )

    if not (
        fast_update > 0.0
    ):
        raise AssertionError(
            "Fast Teacher parameters did not change"
        )

    # Target must still be unchanged before explicit EMA.
    target_before_ema = parameter_rms_difference(
        target_before,
        target_parameters,
    )

    if target_before_ema != 0.0:
        raise AssertionError(
            "EMA Target Teacher changed before update_ema()"
        )

    ema_update = float(
        model
        .training_auxiliary
        .update_ema()
    )

    target_update = parameter_rms_difference(
        target_before,
        target_parameters,
    )

    target_gap = float(
        model
        .training_auxiliary
        .teacher_target_gap()
    )

    if not (
        ema_update > 0.0
    ):
        raise AssertionError(
            "update_ema() reported zero update"
        )

    if not (
        target_update > 0.0
    ):
        raise AssertionError(
            "EMA Target Teacher parameters did not move"
        )

    if not math.isfinite(
        target_gap
    ):
        raise AssertionError(
            "teacher_target_gap is NaN/Inf"
        )

    # Target proposal should now begin to depart from exact identity.
    with torch.no_grad():
        _, auxiliary_after = model(
            image_a,
            image_b,
            target=target,
            teacher_pack=teacher_pack,
            compute_auxiliary=True,
        )

    detail = auxiliary_after[
        "bt_sam_rdt"
    ]

    dynamic_shift = float(
        detail[
            "dynamic_shift"
        ]
    )

    if not math.isfinite(
        dynamic_shift
    ):
        raise AssertionError(
            "Post-update dynamic_shift is NaN/Inf"
        )

    if dynamic_shift <= 0.0:
        raise AssertionError(
            "EMA Teacher updated but dynamic proposal still "
            "exactly copies Student"
        )

    print(
        "[PASS] Fast Teacher + EMA evolution: "
        f"grad={teacher_gradient:.6e}, "
        f"fast_update={fast_update:.6e}, "
        f"ema_update={ema_update:.6e}, "
        f"target_update={target_update:.6e}, "
        f"gap={target_gap:.6e}, "
        f"dynamic_shift={dynamic_shift:.6e}"
    )

    return {
        "teacher_gradient": teacher_gradient,
        "fast_update": fast_update,
        "ema_update": ema_update,
        "target_update": target_update,
        "target_gap": target_gap,
        "dynamic_shift": dynamic_shift,
    }


# ============================================================================
# Deploy contract
# ============================================================================


@torch.no_grad()
def check_deploy_contract(
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
) -> Dict[str, float]:
    model.eval()

    prediction_before = model(
        image_a,
        image_b,
    )

    training_parameters = parameter_count(
        model
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
            "switch_to_deploy() did not remove training_auxiliary"
        )

    # Idempotence.
    model.switch_to_deploy()

    model.eval()

    prediction_after = model(
        image_a,
        image_b,
    )

    maximum = max_prediction_difference(
        prediction_before,
        prediction_after,
    )

    if maximum >= PREDICTION_ATOL:
        raise AssertionError(
            "Deploy conversion changed Student prediction: "
            f"max_abs={maximum:.9e}"
        )

    deploy_parameters = parameter_count(
        model
    )

    if deploy_parameters != EXPECTED_DEPLOY_PARAMS:
        raise AssertionError(
            "Unexpected deployed parameter count: "
            f"expected={EXPECTED_DEPLOY_PARAMS:,}, "
            f"actual={deploy_parameters:,}"
        )

    leaked_keys = [
        key
        for key in model.state_dict().keys()
        if key.startswith(
            "training_auxiliary."
        )
    ]

    if leaked_keys:
        raise AssertionError(
            "Training-only parameters remain after deploy: "
            + ", ".join(
                leaked_keys[:5]
            )
        )

    if model.auxiliary_mode != "none":
        raise AssertionError(
            "auxiliary_mode was not reset to 'none'"
        )

    if model.training_mechanism != "none":
        raise AssertionError(
            "training_mechanism was not reset to 'none'"
        )

    print(
        "[PASS] deploy contract: "
        f"training_params={training_parameters:,}, "
        f"deploy_params={deploy_parameters:,}, "
        f"max_abs={maximum:.9e}"
    )

    return {
        "training_parameters": float(
            training_parameters
        ),
        "deploy_parameters": float(
            deploy_parameters
        ),
        "deploy_error": maximum,
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    device = get_device(
        args
    )

    print(
        "=" * 88
    )

    print(
        "Run3 BT-SAM-RDT synthetic smoke"
    )

    print(
        f"device={device}; "
        f"seed={args.seed}; "
        f"batch={args.batch_size}"
    )

    print(
        "=" * 88
    )

    # ------------------------------------------------------------------
    # A. Pure BT-SAM / task-space tests.
    # ------------------------------------------------------------------

    check_bt_sam_identical_geometry(
        device
    )

    check_bt_sam_disappearance(
        device
    )

    check_bt_sam_appearance(
        device
    )

    check_bt_sam_expansion_localization(
        device
    )

    check_bt_sam_shrinkage_localization(
        device
    )

    check_sam_quality_reliability(
        device
    )

    check_fusion_conflict(
        device
    )

    check_task_space_audit(
        device
    )

    # ------------------------------------------------------------------
    # B. Full synthetic Run3 teacher pack.
    # ------------------------------------------------------------------

    target = make_synthetic_target(
        args.batch_size,
        args.height,
        args.width,
        device,
    )

    teacher_pack = make_synthetic_teacher_pack(
        target
    )

    check_foundation_prior_paths(
        teacher_pack,
        output_size=(
            args.height,
            args.width,
        ),
    )

    # ------------------------------------------------------------------
    # C. Student inputs.
    # ------------------------------------------------------------------

    image_a = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=device,
        dtype=torch.float32,
    )

    image_b = torch.randn_like(
        image_a
    )

    # ------------------------------------------------------------------
    # D. Full Run3 model.
    # ------------------------------------------------------------------

    model = build_run3_model(
        args,
        device,
    )

    check_state_registration(
        model
    )

    check_auxiliary_prediction_invariance(
        model,
        image_a,
        image_b,
        target,
        teacher_pack,
    )

    gradient_stats = check_forward_and_gradient_isolation(
        args,
        model,
        image_a,
        image_b,
        target,
        teacher_pack,
    )

    teacher_stats = check_teacher_optimizer_and_ema(
        args,
        model,
        image_a,
        image_b,
        target,
        teacher_pack,
    )

    deploy_stats = check_deploy_contract(
        model,
        image_a,
        image_b,
    )

    # ------------------------------------------------------------------
    # Final smoke summary.
    # ------------------------------------------------------------------

    print(
        "=" * 88
    )

    print(
        "RUN3 BT-SAM-RDT SMOKE PASSED"
    )

    print(
        "Student grad: "
        f"{gradient_stats['student_gradient']:.6e}"
    )

    print(
        "Fast Teacher grad: "
        f"{gradient_stats['fast_teacher_gradient']:.6e}"
    )

    print(
        "Fast Teacher update RMS: "
        f"{teacher_stats['fast_update']:.6e}"
    )

    print(
        "EMA update RMS: "
        f"{teacher_stats['ema_update']:.6e}"
    )

    print(
        "Post-update dynamic shift: "
        f"{teacher_stats['dynamic_shift']:.6e}"
    )

    print(
        "Deploy params: "
        f"{int(deploy_stats['deploy_parameters']):,}"
    )

    print(
        "Deploy prediction max error: "
        f"{deploy_stats['deploy_error']:.9e}"
    )

    print(
        "=" * 88
    )

    print(
        "Synthetic smoke is complete. "
        "A real-cache aligned-replay dry run is still required "
        "before formal training."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback

        traceback.print_exc()

        sys.exit(
            1
        )
