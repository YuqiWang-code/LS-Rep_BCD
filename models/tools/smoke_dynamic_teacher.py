#!/usr/bin/env python3
"""
Synthetic smoke test for RDT-CD + SCGR.

SCGR = Sparse-Change Gradient-Concordant Routing.

This script is intentionally a correctness smoke test, not a formal experiment.
It uses synthetic images, GT masks, and SAMStruct/OVCDistill-shaped cache data.
It does NOT read the real datasets or Teacher Cache.

What it verifies
----------------
A. SCGR task-space mechanism
   1) zero/non-positive utility -> Reject;
   2) positive Brier utility but negative analytical gradient concordance
      -> Reject;
   3) positive utility + positive gradient concordance -> Accept;
   4) teacher selection chooses the useful teacher (SAM/OV);
   5) sparse change is normalized by remaining error mass rather than HxW,
      so useful sparse supervision is not drowned by background area;
   6) force_action=Reject really disables teacher supervision.

B. RDT-CD integration
   7) dynamic-teacher forward is finite and exposes the SCGR contract;
   8) auxiliary ON/OFF never changes the main student prediction;
   9) Student objective does not update fast-teacher parameters;
  10) Teacher fitting objective does not backpropagate into Student;
  11) one fast-teacher optimization step changes fast teacher;
  12) EMA target teacher updates and remains frozen;
  13) dynamic-teacher state is checkpoint-visible;
  14) switch_to_deploy() physically deletes all training-only auxiliary state;
  15) deploy prediction is unchanged with max absolute error < 1e-6;
  16) deployed parameter count is exactly 2,913,094.

Run only AFTER all RDT-CD + SCGR files have been replaced.

Examples
--------
CUDA:
    python models/tools/smoke_dynamic_teacher.py --device cuda --gpu_id 0

CPU:
    python models/tools/smoke_dynamic_teacher.py --device cpu

No-cache D2 control:
    python models/tools/smoke_dynamic_teacher.py --device cuda \
        --no-cache-conditioning
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import A2Net_LWGANet_L0  # noqa: E402
from models.distill.task_space import (  # noqa: E402
    OV_INDEX,
    REJECT_INDEX,
    SAM_INDEX,
    routed_bernoulli_kd,
    task_space_route,
)


EXPECTED_DEPLOY_PARAMS = 2_913_094
PREDICTION_ATOL = 1e-6
SCGR_POLICY = "scgr"
SCGR_TEACHER = "both"


# ---------------------------------------------------------------------------
# CLI / deterministic setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic P0 smoke test for RDT-CD + SCGR"
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
    parser.add_argument("--scgr_region_size", type=int, default=16)

    parser.add_argument(
        "--no-cache-conditioning",
        action="store_true",
        help=(
            "Smoke the capacity-matched D2 no-cache control. "
            "The default smokes D1 with a synthetic SAM/OV cache pack."
        ),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.height != 256 or args.width != 256:
        raise ValueError(
            "Project deploy contract is defined for 256x256 inputs"
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
    if args.scgr_region_size <= 0:
        raise ValueError("--scgr_region_size must be positive")

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
                "--device cuda requested but CUDA is not available"
            )
        if not 0 <= args.gpu_id < torch.cuda.device_count():
            raise ValueError(
                f"Invalid --gpu_id={args.gpu_id}; "
                f"visible CUDA devices={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(args.gpu_id)
        return torch.device("cuda", args.gpu_id)

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        squared += gradient.square().sum().item()
    return math.sqrt(squared)


def snapshot_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().clone()
        for parameter in parameters
    )


def parameter_rms_difference(
    before: Sequence[torch.Tensor],
    after_parameters: Iterable[torch.nn.Parameter],
) -> float:
    after = tuple(parameter.detach() for parameter in after_parameters)
    if len(before) != len(after):
        raise AssertionError(
            "Parameter snapshot length changed unexpectedly"
        )

    squared = 0.0
    count = 0
    for old, new in zip(before, after):
        delta = new.float() - old.float()
        squared += delta.square().sum().item()
        count += delta.numel()

    return math.sqrt(squared / max(count, 1))


def assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise AssertionError(f"{name} is not a tensor")
    if not bool(torch.isfinite(value).all()):
        raise AssertionError(f"{name} contains NaN/Inf")


def assert_close(
    name: str,
    value: float,
    expected: float,
    atol: float = 1e-6,
) -> None:
    if abs(value - expected) > atol:
        raise AssertionError(
            f"{name}: expected {expected:.9e}, got {value:.9e}"
        )


def max_prediction_difference(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
) -> float:
    if len(left) != len(right):
        raise AssertionError(
            "Prediction tuple lengths do not match"
        )

    maximum = 0.0
    for a, b in zip(left, right):
        if a.shape != b.shape:
            raise AssertionError(
                f"Prediction shape mismatch: {a.shape} vs {b.shape}"
            )
        maximum = max(
            maximum,
            (a.detach().float() - b.detach().float())
            .abs()
            .max()
            .item(),
        )
    return maximum


def _teacher_pair_mean(
    tensor: torch.Tensor,
) -> Tuple[float, float]:
    value = tensor.detach().float()
    if value.ndim == 0:
        raise AssertionError(
            "Expected a teacher-pair tensor, got scalar"
        )
    if value.shape[-1] == 2:
        reduced = value.reshape(-1, 2).mean(0)
    elif value.ndim >= 2 and value.shape[1] == 2:
        reduced = value.transpose(1, -1).reshape(-1, 2).mean(0)
    else:
        raise AssertionError(
            f"Could not locate teacher dimension in shape {tuple(value.shape)}"
        )
    return float(reduced[0]), float(reduced[1])


# ---------------------------------------------------------------------------
# Synthetic task-space fixtures
# ---------------------------------------------------------------------------

def _base_two_error_fixture(
    device: torch.device,
    *,
    height: int = 16,
    width: int = 16,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Construct a single region with one change error and one background error.

    All other pixels are already correct.  Feature is nonzero only at the two
    error pixels, which makes analytical gradient direction easy to control.
    """
    target = torch.zeros(
        1, 1, height, width,
        device=device,
        dtype=torch.float32,
    )
    prediction = torch.zeros_like(target)
    feature = torch.zeros_like(target)

    # Sparse changed pixel.
    target[0, 0, 0, 0] = 1.0
    prediction[0, 0, 0, 0] = 0.2
    feature[0, 0, 0, 0] = 1.0

    # One background false-positive.
    prediction[0, 0, 0, 1] = 0.6
    feature[0, 0, 0, 1] = 1.0

    quality = torch.ones(
        1, 2, height, width,
        device=device,
        dtype=torch.float32,
    )
    return prediction, target, feature, quality


@torch.no_grad()
def check_scgr_rejects_nonpositive_utility(
    device: torch.device,
) -> None:
    prediction, target, feature, quality = _base_two_error_fixture(
        device
    )
    proposals = prediction.expand(-1, 2, -1, -1).clone()

    route = task_space_route(
        prediction,
        target,
        proposals,
        quality,
        feature=feature,
        policy=SCGR_POLICY,
        teacher=SCGR_TEACHER,
        region_size=16,
    )

    if bool(route["accepted_map"].any()):
        raise AssertionError(
            "SCGR accepted an identity/non-positive-utility teacher"
        )
    if bool((route["action"] != REJECT_INDEX).any()):
        raise AssertionError(
            "SCGR action must be Reject when utility is non-positive"
        )

    print(
        "[PASS] SCGR non-positive utility -> Reject"
    )


@torch.no_grad()
def check_scgr_rejects_gradient_conflict(
    device: torch.device,
) -> None:
    """Teacher is Brier-better, but its regional classifier direction conflicts.

    Changed pixel:
        y=1, p=.2, q=.201  -> tiny positive Brier improvement.

    Background pixel:
        y=0, p=.6, q=0     -> large positive Brier improvement.

    After sparse class balancing, GT regional classifier direction is dominated
    by the changed pixel, whereas KD direction is dominated by the background
    correction.  Utility is positive but cosine becomes negative.
    """
    prediction, target, feature, quality = _base_two_error_fixture(
        device
    )

    sam = prediction.clone()
    sam[0, 0, 0, 0] = 0.201
    sam[0, 0, 0, 1] = 0.0

    # Disable OV to isolate the intended SAM counterexample.
    ov = prediction.clone()
    proposals = torch.cat((sam, ov), dim=1)
    quality[:, 1] = 0.0

    route = task_space_route(
        prediction,
        target,
        proposals,
        quality,
        feature=feature,
        policy=SCGR_POLICY,
        teacher=SCGR_TEACHER,
        region_size=16,
    )

    utility = float(route["region_utility"][0, SAM_INDEX, 0, 0])
    cosine = float(
        route["region_concordance"][0, SAM_INDEX, 0, 0]
    )
    conflict = bool(
        route["gradient_conflict_map"][:, SAM_INDEX].any()
    )

    if not utility > 0.0:
        raise AssertionError(
            f"Counterexample did not obtain positive utility: {utility}"
        )
    if not cosine < 0.0:
        raise AssertionError(
            f"Counterexample did not obtain negative concordance: {cosine}"
        )
    if not conflict:
        raise AssertionError(
            "SCGR did not mark the positive-utility negative-cosine region"
        )
    if bool(route["accepted_map"].any()):
        raise AssertionError(
            "SCGR accepted a gradient-conflicting teacher"
        )

    print(
        "[PASS] SCGR Brier-positive + gradient-negative -> Reject: "
        f"utility={utility:.6e}, cosine={cosine:.6f}"
    )


@torch.no_grad()
def check_scgr_accepts_concordant_teacher(
    device: torch.device,
) -> None:
    prediction, target, feature, quality = _base_two_error_fixture(
        device
    )

    sam = prediction.clone()
    sam[0, 0, 0, 0] = 0.8
    sam[0, 0, 0, 1] = 0.2

    ov = prediction.clone()
    proposals = torch.cat((sam, ov), dim=1)
    quality[:, 1] = 0.0

    route = task_space_route(
        prediction,
        target,
        proposals,
        quality,
        feature=feature,
        policy=SCGR_POLICY,
        teacher=SCGR_TEACHER,
        region_size=16,
    )

    utility = float(route["region_utility"][0, SAM_INDEX, 0, 0])
    cosine = float(
        route["region_concordance"][0, SAM_INDEX, 0, 0]
    )

    if not utility > 0.0:
        raise AssertionError(
            f"Useful teacher utility is not positive: {utility}"
        )
    if not cosine > 0.0:
        raise AssertionError(
            f"Useful teacher concordance is not positive: {cosine}"
        )
    if not bool(route["accepted_map"].any()):
        raise AssertionError(
            "SCGR rejected a positive-utility positive-concordance teacher"
        )
    if not bool(
        (route["action"][route["accepted_map"][:, 0]] == SAM_INDEX).all()
    ):
        raise AssertionError(
            "SCGR accepted the wrong teacher"
        )

    kd = routed_bernoulli_kd(
        prediction,
        proposals,
        route,
    )
    if not float(kd["total"]) > 0.0:
        raise AssertionError(
            "Accepted teacher produced zero routed KD"
        )

    print(
        "[PASS] SCGR positive utility + positive concordance -> Accept: "
        f"utility={utility:.6e}, cosine={cosine:.6f}, "
        f"kd={float(kd['total']):.6e}"
    )


@torch.no_grad()
def check_scgr_selects_best_available_teacher(
    device: torch.device,
) -> None:
    prediction, target, feature, quality = _base_two_error_fixture(
        device
    )

    # SAM deliberately harmful.
    sam = prediction.clone()
    sam[0, 0, 0, 0] = 0.0
    sam[0, 0, 0, 1] = 0.95

    # OV useful and gradient-concordant.
    ov = prediction.clone()
    ov[0, 0, 0, 0] = 0.8
    ov[0, 0, 0, 1] = 0.2

    proposals = torch.cat((sam, ov), dim=1)

    route = task_space_route(
        prediction,
        target,
        proposals,
        quality,
        feature=feature,
        policy=SCGR_POLICY,
        teacher=SCGR_TEACHER,
        region_size=16,
    )

    if not bool(route["accepted_map"].any()):
        raise AssertionError(
            "SCGR unexpectedly rejected both teachers"
        )

    accepted_actions = route["action"][
        route["accepted_map"][:, 0]
    ]
    if not bool((accepted_actions == OV_INDEX).all()):
        raise AssertionError(
            f"SCGR should choose OV={OV_INDEX}; "
            f"observed actions={accepted_actions.unique().tolist()}"
        )

    sam_utility = float(
        route["region_utility"][0, SAM_INDEX, 0, 0]
    )
    ov_utility = float(
        route["region_utility"][0, OV_INDEX, 0, 0]
    )
    print(
        "[PASS] SCGR teacher selection: useful OV selected over harmful SAM: "
        f"sam_utility={sam_utility:.6e}, ov_utility={ov_utility:.6e}"
    )


@torch.no_grad()
def _sparse_mass_case(
    device: torch.device,
    size: int,
) -> Tuple[float, float]:
    if size < 32 or size % 16 != 0:
        raise ValueError("Synthetic sparse-mass case requires size >=32, /16")

    target = torch.zeros(
        1, 1, size, size,
        device=device,
        dtype=torch.float32,
    )
    prediction = torch.zeros_like(target)
    feature = torch.ones_like(target)

    # Fixed 16x16 changed region regardless of canvas size.
    target[:, :, :16, :16] = 1.0
    prediction[:, :, :16, :16] = 0.2

    sam = prediction.clone()
    sam[:, :, :16, :16] = 0.8
    ov = prediction.clone()
    proposals = torch.cat((sam, ov), dim=1)

    quality = torch.zeros(
        1, 2, size, size,
        device=device,
        dtype=torch.float32,
    )
    quality[:, SAM_INDEX, :16, :16] = 1.0

    route = task_space_route(
        prediction,
        target,
        proposals,
        quality,
        feature=feature,
        policy=SCGR_POLICY,
        teacher=SCGR_TEACHER,
        region_size=16,
    )
    kd = routed_bernoulli_kd(
        prediction,
        proposals,
        route,
    )

    effective = float(
        kd["effective_per_error_mass_per_image"].mean()
    )
    return float(kd["total"]), effective


@torch.no_grad()
def check_scgr_sparse_change_mass_conservation(
    device: torch.device,
) -> None:
    """Same 16x16 useful change region on small and large canvases.

    HxW normalization would shrink the large-canvas KD substantially.
    Error-mass normalization should keep both nearly identical.
    """
    kd_small, mass_small = _sparse_mass_case(
        device,
        size=64,
    )
    kd_large, mass_large = _sparse_mass_case(
        device,
        size=256,
    )

    if kd_small <= 0.0 or kd_large <= 0.0:
        raise AssertionError(
            "Sparse useful change produced zero KD"
        )

    relative_gap = abs(kd_small - kd_large) / max(
        abs(kd_small),
        abs(kd_large),
        1e-12,
    )
    if relative_gap > 1e-5:
        raise AssertionError(
            "SCGR KD still depends on irrelevant background canvas area: "
            f"small={kd_small:.9e}, large={kd_large:.9e}, "
            f"relative_gap={relative_gap:.3e}"
        )

    assert_close(
        "small effective/error mass",
        mass_small,
        1.0,
        atol=1e-6,
    )
    assert_close(
        "large effective/error mass",
        mass_large,
        1.0,
        atol=1e-6,
    )

    print(
        "[PASS] SCGR sparse-change error-mass conservation: "
        f"kd64={kd_small:.6e}, kd256={kd_large:.6e}, "
        f"relative_gap={relative_gap:.3e}"
    )


@torch.no_grad()
def check_scgr_force_reject(
    device: torch.device,
) -> None:
    prediction, target, feature, quality = _base_two_error_fixture(
        device
    )

    sam = prediction.clone()
    sam[0, 0, 0, 0] = 0.8
    sam[0, 0, 0, 1] = 0.2
    ov = sam.clone()
    proposals = torch.cat((sam, ov), dim=1)

    route = task_space_route(
        prediction,
        target,
        proposals,
        quality,
        feature=feature,
        policy=SCGR_POLICY,
        teacher=SCGR_TEACHER,
        force_action=REJECT_INDEX,
        region_size=16,
    )

    if bool(route["accepted_map"].any()):
        raise AssertionError(
            "force_action=Reject did not disable supervision"
        )

    kd = routed_bernoulli_kd(
        prediction,
        proposals,
        route,
    )
    assert_close(
        "forced-reject KD",
        float(kd["total"]),
        0.0,
        atol=1e-12,
    )

    print(
        "[PASS] SCGR force_action=Reject -> zero teacher supervision"
    )


# ---------------------------------------------------------------------------
# Synthetic CD batch / synthetic cache
# ---------------------------------------------------------------------------

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

    for index in range(batch_size):
        offset = 5 * index

        target[
            index,
            0,
            40 + offset:88 + offset,
            44:104,
        ] = 1.0

        target[
            index,
            0,
            132:174,
            152 - offset:208 - offset,
        ] = 1.0

        # Small sparse object.
        target[
            index,
            0,
            206:218,
            64 + offset:84 + offset,
        ] = 1.0

    return target


def make_synthetic_images(
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = target.shape
    device = target.device

    image_a = torch.rand(
        batch, 3, height, width,
        device=device,
        dtype=torch.float32,
    )

    # Keep the pair realistic enough for the student graph while making the
    # change regions slightly different.
    texture = torch.randn_like(image_a) * 0.03
    change_signal = target.expand(-1, 3, -1, -1) * 0.20

    image_b = (
        image_a
        + texture
        + change_signal
    ).clamp(0.0, 1.0)

    return image_a, image_b


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
    ).view(height, 1)

    xx = torch.arange(
        width,
        device=device,
    ).view(1, width)

    columns = math.ceil(width / block)

    ids = (
        (yy // block) * columns
        + (xx // block)
        + 1
    ).to(torch.int32)

    return (
        ids.view(1, 1, height, width)
        .expand(batch_size, -1, -1, -1)
        .clone()
    )


def make_boundary_from_ids(
    ids: torch.Tensor,
) -> torch.Tensor:
    boundary = torch.zeros_like(
        ids,
        dtype=torch.float32,
    )

    boundary[:, :, 1:, :] = torch.maximum(
        boundary[:, :, 1:, :],
        (ids[:, :, 1:, :] != ids[:, :, :-1, :]).float(),
    )
    boundary[:, :, :-1, :] = torch.maximum(
        boundary[:, :, :-1, :],
        (ids[:, :, 1:, :] != ids[:, :, :-1, :]).float(),
    )

    boundary[:, :, :, 1:] = torch.maximum(
        boundary[:, :, :, 1:],
        (ids[:, :, :, 1:] != ids[:, :, :, :-1]).float(),
    )
    boundary[:, :, :, :-1] = torch.maximum(
        boundary[:, :, :, :-1],
        (ids[:, :, :, 1:] != ids[:, :, :, :-1]).float(),
    )

    return boundary


def make_synthetic_teacher_pack(
    target: torch.Tensor,
) -> Dict:
    """Construct only the fields consumed by build_task_proposals()."""
    batch_size, _, height, width = target.shape
    device = target.device

    ids_t1 = make_instance_grid(
        batch_size,
        height,
        width,
        device,
        block=32,
    )
    ids_t2 = torch.roll(
        ids_t1,
        shifts=(16, 16),
        dims=(-2, -1),
    )

    boundary_t1 = make_boundary_from_ids(ids_t1)
    boundary_t2 = make_boundary_from_ids(ids_t2)

    quality_t1 = torch.full_like(
        target,
        0.90,
    ) * (1.0 - 0.25 * boundary_t1)

    quality_t2 = torch.full_like(
        target,
        0.85,
    ) * (1.0 - 0.25 * boundary_t2)

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

    # Soft prior, not GT copy.
    ov_l1 = (0.80 * ov_l1 + 0.10).clamp(0.0, 1.0)
    ov_l2 = (0.70 * ov_l2 + 0.15).clamp(0.0, 1.0)

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
                "l1": torch.full_like(ov_l1, 0.88),
                "l2": torch.full_like(ov_l2, 0.78),
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
        "policy": SCGR_POLICY,
        "teacher": SCGR_TEACHER,
        "difficulty": True,
        "cache_conditioning": not args.no_cache_conditioning,
        "region_size": args.scgr_region_size,
    }

    model = A2Net_LWGANet_L0(
        pretrained=False,
        pretrained_path=None,
        auxiliary_mode="direction_c",
        routing_cfg=routing_cfg,
    )

    return model.to(device)


# ---------------------------------------------------------------------------
# Integration checks
# ---------------------------------------------------------------------------

def check_state_registration(
    model: A2Net_LWGANet_L0,
) -> None:
    if not hasattr(model, "training_auxiliary"):
        raise AssertionError(
            "Dynamic model has no training_auxiliary"
        )

    state_keys = tuple(model.state_dict().keys())

    fast_present = any(
        key.startswith("training_auxiliary.fast.")
        for key in state_keys
    )
    target_present = any(
        key.startswith("training_auxiliary.target.")
        for key in state_keys
    )

    if not fast_present:
        raise AssertionError(
            "Fast teacher parameters are absent from model.state_dict()"
        )
    if not target_present:
        raise AssertionError(
            "EMA target teacher parameters are absent from model.state_dict()"
        )

    target_parameters = tuple(
        model.training_auxiliary.target_teacher_parameters()
    )
    if not target_parameters:
        raise AssertionError(
            "target_teacher_parameters() returned an empty set"
        )
    if any(parameter.requires_grad for parameter in target_parameters):
        raise AssertionError(
            "EMA target parameters must all have requires_grad=False"
        )

    print(
        "[PASS] state registration: fast + EMA teachers are checkpoint-visible"
    )


def _required_dynamic_keys() -> Tuple[str, ...]:
    return (
        "total",
        "teacher_total",
        "loss_per_teacher",
        "weights",
        "effective_weights",
        "proposals",
        "student_brier",
        "dynamic_teacher_brier",
        "dynamic_teacher_gain",
        "effective_mass",
        "student_error_mass",
        "effective_per_error_mass",
        # SCGR contract used by train.py.
        "scgr_region_accept_ratio",
        "scgr_region_reject_ratio",
        "scgr_positive_utility_ratio",
        "scgr_positive_concordance_ratio",
        "scgr_gradient_conflict_ratio",
        "scgr_negative_cosine_reject_ratio",
        "scgr_change_error_mass",
        "scgr_background_error_mass",
        "scgr_effective_error_mass",
        "scgr_effective_per_error_mass",
        "scgr_region_utility",
        "scgr_region_concordance",
        "scgr_region_quality",
        "scgr_region_score",
        "scgr_teacher_mass",
    )


def check_dynamic_output_contract(
    dynamic: Dict[str, torch.Tensor],
) -> None:
    for key in _required_dynamic_keys():
        if key not in dynamic:
            raise AssertionError(
                f"Missing RDT-CD + SCGR output key: {key}"
            )
        assert_finite_tensor(key, dynamic[key])

    if dynamic["weights"].shape[-1] != 3:
        raise AssertionError(
            "weights must expose [SAM, OV, Reject] in the last dimension"
        )

    for key in (
        "scgr_region_utility",
        "scgr_region_concordance",
        "scgr_region_quality",
        "scgr_region_score",
        "scgr_teacher_mass",
    ):
        _teacher_pair_mean(dynamic[key])

    print(
        "[PASS] dynamic output contract: RDT + SCGR diagnostics are finite"
    )


@torch.no_grad()
def check_auxiliary_prediction_invariance(
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    teacher_pack: Optional[Dict],
    seed: int,
) -> float:
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
            "Auxiliary output is missing direction_c"
        )

    check_dynamic_output_contract(
        auxiliary["direction_c"]
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


def _student_parameters(
    model: A2Net_LWGANet_L0,
) -> Tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for name, parameter in model.named_parameters()
        if (
            parameter.requires_grad
            and not name.startswith("training_auxiliary.")
        )
    )


def check_forward_and_gradient_isolation(
    args: argparse.Namespace,
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    teacher_pack: Optional[Dict],
) -> Dict[str, float]:
    model.train()
    model.zero_grad(set_to_none=True)

    predictions, auxiliary = model(
        image_a,
        image_b,
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )
    dynamic = auxiliary["direction_c"]
    check_dynamic_output_contract(dynamic)

    main_loss = F.binary_cross_entropy(
        predictions[0],
        target,
    )
    student_total = (
        main_loss
        + args.kd_lambda * dynamic["total"]
    )
    teacher_total = dynamic["teacher_total"]

    for name, value in (
        ("main_loss", main_loss),
        ("student_total", student_total),
        ("teacher_total", teacher_total),
    ):
        assert_finite_tensor(name, value)

    student_parameters = _student_parameters(model)
    fast_parameters = tuple(
        model.training_auxiliary.teacher_parameters()
    )

    if not student_parameters:
        raise AssertionError(
            "Student parameter partition is empty"
        )
    if not fast_parameters:
        raise AssertionError(
            "Fast-teacher parameter partition is empty"
        )

    student_ids = {id(parameter) for parameter in student_parameters}
    teacher_ids = {id(parameter) for parameter in fast_parameters}
    if student_ids & teacher_ids:
        raise AssertionError(
            "Student and fast-teacher parameter sets overlap"
        )

    # A) Teacher -> Student graph.
    model.zero_grad(set_to_none=True)
    student_total.backward()

    student_grad = grad_norm(student_parameters)
    teacher_grad_from_student = grad_norm(fast_parameters)

    if not student_grad > 0.0:
        raise AssertionError(
            "Student objective produced zero Student gradient"
        )
    if teacher_grad_from_student != 0.0:
        raise AssertionError(
            "Student objective leaked gradient into fast teachers: "
            f"{teacher_grad_from_student:.9e}"
        )

    print(
        "[PASS] Student backward isolation: "
        f"student_grad={student_grad:.6e}, "
        f"teacher_grad={teacher_grad_from_student:.6e}"
    )

    # B) Student -> Teacher graph.
    model.zero_grad(set_to_none=True)
    teacher_total.backward()

    student_grad_from_teacher = grad_norm(student_parameters)
    teacher_grad = grad_norm(fast_parameters)

    if student_grad_from_teacher != 0.0:
        raise AssertionError(
            "Teacher objective leaked gradient into Student: "
            f"{student_grad_from_teacher:.9e}"
        )
    if not teacher_grad > 0.0:
        raise AssertionError(
            "Teacher objective produced zero fast-teacher gradient"
        )

    print(
        "[PASS] Teacher backward isolation: "
        f"student_grad={student_grad_from_teacher:.6e}, "
        f"teacher_grad={teacher_grad:.6e}"
    )

    return {
        "main_loss": float(main_loss.detach()),
        "student_total": float(student_total.detach()),
        "teacher_total": float(teacher_total.detach()),
        "student_grad": student_grad,
        "teacher_grad": teacher_grad,
        "effective_mass": float(
            dynamic["effective_mass"].detach().float().mean()
        ),
        "effective_per_error_mass": float(
            dynamic["effective_per_error_mass"]
            .detach()
            .float()
            .mean()
        ),
    }


def check_teacher_optimizer_and_ema(
    args: argparse.Namespace,
    model: A2Net_LWGANet_L0,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    target: torch.Tensor,
    teacher_pack: Optional[Dict],
) -> Dict[str, float]:
    model.train()

    fast_parameters = tuple(
        model.training_auxiliary.teacher_parameters()
    )
    target_parameters = tuple(
        model.training_auxiliary.target_teacher_parameters()
    )

    optimizer = torch.optim.Adam(
        fast_parameters,
        lr=args.teacher_lr,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    fast_before = snapshot_parameters(fast_parameters)
    target_before = snapshot_parameters(target_parameters)

    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    _, auxiliary = model(
        image_a,
        image_b,
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )

    teacher_loss = auxiliary["direction_c"]["teacher_total"]
    assert_finite_tensor("teacher_total", teacher_loss)

    teacher_loss.backward()
    teacher_gradient = grad_norm(fast_parameters)

    if not teacher_gradient > 0.0:
        raise AssertionError(
            "Teacher optimizer received zero teacher gradient"
        )

    optimizer.step()

    fast_update = parameter_rms_difference(
        fast_before,
        fast_parameters,
    )
    if not fast_update > 0.0:
        raise AssertionError(
            "Fast-teacher parameters did not change after optimizer.step()"
        )

    ema_reported_update = float(
        model.training_auxiliary.update_ema()
    )
    target_update = parameter_rms_difference(
        target_before,
        target_parameters,
    )
    target_gap = float(
        model.training_auxiliary.teacher_target_gap()
    )

    for name, value in (
        ("ema_reported_update", ema_reported_update),
        ("target_update", target_update),
        ("target_gap", target_gap),
    ):
        if not math.isfinite(value):
            raise AssertionError(
                f"{name} is NaN/Inf"
            )

    if not ema_reported_update > 0.0:
        raise AssertionError(
            "update_ema() reported zero teacher evolution"
        )
    if not target_update > 0.0:
        raise AssertionError(
            "EMA target parameters did not change"
        )

    if any(parameter.requires_grad for parameter in target_parameters):
        raise AssertionError(
            "EMA target parameters became trainable"
        )

    print(
        "[PASS] dynamic teacher update: "
        f"grad={teacher_gradient:.6e}, "
        f"fast_update_rms={fast_update:.6e}, "
        f"ema_update_rms={ema_reported_update:.6e}, "
        f"target_update_rms={target_update:.6e}, "
        f"fast_target_gap={target_gap:.6e}"
    )

    # One forward after teacher evolution.  We require finite dynamic signal,
    # but do NOT require SCGR to accept it: Reject is a valid decision when
    # utility/concordance are not satisfied.
    with torch.no_grad():
        _, auxiliary_after = model(
            image_a,
            image_b,
            target=target,
            teacher_pack=teacher_pack,
            compute_auxiliary=True,
        )

    dynamic_after = auxiliary_after["direction_c"]
    check_dynamic_output_contract(dynamic_after)

    dynamic_shift = dynamic_after.get(
        "dynamic_shift",
        torch.zeros(2, device=image_a.device),
    )
    assert_finite_tensor("dynamic_shift", dynamic_shift)

    print(
        "[PASS] post-update dynamic forward remains finite; "
        "SCGR is allowed to Accept or Reject based on evidence"
    )

    return {
        "teacher_gradient": teacher_gradient,
        "fast_update_rms": fast_update,
        "ema_update_rms": ema_reported_update,
        "target_update_rms": target_update,
        "teacher_target_gap": target_gap,
    }


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
        compute_auxiliary=False,
    )
    training_parameter_count = parameter_count(model)

    if not hasattr(model, "training_auxiliary"):
        raise AssertionError(
            "training_auxiliary disappeared before switch_to_deploy()"
        )

    model.switch_to_deploy()

    if hasattr(model, "training_auxiliary"):
        raise AssertionError(
            "switch_to_deploy() did not delete training_auxiliary"
        )
    if getattr(model, "auxiliary_mode", None) != "none":
        raise AssertionError(
            "switch_to_deploy() did not reset auxiliary_mode"
        )

    prediction_after = model(
        image_a,
        image_b,
        compute_auxiliary=False,
    )

    deploy_parameter_count = parameter_count(model)
    prediction_difference = max_prediction_difference(
        prediction_before,
        prediction_after,
    )

    if deploy_parameter_count != EXPECTED_DEPLOY_PARAMS:
        raise AssertionError(
            "Deploy parameter count mismatch: "
            f"expected={EXPECTED_DEPLOY_PARAMS:,}, "
            f"actual={deploy_parameter_count:,}"
        )

    if prediction_difference >= PREDICTION_ATOL:
        raise AssertionError(
            "Deploy conversion changed main prediction: "
            f"max_abs={prediction_difference:.9e}"
        )

    if training_parameter_count <= deploy_parameter_count:
        raise AssertionError(
            "Training model did not contain removable teacher parameters"
        )

    state_keys = tuple(model.state_dict().keys())
    if any(key.startswith("training_auxiliary.") for key in state_keys):
        raise AssertionError(
            "Deploy state_dict still contains training_auxiliary keys"
        )

    print(
        "[PASS] deploy contract: "
        f"train_params={training_parameter_count:,}, "
        f"deploy_params={deploy_parameter_count:,}, "
        f"max_abs={prediction_difference:.9e}"
    )

    return {
        "training_params": float(training_parameter_count),
        "deploy_params": float(deploy_parameter_count),
        "deploy_max_abs": prediction_difference,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scgr_unit_smoke(
    device: torch.device,
) -> None:
    print("\n=== SCGR TASK-SPACE UNIT SMOKE ===")
    check_scgr_rejects_nonpositive_utility(device)
    check_scgr_rejects_gradient_conflict(device)
    check_scgr_accepts_concordant_teacher(device)
    check_scgr_selects_best_available_teacher(device)
    check_scgr_sparse_change_mass_conservation(device)
    check_scgr_force_reject(device)


def run_model_integration_smoke(
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    print("\n=== RDT-CD + SCGR MODEL INTEGRATION SMOKE ===")

    target = make_synthetic_target(
        args.batch_size,
        args.height,
        args.width,
        device,
    )
    image_a, image_b = make_synthetic_images(target)

    if args.no_cache_conditioning:
        teacher_pack = None
        mode_name = "D2 no-cache control"
    else:
        teacher_pack = make_synthetic_teacher_pack(target)
        mode_name = "D1 cache-conditioned"

    print(f"mode: {mode_name}")

    model = build_dynamic_model(args, device)

    check_state_registration(model)

    auxiliary_difference = check_auxiliary_prediction_invariance(
        model,
        image_a,
        image_b,
        target,
        teacher_pack,
        seed=args.seed + 101,
    )

    isolation = check_forward_and_gradient_isolation(
        args,
        model,
        image_a,
        image_b,
        target,
        teacher_pack,
    )

    teacher_update = check_teacher_optimizer_and_ema(
        args,
        model,
        image_a,
        image_b,
        target,
        teacher_pack,
    )

    deploy = check_deploy_contract(
        model,
        image_a,
        image_b,
    )

    result = {
        "auxiliary_prediction_max_abs": auxiliary_difference,
        **isolation,
        **teacher_update,
        **deploy,
    }

    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args)

    print("RDT-CD + SCGR synthetic smoke")
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"region_size: {args.scgr_region_size}")
    print(
        "cache_conditioning: "
        f"{not args.no_cache_conditioning}"
    )

    run_scgr_unit_smoke(device)

    integration = run_model_integration_smoke(
        args,
        device,
    )

    print("\n=== SMOKE SUMMARY ===")
    for key, value in integration.items():
        if isinstance(value, float):
            print(f"{key}: {value:.9e}")
        else:
            print(f"{key}: {value}")

    print("\nALL RDT-CD + SCGR SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
