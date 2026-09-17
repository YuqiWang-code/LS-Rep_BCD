"""Run3 BT-SAM-RDT trainer for A2Net-LWGANet-L0."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

try:
    from thop import profile
except ImportError:
    profile = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from models import A2Net_LWGANet_L0, build_loss
from models.datasets.cd_dataset import (
    get_loader,
    get_test_loader,
)
from models.distill import PairedTeacherCache
from models.distill.diagnostics import H_NAMES
from models.utils.checkpoint import (
    build_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
)
from models.utils.logger import TrainingLogger
from models.utils.metrics import ConfuseMatrixMeter
from models.utils.scheduler import adjust_learning_rate


# ============================================================================
# Deployment contract
# ============================================================================


EXPECTED_DEPLOY_PARAMS = 2_913_094
EXPECTED_DEPLOY_FLOPS = 2.7475e9
DEPLOY_FLOPS_ATOL = 0.03e9


# ============================================================================
# Run3 experiment matrix
# ============================================================================


EXPERIMENTS = {
    # --------------------------------------------------------------
    # Clean student.
    # --------------------------------------------------------------
    "B0": {
        "name": "B0_Clean_A2Net_LWGANet_L0",
        "auxiliary_mode": "none",
        "use_ov": False,
        "requires_teacher_cache": False,
    },

    # --------------------------------------------------------------
    # Minimal mechanism ablation:
    #
    # BT-SAM structural prior
    #       -> one Fast Teacher
    #       -> EMA Target Teacher
    #
    # OV information is disabled, while teacher capacity remains
    # identical to the full Run3 model.
    # --------------------------------------------------------------
    "R3A": {
        "name": "R3A_BT_SAM_RDT_SAM_Only",
        "auxiliary_mode": "bt_sam_rdt",
        "use_ov": False,
        "requires_teacher_cache": True,
    },

    # --------------------------------------------------------------
    # Run3 main method:
    #
    # BT-SAM structural prior
    #       +
    # OV semantic prior
    #       -> fused foundation prior
    #       -> one Fast Teacher
    #       -> EMA Target Teacher
    #       -> positive-Brier-gain audit
    #       -> Student KD
    # --------------------------------------------------------------
    "R3": {
        "name": "R3_BT_SAM_RDT_Full",
        "auxiliary_mode": "bt_sam_rdt",
        "use_ov": True,
        "requires_teacher_cache": True,
    },
}


AUXILIARY_EXPERIMENTS = {
    "R3A",
    "R3",
}


# ============================================================================
# Arguments
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run3 BT-SAM-RDT trainer: "
            "B0 clean / R3A BT-SAM-only / R3 BT-SAM+OV"
        )
    )

    # ------------------------------------------------------------------
    # Experiment / dataset
    # ------------------------------------------------------------------
    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(EXPERIMENTS),
    )

    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=[
            "CDD",
            "LEVIR",
            "SYSU",
            "WHU",
        ],
    )

    parser.add_argument(
        "--data_root",
        required=True,
    )

    parser.add_argument(
        "--sam_cache_root",
        default=None,
    )

    parser.add_argument(
        "--ov_cache_root",
        default=None,
    )

    # ------------------------------------------------------------------
    # Device / reproducibility
    # ------------------------------------------------------------------
    parser.add_argument(
        "--device",
        choices=[
            "cuda",
            "cpu",
        ],
        default="cuda",
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2333,
    )

    # ------------------------------------------------------------------
    # Run3 detached Student-difficulty diagnosis
    # ------------------------------------------------------------------
    parser.add_argument(
        "--boundary_radius",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--small_area",
        type=int,
        default=64,
    )

    # ------------------------------------------------------------------
    # Run3 Fast / EMA Teacher
    # ------------------------------------------------------------------
    parser.add_argument(
        "--teacher_lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--teacher_hidden",
        type=int,
        default=24,
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
        "--teacher_weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--teacher_grad_clip",
        type=float,
        default=5.0,
    )

    # ------------------------------------------------------------------
    # Student initialization
    # ------------------------------------------------------------------
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--pretrained_path",
        default=None,
    )

    # ------------------------------------------------------------------
    # Formal training protocol
    # ------------------------------------------------------------------
    parser.add_argument(
        "--inWidth",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--inHeight",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=40000,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    # ------------------------------------------------------------------
    # Student optimizer
    # ------------------------------------------------------------------
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--lr_mode",
        default="poly",
        choices=[
            "poly",
            "step",
        ],
    )

    parser.add_argument(
        "--step_loss",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--backbone_lr_mult",
        type=float,
        default=1.0,
    )

    # ------------------------------------------------------------------
    # Student loss
    # ------------------------------------------------------------------
    parser.add_argument(
        "--dice_reduction",
        default="batch",
        choices=[
            "sample",
            "batch",
        ],
    )

    parser.add_argument(
        "--main_loss_weights",
        default="1,1,1,1",
    )

    parser.add_argument(
        "--kd_lambda",
        type=float,
        default=0.06,
    )

    # ------------------------------------------------------------------
    # Output / recovery
    # ------------------------------------------------------------------
    parser.add_argument(
        "--save_dir",
        required=True,
    )

    parser.add_argument(
        "--log_file",
        default="train_log.txt",
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    # ==================================================================
    # Parse / validate
    # ==================================================================

    args.main_loss_weights = tuple(
        float(value)
        for value in args.main_loss_weights.split(",")
    )

    if (
        len(args.main_loss_weights)
        != 4
    ):
        raise ValueError(
            "main_loss_weights must contain four comma-separated values"
        )

    if any(
        (
            not math.isfinite(value)
            or value < 0
        )
        for value in args.main_loss_weights
    ):
        raise ValueError(
            "Invalid main_loss_weights"
        )

    if (
        args.batch_size <= 0
        or args.max_steps <= 0
        or args.num_workers < 0
    ):
        raise ValueError(
            "batch_size/max_steps must be positive and "
            "num_workers must be non-negative"
        )

    if args.lr <= 0:
        raise ValueError(
            "Student learning rate must be positive"
        )

    if args.teacher_lr <= 0:
        raise ValueError(
            "Teacher learning rate must be positive"
        )

    if args.weight_decay < 0:
        raise ValueError(
            "Student weight_decay must be non-negative"
        )

    if (
        args.teacher_weight_decay
        < 0
    ):
        raise ValueError(
            "teacher_weight_decay must be non-negative"
        )

    if (
        args.teacher_grad_clip
        < 0
    ):
        raise ValueError(
            "teacher_grad_clip must be non-negative"
        )

    if (
        args.boundary_radius
        < 1
    ):
        raise ValueError(
            "boundary_radius must be positive"
        )

    if (
        args.small_area
        < 1
    ):
        raise ValueError(
            "small_area must be positive"
        )

    if (
        args.teacher_hidden
        < 1
    ):
        raise ValueError(
            "teacher_hidden must be positive"
        )

    if not (
        0.0
        <= args.teacher_ema
        < 1.0
    ):
        raise ValueError(
            "teacher_ema must satisfy 0 <= teacher_ema < 1"
        )

    if (
        args.max_logit_delta
        <= 0
    ):
        raise ValueError(
            "max_logit_delta must be positive"
        )

    if (
        args.kd_lambda
        < 0
    ):
        raise ValueError(
            "kd_lambda must be non-negative"
        )

    if (
        args.inWidth != 256
        or args.inHeight != 256
    ):
        raise ValueError(
            "Formal Run3 protocol requires 256x256 inputs"
        )

    if (
        args.inWidth
        != args.inHeight
    ):
        raise ValueError(
            "Only square 256x256 inputs are supported"
        )

    if (
        args.pretrained
        and not args.pretrained_path
    ):
        raise ValueError(
            "--pretrained_path is required when --pretrained is enabled"
        )

    # ==================================================================
    # Resolve experiment
    # ==================================================================

    recipe = EXPERIMENTS[
        args.experiment
    ]

    args.experiment_name = (
        recipe[
            "name"
        ]
    )

    args.auxiliary_mode = (
        recipe[
            "auxiliary_mode"
        ]
    )

    args.use_ov = bool(
        recipe[
            "use_ov"
        ]
    )

    args.requires_teacher_cache = bool(
        recipe[
            "requires_teacher_cache"
        ]
    )

    args.teacher_update = (
        args.experiment
        in AUXILIARY_EXPERIMENTS
    )

    args.mechanism = (
        "bt_sam_rdt"
        if args.teacher_update
        else "none"
    )

    # Exact augmentation/cache replay protocol.
    args.cache_replay = "aligned"

    # Fixed Run3 audit policy.
    args.routing_signal = (
        "pixel_positive_brier_gain"
        if args.teacher_update
        else "none"
    )

    args.reject_unit = (
        "pixel"
        if args.teacher_update
        else "none"
    )

    args.implementation_version = (
        "bt_sam_rdt_run3_v1"
    )

    # Run3/R3A intentionally use the same paired-cache loading path.
    # R3A ignores OV inside BTSAMRDT but uses the identical input pipeline.
    if (
        args.requires_teacher_cache
        and not (
            args.sam_cache_root
            and args.ov_cache_root
        )
    ):
        raise ValueError(
            f"{args.experiment} requires "
            "--sam_cache_root AND --ov_cache_root"
        )

    return args


# ============================================================================
# Reproducibility
# ============================================================================


def set_seed(
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    cudnn.deterministic = True
    cudnn.benchmark = False


# ============================================================================
# Nested teacher-pack helpers
# ============================================================================


def nested_to(
    value,
    device,
):
    if torch.is_tensor(
        value
    ):
        return value.to(
            device,
            non_blocking=True,
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            key: nested_to(
                item,
                device,
            )
            for key, item in value.items()
        }

    return value


def nested_add_batch_to(
    value,
    device,
):
    """
    Add batch dimension to one Dataset sample teacher pack.
    """
    if torch.is_tensor(
        value
    ):
        return (
            value
            .unsqueeze(0)
            .to(
                device,
                non_blocking=True,
            )
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            key: nested_add_batch_to(
                item,
                device,
            )
            for key, item in value.items()
        }

    return value


# ============================================================================
# Batch / loss helpers
# ============================================================================


def unpack_batch(
    batch,
):
    if len(batch) == 2:
        return (
            batch[0],
            batch[1],
            None,
            None,
        )

    if len(batch) == 3:
        return (
            batch[0],
            batch[1],
            batch[2],
            None,
        )

    if len(batch) == 4:
        return (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )

    raise ValueError(
        "Unexpected batch structure with "
        f"{len(batch)} fields"
    )


def multiscale_loss(
    predictions,
    target,
    criterion,
    weights,
):
    return sum(
        weight
        * criterion(
            prediction,
            target,
        )
        for (
            weight,
            prediction,
        ) in zip(
            weights,
            predictions,
        )
    )


# ============================================================================
# Optimizers
# ============================================================================


def build_student_optimizer(
    args,
    model,
):
    """
    Build Student optimizer.

    Critical Run3 contract:
        no training_auxiliary parameter may enter this optimizer.
    """
    grouped = {}

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if name.startswith(
            "training_auxiliary."
        ):
            continue

        scale = (
            args.backbone_lr_mult
            if name.startswith(
                "backbone."
            )
            else 1.0
        )

        grouped.setdefault(
            scale,
            [],
        ).append(
            parameter
        )

    if not grouped:
        raise RuntimeError(
            "Student optimizer received no parameters"
        )

    groups = [
        {
            "params": parameters,
            "lr": (
                args.lr
                * scale
            ),
            "lr_scale": scale,
            "weight_decay": (
                args.weight_decay
            ),
        }
        for (
            scale,
            parameters,
        ) in grouped.items()
    ]

    return torch.optim.Adam(
        groups,
        lr=args.lr,
        betas=(
            0.9,
            0.99,
        ),
        eps=1e-8,
    )


def build_teacher_optimizer(
    args,
    model,
):
    """
    Build Run3 Fast Teacher optimizer.

    EMA Target Teacher parameters are frozen and are not returned by
    teacher_parameters().
    """
    if not args.teacher_update:
        return None

    if not model.use_training_auxiliary:
        raise RuntimeError(
            "Run3 selected but model contains no training auxiliary"
        )

    if not hasattr(
        model.training_auxiliary,
        "teacher_parameters",
    ):
        raise RuntimeError(
            "BTSAMRDT does not expose teacher_parameters()"
        )

    parameters = list(
        model
        .training_auxiliary
        .teacher_parameters()
    )

    if not parameters:
        raise RuntimeError(
            "Fast Teacher optimizer received no parameters"
        )

    if any(
        not parameter.requires_grad
        for parameter in parameters
    ):
        raise RuntimeError(
            "Every Fast Teacher parameter must require gradients"
        )

    return torch.optim.Adam(
        [
            {
                "params": parameters,
                "lr": (
                    args.teacher_lr
                ),
                "lr_scale": 1.0,
                "weight_decay": (
                    args.teacher_weight_decay
                ),
            }
        ],
        lr=args.teacher_lr,
        betas=(
            0.9,
            0.99,
        ),
        eps=1e-8,
    )


def adjust_teacher_learning_rate(
    args,
    teacher_optimizer,
    student_base_lr,
):
    """
    Apply the same warmup/decay factor used by the Student while preserving
    the Teacher's independent base learning rate.
    """
    if teacher_optimizer is None:
        return 0.0

    if args.lr <= 0:
        raise ValueError(
            "Student base learning rate must be positive"
        )

    factor = (
        float(
            student_base_lr
        )
        / float(
            args.lr
        )
    )

    teacher_lr = (
        float(
            args.teacher_lr
        )
        * factor
    )

    for group in (
        teacher_optimizer.param_groups
    ):
        group[
            "lr"
        ] = (
            teacher_lr
            * group.get(
                "lr_scale",
                1.0,
            )
        )

    return teacher_lr


def gradient_norm(
    parameters,
):
    """
    Global L2 gradient norm without modifying gradients.
    """
    squared = None

    for parameter in parameters:
        if parameter.grad is None:
            continue

        value = (
            parameter
            .grad
            .detach()
            .float()
            .square()
            .sum()
        )

        squared = (
            value
            if squared is None
            else squared + value
        )

    if squared is None:
        return 0.0

    return float(
        torch.sqrt(
            squared
        ).item()
    )


# ============================================================================
# Model
# ============================================================================


def build_model(
    args,
):
    if args.teacher_update:
        auxiliary_cfg = {
            "hidden": (
                args.teacher_hidden
            ),
            "ema": (
                args.teacher_ema
            ),
            "max_logit_delta": (
                args.max_logit_delta
            ),
            "boundary_radius": (
                args.boundary_radius
            ),
            "small_area": (
                args.small_area
            ),
            "policy": (
                "advantage"
            ),
            "difficulty": True,
            "use_ov": (
                args.use_ov
            ),
        }

        return A2Net_LWGANet_L0(
            pretrained=args.pretrained,
            pretrained_path=args.pretrained_path,
            auxiliary_mode="bt_sam_rdt",
            auxiliary_cfg=auxiliary_cfg,
        )

    return A2Net_LWGANet_L0(
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        auxiliary_mode="none",
        auxiliary_cfg=None,
    )


# ============================================================================
# Training diagnostics
# ============================================================================


def _training_metric_keys():
    keys = [
        # Main losses.
        "total",
        "main",
        "aux_raw",
        "aux_weighted",
        "aux_ratio",

        # Fast Teacher optimization.
        "teacher_total",
        "teacher_lr",
        "teacher_grad_norm",
        "teacher_ema_update_norm",
        "teacher_target_gap",

        # Runtime.
        "data_time",
        "step_time",
    ]

    keys.extend(
        "h_" + name
        for name in H_NAMES
    )

    keys.extend(
        [
            # ----------------------------------------------------------
            # BT-SAM correspondence
            # ----------------------------------------------------------
            "pair_match_ratio",
            "pair_match_iou",
            "pair_match_cov12",
            "pair_match_cov21",

            # ----------------------------------------------------------
            # Reliability / fusion
            # ----------------------------------------------------------
            "sam_reliability_mean",
            "ov_reliability_mean",
            "fused_reliability_mean",
            "sam_ov_conflict",
            "foundation_support_ratio",

            # ----------------------------------------------------------
            # GT safety audit
            # ----------------------------------------------------------
            "pixel_reject_ratio",
            "image_reject_ratio",
            "accepted_change_ratio",
            "accepted_bg_ratio",
            "available_ratio",
            "eligible_ratio",
            "effective_mass",
            "effective_per_error_mass",

            # ----------------------------------------------------------
            # Prior / teacher quality
            # ----------------------------------------------------------
            "student_brier",
            "sam_prior_brier",
            "ov_prior_brier",
            "fusion_prior_brier",
            "dynamic_teacher_brier",
            "fast_teacher_brier",

            "sam_prior_gain",
            "ov_prior_gain",
            "fusion_prior_gain",
            "dynamic_teacher_gain",

            "sam_prior_brier_weighted",
            "ov_prior_brier_weighted",
            "fusion_prior_brier_weighted",
            "dynamic_teacher_brier_weighted",

            # ----------------------------------------------------------
            # Student / teacher displacement
            # ----------------------------------------------------------
            "dynamic_shift",
            "fused_prior_shift",
            "sam_prior_shift",
            "ov_prior_shift",

            "target_residual_magnitude",
            "fast_residual_magnitude",

            # ----------------------------------------------------------
            # Remaining Student error
            # ----------------------------------------------------------
            "student_abs_error",
            "student_error_mass",

            # ----------------------------------------------------------
            # Explicit experiment state
            # ----------------------------------------------------------
            "use_ov",
            "policy_advantage",
        ]
    )

    return tuple(
        keys
    )


def _scalar_metric(
    value,
    reference,
):
    """
    Convert scalar or tensor diagnostic into one detached scalar tensor.
    """
    if value is None:
        return reference.new_zeros(
            ()
        )

    if torch.is_tensor(
        value
    ):
        return (
            value
            .detach()
            .float()
            .mean()
        )

    return reference.new_tensor(
        float(value)
    )


def _populate_run3_metrics(
    values,
    detail,
    reference,
):
    if not detail:
        return

    if "h" in detail:
        h = (
            detail[
                "h"
            ]
            .detach()
            .float()
        )

        if (
            h.ndim != 2
            or h.shape[1]
            != len(H_NAMES)
        ):
            raise ValueError(
                "Run3 diagnostic h must be [B,6]"
            )

        for (
            index,
            name,
        ) in enumerate(
            H_NAMES
        ):
            values[
                "h_" + name
            ] = (
                h[
                    :,
                    index,
                ]
                .mean()
            )

    scalar_names = (
        "pair_match_ratio",
        "pair_match_iou",
        "pair_match_cov12",
        "pair_match_cov21",

        "sam_reliability_mean",
        "ov_reliability_mean",
        "fused_reliability_mean",
        "sam_ov_conflict",
        "foundation_support_ratio",

        "pixel_reject_ratio",
        "image_reject_ratio",
        "accepted_change_ratio",
        "accepted_bg_ratio",
        "available_ratio",
        "eligible_ratio",
        "effective_mass",
        "effective_per_error_mass",

        "student_brier",
        "sam_prior_brier",
        "ov_prior_brier",
        "fusion_prior_brier",
        "dynamic_teacher_brier",
        "fast_teacher_brier",

        "sam_prior_gain",
        "ov_prior_gain",
        "fusion_prior_gain",
        "dynamic_teacher_gain",

        "sam_prior_brier_weighted",
        "ov_prior_brier_weighted",
        "fusion_prior_brier_weighted",
        "dynamic_teacher_brier_weighted",

        "dynamic_shift",
        "fused_prior_shift",
        "sam_prior_shift",
        "ov_prior_shift",

        "target_residual_magnitude",
        "fast_residual_magnitude",

        "student_abs_error",
        "student_error_mass",

        "use_ov",
        "policy_advantage",
    )

    for name in scalar_names:
        if name in detail:
            values[
                name
            ] = _scalar_metric(
                detail[
                    name
                ],
                reference,
            )


# ============================================================================
# One training epoch
# ============================================================================


def train_epoch(
    args,
    loader,
    model,
    criterion,
    student_optimizer,
    epoch,
    global_step,
    device,
    teacher_optimizer=None,
):
    model.train()

    loader.dataset.set_epoch(
        epoch
    )

    loader.generator.manual_seed(
        args.seed
        + epoch
    )

    meter = ConfuseMatrixMeter(
        n_class=2
    )

    keys = (
        _training_metric_keys()
    )

    totals = dict.fromkeys(
        keys,
        0.0,
    )

    batches = 0

    last_lr = (
        args.lr
    )

    last_teacher_lr = 0.0

    data_started = (
        time.perf_counter()
    )

    for batch in loader:
        data_elapsed = (
            time.perf_counter()
            - data_started
        )

        if (
            global_step
            >= args.max_steps
        ):
            break

        step_started = (
            time.perf_counter()
        )

        (
            image,
            target,
            _,
            teacher_pack,
        ) = unpack_batch(
            batch
        )

        pre = (
            image[
                :,
                :3,
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        post = (
            image[
                :,
                3:6,
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        target = (
            target
            .to(
                device,
                non_blocking=True,
            )
            .float()
        )

        teacher_pack = (
            nested_to(
                teacher_pack,
                device,
            )
            if teacher_pack is not None
            else None
        )

        # ==============================================================
        # Learning rates
        # ==============================================================

        last_lr = adjust_learning_rate(
            args,
            student_optimizer,
            epoch,
            global_step,
            len(loader),
        )

        last_teacher_lr = (
            adjust_teacher_learning_rate(
                args,
                teacher_optimizer,
                last_lr,
            )
        )

        # ==============================================================
        # Clear disjoint optimization graphs
        # ==============================================================

        student_optimizer.zero_grad(
            set_to_none=True
        )

        if (
            teacher_optimizer
            is not None
        ):
            teacher_optimizer.zero_grad(
                set_to_none=True
            )

        # ==============================================================
        # Forward
        # ==============================================================

        predictions, auxiliary = model(
            pre,
            post,
            target=target,
            teacher_pack=teacher_pack,
            compute_auxiliary=(
                args.teacher_update
            ),
        )

        main_loss = multiscale_loss(
            predictions,
            target,
            criterion,
            args.main_loss_weights,
        )

        zero = (
            main_loss
            .new_zeros(
                ()
            )
        )

        detail = auxiliary.get(
            "bt_sam_rdt",
            {},
        )

        if (
            args.teacher_update
            and not detail
        ):
            raise RuntimeError(
                "Run3 auxiliary was requested but BTSAMRDT returned no output"
            )

        aux_raw = detail.get(
            "total",
            zero,
        )

        aux_weighted = (
            args.kd_lambda
            * aux_raw
        )

        student_loss = (
            main_loss
            + aux_weighted
        )

        teacher_loss = detail.get(
            "teacher_total",
            zero,
        )

        if (
            teacher_optimizer is not None
            and "teacher_total" not in detail
        ):
            raise RuntimeError(
                "Run3 Fast Teacher optimizer exists but teacher_total is missing"
            )

        # ==============================================================
        # Numerical safety
        # ==============================================================

        finite_values = [
            student_loss,
        ]

        if (
            teacher_optimizer
            is not None
        ):
            finite_values.append(
                teacher_loss
            )

        if not all(
            bool(
                torch.isfinite(
                    value
                )
            )
            for value in finite_values
        ):
            raise FloatingPointError(
                "Non-finite training loss; checkpoint not overwritten"
            )

        # ==============================================================
        # A. Target Teacher -> Student
        #
        # Dynamic target proposal is no-grad inside BTSAMRDT.
        # Fast Teacher parameters are excluded from Student optimizer.
        # ==============================================================

        student_loss.backward()

        student_optimizer.step()

        # ==============================================================
        # B. Student -> Fast Teacher
        #
        # BTSAMRDT explicitly detached all Student inputs on this graph.
        # ==============================================================

        teacher_grad_norm = 0.0
        teacher_ema_update_norm = 0.0
        teacher_target_gap = 0.0

        if (
            teacher_optimizer
            is not None
        ):
            if not teacher_loss.requires_grad:
                raise RuntimeError(
                    "teacher_total has no gradient graph"
                )

            teacher_loss.backward()

            teacher_parameters = list(
                model
                .training_auxiliary
                .teacher_parameters()
            )

            teacher_grad_norm = gradient_norm(
                teacher_parameters
            )

            if not math.isfinite(
                teacher_grad_norm
            ):
                raise FloatingPointError(
                    "Non-finite Fast Teacher gradient norm"
                )

            if (
                args.teacher_grad_clip
                > 0
            ):
                torch.nn.utils.clip_grad_norm_(
                    teacher_parameters,
                    max_norm=(
                        args.teacher_grad_clip
                    ),
                )

            teacher_optimizer.step()

            # ==========================================================
            # C. Fast Teacher -> EMA Target Teacher
            # ==========================================================

            teacher_ema_update_norm = float(
                model
                .training_auxiliary
                .update_ema()
            )

            teacher_target_gap = float(
                model
                .training_auxiliary
                .teacher_target_gap()
            )

        # ==============================================================
        # Training segmentation metrics
        # ==============================================================

        hard_prediction = (
            predictions[
                0
            ]
            .detach()
            > 0.5
        ).long()

        current_f1 = (
            meter.update_cm(
                hard_prediction
                .cpu()
                .numpy(),
                target
                .cpu()
                .numpy(),
            )
        )

        aux_ratio = (
            aux_weighted.detach()
            / main_loss.detach().clamp_min(
                1e-8
            )
        )

        # ==============================================================
        # Diagnostic accumulation
        # ==============================================================

        values = {
            key: zero
            for key in keys
        }

        values.update(
            {
                "total": (
                    student_loss
                ),
                "main": (
                    main_loss
                ),
                "aux_raw": (
                    aux_raw
                ),
                "aux_weighted": (
                    aux_weighted
                ),
                "aux_ratio": (
                    aux_ratio
                ),
                "teacher_total": (
                    teacher_loss
                ),
                "teacher_lr": (
                    zero.new_tensor(
                        last_teacher_lr
                    )
                ),
                "teacher_grad_norm": (
                    zero.new_tensor(
                        teacher_grad_norm
                    )
                ),
                "teacher_ema_update_norm": (
                    zero.new_tensor(
                        teacher_ema_update_norm
                    )
                ),
                "teacher_target_gap": (
                    zero.new_tensor(
                        teacher_target_gap
                    )
                ),
            }
        )

        if detail:
            _populate_run3_metrics(
                values,
                detail,
                zero,
            )

        values[
            "data_time"
        ] = zero.new_tensor(
            data_elapsed
        )

        values[
            "step_time"
        ] = zero.new_tensor(
            time.perf_counter()
            - step_started
        )

        for (
            key,
            value,
        ) in values.items():

            totals[
                key
            ] += float(
                _scalar_metric(
                    value,
                    zero,
                )
            )

        batches += 1
        global_step += 1

        # ==============================================================
        # Console progress
        # ==============================================================

        if (
            global_step
            % 5
            == 0
        ):
            message = (
                f"\rstep [{global_step}/{args.max_steps}] "
                f"F1={current_f1:.3f} "
                f"lr={last_lr:.7f} "
                f"loss={float(student_loss.detach()):.3f}"
            )

            if (
                teacher_optimizer
                is not None
            ):
                message += (
                    f" aux={float(aux_raw.detach()):.4f}"
                    f" reject={float(values['pixel_reject_ratio']):.3f}"
                    f" teacher={float(teacher_loss.detach()):.4f}"
                    f" tg={teacher_grad_norm:.3e}"
                    f" ema={teacher_ema_update_norm:.3e}"
                )

            message += (
                f" data={data_elapsed:.3f}s"
            )

            print(
                message,
                end="",
            )

        data_started = (
            time.perf_counter()
        )

    if not batches:
        raise RuntimeError(
            "No training batches; check batch_size and max_steps"
        )

    averages = {
        key: (
            value
            / batches
        )
        for (
            key,
            value,
        ) in totals.items()
    }

    # Clean baseline logs only contain meaningful baseline fields.
    if not args.teacher_update:
        keep = {
            "total",
            "main",
            "data_time",
            "step_time",
        }

        averages = {
            key: value
            for (
                key,
                value,
            ) in averages.items()
            if key in keep
        }

    return (
        averages,
        meter.get_scores(),
        last_lr,
        global_step,
    )


# ============================================================================
# Validation
# ============================================================================


@torch.no_grad()
def evaluate(
    loader,
    model,
    criterion,
    weights,
    device,
):
    model.eval()

    meter = ConfuseMatrixMeter(
        n_class=2
    )

    losses = []

    for batch in loader:
        (
            image,
            target,
            _,
            _,
        ) = unpack_batch(
            batch
        )

        pre = (
            image[
                :,
                :3,
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        post = (
            image[
                :,
                3:6,
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        target = (
            target
            .to(
                device,
                non_blocking=True,
            )
            .float()
        )

        predictions = model(
            pre,
            post,
        )

        loss = multiscale_loss(
            predictions,
            target,
            criterion,
            weights,
        )

        meter.update_cm(
            (
                predictions[
                    0
                ]
                > 0.5
            )
            .long()
            .cpu()
            .numpy(),
            target
            .cpu()
            .numpy(),
        )

        losses.append(
            float(
                loss
            )
        )

    return (
        sum(
            losses
        )
        / max(
            len(
                losses
            ),
            1,
        ),
        meter.get_scores(),
    )


# ============================================================================
# Deployment invariance tests
# ============================================================================


@torch.no_grad()
def auxiliary_toggle_consistency(
    model,
    dataset,
    device,
):
    """
    Prove that Run3 auxiliary ON/OFF cannot alter Student main predictions.
    """
    sample = (
        dataset[
            0
        ]
    )

    (
        image,
        target,
        _,
        teacher_pack,
    ) = unpack_batch(
        sample
    )

    image = (
        image
        .unsqueeze(
            0
        )
        .to(
            device,
            non_blocking=True,
        )
    )

    target = (
        target
        .unsqueeze(
            0
        )
        .to(
            device,
            non_blocking=True,
        )
        .float()
    )

    teacher_pack = (
        nested_add_batch_to(
            teacher_pack,
            device,
        )
        if teacher_pack is not None
        else None
    )

    # Make every child deterministic.
    model.eval()

    # Root only enters the training-return branch.
    model.training = True

    without_auxiliary, _ = model(
        image[
            :,
            :3,
        ],
        image[
            :,
            3:6,
        ],
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=False,
    )

    with_auxiliary, _ = model(
        image[
            :,
            :3,
        ],
        image[
            :,
            3:6,
        ],
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )

    model.training = False

    max_error = max(
        (
            left
            - right
        )
        .abs()
        .max()
        .item()
        for (
            left,
            right,
        ) in zip(
            without_auxiliary,
            with_auxiliary,
        )
    )

    if (
        max_error
        != 0.0
    ):
        raise RuntimeError(
            "Auxiliary changed main prediction: "
            f"max_error={max_error:.8e}"
        )

    return max_error


@torch.no_grad()
def deploy_consistency(
    model,
    loader,
    device,
):
    """
    Compare predictions immediately before and after switch_to_deploy().
    """
    model.eval()

    (
        image,
        _,
        _,
        _,
    ) = unpack_batch(
        next(
            iter(
                loader
            )
        )
    )

    pre = (
        image[
            :1,
            :3,
        ]
        .to(
            device
        )
    )

    post = (
        image[
            :1,
            3:6,
        ]
        .to(
            device
        )
    )

    before = model(
        pre,
        post,
    )

    model.switch_to_deploy()
    model.eval()

    after = model(
        pre,
        post,
    )

    max_error = max(
        (
            left
            - right
        )
        .abs()
        .max()
        .item()
        for (
            left,
            right,
        ) in zip(
            before,
            after,
        )
    )

    if (
        max_error
        >= 1e-6
    ):
        raise RuntimeError(
            "Deploy consistency failed: "
            f"max_error={max_error:.8e}"
        )

    return max_error


# ============================================================================
# Deployed test
# ============================================================================


@torch.no_grad()
def test_deployed(
    args,
    loader,
    model,
    device,
):
    model.eval()

    meter = ConfuseMatrixMeter(
        n_class=2
    )

    for batch in loader:
        (
            image,
            target,
            _,
            _,
        ) = unpack_batch(
            batch
        )

        output = model(
            image[
                :,
                :3,
            ].to(
                device,
                non_blocking=True,
            ),
            image[
                :,
                3:6,
            ].to(
                device,
                non_blocking=True,
            ),
        )[0]

        meter.update_cm(
            (
                output
                > 0.5
            )
            .long()
            .cpu()
            .numpy(),
            target
            .cpu()
            .numpy(),
        )

    infer_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    flops = None

    if profile is not None:
        dummy = torch.randn(
            1,
            3,
            args.inHeight,
            args.inWidth,
            device=device,
        )

        flops, _ = profile(
            model,
            inputs=(
                dummy,
                dummy,
            ),
            verbose=False,
        )

    return (
        meter.get_scores(),
        infer_params,
        flops,
    )


# ============================================================================
# Authoritative test block
# ============================================================================


def append_test_results(
    logger,
    args,
    scores,
    train_params,
    trainable_params,
    auxiliary_params,
    infer_params,
    flops,
    auxiliary_error,
    deploy_error,
    elapsed,
):
    """
    Append the single authoritative final test block.

    Formal result extraction must use the final complete block between:
        === TEST RESULTS ===
        === END TEST RESULTS ===
    """
    logger.log_message(
        "=" * 100
    )

    logger.log_message(
        "=== TEST RESULTS ==="
    )

    logger.log_message(
        "Metric Split: test"
    )

    logger.log_message(
        f"Dataset: {args.dataset_name}"
    )

    logger.log_message(
        f"Experiment: {args.experiment}/{args.experiment_name}"
    )

    logger.log_message(
        f"Implementation: {args.implementation_version}"
    )

    logger.log_message(
        f"Seed: {args.seed}"
    )

    logger.log_message(
        f"Max Steps: {args.max_steps}"
    )

    logger.log_message(
        f"Batch Size: {args.batch_size}"
    )

    logger.log_message(
        f"Train Params: {train_params / 1e6:.4f}M"
    )

    logger.log_message(
        f"Trainable Params: {trainable_params / 1e6:.4f}M"
    )

    logger.log_message(
        f"Auxiliary Params: {auxiliary_params / 1e6:.4f}M"
    )

    logger.log_message(
        f"Infer Params: {infer_params / 1e6:.4f}M"
    )

    if flops is not None:
        logger.log_message(
            f"FLOPs: {flops / 1e9:.4f}G"
        )

    else:
        logger.log_message(
            "FLOPs: unavailable"
        )

    logger.log_message(
        "Auxiliary toggle max error: "
        f"{auxiliary_error:.8e}"
    )

    logger.log_message(
        "Deploy max error: "
        f"{deploy_error:.8e}"
    )

    labels = {
        "recall": "Recall",
        "precision": "Precision",
        "OA": "OA",
        "F1": "F1",
        "IoU": "IoU",
        "Kappa": "Kappa",
    }

    for (
        key,
        label,
    ) in labels.items():
        logger.log_message(
            f"{label}: {scores[key]:.6f}"
        )

    logger.log_message(
        f"Total time: {elapsed}"
    )

    logger.log_message(
        "=== END TEST RESULTS ==="
    )


# ============================================================================
# Exact-resume validation
# ============================================================================


def validate_resume(
    args,
    checkpoint,
):
    """
    Run3 exact-resume validation.

    Clean Run3 does not resume historical Run1/Run2 checkpoints.
    """
    version = checkpoint.get(
        "format_version"
    )

    if version != 3:
        raise ValueError(
            "Run3 requires checkpoint format_version=3; "
            "do not resume historical Run1/Run2 checkpoints"
        )

    saved = checkpoint.get(
        "args",
        {},
    )

    keys = [
        # Identity.
        "implementation_version",
        "experiment",
        "experiment_name",
        "dataset_name",
        "mechanism",

        # Protocol.
        "batch_size",
        "max_steps",
        "seed",
        "inWidth",
        "inHeight",

        # Student optimization.
        "lr",
        "lr_mode",
        "step_loss",
        "weight_decay",
        "backbone_lr_mult",
        "dice_reduction",
        "main_loss_weights",
        "kd_lambda",

        # Run3 mechanism.
        "boundary_radius",
        "small_area",
        "teacher_lr",
        "teacher_hidden",
        "teacher_ema",
        "max_logit_delta",
        "teacher_weight_decay",
        "teacher_grad_clip",
        "use_ov",
        "teacher_update",
        "requires_teacher_cache",
        "routing_signal",
        "reject_unit",

        # Data/cache identity.
        "cache_replay",
        "data_fingerprint",
        "cache_fingerprint",
    ]

    for key in keys:
        saved_value = saved.get(
            key
        )

        current_value = getattr(
            args,
            key,
            None,
        )

        if (
            saved_value
            != current_value
        ):
            raise ValueError(
                "Resume configuration mismatch for "
                f"{key}: "
                f"{saved_value!r} vs {current_value!r}"
            )


# ============================================================================
# Dataset fingerprints
# ============================================================================


def _split_fingerprints(
    data_root,
):
    root = Path(
        data_root
    )

    return {
        split: hashlib.sha256(
            (
                root
                / "list"
                / f"{split}.txt"
            ).read_bytes()
        ).hexdigest()
        for split in (
            "train",
            "val",
            "test",
        )
    }


def _read_split_ids(
    data_root,
):
    root = Path(
        data_root
    )

    result = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        path = (
            root
            / "list"
            / f"{split}.txt"
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        ids = [
            value.strip()
            for value in (
                path
                .read_text(
                    encoding="utf-8"
                )
                .splitlines()
            )
            if value.strip()
        ]

        if not ids:
            raise ValueError(
                f"Empty split: {split}"
            )

        if (
            len(ids)
            != len(
                set(
                    ids
                )
            )
        ):
            raise ValueError(
                f"Duplicate IDs in split: {split}"
            )

        result[
            split
        ] = ids

    for (
        left,
        right,
    ) in (
        (
            "train",
            "val",
        ),
        (
            "train",
            "test",
        ),
        (
            "val",
            "test",
        ),
    ):
        overlap = (
            set(
                result[
                    left
                ]
            )
            & set(
                result[
                    right
                ]
            )
        )

        if overlap:
            raise ValueError(
                "Sample IDs overlap between "
                f"{left}/{right}: {len(overlap)}"
            )

    return result


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    # ==================================================================
    # Device
    # ==================================================================

    if (
        args.device
        == "cuda"
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; use --device cpu only for local verification"
            )

        torch.cuda.set_device(
            args.gpu_id
        )

    device = torch.device(
        (
            f"cuda:{args.gpu_id}"
            if args.device == "cuda"
            else "cpu"
        )
    )

    set_seed(
        args.seed
    )

    # ==================================================================
    # Output directory
    # ==================================================================

    save_dir = Path(
        args.save_dir
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================================
    # Model
    # ==================================================================

    model = build_model(
        args
    ).to(
        device
    )

    train_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    auxiliary_params = sum(
        parameter.numel()
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if name.startswith(
            "training_auxiliary."
        )
    )

    fast_teacher_params = sum(
        parameter.numel()
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if name.startswith(
            "training_auxiliary.fast."
        )
    )

    target_teacher_params = sum(
        parameter.numel()
        for (
            name,
            parameter,
        ) in model.named_parameters()
        if name.startswith(
            "training_auxiliary.target."
        )
    )

    # Target teacher must be fully frozen.
    if args.teacher_update:
        target_trainable = [
            name
            for (
                name,
                parameter,
            ) in model.named_parameters()
            if (
                name.startswith(
                    "training_auxiliary.target."
                )
                and parameter.requires_grad
            )
        ]

        if target_trainable:
            raise RuntimeError(
                "EMA Target Teacher contains trainable parameters: "
                + ", ".join(
                    target_trainable
                )
            )

    # ==================================================================
    # Optimizers / criterion
    # ==================================================================

    student_optimizer = (
        build_student_optimizer(
            args,
            model,
        )
    )

    teacher_optimizer = (
        build_teacher_optimizer(
            args,
            model,
        )
    )

    criterion = build_loss(
        args.dice_reduction
    )

    # ==================================================================
    # Data/cache identity
    # ==================================================================

    args.data_fingerprint = (
        _split_fingerprints(
            args.data_root
        )
    )

    split_ids = (
        _read_split_ids(
            args.data_root
        )
    )

    train_list = (
        split_ids[
            "train"
        ]
    )

    if args.requires_teacher_cache:
        teacher_cache = (
            PairedTeacherCache(
                args.sam_cache_root,
                args.ov_cache_root,
                args.dataset_name,
                train_list,
            )
        )

    else:
        teacher_cache = None

    args.cache_fingerprint = (
        teacher_cache.fingerprint
        if teacher_cache is not None
        else None
    )

    # ==================================================================
    # DataLoaders
    # ==================================================================

    train_loader = get_loader(
        args.data_root,
        os.path.join(
            args.data_root,
            "list",
            "train.txt",
        ),
        batchsize=args.batch_size,
        trainsize=args.inWidth,
        num_workers=args.num_workers,
        teacher_cache=teacher_cache,
        seed=args.seed,
        cache_replay=args.cache_replay,
    )

    if (
        len(train_loader)
        == 0
    ):
        raise ValueError(
            "Training dataset is smaller than batch_size "
            "with drop_last=True"
        )

    val_loader = get_test_loader(
        args.data_root,
        os.path.join(
            args.data_root,
            "list",
            "val.txt",
        ),
        batchsize=args.batch_size,
        testsize=args.inWidth,
        num_workers=args.num_workers,
        return_meta=True,
    )

    test_loader = get_test_loader(
        args.data_root,
        os.path.join(
            args.data_root,
            "list",
            "test.txt",
        ),
        batchsize=args.batch_size,
        testsize=args.inWidth,
        num_workers=args.num_workers,
        return_meta=True,
    )

    args.max_epochs = int(
        np.ceil(
            args.max_steps
            / len(
                train_loader
            )
        )
    )

    # ==================================================================
    # Resume state
    # ==================================================================

    start_epoch = 0
    global_step = 0
    best_val_f1 = -1.0

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        validate_resume(
            args,
            checkpoint,
        )

        model.load_state_dict(
            checkpoint[
                "model"
            ]
        )

        student_optimizer.load_state_dict(
            checkpoint[
                "optimizer"
            ]
        )

        if (
            teacher_optimizer
            is not None
        ):
            teacher_state = checkpoint.get(
                "teacher_optimizer"
            )

            if teacher_state is None:
                raise ValueError(
                    "Run3 checkpoint is missing teacher_optimizer state"
                )

            teacher_optimizer.load_state_dict(
                teacher_state
            )

        elif (
            checkpoint.get(
                "teacher_optimizer"
            )
            is not None
        ):
            raise ValueError(
                "B0 checkpoint unexpectedly contains Teacher optimizer state"
            )

        start_epoch = (
            checkpoint[
                "epoch"
            ]
            + 1
        )

        global_step = int(
            checkpoint[
                "global_step"
            ]
        )

        best_val_f1 = float(
            checkpoint[
                "best_val_f1"
            ]
        )

        restore_rng_state(
            checkpoint.get(
                "rng"
            )
        )

    # ==================================================================
    # Log path / overwrite protection
    # ==================================================================

    log_path = Path(
        args.log_file
    )

    if not log_path.is_absolute():
        log_path = (
            save_dir
            / log_path
        )

    last_path = (
        save_dir
        / "last_checkpoint.pth"
    )

    if (
        not args.resume
        and (
            log_path.exists()
            or last_path.exists()
        )
    ):
        raise FileExistsError(
            "Existing run found; use --resume or a fresh save/log directory"
        )

    # ==================================================================
    # Logger
    # ==================================================================

    logger = TrainingLogger(
        str(
            log_path
        ),
        {
            **vars(
                args
            ),

            "train_params": (
                f"{train_params / 1e6:.4f}M"
            ),

            "trainable_params": (
                f"{trainable_params / 1e6:.4f}M"
            ),

            "auxiliary_params": (
                f"{auxiliary_params / 1e6:.6f}M"
            ),

            "fast_teacher_params": (
                f"{fast_teacher_params / 1e6:.6f}M"
            ),

            "target_teacher_params": (
                f"{target_teacher_params / 1e6:.6f}M"
            ),

            "n_train": (
                len(
                    train_loader.dataset
                )
            ),

            "n_val": (
                len(
                    val_loader.dataset
                )
            ),

            "n_test": (
                len(
                    test_loader.dataset
                )
            ),
        },
        append=bool(
            args.resume
        ),
    )

    # ==================================================================
    # Best-checkpoint path
    # ==================================================================

    best_path = None

    if args.resume:
        candidate = (
            save_dir
            / (
                "best_model_F1="
                f"{best_val_f1:.6f}.pth"
            )
        )

        if candidate.is_file():
            best_path = (
                candidate
            )

    started = (
        datetime.datetime.now()
    )

    # ==================================================================
    # Train
    # ==================================================================

    for epoch in range(
        start_epoch,
        args.max_epochs,
    ):
        if (
            global_step
            >= args.max_steps
        ):
            break

        (
            train_losses,
            _,
            current_lr,
            global_step,
        ) = train_epoch(
            args,
            train_loader,
            model,
            criterion,
            student_optimizer,
            epoch,
            global_step,
            device,
            teacher_optimizer=(
                teacher_optimizer
            ),
        )

        # ==============================================================
        # Validation
        # ==============================================================

        (
            val_loss,
            val_scores,
        ) = evaluate(
            val_loader,
            model,
            criterion,
            args.main_loss_weights,
            device,
        )

        is_best = (
            val_scores[
                "F1"
            ]
            > best_val_f1
        )

        if is_best:
            best_val_f1 = (
                val_scores[
                    "F1"
                ]
            )

        # ==============================================================
        # Checkpoint
        # ==============================================================

        checkpoint = build_checkpoint(
            model,
            student_optimizer,
            epoch,
            global_step,
            best_val_f1,
            args,
            teacher_optimizer=(
                teacher_optimizer
            ),
        )

        save_checkpoint_atomic(
            checkpoint,
            last_path,
        )

        if is_best:
            new_best = (
                save_dir
                / (
                    "best_model_F1="
                    f"{best_val_f1:.6f}.pth"
                )
            )

            save_checkpoint_atomic(
                checkpoint,
                new_best,
            )

            if (
                best_path is not None
                and best_path
                != new_best
                and best_path.name.startswith(
                    "best_model_F1="
                )
            ):
                best_path.unlink(
                    missing_ok=True
                )

            best_path = (
                new_best
            )

        # ==============================================================
        # Epoch log
        # ==============================================================

        logger.log_epoch(
            epoch,
            args.max_epochs,
            train_losses,
            {
                "f1": (
                    val_scores[
                        "F1"
                    ]
                ),
                "iou": (
                    val_scores[
                        "IoU"
                    ]
                ),
                "kappa": (
                    val_scores[
                        "Kappa"
                    ]
                ),
                "recall": (
                    val_scores[
                        "recall"
                    ]
                ),
                "precision": (
                    val_scores[
                        "precision"
                    ]
                ),
                "oa": (
                    val_scores[
                        "OA"
                    ]
                ),
            },
            current_lr,
            (
                torch.cuda.max_memory_allocated(
                    device
                )
                / 1e9
                if device.type
                == "cuda"
                else 0.0
            ),
            is_best,
        )

        logger.log_message(
            "Val loss: "
            f"{val_loss:.6f}; "
            f"global_step: {global_step}"
        )

        if (
            global_step
            >= args.max_steps
        ):
            break

    # ==================================================================
    # Load validation-selected best checkpoint
    # ==================================================================

    if (
        best_path is None
        or not best_path.is_file()
    ):
        raise RuntimeError(
            "Training finished without a validation-selected best checkpoint"
        )

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    # ==================================================================
    # Deployment correctness
    # ==================================================================

    # 1. Training auxiliary ON/OFF cannot change main predictions.
    auxiliary_error = (
        auxiliary_toggle_consistency(
            model,
            train_loader.dataset,
            device,
        )
    )

    # 2. Physically remove complete training auxiliary.
    deploy_error = (
        deploy_consistency(
            model,
            test_loader,
            device,
        )
    )

    # 3. Test only the deployed Student.
    (
        test_scores,
        infer_params,
        flops,
    ) = test_deployed(
        args,
        test_loader,
        model,
        device,
    )

    # ==================================================================
    # Hard deployment contracts
    # ==================================================================

    if (
        infer_params
        != EXPECTED_DEPLOY_PARAMS
    ):
        raise RuntimeError(
            "Unexpected deploy parameter count: "
            f"{infer_params:,}; "
            f"expected {EXPECTED_DEPLOY_PARAMS:,}"
        )

    if (
        flops is not None
        and abs(
            flops
            - EXPECTED_DEPLOY_FLOPS
        )
        > DEPLOY_FLOPS_ATOL
    ):
        raise RuntimeError(
            "Unexpected deploy FLOPs: "
            f"{flops / 1e9:.6f}G; "
            f"expected approximately "
            f"{EXPECTED_DEPLOY_FLOPS / 1e9:.6f}G"
        )

    # ==================================================================
    # Authoritative final result
    # ==================================================================

    append_test_results(
        logger,
        args,
        test_scores,
        train_params,
        trainable_params,
        auxiliary_params,
        infer_params,
        flops,
        auxiliary_error,
        deploy_error,
        (
            datetime.datetime.now()
            - started
        ),
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