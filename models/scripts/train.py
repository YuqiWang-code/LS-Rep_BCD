"""Formal trainer for A2Net-LWGANet-L0 with RDT-CD + SCGR.

Active experiments
------------------
B0
    Clean fully supervised A2Net-LWGANet-L0 baseline.

D1
    Full reciprocal dynamic teacher (RDT-CD) with SAMStruct + OVCDistill
    cache priors and SCGR routing.

D2
    Capacity-matched no-cache control.  The same online/EMA teacher experts
    are trained, but cache conditioning is disabled.  SCGR remains unchanged.

SCGR = Sparse-Change Gradient-Concordant Routing.

The deploy graph is never changed by RDT-CD/SCGR.  Every teacher/router/cache
component is training-only and must be physically removed by switch_to_deploy().
"""
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
from typing import Dict, Iterable, Optional, Tuple

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
from models.datasets.cd_dataset import get_loader, get_test_loader
from models.distill import PairedTeacherCache
from models.utils.checkpoint import (
    build_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
)
from models.utils.logger import TrainingLogger
from models.utils.metrics import ConfuseMatrixMeter
from models.utils.scheduler import adjust_learning_rate


EXPECTED_DEPLOY_PARAMS = 2_913_094
EXPECTED_DEPLOY_FLOPS = 2.7475e9
DEPLOY_FLOPS_ATOL = 0.03e9
IMPLEMENTATION_VERSION = "rdt_scgr_v1"
SCGR_POLICY = "scgr"
SCGR_TEACHER = "both"

EXPERIMENTS = {
    "B0": {
        "name": "Baseline_A2Net_LWGANet_L0",
        "auxiliary_mode": "none",
        "mechanism": "none",
        "cache_conditioning": False,
    },
    "D1": {
        "name": "D1_RDT_SCGR_Full",
        "auxiliary_mode": "direction_c",
        "mechanism": "dynamic_teacher",
        "cache_conditioning": True,
    },
    "D2": {
        "name": "D2_RDT_SCGR_NoCache",
        "auxiliary_mode": "direction_c",
        "mechanism": "dynamic_teacher",
        "cache_conditioning": False,
    },
}
DYNAMIC_EXPERIMENTS = {"D1", "D2"}


# Pair-valued diagnostics emitted as [B,2] or [2].
PAIR_DETAIL_KEYS = {
    "q": "q_",
    "weights": "w_",
    "effective_weights": "effective_",
    "proposal_brier": "proposal_brier_",
    "dynamic_teacher_brier": "dynamic_teacher_brier_",
    "fast_teacher_brier": "fast_teacher_brier_",
    "static_teacher_brier": "static_teacher_brier_",
    "dynamic_teacher_gain": "dynamic_teacher_gain_",
    "static_teacher_gain": "static_teacher_gain_",
    "static_shift": "static_shift_",
    "target_residual_magnitude": "target_residual_",
    "fast_residual_magnitude": "fast_residual_",
    "cache_support_ratio": "cache_support_",
    # SCGR region diagnostics.  The dynamic teacher should summarize region
    # tensors to a teacher pair before exposing them to this trainer.
    "scgr_region_utility": "scgr_region_utility_",
    "scgr_region_concordance": "scgr_region_concordance_",
    "scgr_region_quality": "scgr_region_quality_",
    "scgr_region_score": "scgr_region_score_",
    "scgr_teacher_mass": "scgr_teacher_mass_",
}

SCALAR_DETAIL_KEYS = (
    # Existing RDT diagnostics retained for direct before/after comparison.
    "pixel_reject_ratio",
    "image_reject_ratio",
    "accepted_change_ratio",
    "accepted_bg_ratio",
    "effective_mass",
    "student_brier",
    "sam_transport",
    "ov_task",
    "dynamic_shift_sam",
    "dynamic_shift_ov",
    "teacher_fit_sam",
    "teacher_fit_ov",
    "student_abs_error",
    "student_error_mass",
    "effective_per_error_mass",
    "cache_conditioning",
    # SCGR-specific observables.
    "scgr_region_accept_ratio",
    "scgr_region_reject_ratio",
    "scgr_change_accept_ratio",
    "scgr_background_accept_ratio",
    "scgr_positive_utility_ratio",
    "scgr_positive_concordance_ratio",
    "scgr_gradient_conflict_ratio",
    "scgr_negative_cosine_reject_ratio",
    "scgr_change_error_mass",
    "scgr_background_error_mass",
    "scgr_effective_error_mass",
    "scgr_effective_per_error_mass",
)

# These are required for a dynamic-teacher forward.  Failing loudly here is
# preferable to silently training the old router while calling the run SCGR.
REQUIRED_SCGR_DETAIL_KEYS = (
    "total",
    "teacher_total",
    "scgr_region_accept_ratio",
    "scgr_positive_utility_ratio",
    "scgr_positive_concordance_ratio",
    "scgr_gradient_conflict_ratio",
    "scgr_effective_per_error_mass",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="A2Net baseline / RDT-CD + SCGR trainer"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(EXPERIMENTS),
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=["CDD", "LEVIR", "SYSU", "WHU"],
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sam_cache_root", default=None)
    parser.add_argument("--ov_cache_root", default=None)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2333)

    # Shared difficulty construction used by the reciprocal dynamic teacher.
    parser.add_argument("--boundary_radius", type=int, default=2)
    parser.add_argument("--small_area", type=int, default=64)

    # SCGR is parameter-free.  region_size defines non-overlapping full-res
    # routing regions in task_space.py.  16 means a 16x16 pixel region on a
    # 256x256 training patch.
    parser.add_argument("--scgr_region_size", type=int, default=16)

    # RDT-CD online teacher controls.  Their optimization schedule is preserved
    # from the current RDT implementation so routing is the only method change.
    parser.add_argument("--teacher_lr", type=float, default=5e-4)
    parser.add_argument("--teacher_hidden", type=int, default=24)
    parser.add_argument("--teacher_ema", type=float, default=0.99)
    parser.add_argument("--max_logit_delta", type=float, default=2.0)
    parser.add_argument("--teacher_weight_decay", type=float, default=1e-4)
    parser.add_argument("--teacher_grad_clip", type=float, default=5.0)

    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pretrained_path", default=None)

    # Formal protocol.
    parser.add_argument("--inWidth", type=int, default=256)
    parser.add_argument("--inHeight", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr_mode",
        default="poly",
        choices=["poly", "step"],
    )
    parser.add_argument("--step_loss", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=1.0)
    parser.add_argument(
        "--dice_reduction",
        default="batch",
        choices=["sample", "batch"],
    )
    parser.add_argument("--main_loss_weights", default="1,1,1,1")
    parser.add_argument("--kd_lambda", type=float, default=0.06)

    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_file", default="train_log.txt")
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()
    args.main_loss_weights = tuple(
        float(value) for value in args.main_loss_weights.split(",")
    )
    if len(args.main_loss_weights) != 4:
        raise ValueError(
            "main_loss_weights must contain four comma-separated values"
        )

    if args.batch_size <= 0 or args.max_steps <= 0 or args.num_workers < 0:
        raise ValueError(
            "batch_size/max_steps must be positive and num_workers non-negative"
        )
    if args.inWidth != 256 or args.inHeight != 256:
        raise ValueError("Formal trainer preserves the existing 256x256 protocol")
    if args.inWidth != args.inHeight:
        raise ValueError("Only square 256x256 inputs are supported")
    if args.lr <= 0 or args.teacher_lr <= 0:
        raise ValueError("Student and teacher learning rates must be positive")
    if args.weight_decay < 0 or args.teacher_weight_decay < 0:
        raise ValueError("Weight decay must be nonnegative")
    if args.teacher_grad_clip < 0:
        raise ValueError("teacher_grad_clip must be nonnegative")
    if args.kd_lambda < 0:
        raise ValueError("kd_lambda must be nonnegative")
    if args.boundary_radius < 1 or args.small_area < 1:
        raise ValueError("boundary_radius/small_area must be positive")
    if args.scgr_region_size < 1:
        raise ValueError("scgr_region_size must be positive")
    if args.teacher_hidden < 1:
        raise ValueError("teacher_hidden must be positive")
    if not 0.0 <= args.teacher_ema < 1.0:
        raise ValueError("teacher_ema must satisfy 0 <= teacher_ema < 1")
    if args.max_logit_delta <= 0:
        raise ValueError("max_logit_delta must be positive")
    if any(
        (not math.isfinite(value)) or value < 0
        for value in args.main_loss_weights
    ):
        raise ValueError("Invalid main loss weights")
    if args.pretrained and not args.pretrained_path:
        raise ValueError(
            "--pretrained_path is required unless --no-pretrained is given"
        )

    recipe = EXPERIMENTS[args.experiment]
    args.experiment_name = recipe["name"]
    args.auxiliary_mode = recipe["auxiliary_mode"]
    args.mechanism = recipe["mechanism"]
    args.cache_conditioning = bool(recipe["cache_conditioning"])

    args.cache_replay = "aligned"
    args.teacher_update = args.mechanism == "dynamic_teacher"
    args.implementation_version = IMPLEMENTATION_VERSION
    args.routing_signal = SCGR_POLICY if args.teacher_update else "none"
    args.reject_unit = "region" if args.teacher_update else "none"
    args.requires_teacher_cache = (
        args.teacher_update and args.cache_conditioning
    )

    if args.requires_teacher_cache and not (
        args.sam_cache_root and args.ov_cache_root
    ):
        raise ValueError(
            f"{args.experiment} requires --sam_cache_root AND --ov_cache_root"
        )

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def nested_to(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: nested_to(item, device) for key, item in value.items()}
    return value


def nested_add_batch_to(value, device):
    """Add a batch dimension to one Dataset sample and move it to device."""
    if torch.is_tensor(value):
        return value.unsqueeze(0).to(device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: nested_add_batch_to(item, device)
            for key, item in value.items()
        }
    return value


def unpack_batch(batch):
    if len(batch) == 2:
        return batch[0], batch[1], None, None
    if len(batch) == 3:
        return batch[0], batch[1], batch[2], None
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    raise ValueError(f"Unexpected batch structure with {len(batch)} fields")


def multiscale_loss(predictions, target, criterion, weights):
    if len(predictions) != len(weights):
        raise ValueError(
            f"Prediction/weight count mismatch: {len(predictions)} vs {len(weights)}"
        )
    return sum(
        weight * criterion(prediction, target)
        for weight, prediction in zip(weights, predictions)
    )


def build_optimizer(args, model):
    """Build the STUDENT optimizer only.

    Every training_auxiliary parameter is excluded.  Fast teacher parameters
    have their own optimizer; EMA target teachers are never optimized.
    """
    grouped: Dict[float, list] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("training_auxiliary."):
            continue

        scale = (
            args.backbone_lr_mult
            if name.startswith("backbone.")
            else 1.0
        )
        grouped.setdefault(scale, []).append(parameter)

    if not grouped:
        raise RuntimeError("Student optimizer received no parameters")

    groups = [
        {
            "params": parameters,
            "lr": args.lr * scale,
            "lr_scale": scale,
            "weight_decay": args.weight_decay,
        }
        for scale, parameters in grouped.items()
    ]
    return torch.optim.Adam(
        groups,
        lr=args.lr,
        betas=(0.9, 0.99),
        eps=1e-8,
    )


def build_teacher_optimizer(args, model):
    """Build the separate FAST-teacher optimizer for D1/D2."""
    if args.mechanism != "dynamic_teacher":
        return None
    if not model.use_training_auxiliary:
        raise RuntimeError(
            "dynamic_teacher selected but model has no training auxiliary"
        )
    if not hasattr(model.training_auxiliary, "teacher_parameters"):
        raise RuntimeError(
            "dynamic_teacher auxiliary does not expose teacher_parameters()"
        )

    parameters = list(model.training_auxiliary.teacher_parameters())
    if not parameters:
        raise RuntimeError("Dynamic teacher optimizer received no parameters")
    if any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("Fast teacher parameters must all require gradients")

    return torch.optim.Adam(
        [
            {
                "params": parameters,
                "lr": args.teacher_lr,
                "lr_scale": 1.0,
                "weight_decay": args.teacher_weight_decay,
            }
        ],
        lr=args.teacher_lr,
        betas=(0.9, 0.99),
        eps=1e-8,
    )


def adjust_teacher_learning_rate(args, teacher_optimizer, student_base_lr):
    """Preserve the current RDT teacher schedule for a clean SCGR ablation.

    The teacher keeps its own base LR but follows the same warmup/decay factor
    as the student.  Teacher-plasticity changes are intentionally NOT mixed
    into the first SCGR experiment.
    """
    if teacher_optimizer is None:
        return 0.0
    if args.lr <= 0:
        raise ValueError("Student base LR must be positive")

    factor = float(student_base_lr) / float(args.lr)
    teacher_lr = float(args.teacher_lr) * factor
    for group in teacher_optimizer.param_groups:
        group["lr"] = teacher_lr * group.get("lr_scale", 1.0)
    return teacher_lr


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Return global L2 gradient norm without modifying gradients."""
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    if squared is None:
        return 0.0
    return float(torch.sqrt(squared).item())


def build_model(args):
    if args.mechanism == "dynamic_teacher":
        cfg = {
            "mechanism": "dynamic_teacher",
            "hidden": args.teacher_hidden,
            "ema": args.teacher_ema,
            "max_logit_delta": args.max_logit_delta,
            "boundary_radius": args.boundary_radius,
            "small_area": args.small_area,
            "policy": SCGR_POLICY,
            "teacher": SCGR_TEACHER,
            "difficulty": True,
            "cache_conditioning": args.cache_conditioning,
            "region_size": args.scgr_region_size,
        }
        return A2Net_LWGANet_L0(
            pretrained=args.pretrained,
            pretrained_path=args.pretrained_path,
            auxiliary_mode=args.auxiliary_mode,
            routing_cfg=cfg,
        )

    return A2Net_LWGANet_L0(
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        auxiliary_mode=args.auxiliary_mode,
    )


def _training_metric_keys() -> Tuple[str, ...]:
    keys = [
        "total",
        "main",
        "aux_raw",
        "aux_weighted",
        "aux_ratio",
        "teacher_total",
        "teacher_lr",
        "teacher_grad_norm",
        "teacher_ema_update_norm",
        "teacher_target_gap",
        "probe_cls_kd_gt_ratio",
        "probe_cls_kd_gt_cosine",
        "data_time",
        "step_time",
    ]

    for prefix in PAIR_DETAIL_KEYS.values():
        for name in ("sam", "ov"):
            keys.append(prefix + name)

    # Explicit reject mass is useful beside the two teacher masses.
    keys.append("w_reject")
    keys.extend(SCALAR_DETAIL_KEYS)
    return tuple(dict.fromkeys(keys))


def _mean_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.float().mean()


def _set_pair_metrics(values, detail, key, prefix):
    if key not in detail:
        return
    tensor = detail[key]
    if not torch.is_tensor(tensor) or tensor.ndim == 0:
        return

    # Accept [2], [B,2], or higher-dimensional tensors whose teacher dimension
    # is the last dimension.  DynamicTeacher will expose SCGR summaries in this
    # layout to keep the trainer independent of region-grid geometry.
    if tensor.shape[-1] != 2:
        return
    for index, name in enumerate(("sam", "ov")):
        values[prefix + name] = tensor[..., index].float().mean()


def _validate_scgr_detail(args, detail):
    if args.mechanism != "dynamic_teacher":
        return
    missing = [key for key in REQUIRED_SCGR_DETAIL_KEYS if key not in detail]
    if missing:
        raise RuntimeError(
            "Dynamic teacher did not expose required SCGR outputs: "
            + ", ".join(missing)
            + ". Replace dynamic_teacher.py with the SCGR-compatible version."
        )


def _populate_detail_metrics(values, detail):
    for key, prefix in PAIR_DETAIL_KEYS.items():
        _set_pair_metrics(values, detail, key, prefix)

    if "weights" in detail:
        weights = detail["weights"]
        if torch.is_tensor(weights) and weights.ndim > 0 and weights.shape[-1] == 3:
            values["w_reject"] = weights[..., 2].float().mean()

    for name in SCALAR_DETAIL_KEYS:
        if name in detail and torch.is_tensor(detail[name]):
            values[name] = detail[name].float().mean()


def _exact_classifier_probe(main_loss, kd_loss, classifier_weight):
    """Exact diagnostic: KD gradient vs supervised gradient at decoder.cls.

    This is diagnostic only.  SCGR itself uses the low-cost analytical regional
    proxy implemented in task_space.py.  Keeping this exact probe lets us test
    the falsifiable claim that SCGR reduces real classifier-gradient conflict.
    """
    main_gradient = torch.autograd.grad(
        main_loss,
        classifier_weight,
        retain_graph=True,
        allow_unused=False,
    )[0].detach().float()

    kd_gradient = torch.autograd.grad(
        kd_loss,
        classifier_weight,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if kd_gradient is None:
        return 0.0, 0.0

    kd_gradient = kd_gradient.detach().float()
    main_norm = main_gradient.norm().clamp_min(1e-12)
    kd_norm = kd_gradient.norm()

    ratio = float(kd_norm / main_norm)
    if float(kd_norm) <= 1e-12:
        cosine = 0.0
    else:
        cosine = float(
            (main_gradient * kd_gradient).sum()
            / (main_norm * kd_norm.clamp_min(1e-12))
        )
    return ratio, cosine


def train_epoch(
    args,
    loader,
    model,
    criterion,
    optimizer,
    epoch,
    global_step,
    device,
    teacher_optimizer=None,
):
    model.train()
    loader.dataset.set_epoch(epoch)
    loader.generator.manual_seed(args.seed + epoch)

    meter = ConfuseMatrixMeter(n_class=2)
    keys = _training_metric_keys()
    totals = dict.fromkeys(keys, 0.0)

    batches = 0
    last_lr = args.lr
    last_teacher_lr = 0.0
    data_started = time.perf_counter()

    # Exact probe is sampled on the first batch of every epoch.  This is the
    # same low-frequency diagnostic spirit as the previous trainer and avoids
    # paying two autograd.grad calls on every optimization step.
    epoch_probe_ratio = 0.0
    epoch_probe_cosine = 0.0

    for batch in loader:
        data_elapsed = time.perf_counter() - data_started
        step_started = time.perf_counter()

        if global_step >= args.max_steps:
            break

        image, target, _, pack = unpack_batch(batch)
        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        pack = nested_to(pack, device) if pack is not None else None

        last_lr = adjust_learning_rate(
            args,
            optimizer,
            epoch,
            global_step,
            len(loader),
        )
        last_teacher_lr = adjust_teacher_learning_rate(
            args,
            teacher_optimizer,
            last_lr,
        )

        optimizer.zero_grad(set_to_none=True)
        if teacher_optimizer is not None:
            teacher_optimizer.zero_grad(set_to_none=True)

        # One student forward.  a2net.py computes the unchanged deploy path
        # first and then invokes the loss-only training auxiliary.
        predictions, auxiliary = model(
            pre,
            post,
            target=target,
            teacher_pack=pack,
            compute_auxiliary=(args.auxiliary_mode != "none"),
        )

        main_loss = multiscale_loss(
            predictions,
            target,
            criterion,
            args.main_loss_weights,
        )
        zero = main_loss.new_zeros(())
        detail = auxiliary.get("direction_c", {})

        if detail:
            _validate_scgr_detail(args, detail)

        aux_raw = detail.get("total", zero)
        aux_weighted = args.kd_lambda * aux_raw
        teacher_loss = detail.get("teacher_total", zero)
        student_loss = main_loss + aux_weighted

        if batches == 0 and detail:
            (
                epoch_probe_ratio,
                epoch_probe_cosine,
            ) = _exact_classifier_probe(
                main_loss,
                aux_weighted,
                model.decoder.cls.weight,
            )

        finite_values = [student_loss]
        if teacher_optimizer is not None:
            finite_values.append(teacher_loss)
        if not all(bool(torch.isfinite(value)) for value in finite_values):
            raise FloatingPointError(
                "Non-finite training loss; checkpoint not overwritten"
            )

        # --------------------------------------------------------------
        # A. Teacher -> Student.
        # Target-teacher proposals and SCGR route evidence are detached, so
        # this backward updates the student only.
        # --------------------------------------------------------------
        student_loss.backward()
        optimizer.step()

        # --------------------------------------------------------------
        # B. Student -> Fast Teacher.
        # Student/cache inputs are detached inside dynamic_teacher.py.
        # --------------------------------------------------------------
        teacher_grad_norm = 0.0
        teacher_ema_update_norm = 0.0
        teacher_target_gap = 0.0
        if teacher_optimizer is not None:
            if not teacher_loss.requires_grad:
                raise RuntimeError("Dynamic teacher loss has no gradient graph")

            teacher_loss.backward()
            teacher_parameters = list(
                model.training_auxiliary.teacher_parameters()
            )
            teacher_grad_norm = gradient_norm(teacher_parameters)
            if not math.isfinite(teacher_grad_norm):
                raise FloatingPointError(
                    "Non-finite dynamic-teacher gradient norm"
                )

            if args.teacher_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    teacher_parameters,
                    max_norm=args.teacher_grad_clip,
                )
            teacher_optimizer.step()

            # ----------------------------------------------------------
            # C. Fast Teacher -> EMA Target Teacher.
            # Current-step GT-fitting is not exposed to the same student step.
            # ----------------------------------------------------------
            teacher_ema_update_norm = float(
                model.training_auxiliary.update_ema()
            )
            teacher_target_gap = float(
                model.training_auxiliary.teacher_target_gap()
            )

        prediction = (predictions[0].detach() > 0.5).long()
        current_f1 = meter.update_cm(
            prediction.cpu().numpy(),
            target.cpu().numpy(),
        )
        aux_ratio = (
            aux_weighted.detach()
            / main_loss.detach().clamp_min(1e-8)
        )

        values = {key: zero for key in keys}
        values.update(
            total=student_loss,
            main=main_loss,
            aux_raw=aux_raw,
            aux_weighted=aux_weighted,
            aux_ratio=aux_ratio,
            teacher_total=teacher_loss,
            teacher_lr=zero.new_tensor(last_teacher_lr),
            teacher_grad_norm=zero.new_tensor(teacher_grad_norm),
            teacher_ema_update_norm=zero.new_tensor(
                teacher_ema_update_norm
            ),
            teacher_target_gap=zero.new_tensor(teacher_target_gap),
            probe_cls_kd_gt_ratio=zero.new_tensor(epoch_probe_ratio),
            probe_cls_kd_gt_cosine=zero.new_tensor(epoch_probe_cosine),
            data_time=zero.new_tensor(data_elapsed),
            step_time=zero.new_tensor(time.perf_counter() - step_started),
        )
        if detail:
            _populate_detail_metrics(values, detail)

        for key, value in values.items():
            if torch.is_tensor(value):
                scalar = float(value.detach().float().mean())
            else:
                scalar = float(value)
            if not math.isfinite(scalar):
                raise FloatingPointError(f"Non-finite training metric: {key}")
            totals[key] += scalar

        batches += 1
        global_step += 1

        if global_step % 5 == 0:
            message = (
                f"\rstep [{global_step}/{args.max_steps}] "
                f"F1={current_f1:.3f} "
                f"lr={last_lr:.7f} "
                f"loss={float(student_loss.detach()):.3f} "
                f"aux_ratio={float(aux_ratio):.3f}"
            )
            if detail:
                message += (
                    " scgr_acc="
                    f"{float(values['scgr_region_accept_ratio']):.3f}"
                    " scgr_cos+="
                    f"{float(values['scgr_positive_concordance_ratio']):.3f}"
                )
            if teacher_optimizer is not None:
                message += (
                    f" teacher={float(teacher_loss.detach()):.3f}"
                    f" tg={teacher_grad_norm:.3e}"
                    f" ema={teacher_ema_update_norm:.3e}"
                )
            message += f" data={data_elapsed:.3f}s"
            print(message, end="")

        data_started = time.perf_counter()

    if not batches:
        raise RuntimeError("No training batches; check batch size and max_steps")

    print()
    averages = {
        key: value / batches
        for key, value in totals.items()
    }

    # Keep clean B0 logs concise instead of fabricating zero teacher/SCGR
    # statistics that were never measured.
    if args.mechanism != "dynamic_teacher":
        auxiliary_keys = {
            "aux_raw",
            "aux_weighted",
            "aux_ratio",
            "teacher_total",
            "teacher_lr",
            "teacher_grad_norm",
            "teacher_ema_update_norm",
            "teacher_target_gap",
            "probe_cls_kd_gt_ratio",
            "probe_cls_kd_gt_cosine",
            "w_reject",
            *PAIR_DETAIL_KEYS.values(),
            *SCALAR_DETAIL_KEYS,
        }
        expanded = set()
        for item in auxiliary_keys:
            if isinstance(item, str) and item.endswith("_"):
                expanded.add(item + "sam")
                expanded.add(item + "ov")
            else:
                expanded.add(item)
        averages = {
            key: value
            for key, value in averages.items()
            if key not in expanded
        }

    return averages, meter.get_scores(), last_lr, global_step


@torch.no_grad()
def evaluate(loader, model, criterion, weights, device):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []

    for batch in loader:
        image, target, _, _ = unpack_batch(batch)
        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        predictions = model(pre, post)
        loss = multiscale_loss(predictions, target, criterion, weights)
        meter.update_cm(
            (predictions[0] > 0.5).long().cpu().numpy(),
            target.cpu().numpy(),
        )
        losses.append(float(loss))

    return (
        sum(losses) / max(len(losses), 1),
        meter.get_scores(),
    )


@torch.no_grad()
def auxiliary_toggle_consistency(model, dataset, device):
    """Prove that the training-only auxiliary cannot alter main prediction."""
    sample = dataset[0]
    image, target, _, teacher_pack = unpack_batch(sample)
    image = image.unsqueeze(0).to(device, non_blocking=True)
    target = target.unsqueeze(0).to(device, non_blocking=True).float()
    teacher_pack = (
        nested_add_batch_to(teacher_pack, device)
        if teacher_pack is not None
        else None
    )

    # Child modules stay deterministic in eval mode.  Only the root training
    # flag is toggled so A2Net returns (predictions, auxiliary).
    model.eval()
    model.training = True
    without_auxiliary, _ = model(
        image[:, :3],
        image[:, 3:6],
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=False,
    )
    with_auxiliary, _ = model(
        image[:, :3],
        image[:, 3:6],
        target=target,
        teacher_pack=teacher_pack,
        compute_auxiliary=True,
    )
    model.training = False

    max_error = max(
        (left - right).abs().max().item()
        for left, right in zip(without_auxiliary, with_auxiliary)
    )
    if max_error != 0.0:
        raise RuntimeError(
            "Auxiliary changed the main output: "
            f"max_error={max_error:.8e}"
        )
    return max_error


@torch.no_grad()
def deploy_consistency(model, loader, device):
    model.eval()
    image, _, _, _ = unpack_batch(next(iter(loader)))
    pre = image[:1, :3].to(device)
    post = image[:1, 3:6].to(device)

    before = model(pre, post)
    model.switch_to_deploy().eval()
    after = model(pre, post)

    max_error = max(
        (left - right).abs().max().item()
        for left, right in zip(before, after)
    )
    if max_error >= 1e-6:
        raise RuntimeError(
            "Deploy consistency failed: "
            f"max_error={max_error:.8e}"
        )
    return max_error


@torch.no_grad()
def test_deployed(args, loader, model, device):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)

    for batch in loader:
        image, target, _, _ = unpack_batch(batch)
        output = model(
            image[:, :3].to(device, non_blocking=True),
            image[:, 3:6].to(device, non_blocking=True),
        )[0]
        meter.update_cm(
            (output > 0.5).long().cpu().numpy(),
            target.cpu().numpy(),
        )

    infer_params = sum(parameter.numel() for parameter in model.parameters())
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
            inputs=(dummy, dummy),
            verbose=False,
        )

    return meter.get_scores(), infer_params, flops


def append_test_results(
    logger,
    args,
    scores,
    train_params,
    infer_params,
    flops,
    auxiliary_error,
    deploy_error,
    elapsed,
):
    """Append the single authoritative final test block to train_log.txt."""
    logger.log_message("=" * 100)
    logger.log_message("=== TEST RESULTS ===")
    logger.log_message("Metric Split: test")
    logger.log_message(f"Dataset: {args.dataset_name}")
    logger.log_message(
        f"Experiment: {args.experiment}/{args.experiment_name}"
    )
    logger.log_message(f"Train Params: {train_params / 1e6:.4f}M")
    logger.log_message(f"Infer Params: {infer_params / 1e6:.4f}M")
    if flops is not None:
        logger.log_message(f"FLOPs: {flops / 1e9:.4f}G")
    else:
        logger.log_message("FLOPs: unavailable")

    logger.log_message(
        "Auxiliary toggle max error: "
        f"{auxiliary_error:.8e}"
    )
    logger.log_message(f"Deploy max error: {deploy_error:.8e}")

    labels = {
        "recall": "Recall",
        "precision": "Precision",
        "F1": "F1",
        "IoU": "IoU",
        "OA": "OA",
        "Kappa": "Kappa",
    }
    for key, label in labels.items():
        logger.log_message(f"{label}: {scores[key]:.6f}")

    logger.log_message(f"Total time: {elapsed}")
    logger.log_message("=== END TEST RESULTS ===")


def _split_fingerprints(data_root):
    root = Path(data_root)
    return {
        split: hashlib.sha256(
            (root / "list" / f"{split}.txt").read_bytes()
        ).hexdigest()
        for split in ("train", "val", "test")
    }


def _read_split_ids(data_root):
    root = Path(data_root)
    result = {}
    for split in ("train", "val", "test"):
        path = root / "list" / f"{split}.txt"
        ids = [
            value.strip()
            for value in path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        if not ids:
            raise ValueError(f"Empty split: {split}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in split: {split}")
        result[split] = ids

    if any(
        set(result[left]) & set(result[right])
        for left, right in (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        )
    ):
        raise ValueError("Sample IDs overlap between train/val/test")
    return result


def validate_resume(args, checkpoint):
    """Validate exact resume for the new RDT+SCGR code line.

    Old RDT checkpoints intentionally do not resume into SCGR runs because the
    routing rule changed.  A fresh SCGR run can resume exactly from checkpoints
    created by this implementation.
    """
    version = checkpoint.get("format_version")
    if version != 3:
        raise ValueError(
            "RDT+SCGR requires checkpoint format_version == 3; start fresh"
        )

    saved = checkpoint.get("args", {})
    keys = [
        "implementation_version",
        "experiment",
        "dataset_name",
        "batch_size",
        "max_steps",
        "seed",
        "lr",
        "lr_mode",
        "step_loss",
        "weight_decay",
        "backbone_lr_mult",
        "dice_reduction",
        "main_loss_weights",
        "boundary_radius",
        "small_area",
        "scgr_region_size",
        "kd_lambda",
        "inWidth",
        "inHeight",
        "data_fingerprint",
        "cache_fingerprint",
        "mechanism",
        "cache_replay",
        "requires_teacher_cache",
        "cache_conditioning",
        "teacher_update",
        "teacher_lr",
        "teacher_hidden",
        "teacher_ema",
        "max_logit_delta",
        "teacher_weight_decay",
        "teacher_grad_clip",
        "routing_signal",
        "reject_unit",
    ]
    for key in keys:
        saved_value = saved.get(key)
        current_value = getattr(args, key, None)
        if saved_value != current_value:
            raise ValueError(
                "Resume configuration mismatch for "
                f"{key}: {saved_value!r} vs {current_value!r}"
            )


def _count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    auxiliary = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("training_auxiliary.")
    )
    fast = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("training_auxiliary.fast.")
    )
    target = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("training_auxiliary.target.")
    )
    return total, trainable, auxiliary, fast, target


def main():
    args = parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; use --device cpu only for local verification"
            )
        torch.cuda.set_device(args.gpu_id)

    device = torch.device(
        f"cuda:{args.gpu_id}"
        if args.device == "cuda"
        else "cpu"
    )
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args).to(device)
    (
        train_params,
        trainable_params,
        auxiliary_params,
        fast_teacher_params,
        target_teacher_params,
    ) = _count_parameters(model)

    optimizer = build_optimizer(args, model)
    teacher_optimizer = build_teacher_optimizer(args, model)
    criterion = build_loss(args.dice_reduction)

    args.data_fingerprint = _split_fingerprints(args.data_root)
    split_ids = _read_split_ids(args.data_root)
    train_list = split_ids["train"]

    if args.requires_teacher_cache:
        teacher_cache = PairedTeacherCache(
            args.sam_cache_root,
            args.ov_cache_root,
            args.dataset_name,
            train_list,
        )
    else:
        teacher_cache = None

    args.cache_fingerprint = (
        teacher_cache.fingerprint
        if teacher_cache is not None
        else None
    )

    train_loader = get_loader(
        args.data_root,
        os.path.join(args.data_root, "list", "train.txt"),
        batchsize=args.batch_size,
        trainsize=args.inWidth,
        num_workers=args.num_workers,
        teacher_cache=teacher_cache,
        seed=args.seed,
        cache_replay=args.cache_replay,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training dataset smaller than batch_size with drop_last=True"
        )

    val_loader = get_test_loader(
        args.data_root,
        os.path.join(args.data_root, "list", "val.txt"),
        batchsize=args.batch_size,
        testsize=args.inWidth,
        num_workers=args.num_workers,
        return_meta=True,
    )
    test_loader = get_test_loader(
        args.data_root,
        os.path.join(args.data_root, "list", "test.txt"),
        batchsize=args.batch_size,
        testsize=args.inWidth,
        num_workers=args.num_workers,
        return_meta=True,
    )

    args.max_epochs = int(
        np.ceil(args.max_steps / len(train_loader))
    )

    start_epoch = 0
    global_step = 0
    best_val_f1 = -1.0

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        validate_resume(args, checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        if teacher_optimizer is not None:
            state = checkpoint.get("teacher_optimizer")
            if state is None:
                raise ValueError("Missing dynamic-teacher optimizer state")
            teacher_optimizer.load_state_dict(state)

        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_f1 = checkpoint["best_val_f1"]
        restore_rng_state(checkpoint.get("rng"))

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = save_dir / log_path

    last_path = save_dir / "last_checkpoint.pth"
    if (
        not args.resume
        and (log_path.exists() or last_path.exists())
    ):
        raise FileExistsError(
            "Existing run found: use --resume or a fresh save/log directory"
        )

    logger = TrainingLogger(
        str(log_path),
        {
            **vars(args),
            "train_params": f"{train_params / 1e6:.4f}M",
            "trainable_params": f"{trainable_params / 1e6:.4f}M",
            "auxiliary_params": f"{auxiliary_params / 1e6:.6f}M",
            "fast_teacher_params": f"{fast_teacher_params / 1e6:.6f}M",
            "target_teacher_params": f"{target_teacher_params / 1e6:.6f}M",
            "deploy_params_expected": EXPECTED_DEPLOY_PARAMS,
            "deploy_flops_expected": EXPECTED_DEPLOY_FLOPS,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "n_test": len(test_loader.dataset),
        },
        append=bool(args.resume),
    )

    best_path: Optional[Path] = None
    if args.resume:
        candidate = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
        if candidate.is_file():
            best_path = candidate

    started = datetime.datetime.now()

    for epoch in range(start_epoch, args.max_epochs):
        if global_step >= args.max_steps:
            break

        losses, _, lr, global_step = train_epoch(
            args,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            global_step,
            device,
            teacher_optimizer=teacher_optimizer,
        )

        val_loss, val_scores = evaluate(
            val_loader,
            model,
            criterion,
            args.main_loss_weights,
            device,
        )
        is_best = val_scores["F1"] > best_val_f1
        if is_best:
            best_val_f1 = val_scores["F1"]

        checkpoint = build_checkpoint(
            model,
            optimizer,
            epoch,
            global_step,
            best_val_f1,
            args,
            router_optimizer=None,
            teacher_optimizer=teacher_optimizer,
        )
        save_checkpoint_atomic(checkpoint, last_path)

        if is_best:
            new_best = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
            save_checkpoint_atomic(checkpoint, new_best)
            if (
                best_path is not None
                and best_path != new_best
                and best_path.name.startswith("best_model_F1=")
            ):
                best_path.unlink(missing_ok=True)
            best_path = new_best

        logger.log_epoch(
            epoch,
            args.max_epochs,
            losses,
            {
                "f1": val_scores["F1"],
                "iou": val_scores["IoU"],
                "kappa": val_scores["Kappa"],
                "recall": val_scores["recall"],
                "precision": val_scores["precision"],
                "oa": val_scores["OA"],
            },
            lr,
            (
                torch.cuda.max_memory_allocated(device) / 1e9
                if device.type == "cuda"
                else 0.0
            ),
            is_best,
        )
        logger.log_message(
            f"Val loss: {val_loss:.6f}; global_step: {global_step}"
        )

        if global_step >= args.max_steps:
            break

    if best_path is None or not best_path.is_file():
        raise RuntimeError(
            "Training finished without a validation-selected best checkpoint"
        )

    checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])

    # 1) Loss-only auxiliary must not modify the main prediction.
    auxiliary_error = auxiliary_toggle_consistency(
        model,
        train_loader.dataset,
        device,
    )

    # 2) Physically remove training-only modules and verify deploy equivalence.
    deploy_error = deploy_consistency(model, test_loader, device)

    # 3) Test only the final deploy graph.
    scores, infer_params, flops = test_deployed(
        args,
        test_loader,
        model,
        device,
    )

    if infer_params != EXPECTED_DEPLOY_PARAMS:
        raise RuntimeError(
            "Unexpected deploy parameter count: "
            f"{infer_params:,}"
        )
    if (
        flops is not None
        and abs(flops - EXPECTED_DEPLOY_FLOPS) > DEPLOY_FLOPS_ATOL
    ):
        raise RuntimeError(
            "Unexpected deploy FLOPs: "
            f"{flops / 1e9:.6f}G"
        )

    append_test_results(
        logger,
        args,
        scores,
        train_params,
        infer_params,
        flops,
        auxiliary_error,
        deploy_error,
        datetime.datetime.now() - started,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
