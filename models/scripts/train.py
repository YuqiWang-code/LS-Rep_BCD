"""Unified trainer for SAM-HSD Run1 and EIR-HSD Run2 ablations."""

from __future__ import annotations

import argparse
import datetime
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
from models.datasets.cd_dataset import get_loader, get_test_loader
from models.distill import TeacherCache
from models.utils.checkpoint import build_checkpoint, restore_rng_state, save_checkpoint_atomic
from models.utils.logger import TrainingLogger
from models.utils.metrics import ConfuseMatrixMeter
from models.utils.scheduler import adjust_learning_rate


EXPERIMENTS = {
    "H0": {"name": "H0_Clean_Anchor", "auxiliary_mode": "none"},
    "H1": {"name": "H1_Legacy_SAMStruct", "auxiliary_mode": "legacy_sam"},
    "H2": {"name": "H2_Encoder_Directional", "auxiliary_mode": "sam_hsd",
           "hsd_mode": "encoder", "evidence_mode": "directional", "use_scgr": False,
           "robust_filter": True},
    "H3": {"name": "H3_Decoder_SCGR", "auxiliary_mode": "sam_hsd",
           "hsd_mode": "decoder", "evidence_mode": "directional", "use_scgr": True,
           "robust_filter": True},
    "H4": {"name": "H4_Full_SAM_HSD", "auxiliary_mode": "sam_hsd",
           "hsd_mode": "full", "evidence_mode": "directional", "use_scgr": True,
           "robust_filter": True},
    "H5": {"name": "H5_Boundary_Scalar", "auxiliary_mode": "sam_hsd",
           "hsd_mode": "full", "evidence_mode": "boundary_scalar", "use_scgr": True,
           "robust_filter": True},
    "H6": {"name": "H6_Unsigned_Evidence", "auxiliary_mode": "sam_hsd",
           "hsd_mode": "full", "evidence_mode": "unsigned", "use_scgr": True,
           "robust_filter": True},
    "H7": {"name": "H7_No_SCGR", "auxiliary_mode": "sam_hsd",
           "hsd_mode": "full", "evidence_mode": "directional", "use_scgr": False,
           "robust_filter": True},
    "H8": {"name": "H8_No_Robust_Filter", "auxiliary_mode": "sam_hsd",
           "hsd_mode": "full", "evidence_mode": "directional", "use_scgr": True,
           "robust_filter": False},
    "R0": {"name": "R0_Clean_Anchor", "auxiliary_mode": "none"},
    "R1": {"name": "R1_Run1_Unsigned_Reference", "auxiliary_mode": "sam_hsd",
           "hsd_mode": "full", "evidence_mode": "unsigned", "use_scgr": True,
           "robust_filter": True},
    "R2": {"name": "R2_Exchange_Invariant_Code", "auxiliary_mode": "eir_hsd",
           "hsd_mode": "encoder", "spatial_adapter": False,
           "use_correction": False, "fixed_fusion": False,
           "directional_restore": False},
    "R3": {"name": "R3_Spatial_Residual_Encoder", "auxiliary_mode": "eir_hsd",
           "hsd_mode": "encoder", "spatial_adapter": True,
           "use_correction": False, "fixed_fusion": False,
           "directional_restore": False},
    "R4": {"name": "R4_Residual_Decoder_Head", "auxiliary_mode": "eir_hsd",
           "hsd_mode": "decoder", "spatial_adapter": False,
           "use_correction": True, "fixed_fusion": False,
           "directional_restore": False},
    "R5": {"name": "R5_Full_EIR_HSD", "auxiliary_mode": "eir_hsd",
           "hsd_mode": "full", "spatial_adapter": True,
           "use_correction": True, "fixed_fusion": False,
           "directional_restore": False},
    "R6": {"name": "R6_Fixed_Fusion", "auxiliary_mode": "eir_hsd",
           "hsd_mode": "full", "spatial_adapter": True,
           "use_correction": True, "fixed_fusion": True,
           "directional_restore": False},
    "R7": {"name": "R7_No_Residual_Correction", "auxiliary_mode": "eir_hsd",
           "hsd_mode": "full", "spatial_adapter": True,
           "use_correction": False, "fixed_fusion": False,
           "directional_restore": False},
    "R8": {"name": "R8_Directional_Restore", "auxiliary_mode": "eir_hsd",
           "hsd_mode": "full", "spatial_adapter": True,
           "use_correction": True, "fixed_fusion": False,
           "directional_restore": True},
}


def parse_args():
    parser = argparse.ArgumentParser(description="SAM-HSD / EIR-HSD training")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--dataset_name", required=True, choices=["SYSU", "WHU"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_cache_root", default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--inWidth", type=int, default=256)
    parser.add_argument("--inHeight", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_mode", default="poly", choices=["poly", "step"])
    parser.add_argument("--step_loss", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=1.0)
    parser.add_argument("--dice_reduction", default="batch", choices=["sample", "batch"])
    parser.add_argument("--main_loss_weights", default="1,1,1,1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hsd_lambda", type=float, default=0.06)
    parser.add_argument("--hsd_max_ratio", type=float, default=0.12)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_file", default="train_log.txt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2333)
    args = parser.parse_args()
    args.main_loss_weights = tuple(float(value) for value in args.main_loss_weights.split(","))
    if len(args.main_loss_weights) != 4:
        raise ValueError("main_loss_weights must contain four comma-separated values")
    if args.batch_size <= 0 or args.max_steps <= 0 or args.num_workers < 0:
        raise ValueError("batch_size/max_steps must be positive and num_workers non-negative")
    if not 0.0 <= args.hsd_max_ratio <= 1.0 or args.hsd_lambda < 0:
        raise ValueError("Invalid SAM-HSD weight or cap")
    recipe = EXPERIMENTS[args.experiment]
    args.experiment_name = recipe["name"]
    args.auxiliary_mode = recipe["auxiliary_mode"]
    if args.auxiliary_mode != "none" and not args.teacher_cache_root:
        raise ValueError("Every non-anchor SAM-HSD/EIR-HSD run requires --teacher_cache_root")
    return args


def set_seed(seed):
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
    """Batch one Dataset sample without materializing a full training batch."""
    if torch.is_tensor(value):
        return value.unsqueeze(0).to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: nested_add_batch_to(item, device) for key, item in value.items()}
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
    return sum(weight * criterion(prediction, target)
               for weight, prediction in zip(weights, predictions))


def build_optimizer(args, model):
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scale = args.backbone_lr_mult if name.startswith("backbone.") else 1.0
        grouped.setdefault(scale, []).append(parameter)
    groups = [
        {"params": parameters, "lr": args.lr * scale, "lr_scale": scale,
         "weight_decay": args.weight_decay}
        for scale, parameters in grouped.items()
    ]
    return torch.optim.Adam(groups, lr=args.lr, betas=(0.9, 0.99), eps=1e-8)


def capped_auxiliary(raw_loss, main_loss, max_ratio):
    cap = max_ratio * main_loss.detach()
    scale = torch.clamp(cap / (raw_loss.detach() + 1e-8), max=1.0)
    weighted = raw_loss * scale.detach()
    return weighted, weighted.detach() / main_loss.detach().clamp_min(1e-8)


def legacy_multiplier(step, total):
    progress = step / max(total, 1)
    if progress < 0.10:
        return 0.0
    if progress < 0.20:
        return (progress - 0.10) / 0.10
    if progress >= 0.80:
        return max(0.0, (1.0 - progress) / 0.20)
    return 1.0


def hsd_multiplier(step, total):
    """OFF 0-5%, warmup 5-15%, plateau 15-80%, cosine to 0.3."""
    progress = min(max(step / max(total, 1), 0.0), 1.0)
    if progress < 0.05:
        return 0.0
    if progress < 0.15:
        return (progress - 0.05) / 0.10
    if progress <= 0.80:
        return 1.0
    phase = (progress - 0.80) / 0.20
    return 0.30 + 0.70 * 0.5 * (1.0 + math.cos(math.pi * phase))


def build_model(args):
    recipe = EXPERIMENTS[args.experiment]
    hsd_cfg = None
    eir_cfg = None
    legacy_cfg = None
    if args.auxiliary_mode == "sam_hsd":
        hsd_cfg = {
            "mode": recipe["hsd_mode"],
            "evidence_mode": recipe["evidence_mode"],
            "use_scgr": recipe["use_scgr"],
            "robust_filter": recipe["robust_filter"],
            "boundary_band": 7,
        }
    elif args.auxiliary_mode == "eir_hsd":
        eir_cfg = {
            "mode": recipe["hsd_mode"],
            "spatial_adapter": recipe["spatial_adapter"],
            "use_correction": recipe["use_correction"],
            "fixed_fusion": recipe["fixed_fusion"],
            "directional_restore": recipe["directional_restore"],
            "boundary_tolerance": 2,
            "trust_gamma": 1.0,
        }
    elif args.auxiliary_mode == "legacy_sam":
        legacy_cfg = {"boundary_weight": 0.7, "affinity_weight": 0.3, "boundary_band": 7}
    return A2Net_LWGANet_L0(
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        auxiliary_mode=args.auxiliary_mode,
        sam_hsd_cfg=hsd_cfg,
        eir_hsd_cfg=eir_cfg,
        legacy_sam_cfg=legacy_cfg,
    )


def train_epoch(args, loader, model, criterion, optimizer, epoch, global_step, device):
    model.train()
    meter = ConfuseMatrixMeter(n_class=2)
    metric_keys = (
        "total", "main", "aux_raw", "aux_weighted", "aux_ratio", "aux_factor",
        "encoder", "decoder", "scgr", "boundary", "relation", "affinity",
        "boundary_positive", "boundary_negative", "relation_changed",
        "stable_suppression", "prediction_contrast", "feature_relation",
        "e_plus", "e_minus", "e_stable", "e_uncertain",
        "plus_conf", "minus_conf", "stable_conf", "change_conf",
        "reliable_positive", "reliable_negative", "hard_negative", "hard_positive",
        "decoder_structure", "relation_same", "relation_contrast",
        "correction", "correction_fn", "correction_fp",
        "code_boundary", "code_local", "code_geometry", "code_stable",
        "trust_change", "trust_stable", "coverage_change", "coverage_stable",
        "structural_uncertainty", "fn_correction", "fp_correction",
        "data_time", "step_time",
    )
    totals = {key: 0.0 for key in metric_keys}
    batches, last_lr = 0, args.lr
    data_started = time.perf_counter()
    for batch in loader:
        data_elapsed = time.perf_counter() - data_started
        step_started = time.perf_counter()
        if global_step >= args.max_steps:
            break
        image, target, _, teacher_pack = unpack_batch(batch)
        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        teacher_pack = nested_to(teacher_pack, device) if teacher_pack is not None else None
        last_lr = adjust_learning_rate(args, optimizer, epoch, global_step, len(loader))
        optimizer.zero_grad(set_to_none=True)
        predictions, auxiliary = model(
            pre, post, target=target, teacher_pack=teacher_pack,
            compute_auxiliary=args.auxiliary_mode != "none",
        )
        main_loss = multiscale_loss(
            predictions, target, criterion, args.main_loss_weights,
        )
        zero = main_loss.new_zeros(())
        aux_raw = aux_weighted = aux_ratio = zero
        factor = 0.0
        details = {}
        if "legacy_sam" in auxiliary:
            details = auxiliary["legacy_sam"]
            aux_raw = details["total"]
            factor = legacy_multiplier(global_step, args.max_steps)
            aux_weighted, aux_ratio = capped_auxiliary(
                0.05 * factor * aux_raw, main_loss, 0.08,
            )
        elif "sam_hsd" in auxiliary:
            details = auxiliary["sam_hsd"]
            aux_raw = details["total"]
            factor = hsd_multiplier(global_step, args.max_steps)
            aux_weighted, aux_ratio = capped_auxiliary(
                args.hsd_lambda * factor * aux_raw, main_loss, args.hsd_max_ratio,
            )
        elif "eir_hsd" in auxiliary:
            details = auxiliary["eir_hsd"]
            aux_raw = details["total"]
            factor = hsd_multiplier(global_step, args.max_steps)
            aux_weighted, aux_ratio = capped_auxiliary(
                args.hsd_lambda * factor * aux_raw, main_loss, args.hsd_max_ratio,
            )
        loss = main_loss + aux_weighted
        loss.backward()
        optimizer.step()

        prediction = (predictions[0].detach() > 0.5).long()
        current_f1 = meter.update_cm(prediction.cpu().numpy(), target.cpu().numpy())
        values = {
            "total": loss, "main": main_loss, "aux_raw": aux_raw,
            "aux_weighted": aux_weighted, "aux_ratio": aux_ratio,
            "aux_factor": loss.new_tensor(factor),
            "encoder": details.get("encoder", zero), "decoder": details.get("decoder", zero),
            "scgr": details.get("scgr", zero), "boundary": details.get("boundary", zero),
            "relation": details.get("relation", zero), "affinity": details.get("affinity", zero),
            "boundary_positive": details.get("boundary_positive", zero),
            "boundary_negative": details.get("boundary_negative", zero),
            "relation_changed": details.get("relation_changed", zero),
            "stable_suppression": details.get("stable_suppression", zero),
            "prediction_contrast": details.get("prediction_contrast", zero),
            "feature_relation": details.get("feature_relation", zero),
            "e_plus": details.get("e_plus_mean", zero),
            "e_minus": details.get("e_minus_mean", zero),
            "e_stable": details.get("e_stable_mean", zero),
            "e_uncertain": details.get("e_uncertain_mean", zero),
            "plus_conf": details.get("plus_conf_mean", zero),
            "minus_conf": details.get("minus_conf_mean", zero),
            "stable_conf": details.get("stable_conf_mean", zero),
            "change_conf": details.get("change_conf_mean", zero),
            "reliable_positive": details.get("reliable_positive_ratio", zero),
            "reliable_negative": details.get("reliable_negative_ratio", zero),
            "hard_negative": details.get("hard_negative_ratio", zero),
            "hard_positive": details.get("hard_positive_ratio", zero),
            "decoder_structure": details.get("decoder_structure", zero),
            "relation_same": details.get("relation_same", zero),
            "relation_contrast": details.get("relation_contrast", zero),
            "correction": details.get("correction", zero),
            "correction_fn": details.get("correction_fn", zero),
            "correction_fp": details.get("correction_fp", zero),
            "code_boundary": details.get("code_boundary_mean", zero),
            "code_local": details.get("code_local_mean", zero),
            "code_geometry": details.get("code_geometry_mean", zero),
            "code_stable": details.get("code_stable_mean", zero),
            "trust_change": details.get("trust_change_mean", zero),
            "trust_stable": details.get("trust_stable_mean", zero),
            "coverage_change": details.get("coverage_change_mean", zero),
            "coverage_stable": details.get("coverage_stable_mean", zero),
            "structural_uncertainty": details.get(
                "structural_uncertainty_mean", zero,
            ),
            "fn_correction": details.get("fn_correction_ratio", zero),
            "fp_correction": details.get("fp_correction_ratio", zero),
            "data_time": loss.new_tensor(data_elapsed),
            "step_time": loss.new_tensor(time.perf_counter() - step_started),
        }
        for key, value in values.items():
            totals[key] += float(value.detach())
        batches += 1
        global_step += 1
        if global_step % 5 == 0:
            print(
                f"\rstep [{global_step}/{args.max_steps}] F1={current_f1:.3f} "
                f"lr={last_lr:.7f} loss={float(loss.detach()):.3f} "
                f"aux_ratio={float(aux_ratio):.3f} factor={factor:.3f} "
                f"data={data_elapsed:.3f}s", end="",
            )
        data_started = time.perf_counter()
    return (
        {key: value / max(batches, 1) for key, value in totals.items()},
        meter.get_scores(), last_lr, global_step,
    )


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
        meter.update_cm((predictions[0] > 0.5).long().cpu().numpy(), target.cpu().numpy())
        losses.append(float(loss))
    return sum(losses) / max(len(losses), 1), meter.get_scores()


@torch.no_grad()
def deploy_consistency(model, loader, device):
    model.eval()
    image, _, _, _ = unpack_batch(next(iter(loader)))
    pre, post = image[:1, :3].to(device), image[:1, 3:6].to(device)
    before = model(pre, post)
    model.switch_to_deploy().eval()
    after = model(pre, post)
    max_error = max((left - right).abs().max().item() for left, right in zip(before, after))
    if max_error >= 1e-6:
        raise RuntimeError(f"Deploy consistency failed: max_error={max_error:.8e}")
    return max_error


@torch.no_grad()
def auxiliary_toggle_consistency(model, dataset, device):
    """Prove the loss-only branch cannot alter the student's main output."""
    sample = dataset[0]
    image, target, _, teacher_pack = unpack_batch(sample)
    image = image.unsqueeze(0).to(device, non_blocking=True)
    target = target.unsqueeze(0).to(device, non_blocking=True).float()
    teacher_pack = nested_add_batch_to(teacher_pack, device) \
        if teacher_pack is not None else None

    # Keep every child module in deterministic eval mode while making only the
    # root execute its training return/auxiliary branch.
    model.eval()
    model.training = True
    without_auxiliary, _ = model(
        image[:, :3], image[:, 3:6], target=target,
        teacher_pack=teacher_pack, compute_auxiliary=False,
    )
    with_auxiliary, _ = model(
        image[:, :3], image[:, 3:6], target=target,
        teacher_pack=teacher_pack, compute_auxiliary=True,
    )
    model.training = False
    max_error = max(
        (left - right).abs().max().item()
        for left, right in zip(without_auxiliary, with_auxiliary)
    )
    if max_error != 0.0:
        raise RuntimeError(f"Auxiliary changed the main output: max_error={max_error:.8e}")
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
            (output > 0.5).long().cpu().numpy(), target.cpu().numpy(),
        )
    infer_params = sum(parameter.numel() for parameter in model.parameters())
    flops = None
    if profile is not None:
        dummy = torch.randn(1, 3, args.inHeight, args.inWidth, device=device)
        flops, _ = profile(model, inputs=(dummy, dummy), verbose=False)
    return meter.get_scores(), infer_params, flops


def append_test_results(logger, args, scores, train_params, infer_params, flops,
                        auxiliary_error, deploy_error, elapsed):
    """The test record lives at the end of train_log.txt; no test_infer directory."""
    logger.log_message("=" * 100)
    logger.log_message("=== TEST RESULTS ===")
    logger.log_message("Metric Split: test")
    logger.log_message(f"Dataset: {args.dataset_name}")
    logger.log_message(f"Experiment: {args.experiment}/{args.experiment_name}")
    logger.log_message(f"Train Params: {train_params / 1e6:.4f}M")
    logger.log_message(f"Infer Params: {infer_params / 1e6:.4f}M")
    logger.log_message(f"FLOPs: {flops / 1e9:.4f}G" if flops is not None else "FLOPs: unavailable")
    logger.log_message(f"Auxiliary toggle max error: {auxiliary_error:.8e}")
    logger.log_message(f"Deploy max error: {deploy_error:.8e}")
    labels = {
        "recall": "Recall", "precision": "Precision", "F1": "F1",
        "IoU": "IoU", "OA": "OA", "Kappa": "Kappa",
    }
    for key, label in labels.items():
        logger.log_message(f"{label}: {scores[key]:.6f}")
    logger.log_message(f"Total time: {elapsed}")
    logger.log_message("=== END TEST RESULTS ===")


def validate_resume(args, checkpoint):
    saved = checkpoint.get("args", {})
    for key in ("experiment", "dataset_name", "batch_size", "max_steps", "seed"):
        if key in saved and saved[key] != getattr(args, key):
            raise ValueError(
                f"Resume configuration mismatch for {key}: checkpoint={saved[key]!r}, "
                f"current={getattr(args, key)!r}"
            )


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args).to(device)
    train_params = sum(parameter.numel() for parameter in model.parameters())
    auxiliary_params = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name.startswith("training_auxiliary.")
    )
    optimizer = build_optimizer(args, model)
    criterion = build_loss(args.dice_reduction)

    teacher_cache = TeacherCache(args.teacher_cache_root, require_validation=True) \
        if args.auxiliary_mode != "none" else None
    if teacher_cache is not None:
        manifest = teacher_cache.manifest
        if manifest.get("source_dataset") != args.dataset_name:
            raise ValueError(
                f"Teacher cache dataset={manifest.get('source_dataset')} "
                f"but training dataset={args.dataset_name}"
            )
        if manifest.get("teacher_type") != "sam2_struct_v2":
            raise ValueError("SAM-HSD/EIR-HSD requires a sam2_struct_v2 cache")

    train_loader = get_loader(
        args.data_root, os.path.join(args.data_root, "list", "train.txt"),
        batchsize=args.batch_size, trainsize=args.inWidth,
        num_workers=args.num_workers, teacher_cache=teacher_cache,
        derive_sam_hsd=args.auxiliary_mode in {"sam_hsd", "eir_hsd"},
    )
    if teacher_cache is not None and set(teacher_cache.entries) != set(train_loader.dataset.file_list):
        raise ValueError("Teacher cache coverage does not exactly match the training list")
    val_loader = get_test_loader(
        args.data_root, os.path.join(args.data_root, "list", "val.txt"),
        batchsize=args.batch_size, testsize=args.inWidth,
        num_workers=args.num_workers, return_meta=True,
    )
    test_loader = get_test_loader(
        args.data_root, os.path.join(args.data_root, "list", "test.txt"),
        batchsize=args.batch_size, testsize=args.inWidth,
        num_workers=args.num_workers, return_meta=True,
    )
    args.max_epochs = int(np.ceil(args.max_steps / len(train_loader)))

    start_epoch, global_step, best_val_f1 = 0, 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(args, checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_f1 = checkpoint["best_val_f1"]
        restore_rng_state(checkpoint.get("rng"))

    logger = TrainingLogger(
        str(save_dir / args.log_file),
        {
            **vars(args),
            "train_params": f"{train_params / 1e6:.4f}M",
            "auxiliary_params": f"{auxiliary_params / 1e6:.6f}M",
            "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
            "n_test": len(test_loader.dataset),
        },
        append=bool(args.resume),
    )
    best_path = None
    if args.resume:
        candidate = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
        if candidate.is_file():
            best_path = candidate
    last_path = save_dir / "last_checkpoint.pth"
    started = datetime.datetime.now()

    for epoch in range(start_epoch, args.max_epochs):
        losses, _, lr, global_step = train_epoch(
            args, train_loader, model, criterion, optimizer,
            epoch, global_step, device,
        )
        val_loss, val_scores = evaluate(
            val_loader, model, criterion, args.main_loss_weights, device,
        )
        is_best = val_scores["F1"] > best_val_f1
        if is_best:
            best_val_f1 = val_scores["F1"]
        checkpoint = build_checkpoint(
            model, optimizer, epoch, global_step, best_val_f1, args,
        )
        save_checkpoint_atomic(checkpoint, last_path)
        if is_best:
            new_best = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
            save_checkpoint_atomic(checkpoint, new_best)
            if best_path is not None and best_path != new_best and best_path.name.startswith("best_model_F1="):
                best_path.unlink(missing_ok=True)
            best_path = new_best
        logger.log_epoch(
            epoch, args.max_epochs, losses,
            {"f1": val_scores["F1"], "iou": val_scores["IoU"],
             "kappa": val_scores["Kappa"], "recall": val_scores["recall"],
             "precision": val_scores["precision"], "oa": val_scores["OA"]},
            lr, torch.cuda.max_memory_allocated(device) / 1e9, is_best,
        )
        logger.log_message(f"Val loss: {val_loss:.6f}; global_step: {global_step}")
        if global_step >= args.max_steps:
            break

    if best_path is None or not best_path.is_file():
        raise RuntimeError("Training finished without a validation-selected best checkpoint")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    auxiliary_error = auxiliary_toggle_consistency(model, train_loader.dataset, device)
    deploy_error = deploy_consistency(model, test_loader, device)
    scores, infer_params, flops = test_deployed(args, test_loader, model, device)
    if infer_params != 2_913_094:
        raise RuntimeError(f"Unexpected deploy parameter count: {infer_params:,}")
    if flops is not None and abs(flops - 2.7475e9) > 0.01e9:
        raise RuntimeError(f"Unexpected deploy FLOPs: {flops / 1e9:.6f}G")
    append_test_results(
        logger, args, scores, train_params, infer_params, flops,
        auxiliary_error, deploy_error, datetime.datetime.now() - started,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
