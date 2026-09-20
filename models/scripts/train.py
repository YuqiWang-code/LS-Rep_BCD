"""SCTC trainer for A2Net-LWGANet-L0.

Formal experiment matrix
------------------------
S0: clean A2Net baseline, no temporal calibration.
S1: full-pixel symmetric temporal calibration.
S2: SCTC -- Unchanged-Aware Symmetric Cross-Temporal Calibration.

This trainer contains no Teacher, Foundation Model, cache, distillation,
routing, EMA, or training-only auxiliary path.
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
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn

try:
    from thop import profile
except ImportError:
    profile = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from models import A2Net_LWGANet_L0, build_loss  # noqa: E402
from models.datasets.cd_dataset import get_loader, get_test_loader  # noqa: E402
from models.decoder.temporal_calibration import SCTC_EPS  # noqa: E402
from models.utils.metrics import ConfuseMatrixMeter  # noqa: E402
from models.utils.scheduler import adjust_learning_rate  # noqa: E402


# =============================================================================
# Hard deployment contract
# =============================================================================

EXPECTED_DEPLOY_PARAMS = 2_913_094
EXPECTED_DEPLOY_FLOPS = 2.7475e9
DEPLOY_FLOPS_ATOL = 0.03e9
SWAP_ATOL = 1e-6
DEPLOY_ATOL = 1e-6

CHECKPOINT_FORMAT_VERSION = 4
IMPLEMENTATION_VERSION = "sctc_v1"


# =============================================================================
# Experiment matrix
# =============================================================================

EXPERIMENTS = {
    "S0": {
        "name": "S0_Clean_A2Net_LWGANet_L0",
        "temporal_calibration_mode": "none",
        "mechanism": "none",
    },
    "S1": {
        "name": "S1_FullPixel_Symmetric_Calibration",
        "temporal_calibration_mode": "symmetric",
        "mechanism": "symmetric_pair_calibration",
    },
    "S2": {
        "name": "S2_SCTC_UnchangedAware_Symmetric_CrossTemporal_Calibration",
        "temporal_calibration_mode": "sctc",
        "mechanism": "sctc",
    },
}


# =============================================================================
# Arguments
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCTC trainer: S0 clean / S1 symmetric / S2 unchanged-aware SCTC"
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(EXPERIMENTS),
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=("CDD", "LEVIR", "SYSU", "WHU"),
    )
    parser.add_argument("--data_root", required=True)

    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2333)

    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pretrained_path", default=None)

    parser.add_argument("--inWidth", type=int, default=256)
    parser.add_argument("--inHeight", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr_mode",
        default="poly",
        choices=("poly", "step"),
    )
    parser.add_argument("--step_loss", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=1.0)

    parser.add_argument(
        "--dice_reduction",
        default="batch",
        choices=("sample", "batch"),
    )
    parser.add_argument("--main_loss_weights", default="1,1,1,1")

    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_file", default="train_log.txt")
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()

    args.main_loss_weights = tuple(
        float(value) for value in args.main_loss_weights.split(",")
    )
    if len(args.main_loss_weights) != 4:
        raise ValueError("main_loss_weights must contain four comma-separated values")
    if any(
        (not math.isfinite(value) or value < 0)
        for value in args.main_loss_weights
    ):
        raise ValueError("Invalid main_loss_weights")

    if args.batch_size <= 0 or args.max_steps <= 0 or args.num_workers < 0:
        raise ValueError(
            "batch_size/max_steps must be positive and num_workers non-negative"
        )
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.backbone_lr_mult <= 0:
        raise ValueError("backbone_lr_mult must be positive")

    if args.inWidth != 256 or args.inHeight != 256:
        raise ValueError("Formal SCTC protocol requires 256x256 inputs")

    if args.pretrained and not args.pretrained_path:
        raise ValueError(
            "--pretrained_path is required when --pretrained is enabled"
        )

    recipe = EXPERIMENTS[args.experiment]
    args.experiment_name = recipe["name"]
    args.temporal_calibration_mode = recipe["temporal_calibration_mode"]
    args.mechanism = recipe["mechanism"]
    args.implementation_version = IMPLEMENTATION_VERSION
    args.calibration_eps = SCTC_EPS

    return args


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# =============================================================================
# Generic logging
# =============================================================================


class TrainingLogger:
    def __init__(
        self,
        log_file: str,
        config: Mapping[str, Any],
        append: bool = False,
    ) -> None:
        self.log_file = str(log_file)
        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as handle:
            if append:
                handle.write("\n" + "=" * 100 + "\nRUN RESUME\n" + "=" * 100 + "\n")
            else:
                handle.write("=" * 100 + "\n")
                handle.write("A2Net-LWGANet-L0 SCTC Training Log\n")
                handle.write("=" * 100 + "\n")
            for key, value in dict(config).items():
                value_text = str(value).replace("\n", "\\n")
                handle.write(f"{key}: {value_text}\n")
            handle.write("-" * 100 + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

    def log_message(self, message: Any) -> None:
        text = str(message).rstrip("\n")
        print(text, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_losses: Mapping[str, float],
        val_metrics: Mapping[str, float],
        lr: float,
        gpu_mem: float,
        is_best: bool,
    ) -> None:
        losses = " ".join(
            f"{key}={float(value):.6f}" for key, value in train_losses.items()
        )
        line = (
            f"Epoch [{epoch}/{total_epochs}] {losses} "
            f"F1={float(val_metrics['f1']):.6f} "
            f"IoU={float(val_metrics['iou']):.6f} "
            f"Recall={float(val_metrics['recall']):.6f} "
            f"Precision={float(val_metrics['precision']):.6f} "
            f"OA={float(val_metrics['oa']):.6f} "
            f"Kappa={float(val_metrics['kappa']):.6f} "
            f"LR={float(lr):.8f} GPU={float(gpu_mem):.3f}GB"
        )
        if is_best:
            line += " BEST"
        self.log_message(line)


# =============================================================================
# Checkpoint / exact resume
# =============================================================================


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    cuda_state = state.get("cuda")
    if cuda_state is None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Checkpoint contains CUDA RNG state but CUDA is unavailable"
        )
    if len(cuda_state) != torch.cuda.device_count():
        raise RuntimeError(
            "CUDA device-count mismatch during exact resume: "
            f"checkpoint={len(cuda_state)}, visible={torch.cuda.device_count()}"
        )
    torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint_atomic(checkpoint: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        if temporary.is_dir():
            raise IsADirectoryError(temporary)
        temporary.unlink()

    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    except Exception:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise


def build_checkpoint(
    model,
    optimizer,
    epoch: int,
    global_step: int,
    best_val_f1: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_f1": float(best_val_f1),
        "args": dict(vars(args)),
        "rng": capture_rng_state(),
    }


def validate_resume(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
) -> None:
    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "SCTC requires checkpoint format_version="
            f"{CHECKPOINT_FORMAT_VERSION}; old Run3/Teacher checkpoints "
            "must not be resumed into SCTC experiments"
        )

    saved = checkpoint.get("args", {})
    keys = (
        "implementation_version",
        "experiment",
        "experiment_name",
        "dataset_name",
        "mechanism",
        "temporal_calibration_mode",
        "calibration_eps",
        "batch_size",
        "max_steps",
        "seed",
        "inWidth",
        "inHeight",
        "lr",
        "lr_mode",
        "step_loss",
        "weight_decay",
        "backbone_lr_mult",
        "dice_reduction",
        "main_loss_weights",
        "data_fingerprint",
    )

    for key in keys:
        saved_value = saved.get(key)
        current_value = getattr(args, key, None)
        if saved_value != current_value:
            raise ValueError(
                f"Resume configuration mismatch for {key}: "
                f"{saved_value!r} vs {current_value!r}"
            )


# =============================================================================
# Dataset audit / identity
# =============================================================================


def _split_fingerprints(data_root: str) -> Dict[str, str]:
    root = Path(data_root)
    result = {}
    for split in ("train", "val", "test"):
        path = root / "list" / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _validate_split_ids(data_root: str) -> None:
    root = Path(data_root)
    split_ids = {}

    for split in ("train", "val", "test"):
        path = root / "list" / f"{split}.txt"
        ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not ids:
            raise ValueError(f"Empty split: {split}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in split: {split}")
        split_ids[split] = ids

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(split_ids[left]) & set(split_ids[right])
        if overlap:
            raise ValueError(
                f"Sample IDs overlap between {left}/{right}: {len(overlap)}"
            )


# =============================================================================
# Batch / loss helpers
# =============================================================================


def unpack_batch(batch):
    if not isinstance(batch, (tuple, list)):
        raise TypeError("DataLoader batch must be tuple/list")
    if len(batch) < 2:
        raise ValueError("DataLoader batch must contain image and target")
    return batch[0], batch[1]


def multiscale_loss(predictions, target, criterion, weights):
    if len(predictions) != len(weights):
        raise ValueError(
            f"Prediction/weight count mismatch: {len(predictions)} vs {len(weights)}"
        )
    return sum(
        weight * criterion(prediction, target)
        for weight, prediction in zip(weights, predictions)
    )


# =============================================================================
# Model / optimizer
# =============================================================================


def build_model(args: argparse.Namespace) -> A2Net_LWGANet_L0:
    return A2Net_LWGANet_L0(
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        temporal_calibration_mode=args.temporal_calibration_mode,
    )


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module):
    grouped: Dict[float, list] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scale = args.backbone_lr_mult if name.startswith("backbone.") else 1.0
        grouped.setdefault(scale, []).append(parameter)

    if not grouped:
        raise RuntimeError("Optimizer received no trainable parameters")

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


# =============================================================================
# Train / validation
# =============================================================================


def train_epoch(
    args,
    loader,
    model,
    criterion,
    optimizer,
    epoch,
    global_step,
    device,
):
    model.train()

    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    if getattr(loader, "generator", None) is not None:
        loader.generator.manual_seed(args.seed + epoch)

    meter = ConfuseMatrixMeter(n_class=2)
    totals = {
        "total": 0.0,
        "main": 0.0,
        "data_time": 0.0,
        "step_time": 0.0,
    }
    batches = 0
    last_lr = args.lr
    data_started = time.perf_counter()

    for batch in loader:
        data_elapsed = time.perf_counter() - data_started
        if global_step >= args.max_steps:
            break

        step_started = time.perf_counter()
        image, target = unpack_batch(batch)

        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        last_lr = adjust_learning_rate(
            args,
            optimizer,
            epoch,
            global_step,
            len(loader),
        )

        optimizer.zero_grad(set_to_none=True)
        predictions = model(pre, post)
        main_loss = multiscale_loss(
            predictions,
            target,
            criterion,
            args.main_loss_weights,
        )

        if not bool(torch.isfinite(main_loss)):
            raise FloatingPointError(
                "Non-finite training loss; checkpoint not overwritten"
            )

        main_loss.backward()
        optimizer.step()

        hard_prediction = (predictions[0].detach() > 0.5).long()
        current_f1 = meter.update_cm(
            hard_prediction.cpu().numpy(),
            target.cpu().numpy(),
        )

        totals["total"] += float(main_loss.detach())
        totals["main"] += float(main_loss.detach())
        totals["data_time"] += float(data_elapsed)
        totals["step_time"] += float(time.perf_counter() - step_started)
        batches += 1
        global_step += 1

        if global_step % 5 == 0:
            print(
                f"\rstep [{global_step}/{args.max_steps}] "
                f"F1={current_f1:.3f} lr={last_lr:.7f} "
                f"loss={float(main_loss.detach()):.3f} "
                f"data={data_elapsed:.3f}s",
                end="",
                flush=True,
            )

        data_started = time.perf_counter()

    if batches == 0:
        raise RuntimeError("No training batches; check batch_size/max_steps")

    print(flush=True)
    averages = {key: value / batches for key, value in totals.items()}
    return averages, meter.get_scores(), last_lr, global_step


@torch.no_grad()
def evaluate(loader, model, criterion, weights, device):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []

    for batch in loader:
        image, target = unpack_batch(batch)
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

    return sum(losses) / max(len(losses), 1), meter.get_scores()


# =============================================================================
# Symmetry / deployment correctness
# =============================================================================


@torch.no_grad()
def temporal_swap_consistency(model, loader, device) -> float:
    model.eval()
    image, _ = unpack_batch(next(iter(loader)))
    pre = image[:1, :3].to(device)
    post = image[:1, 3:6].to(device)

    output_ab = model(pre, post)
    output_ba = model(post, pre)

    max_error = max(
        (left - right).abs().max().item()
        for left, right in zip(output_ab, output_ba)
    )
    if max_error >= SWAP_ATOL:
        raise RuntimeError(
            f"Temporal swap consistency failed: max_error={max_error:.8e}"
        )
    return max_error


@torch.no_grad()
def deploy_consistency(model, loader, device) -> float:
    model.eval()
    image, _ = unpack_batch(next(iter(loader)))
    pre = image[:1, :3].to(device)
    post = image[:1, 3:6].to(device)

    before = model(pre, post)
    model.switch_to_deploy()
    model.eval()
    after = model(pre, post)

    max_error = max(
        (left - right).abs().max().item()
        for left, right in zip(before, after)
    )
    if max_error >= DEPLOY_ATOL:
        raise RuntimeError(
            f"Deploy consistency failed: max_error={max_error:.8e}"
        )
    return max_error


# =============================================================================
# Final deployed test
# =============================================================================


@torch.no_grad()
def test_deployed(args, loader, model, device):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)

    for batch in loader:
        image, target = unpack_batch(batch)
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
        dummy1 = torch.randn(
            1,
            3,
            args.inHeight,
            args.inWidth,
            device=device,
        )
        dummy2 = torch.randn_like(dummy1)
        flops, _ = profile(
            model,
            inputs=(dummy1, dummy2),
            verbose=False,
        )

    return meter.get_scores(), infer_params, flops


# =============================================================================
# Authoritative result block
# =============================================================================


def append_test_results(
    logger,
    args,
    scores,
    train_params,
    trainable_params,
    calibration_params,
    infer_params,
    flops,
    swap_error,
    deploy_error,
    elapsed,
):
    logger.log_message("=" * 100)
    logger.log_message("=== TEST RESULTS ===")
    logger.log_message("Metric Split: test")
    logger.log_message(f"Dataset: {args.dataset_name}")
    logger.log_message(f"Experiment: {args.experiment}/{args.experiment_name}")
    logger.log_message(f"Implementation: {args.implementation_version}")
    logger.log_message(f"Temporal Calibration: {args.temporal_calibration_mode}")
    logger.log_message(f"Seed: {args.seed}")
    logger.log_message(f"Max Steps: {args.max_steps}")
    logger.log_message(f"Batch Size: {args.batch_size}")
    logger.log_message(f"Train Params: {train_params / 1e6:.4f}M")
    logger.log_message(f"Trainable Params: {trainable_params / 1e6:.4f}M")
    logger.log_message(f"Calibration Params: {calibration_params}")
    logger.log_message(f"Infer Params: {infer_params / 1e6:.4f}M")
    logger.log_message(
        f"FLOPs: {flops / 1e9:.4f}G" if flops is not None else "FLOPs: unavailable"
    )
    logger.log_message(f"Temporal swap max error: {swap_error:.8e}")
    logger.log_message(f"Deploy max error: {deploy_error:.8e}")

    labels = {
        "recall": "Recall",
        "precision": "Precision",
        "OA": "OA",
        "F1": "F1",
        "IoU": "IoU",
        "Kappa": "Kappa",
    }
    for key, label in labels.items():
        logger.log_message(f"{label}: {scores[key]:.6f}")

    logger.log_message(f"Total time: {elapsed}")
    logger.log_message("=== END TEST RESULTS ===")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; use --device cpu only for local verification"
            )
        torch.cuda.set_device(args.gpu_id)

    device = torch.device(
        f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu"
    )
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    args.data_fingerprint = _split_fingerprints(args.data_root)
    _validate_split_ids(args.data_root)

    model = build_model(args).to(device)

    train_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    calibration_params = sum(
        parameter.numel()
        for parameter in model.tfm.temporal_calibration.parameters()
    )

    if calibration_params != 0:
        raise RuntimeError(
            f"SCTC must remain parameter-free, got {calibration_params} parameters"
        )
    if train_params != EXPECTED_DEPLOY_PARAMS:
        raise RuntimeError(
            f"Unexpected model parameter count before training: {train_params:,}; "
            f"expected {EXPECTED_DEPLOY_PARAMS:,}"
        )

    optimizer = build_optimizer(args, model)
    criterion = build_loss(args.dice_reduction)

    train_loader = get_loader(
        args.data_root,
        os.path.join(args.data_root, "list", "train.txt"),
        batchsize=args.batch_size,
        trainsize=args.inWidth,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if len(train_loader) == 0:
        raise ValueError(
            "Training dataset is smaller than batch_size with drop_last=True"
        )

    val_loader = get_test_loader(
        args.data_root,
        os.path.join(args.data_root, "list", "val.txt"),
        batchsize=args.batch_size,
        testsize=args.inWidth,
        num_workers=args.num_workers,
        return_meta=False,
    )
    test_loader = get_test_loader(
        args.data_root,
        os.path.join(args.data_root, "list", "test.txt"),
        batchsize=args.batch_size,
        testsize=args.inWidth,
        num_workers=args.num_workers,
        return_meta=False,
    )

    args.max_epochs = int(np.ceil(args.max_steps / len(train_loader)))

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
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_val_f1 = float(checkpoint["best_val_f1"])
        restore_rng_state(checkpoint.get("rng"))

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = save_dir / log_path
    last_path = save_dir / "last_checkpoint.pth"

    if not args.resume and (log_path.exists() or last_path.exists()):
        raise FileExistsError(
            "Existing run found; use --resume or a fresh save/log directory"
        )

    logger = TrainingLogger(
        str(log_path),
        {
            **vars(args),
            "train_params": f"{train_params / 1e6:.4f}M",
            "trainable_params": f"{trainable_params / 1e6:.4f}M",
            "calibration_params": calibration_params,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "n_test": len(test_loader.dataset),
        },
        append=bool(args.resume),
    )

    best_path = None
    if args.resume:
        candidate = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
        if candidate.is_file():
            best_path = candidate

    started = datetime.datetime.now()

    for epoch in range(start_epoch, args.max_epochs):
        if global_step >= args.max_steps:
            break

        train_losses, _, current_lr, global_step = train_epoch(
            args,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            global_step,
            device,
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
            best_val_f1 = float(val_scores["F1"])

        checkpoint = build_checkpoint(
            model,
            optimizer,
            epoch,
            global_step,
            best_val_f1,
            args,
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
            train_losses,
            {
                "f1": val_scores["F1"],
                "iou": val_scores["IoU"],
                "kappa": val_scores["Kappa"],
                "recall": val_scores["recall"],
                "precision": val_scores["precision"],
                "oa": val_scores["OA"],
            },
            current_lr,
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
    model.load_state_dict(checkpoint["model"], strict=True)

    swap_error = temporal_swap_consistency(model, test_loader, device)
    deploy_error = deploy_consistency(model, test_loader, device)

    test_scores, infer_params, flops = test_deployed(
        args,
        test_loader,
        model,
        device,
    )

    if infer_params != EXPECTED_DEPLOY_PARAMS:
        raise RuntimeError(
            f"Unexpected deploy parameter count: {infer_params:,}; "
            f"expected {EXPECTED_DEPLOY_PARAMS:,}"
        )

    if (
        flops is not None
        and abs(flops - EXPECTED_DEPLOY_FLOPS) > DEPLOY_FLOPS_ATOL
    ):
        raise RuntimeError(
            f"Unexpected profiled deploy FLOPs: {flops / 1e9:.6f}G; "
            f"expected approximately {EXPECTED_DEPLOY_FLOPS / 1e9:.6f}G"
        )

    append_test_results(
        logger,
        args,
        test_scores,
        train_params,
        trainable_params,
        calibration_params,
        infer_params,
        flops,
        swap_error,
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
