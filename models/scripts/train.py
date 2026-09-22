"""FA-SCRD trainer for A2Net-LWGANet-L0.

Formal experiment matrix (minimal ablation, single seed):
    C0: clean A2Net (no teacher, no joint BN)            — clean anchor
    C1: C0 + joint temporal BN                            — normalization control
    A1: C1 + fixed-teacher symmetric relation KD (uniform weight)
    M1: A1 + failure-aware soft weighting                 — full FA-SCRD

The DINOv3 teacher / cache / router / relation-KD are all training-only; the
deployed student is unchanged (2,913,094 params, ~2.77G FLOPs).
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
import torch.nn.functional as F

try:
    from thop import profile
except ImportError:
    profile = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import A2Net_LWGANet_L0, build_loss  # noqa: E402
from models.datasets.cd_dataset import get_loader, get_test_loader  # noqa: E402
from models.distill.cache_dataset import TeacherCacheReader, apply_cache_state  # noqa: E402
from models.distill.failure_router import compute_failure_weight  # noqa: E402
from models.distill.relation_kd import symmetric_relation_kd  # noqa: E402
from models.utils.metrics import ConfuseMatrixMeter  # noqa: E402
from models.utils.scheduler import adjust_learning_rate  # noqa: E402


EXPECTED_DEPLOY_PARAMS = 2_913_094
EXPECTED_DEPLOY_FLOPS = 2.7475e9
DEPLOY_FLOPS_ATOL = 0.03e9
SWAP_ATOL = 1e-6
DEPLOY_ATOL = 1e-6

CHECKPOINT_FORMAT_VERSION = 5
IMPLEMENTATION_VERSION = "fa_scrd_v1"

EXPERIMENTS = {
    "C0": {"name": "C0_Clean_A2Net", "joint_bn": False, "teacher": False, "failure_weight": False},
    "C1": {"name": "C1_JointTemporalBN", "joint_bn": True, "teacher": False, "failure_weight": False},
    "A1": {"name": "A1_FixedTeacherRelationKD", "joint_bn": True, "teacher": True, "failure_weight": False},
    "M1": {"name": "M1_FASCRD", "joint_bn": True, "teacher": True, "failure_weight": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FA-SCRD trainer: C0/C1/A1/M1")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--dataset_name", required=True, choices=("CDD", "LEVIR", "SYSU", "WHU"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cache_root", default=None, help="teacher cache root (A1/M1)")

    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2333)

    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained_path", default=None)

    parser.add_argument("--inWidth", type=int, default=256)
    parser.add_argument("--inHeight", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_mode", default="poly", choices=("poly", "step"))
    parser.add_argument("--step_loss", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=1.0)
    parser.add_argument("--dice_reduction", default="batch", choices=("sample", "batch"))
    parser.add_argument("--main_loss_weights", default="1,1,1,1")

    parser.add_argument("--kd_lambda", type=float, default=0.5)
    parser.add_argument("--kd_lambda_mid", type=float, default=1.0)
    parser.add_argument("--kd_lambda_deep", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--tau_a", type=float, default=0.5)

    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_file", default="train_log.txt")
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()

    args.main_loss_weights = tuple(float(v) for v in args.main_loss_weights.split(","))
    if len(args.main_loss_weights) != 4:
        raise ValueError("main_loss_weights must contain four comma-separated values")

    recipe = EXPERIMENTS[args.experiment]
    args.experiment_name = recipe["name"]
    args.joint_bn = recipe["joint_bn"]
    args.use_teacher = recipe["teacher"]
    args.failure_weight = recipe["failure_weight"]
    args.implementation_version = IMPLEMENTATION_VERSION

    if args.use_teacher and not args.cache_root:
        raise ValueError("--cache_root is required for teacher arms (A1/M1)")
    if args.pretrained and not args.pretrained_path:
        raise ValueError("--pretrained_path is required when --pretrained is enabled")

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class TrainingLogger:
    def __init__(self, log_file: str, config: Mapping[str, Any], append: bool = False) -> None:
        self.log_file = str(log_file)
        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as handle:
            if append:
                handle.write("\n" + "=" * 100 + "\nRUN RESUME\n" + "=" * 100 + "\n")
            else:
                handle.write("=" * 100 + "\n")
                handle.write("A2Net-LWGANet-L0 FA-SCRD Training Log\n")
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

    def log_epoch(self, epoch, total_epochs, train_losses, val_metrics, lr, gpu_mem, is_best) -> None:
        losses = " ".join(f"{k}={float(v):.6f}" for k, v in train_losses.items())
        line = (
            f"Epoch [{epoch}/{total_epochs}] {losses} "
            f"F1={float(val_metrics['f1']):.6f} IoU={float(val_metrics['iou']):.6f} "
            f"Recall={float(val_metrics['recall']):.6f} Precision={float(val_metrics['precision']):.6f} "
            f"OA={float(val_metrics['oa']):.6f} Kappa={float(val_metrics['kappa']):.6f} "
            f"LR={float(lr):.8f} GPU={float(gpu_mem):.3f}GB"
        )
        if is_best:
            line += " BEST"
        self.log_message(line)


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
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint_atomic(checkpoint: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def build_checkpoint(model, optimizer, epoch, global_step, best_val_f1, args) -> Dict[str, Any]:
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


def validate_resume(args, checkpoint: Mapping[str, Any]) -> None:
    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"FA-SCRD requires format_version={CHECKPOINT_FORMAT_VERSION}; "
            f"got {version}. Old SCTC/Run3 checkpoints must not be resumed."
        )
    saved = checkpoint.get("args", {})
    for key in ("implementation_version", "experiment", "dataset_name", "batch_size",
                "max_steps", "seed", "inWidth", "inHeight", "lr", "weight_decay",
                "backbone_lr_mult", "dice_reduction", "data_fingerprint"):
        if saved.get(key) != getattr(args, key, None):
            raise ValueError(f"Resume config mismatch for {key}: {saved.get(key)!r} vs {getattr(args, key, None)!r}")


def _split_fingerprints(data_root: str) -> Dict[str, str]:
    root = Path(data_root)
    return {split: hashlib.sha256((root / "list" / f"{split}.txt").read_bytes()).hexdigest()
            for split in ("train", "val", "test")}


def unpack_batch(batch):
    if not isinstance(batch, (tuple, list)):
        raise TypeError("DataLoader batch must be tuple/list")
    return batch[0], batch[1]


class CacheAwareDataset(torch.utils.data.Dataset):
    """Wraps a state-returning dataset and replays the augmentation onto the cache."""

    def __init__(self, base, cache_reader, split: str):
        self.base = base
        self.reader = cache_reader
        self.split = split

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __getitem__(self, idx: int):
        image, label, sample_id, state = self.base[idx]
        cache = self.reader.load(self.split, sample_id)
        cache = apply_cache_state(cache, state)
        return (
            image, label,
            cache["t1_mid"].float(), cache["t2_mid"].float(),
            cache["t1_deep"].float(), cache["t2_deep"].float(),
            cache["teacher_change_logit"].float(),
        )


def build_model(args) -> A2Net_LWGANet_L0:
    return A2Net_LWGANet_L0(
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        temporal_calibration_mode="none",
        joint_temporal_bn=args.joint_bn,
    )


def build_optimizer(args, model: torch.nn.Module):
    grouped: Dict[float, list] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scale = args.backbone_lr_mult if name.startswith("backbone.") else 1.0
        grouped.setdefault(scale, []).append(parameter)
    groups = [
        {"params": params, "lr": args.lr * scale, "lr_scale": scale, "weight_decay": args.weight_decay}
        for scale, params in grouped.items()
    ]
    return torch.optim.Adam(groups, lr=args.lr, betas=(0.9, 0.99), eps=1e-8)


def multiscale_loss(predictions, target, criterion, weights):
    return sum(w * criterion(p, target) for w, p in zip(weights, predictions))


def teacher_change_features(cache):
    """Compute C_T^mid and C_T^deep = |Norm(F_T1) - Norm(F_T2)|."""
    c_mid = (F.normalize(cache["t1_mid"], p=2, dim=1) - F.normalize(cache["t2_mid"], p=2, dim=1)).abs()
    c_deep = (F.normalize(cache["t1_deep"], p=2, dim=1) - F.normalize(cache["t2_deep"], p=2, dim=1)).abs()
    return c_mid, c_deep


def relation_kd_loss(student_c, c_mid, c_deep, weight, args):
    l_mid = symmetric_relation_kd(student_c, c_mid, weight, tau=args.tau)
    l_deep = symmetric_relation_kd(student_c, c_deep, weight, tau=args.tau)
    return args.kd_lambda_mid * l_mid + args.kd_lambda_deep * l_deep


def train_epoch(args, loader, model, criterion, optimizer, epoch, global_step, device):
    model.train()
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    if getattr(loader, "generator", None) is not None:
        loader.generator.manual_seed(args.seed + epoch)

    meter = ConfuseMatrixMeter(n_class=2)
    totals = {"total": 0.0, "main": 0.0, "kd": 0.0}
    batches = 0
    last_lr = args.lr
    use_teacher = args.use_teacher

    for batch in loader:
        if global_step >= args.max_steps:
            break
        if use_teacher:
            image, target = batch[0], batch[1]
            t1_mid, t2_mid = batch[2].to(device, non_blocking=True), batch[3].to(device, non_blocking=True)
            t1_deep, t2_deep = batch[4].to(device, non_blocking=True), batch[5].to(device, non_blocking=True)
            teacher_logit = batch[6].to(device, non_blocking=True)
        else:
            image, target = unpack_batch(batch)

        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        last_lr = adjust_learning_rate(args, optimizer, epoch, global_step, len(loader))
        optimizer.zero_grad(set_to_none=True)

        predictions, change = model(pre, post, return_change_features=True)
        main_loss = multiscale_loss(predictions, target, criterion, args.main_loss_weights)

        kd_loss = torch.zeros((), device=device)
        if use_teacher:
            student_c = change[2]  # c4 at 1/16 resolution
            c_mid, c_deep = teacher_change_features({"t1_mid": t1_mid, "t2_mid": t2_mid,
                                                     "t1_deep": t1_deep, "t2_deep": t2_deep})
            if args.failure_weight:
                p_s = F.avg_pool2d(predictions[0].detach(), 16)
                p_t = F.avg_pool2d(teacher_logit, 16)
                y = F.avg_pool2d(target, 16)
                weight = compute_failure_weight(p_s, p_t, y, tau_a=args.tau_a)
            else:
                weight = torch.ones((target.shape[0], 1, 16, 16), device=device)
            kd_loss = relation_kd_loss(student_c, c_mid, c_deep, weight, args)

        total = main_loss + args.kd_lambda * kd_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Non-finite training loss; checkpoint not overwritten")

        total.backward()
        optimizer.step()

        hard_prediction = (predictions[0].detach() > 0.5).long()
        current_f1 = meter.update_cm(hard_prediction.cpu().numpy(), target.cpu().numpy())
        totals["total"] += float(total.detach())
        totals["main"] += float(main_loss.detach())
        totals["kd"] += float(kd_loss.detach())
        batches += 1
        global_step += 1

        if global_step % 5 == 0:
            print(f"\rstep [{global_step}/{args.max_steps}] F1={current_f1:.3f} "
                  f"main={float(main_loss.detach()):.3f} kd={float(kd_loss.detach()):.4f} "
                  f"lr={last_lr:.7f}", end="", flush=True)

    if batches == 0:
        raise RuntimeError("No training batches")
    print(flush=True)
    averages = {k: v / batches for k, v in totals.items()}
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
        meter.update_cm((predictions[0] > 0.5).long().cpu().numpy(), target.cpu().numpy())
        losses.append(float(loss))
    return sum(losses) / max(len(losses), 1), meter.get_scores()


@torch.no_grad()
def temporal_swap_consistency(model, loader, device) -> float:
    model.eval()
    image, _ = unpack_batch(next(iter(loader)))
    pre = image[:1, :3].to(device)
    post = image[:1, 3:6].to(device)
    out_ab = model(pre, post)
    out_ba = model(post, pre)
    return max((a - b).abs().max().item() for a, b in zip(out_ab, out_ba))


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
    return max((a - b).abs().max().item() for a, b in zip(before, after))


@torch.no_grad()
def test_deployed(args, loader, model, device):
    model.eval()
    joint_bn = model.joint_temporal_bn
    model.joint_temporal_bn = False  # deploy graph uses separate temporal forward
    meter = ConfuseMatrixMeter(n_class=2)
    for batch in loader:
        image, target = unpack_batch(batch)
        output = model(image[:, :3].to(device, non_blocking=True),
                       image[:, 3:6].to(device, non_blocking=True))[0]
        meter.update_cm((output > 0.5).long().cpu().numpy(), target.cpu().numpy())

    infer_params = sum(p.numel() for p in model.parameters())
    flops = None
    if profile is not None:
        dummy1 = torch.randn(1, 3, args.inHeight, args.inWidth, device=device)
        dummy2 = torch.randn_like(dummy1)
        flops, _ = profile(model, inputs=(dummy1, dummy2), verbose=False)
    model.joint_temporal_bn = joint_bn
    return meter.get_scores(), infer_params, flops


def append_test_results(logger, args, scores, train_params, trainable_params, infer_params, flops,
                        swap_error, deploy_error, elapsed):
    logger.log_message("=" * 100)
    logger.log_message("=== TEST RESULTS ===")
    logger.log_message("Metric Split: test")
    logger.log_message(f"Dataset: {args.dataset_name}")
    logger.log_message(f"Experiment: {args.experiment}/{args.experiment_name}")
    logger.log_message(f"Implementation: {args.implementation_version}")
    logger.log_message(f"Joint Temporal BN: {args.joint_bn}")
    logger.log_message(f"Teacher KD: {args.use_teacher}")
    logger.log_message(f"Failure Weighting: {args.failure_weight}")
    logger.log_message(f"Seed: {args.seed}")
    logger.log_message(f"Max Steps: {args.max_steps}")
    logger.log_message(f"Batch Size: {args.batch_size}")
    logger.log_message(f"Train Params: {train_params / 1e6:.4f}M")
    logger.log_message(f"Trainable Params: {trainable_params / 1e6:.4f}M")
    logger.log_message(f"Infer Params: {infer_params / 1e6:.4f}M")
    logger.log_message(f"FLOPs: {flops / 1e9:.4f}G" if flops is not None else "FLOPs: unavailable")
    logger.log_message(f"Temporal swap max error: {swap_error:.8e}")
    logger.log_message(f"Deploy max error: {deploy_error:.8e}")
    for key, label in {"recall": "Recall", "precision": "Precision", "OA": "OA",
                       "F1": "F1", "IoU": "IoU", "Kappa": "Kappa"}.items():
        logger.log_message(f"{label}: {scores[key]:.6f}")
    logger.log_message(f"Total time: {elapsed}")
    logger.log_message("=== END TEST RESULTS ===")


def main() -> None:
    args = parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu")
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args.data_fingerprint = _split_fingerprints(args.data_root)

    model = build_model(args).to(device)
    train_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if train_params != EXPECTED_DEPLOY_PARAMS:
        raise RuntimeError(f"Unexpected model params: {train_params:,} != {EXPECTED_DEPLOY_PARAMS:,}")

    optimizer = build_optimizer(args, model)
    criterion = build_loss(args.dice_reduction)

    if args.use_teacher:
        reader = TeacherCacheReader(args.cache_root, args.dataset_name)
        base_dataset = get_loader(
            args.data_root, os.path.join(args.data_root, "list", "train.txt"),
            batchsize=1, trainsize=args.inWidth, num_workers=args.num_workers,
            return_state=True, seed=args.seed,
        ).dataset
        train_dataset = CacheAwareDataset(base_dataset, reader, "train")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            generator=torch.Generator().manual_seed(args.seed), persistent_workers=False,
        )
    else:
        train_loader = get_loader(
            args.data_root, os.path.join(args.data_root, "list", "train.txt"),
            batchsize=args.batch_size, trainsize=args.inWidth, num_workers=args.num_workers,
            seed=args.seed,
        )

    val_loader = get_test_loader(
        args.data_root, os.path.join(args.data_root, "list", "val.txt"),
        batchsize=args.batch_size, testsize=args.inWidth, num_workers=args.num_workers, return_meta=False,
    )
    test_loader = get_test_loader(
        args.data_root, os.path.join(args.data_root, "list", "test.txt"),
        batchsize=args.batch_size, testsize=args.inWidth, num_workers=args.num_workers, return_meta=False,
    )

    args.max_epochs = int(np.ceil(args.max_steps / len(train_loader)))
    start_epoch, global_step, best_val_f1 = 0, 0, -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
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
        raise FileExistsError("Existing run found; use --resume or a fresh directory")

    logger = TrainingLogger(str(log_path), {
        **vars(args),
        "train_params": f"{train_params / 1e6:.4f}M",
        "trainable_params": f"{trainable_params / 1e6:.4f}M",
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
    }, append=bool(args.resume))

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
            args, train_loader, model, criterion, optimizer, epoch, global_step, device)
        val_loss, val_scores = evaluate(val_loader, model, criterion, args.main_loss_weights, device)

        is_best = val_scores["F1"] > best_val_f1
        if is_best:
            best_val_f1 = float(val_scores["F1"])

        checkpoint = build_checkpoint(model, optimizer, epoch, global_step, best_val_f1, args)
        save_checkpoint_atomic(checkpoint, last_path)
        if is_best:
            new_best = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
            save_checkpoint_atomic(checkpoint, new_best)
            if best_path is not None and best_path != new_best and best_path.name.startswith("best_model_F1="):
                best_path.unlink(missing_ok=True)
            best_path = new_best

        logger.log_epoch(epoch, args.max_epochs, train_losses,
                         {"f1": val_scores["F1"], "iou": val_scores["IoU"], "kappa": val_scores["Kappa"],
                          "recall": val_scores["recall"], "precision": val_scores["precision"],
                          "oa": val_scores["OA"]},
                         current_lr,
                         torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
                         is_best)
        logger.log_message(f"Val loss: {val_loss:.6f}; global_step: {global_step}")
        if global_step >= args.max_steps:
            break

    if best_path is None or not best_path.is_file():
        raise RuntimeError("Training finished without a validation-selected best checkpoint")

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)

    swap_error = temporal_swap_consistency(model, test_loader, device)
    deploy_error = deploy_consistency(model, test_loader, device)
    test_scores, infer_params, flops = test_deployed(args, test_loader, model, device)

    if infer_params != EXPECTED_DEPLOY_PARAMS:
        raise RuntimeError(f"Unexpected deploy params: {infer_params:,} != {EXPECTED_DEPLOY_PARAMS:,}")
    if flops is not None and abs(flops - EXPECTED_DEPLOY_FLOPS) > DEPLOY_FLOPS_ATOL:
        raise RuntimeError(f"Unexpected FLOPs: {flops / 1e9:.6f}G (expected ~{EXPECTED_DEPLOY_FLOPS / 1e9:.6f}G)")

    append_test_results(logger, args, test_scores, train_params, trainable_params, infer_params, flops,
                        swap_error, deploy_error, datetime.datetime.now() - started)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
