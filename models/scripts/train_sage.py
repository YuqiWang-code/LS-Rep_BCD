"""SAGE-CD trainer (first gate: G0 / G1 / G2 Safe-DINO).

    G0: clean A2Net (no teacher, no joint BN)   — clean anchor
    G1: G0 + joint temporal BN                   — normalization control
    G2: G1 + full-res DINO semantic target + train-only semantic head
             + logit standardization + gradient budget  (Safe-DINO)

The SAM2 structural expert and soft router are the SECOND stage (only after G2
passes the gate). Deployed student stays 2,913,094 params / ~2.77G FLOPs.
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
from typing import Any, Dict, Optional

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
from models.distill.cache_dataset import TeacherCacheReader, apply_cache_state  # noqa: E402
from models.distill.sage_heads import AuxSemanticHead  # noqa: E402
from models.distill.sage_kd import grad_norm, gradient_budget_lambda, semantic_kd_loss  # noqa: E402
from models.utils.metrics import ConfuseMatrixMeter  # noqa: E402
from models.utils.scheduler import adjust_learning_rate  # noqa: E402

EXPECTED_DEPLOY_PARAMS = 2_913_094
EXPECTED_DEPLOY_FLOPS = 2.7475e9
DEPLOY_FLOPS_ATOL = 0.03e9
SWAP_ATOL = 1e-6
DEPLOY_ATOL = 1e-6

CHECKPOINT_FORMAT_VERSION = 6
IMPLEMENTATION_VERSION = "sage_cd_v1"

EXPERIMENTS = {
    "G0": {"name": "G0_Clean", "joint_bn": False, "teacher": False},
    "G1": {"name": "G1_JointBN", "joint_bn": True, "teacher": False},
    "G2": {"name": "G2_SafeDINO", "joint_bn": True, "teacher": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAGE-CD trainer: G0/G1/G2")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--dataset_name", required=True, choices=("CDD", "LEVIR", "SYSU", "WHU"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cache_root", default=None, help="SAGE DINO semantic cache root (G2)")

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

    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--lambda_max", type=float, default=1.0)

    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_file", default="train_log.txt")
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()
    args.main_loss_weights = tuple(float(v) for v in args.main_loss_weights.split(","))
    if len(args.main_loss_weights) != 4:
        raise ValueError("main_loss_weights must contain four values")

    recipe = EXPERIMENTS[args.experiment]
    args.experiment_name = recipe["name"]
    args.joint_bn = recipe["joint_bn"]
    args.use_teacher = recipe["teacher"]
    args.implementation_version = IMPLEMENTATION_VERSION

    if args.use_teacher and not args.cache_root:
        raise ValueError("--cache_root is required for G2")
    if args.pretrained and not args.pretrained_path:
        raise ValueError("--pretrained_path is required when --pretrained")
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
    def __init__(self, log_file: str, config: Dict[str, Any], append: bool = False) -> None:
        self.log_file = str(log_file)
        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as h:
            if append:
                h.write("\n" + "=" * 100 + "\nRUN RESUME\n" + "=" * 100 + "\n")
            else:
                h.write("=" * 100 + "\nA2Net-LWGANet-L0 SAGE-CD Training Log\n" + "=" * 100 + "\n")
            for k, v in dict(config).items():
                value_text = str(v).replace("\n", "\\n")
                h.write(f"{k}: {value_text}\n")
            h.write("-" * 100 + "\n")
            h.flush()

    def log_message(self, message) -> None:
        text = str(message).rstrip("\n")
        print(text, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as h:
            h.write(text + "\n")
            h.flush()

    def log_epoch(self, epoch, total_epochs, train_losses, val_metrics, lr, gpu_mem, is_best) -> None:
        losses = " ".join(f"{k}={float(v):.6f}" for k, v in train_losses.items())
        line = (f"Epoch [{epoch}/{total_epochs}] {losses} "
                f"F1={float(val_metrics['f1']):.6f} IoU={float(val_metrics['iou']):.6f} "
                f"Recall={float(val_metrics['recall']):.6f} Precision={float(val_metrics['precision']):.6f} "
                f"OA={float(val_metrics['oa']):.6f} Kappa={float(val_metrics['kappa']):.6f} "
                f"LR={float(lr):.8f} GPU={float(gpu_mem):.3f}GB")
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


def restore_rng_state(state) -> None:
    if state is None:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint_atomic(checkpoint: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def build_checkpoint(model, aux_head, optimizer, aux_optimizer, epoch, global_step, best_val_f1, args) -> Dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "aux_head": aux_head.state_dict() if aux_head is not None else None,
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_f1": float(best_val_f1),
        "args": dict(vars(args)),
        "rng": capture_rng_state(),
    }


def _split_fingerprints(data_root: str) -> Dict[str, str]:
    root = Path(data_root)
    return {s: hashlib.sha256((root / "list" / f"{s}.txt").read_bytes()).hexdigest()
            for s in ("train", "val", "test")}


def unpack_batch(batch):
    return batch[0], batch[1]


class SageCacheDataset(torch.utils.data.Dataset):
    """Replays augmentation onto the full-res DINO semantic target."""

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
        return image, label, cache["dino_change_logit"].float()


def build_model(args) -> A2Net_LWGANet_L0:
    return A2Net_LWGANet_L0(
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        temporal_calibration_mode="none",
        joint_temporal_bn=args.joint_bn,
    )


def build_optimizer(args, params) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=args.weight_decay)


def multiscale_loss(predictions, target, criterion, weights):
    return sum(w * criterion(p, target) for w, p in zip(weights, predictions))


def train_epoch(args, loader, model, aux_head, criterion, optimizer, aux_optimizer, epoch, global_step, device):
    model.train()
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    if getattr(loader, "generator", None) is not None:
        loader.generator.manual_seed(args.seed + epoch)

    meter = ConfuseMatrixMeter(n_class=2)
    totals = {"total": 0.0, "main": 0.0, "kd": 0.0, "lambda_kd": 0.0, "grad_ratio": 0.0}
    batches = 0
    last_lr = args.lr
    use_teacher = args.use_teacher
    if use_teacher:
        aux_head.train()

    for batch in loader:
        if global_step >= args.max_steps:
            break
        if use_teacher:
            image, target = batch[0], batch[1]
            teacher_logit = batch[2].to(device, non_blocking=True)
        else:
            image, target = unpack_batch(batch)

        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        last_lr = adjust_learning_rate(args, optimizer, epoch, global_step, len(loader))
        optimizer.zero_grad(set_to_none=True)
        if aux_optimizer is not None:
            aux_optimizer.zero_grad(set_to_none=True)

        predictions, change = model(pre, post, return_change_features=True)
        main_loss = multiscale_loss(predictions, target, criterion, args.main_loss_weights)

        lambda_kd = torch.zeros((), device=device)
        kd_loss = torch.zeros((), device=device)
        grad_ratio = torch.zeros((), device=device)

        if use_teacher:
            c4 = change[2]  # 1/16 semantic change feature
            aux_logit = aux_head(c4)  # [B,1,256,256]
            kd_loss = semantic_kd_loss(aux_logit, teacher_logit, temperature=args.kd_temperature)
            lambda_kd = gradient_budget_lambda(main_loss, kd_loss, c4, rho=args.rho, lambda_max=args.lambda_max)
            g_main = grad_norm(main_loss, c4)
            g_kd = grad_norm(kd_loss, c4)
            grad_ratio = g_kd / (g_main + 1e-5)

        total = main_loss + lambda_kd * kd_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Non-finite training loss")

        total.backward()
        optimizer.step()
        if aux_optimizer is not None:
            aux_optimizer.step()

        hard_prediction = (predictions[0].detach() > 0.5).long()
        current_f1 = meter.update_cm(hard_prediction.cpu().numpy(), target.cpu().numpy())
        totals["total"] += float(total.detach())
        totals["main"] += float(main_loss.detach())
        totals["kd"] += float(kd_loss.detach())
        totals["lambda_kd"] += float(lambda_kd.detach())
        totals["grad_ratio"] += float(grad_ratio.detach())
        batches += 1
        global_step += 1

        if global_step % 5 == 0:
            print(f"\rstep [{global_step}/{args.max_steps}] F1={current_f1:.3f} "
                  f"main={float(main_loss.detach()):.3f} kd={float(kd_loss.detach()):.4f} "
                  f"lambda_kd={float(lambda_kd.detach()):.3f}", end="", flush=True)

    if batches == 0:
        raise RuntimeError("No training batches")
    print(flush=True)
    return {k: v / batches for k, v in totals.items()}, meter.get_scores(), last_lr, global_step


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
    model.joint_temporal_bn = False
    meter = ConfuseMatrixMeter(n_class=2)
    for batch in loader:
        image, target = unpack_batch(batch)
        output = model(image[:, :3].to(device, non_blocking=True),
                       image[:, 3:6].to(device, non_blocking=True))[0]
        meter.update_cm((output > 0.5).long().cpu().numpy(), target.cpu().numpy())
    infer_params = sum(p.numel() for p in model.parameters())
    flops = None
    if profile is not None:
        d1 = torch.randn(1, 3, args.inHeight, args.inWidth, device=device)
        d2 = torch.randn_like(d1)
        flops, _ = profile(model, inputs=(d1, d2), verbose=False)
    model.joint_temporal_bn = joint_bn
    return meter.get_scores(), infer_params, flops


def append_test_results(logger, args, scores, train_params, infer_params, flops, swap_error, deploy_error, elapsed):
    logger.log_message("=" * 100)
    logger.log_message("=== TEST RESULTS ===")
    logger.log_message("Metric Split: test")
    logger.log_message(f"Dataset: {args.dataset_name}")
    logger.log_message(f"Experiment: {args.experiment}/{args.experiment_name}")
    logger.log_message(f"Implementation: {args.implementation_version}")
    logger.log_message(f"Joint Temporal BN: {args.joint_bn}")
    logger.log_message(f"Teacher KD: {args.use_teacher}")
    logger.log_message(f"Seed: {args.seed}")
    logger.log_message(f"Max Steps: {args.max_steps}")
    logger.log_message(f"Batch Size: {args.batch_size}")
    logger.log_message(f"Train Params: {train_params / 1e6:.4f}M")
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
    if train_params != EXPECTED_DEPLOY_PARAMS:
        raise RuntimeError(f"Unexpected params: {train_params:,}")

    aux_head = AuxSemanticHead().to(device) if args.use_teacher else None

    optimizer = build_optimizer(args, model.parameters())
    aux_optimizer = build_optimizer(args, aux_head.parameters()) if aux_head is not None else None
    criterion = build_loss(args.dice_reduction)

    if args.use_teacher:
        reader = TeacherCacheReader(args.cache_root, args.dataset_name)
        base_dataset = get_loader(
            args.data_root, os.path.join(args.data_root, "list", "train.txt"),
            batchsize=1, trainsize=args.inWidth, num_workers=args.num_workers,
            return_state=True, seed=args.seed,
        ).dataset
        train_dataset = SageCacheDataset(base_dataset, reader, "train")
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

    val_loader = get_test_loader(args.data_root, os.path.join(args.data_root, "list", "val.txt"),
                                 batchsize=args.batch_size, testsize=args.inWidth, num_workers=args.num_workers)
    test_loader = get_test_loader(args.data_root, os.path.join(args.data_root, "list", "test.txt"),
                                  batchsize=args.batch_size, testsize=args.inWidth, num_workers=args.num_workers)

    args.max_epochs = int(np.ceil(args.max_steps / len(train_loader)))
    start_epoch, global_step, best_val_f1 = 0, 0, -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("SAGE-CD requires format_version=6")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if aux_head is not None and ckpt.get("aux_head"):
            aux_head.load_state_dict(ckpt["aux_head"])
        if aux_optimizer is not None and ckpt.get("aux_optimizer"):
            aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        best_val_f1 = float(ckpt["best_val_f1"])
        restore_rng_state(ckpt.get("rng"))

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = save_dir / log_path
    last_path = save_dir / "last_checkpoint.pth"
    if not args.resume and (log_path.exists() or last_path.exists()):
        raise FileExistsError("Existing run found; use --resume or a fresh directory")

    logger = TrainingLogger(str(log_path), {**vars(args),
                                            "train_params": f"{train_params / 1e6:.4f}M",
                                            "n_train": len(train_loader.dataset),
                                            "n_val": len(val_loader.dataset),
                                            "n_test": len(test_loader.dataset)}, append=bool(args.resume))

    best_path = None
    if args.resume:
        cand = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
        if cand.is_file():
            best_path = cand

    started = datetime.datetime.now()
    for epoch in range(start_epoch, args.max_epochs):
        if global_step >= args.max_steps:
            break
        train_losses, _, current_lr, global_step = train_epoch(
            args, train_loader, model, aux_head, criterion, optimizer, aux_optimizer,
            epoch, global_step, device)
        val_loss, val_scores = evaluate(val_loader, model, criterion, args.main_loss_weights, device)

        is_best = val_scores["F1"] > best_val_f1
        if is_best:
            best_val_f1 = float(val_scores["F1"])

        checkpoint = build_checkpoint(model, aux_head, optimizer, aux_optimizer,
                                      epoch, global_step, best_val_f1, args)
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
        raise RuntimeError("Training finished without a best checkpoint")

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    swap_error = temporal_swap_consistency(model, test_loader, device)
    deploy_error = deploy_consistency(model, test_loader, device)
    test_scores, infer_params, flops = test_deployed(args, test_loader, model, device)

    if infer_params != EXPECTED_DEPLOY_PARAMS:
        raise RuntimeError(f"Unexpected deploy params: {infer_params:,}")
    if flops is not None and abs(flops - EXPECTED_DEPLOY_FLOPS) > DEPLOY_FLOPS_ATOL:
        raise RuntimeError(f"Unexpected FLOPs: {flops / 1e9:.6f}G")

    append_test_results(logger, args, test_scores, train_params, infer_params, flops,
                        swap_error, deploy_error, datetime.datetime.now() - started)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
