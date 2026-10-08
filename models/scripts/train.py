"""CATA-CD v2 trainer.

Single-student change detector with an optional Deployable Change Adapter (DCA)
and an optional single Teacher Package (compact cache v2 + dense-change KD).

Arms:
    C0           clean A2Net-LWGANet-L0 (dca_mode=none, no teacher)
    C1           C0 + DCA (dca_mode=moe128, no teacher)
    TV-*         C1 + one Teacher Package (dca_mode=moe128, teacher_package=<id>)

Teacher utility is reported relative to the C1 anchor. Teacher/cache/aux-head are
training-only; deploy params stay <5M (C0=2,913,094; C1=3,280,274).
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
from models.a2net import SUPPORTED_DCA_MODES  # noqa: E402
from models.datasets.cd_dataset import get_loader, get_test_loader  # noqa: E402
from models.distill.cache_v2 import TeacherCacheReaderV2, apply_cache_state  # noqa: E402
from models.distill.kd import ChangeEvidenceHead, dense_change_kd, gradient_budget_lambda  # noqa: E402
from models.distill.task_reliable_kd import (  # noqa: E402
    AUX_TASKS,
    TaskReliableAuxHead,
    TaskReliableConfig,
    aux_task_loss,
)
from models.distill.teacher_package import TEACHER_REGISTRY  # noqa: E402
from models.utils.metrics import ConfuseMatrixMeter  # noqa: E402
from models.utils.scheduler import adjust_learning_rate  # noqa: E402

EXPECTED_C0_PARAMS = 2_913_094
EXPECTED_C1_PARAMS = 3_280_274
PARAM_CAP = 5_000_000
SWAP_ATOL = 1e-6
DEPLOY_ATOL = 1e-6

CHECKPOINT_FORMAT_VERSION = 8
IMPLEMENTATION_VERSION = "cata_cd_v2"

TEACHER_CHOICES = ["none"] + list(TEACHER_REGISTRY.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CATA-CD v2 trainer")
    p.add_argument("--experiment_id", required=True, help="e.g. C0 / C1 / TV-SAM")
    p.add_argument("--dca_mode", default="none", choices=SUPPORTED_DCA_MODES)
    p.add_argument("--teacher_package", default="none", choices=TEACHER_CHOICES)
    p.add_argument("--dataset_name", required=True, choices=("CDD", "LEVIR", "SYSU", "WHU"))
    p.add_argument("--data_root", required=True)
    p.add_argument("--teacher_cache_root", default=None)

    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=2333)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pretrained_path", default=None)

    p.add_argument("--inWidth", type=int, default=256)
    p.add_argument("--inHeight", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=40000)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr_mode", default="poly", choices=("poly", "step"))
    p.add_argument("--step_loss", type=int, default=30)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dice_reduction", default="batch", choices=("sample", "batch"))
    p.add_argument("--main_loss_weights", default="1,1,1,1")

    p.add_argument("--kd_cap", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--lambda_max", type=float, default=1.0)

    # J1 mechanism arms (review §4.3). Default "none" reproduces the historical
    # C0/C1/TV-* protocol byte-for-byte: dense 128ch cosine KD when a teacher is set.
    p.add_argument("--aux_task", default="none", choices=AUX_TASKS,
                   help="none = legacy dense-change KD (or no aux at all); "
                        "gt = 1ch GT-only aux; gate = GT target on the teacher-consistent "
                        "region G; taskkd = teacher soft response q_T on the same region G")
    p.add_argument("--taskkd_calibration", default=None,
                   help="frozen train-only calibration JSON for --aux_task gate/taskkd")

    p.add_argument("--save_dir", required=True)
    p.add_argument("--log_file", default="train_log.txt")
    p.add_argument("--resume", default=None)

    args = p.parse_args()
    args.main_loss_weights = tuple(float(v) for v in args.main_loss_weights.split(","))
    if len(args.main_loss_weights) != 4:
        raise ValueError("main_loss_weights must contain four values")
    args.use_teacher = args.teacher_package != "none"
    # Derived protocol flags. ``use_dense_kd`` is exactly the legacy behaviour.
    args.use_dense_kd = (args.aux_task == "none") and args.use_teacher
    args.use_taskkd = args.aux_task != "none"
    # J1 requires the cache for gate/taskkd (the teacher signal) and, for gt, so that
    # GT/GATE/TASKKD share ONE data stream -- otherwise the "same pixels" control
    # would be confounded by a different augmentation RNG stream.
    args.load_cache = args.use_dense_kd or args.use_taskkd
    args.implementation_version = IMPLEMENTATION_VERSION
    if args.load_cache and not args.teacher_cache_root:
        raise ValueError("--teacher_cache_root required when a teacher package or --aux_task is used")
    if args.use_taskkd and args.teacher_package == "none":
        raise ValueError("--aux_task requires --teacher_package (cache is needed for loader parity)")
    if args.aux_task in ("gate", "taskkd") and not args.taskkd_calibration:
        raise ValueError(f"--aux_task {args.aux_task} requires --taskkd_calibration")
    if args.pretrained and not args.pretrained_path:
        raise ValueError("--pretrained_path required when --pretrained")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class TrainingLogger:
    def __init__(self, log_file, config, append=False):
        self.log_file = str(log_file)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(self.log_file, mode, encoding="utf-8") as h:
            if append:
                h.write("\n" + "=" * 100 + "\nRUN RESUME\n" + "=" * 100 + "\n")
            else:
                h.write("=" * 100 + "\nCATA-CD v2 Log\n" + "=" * 100 + "\n")
            for k, v in dict(config).items():
                h.write(f"{k}: {str(v)}\n")
            h.write("-" * 100 + "\n")
            h.flush()

    def log_message(self, m):
        t = str(m).rstrip("\n")
        print(t, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as h:
            h.write(t + "\n")
            h.flush()

    def log_epoch(self, epoch, total, losses, val, lr, gpu, best):
        ls = " ".join(f"{k}={float(v):.6f}" for k, v in losses.items())
        line = (f"Epoch [{epoch}/{total}] {ls} F1={float(val['f1']):.6f} IoU={float(val['iou']):.6f} "
                f"Recall={float(val['recall']):.6f} Precision={float(val['precision']):.6f} "
                f"OA={float(val['oa']):.6f} Kappa={float(val['kappa']):.6f} LR={float(lr):.8f} GPU={float(gpu):.3f}GB")
        if best:
            line += " BEST"
        self.log_message(line)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(s):
    if s is None:
        return
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None:
        torch.cuda.set_rng_state_all(s["cuda"])


def save_ckpt_atomic(ckpt, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def build_ckpt(model, aux, optimizer, epoch, gs, best, args):
    return {"format_version": CHECKPOINT_FORMAT_VERSION, "model": model.state_dict(),
            "aux": aux.state_dict() if aux is not None else None, "optimizer": optimizer.state_dict(),
            "epoch": epoch, "global_step": gs, "best_val_f1": best,
            "args": dict(vars(args)), "rng": capture_rng()}


def _fingerprints(data_root):
    root = Path(data_root)
    return {s: hashlib.sha256((root / "list" / f"{s}.txt").read_bytes()).hexdigest() for s in ("train", "val", "test")}


def unpack(batch):
    return batch[0], batch[1]


class CataCacheDataset(torch.utils.data.Dataset):
    def __init__(self, base, reader, split):
        self.base = base
        self.reader = reader
        self.split = split

    def __len__(self):
        return len(self.base)

    def set_epoch(self, e):
        self.base.set_epoch(e)

    def __getitem__(self, idx):
        image, label, sample_id, state = self.base[idx]
        cache = apply_cache_state(self.reader.load(self.split, sample_id), state)
        return (image, label, cache["local_change"].float(), cache["confidence"].float())


def build_model(args):
    return A2Net_LWGANet_L0(pretrained=args.pretrained, pretrained_path=args.pretrained_path,
                            dca_mode=args.dca_mode)


def build_optimizer(args, params):
    return torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=args.weight_decay)


def multiscale_loss(preds, target, crit, ws):
    return sum(w * crit(p, target) for w, p in zip(ws, preds))


def train_epoch(args, loader, model, aux, criterion, optimizer, epoch, gs, device,
                taskcfg: TaskReliableConfig | None = None):
    model.train()
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    if getattr(loader, "generator", None) is not None:
        loader.generator.manual_seed(args.seed + epoch)

    meter = ConfuseMatrixMeter(n_class=2)
    totals = {"total": 0.0, "main": 0.0, "kd": 0.0, "lam": 0.0}
    batches = 0
    last_lr = args.lr
    load_cache = args.load_cache
    aux_skipped = 0
    region_acc: dict[str, float] = {}
    region_n = 0
    if aux is not None:
        aux.train()

    for batch in loader:
        if gs >= args.max_steps:
            break
        if load_cache:
            image, target = batch[0], batch[1]
            local_change = batch[2].to(device, non_blocking=True)
            confidence = batch[3].to(device, non_blocking=True)
        else:
            image, target = unpack(batch)

        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        last_lr = adjust_learning_rate(args, optimizer, epoch, gs, len(loader))
        optimizer.zero_grad(set_to_none=True)

        predictions, change = model(pre, post, return_change_features=True)
        main_loss = multiscale_loss(predictions, target, criterion, args.main_loss_weights)

        kd_loss = torch.zeros((), device=device)
        lam = torch.zeros((), device=device)
        if args.use_dense_kd:
            student_evidence = aux(change)  # [B,128,h,w] at scale 2 (stride 16)
            kd_loss = dense_change_kd(student_evidence, local_change, confidence, cap=args.kd_cap)
            lam = gradient_budget_lambda(main_loss, kd_loss, change[2], rho=args.rho, lam_max=args.lambda_max)
        elif args.use_taskkd:
            # J1: 1-channel head on c2 (stride 4). Same head/capacity for gt/gate/taskkd;
            # only the supervised region and/or the target differ.
            logits = aux(change)  # [B,1,h,w]
            task_loss, _stats = aux_task_loss(args.aux_task, logits, target, taskcfg, confidence)
            if task_loss is None:
                # Empty region: skip the auxiliary term entirely, main loss continues.
                aux_skipped += 1
            else:
                kd_loss = task_loss
                lam = gradient_budget_lambda(main_loss, kd_loss, change[0],
                                             rho=args.rho, lam_max=args.lambda_max)
                for _k, _v in _stats.items():
                    if isinstance(_v, (int, float)) and _k != "skipped":
                        region_acc[_k] = region_acc.get(_k, 0.0) + float(_v)
                region_n += 1

        total = main_loss + lam * kd_loss
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Non-finite loss")
        total.backward()
        optimizer.step()

        hard = (predictions[0].detach() > 0.5).long()
        f1 = meter.update_cm(hard.cpu().numpy(), target.cpu().numpy())
        totals["total"] += float(total.detach())
        totals["main"] += float(main_loss.detach())
        totals["kd"] += float(kd_loss.detach())
        totals["lam"] += float(lam.detach())
        batches += 1
        gs += 1
        if gs % 5 == 0:
            print(f"\rstep [{gs}/{args.max_steps}] F1={f1:.3f} main={float(main_loss.detach()):.3f} "
                  f"kd={float(kd_loss.detach()):.4f} lam={float(lam.detach()):.3f}", end="", flush=True)

    if batches == 0:
        raise RuntimeError("No batches")
    print(flush=True)
    aux_info = {"aux_skipped": aux_skipped, "n_batches": batches, "region_batches": region_n}
    for k, v in region_acc.items():
        aux_info[k] = v / max(region_n, 1)
    return {k: v / batches for k, v in totals.items()}, meter.get_scores(), last_lr, gs, aux_info


@torch.no_grad()
def evaluate(loader, model, criterion, ws, device):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    for batch in loader:
        image, target = unpack(batch)
        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        preds = model(pre, post)
        losses.append(float(multiscale_loss(preds, target, criterion, ws)))
        meter.update_cm((preds[0] > 0.5).long().cpu().numpy(), target.cpu().numpy())
    return sum(losses) / max(len(losses), 1), meter.get_scores()


@torch.no_grad()
def swap_consistency(model, loader, device):
    model.eval()
    image, _ = unpack(next(iter(loader)))
    pre = image[:1, :3].to(device)
    post = image[:1, 3:6].to(device)
    ab = model(pre, post)
    ba = model(post, pre)
    return max((a - b).abs().max().item() for a, b in zip(ab, ba))


@torch.no_grad()
def deploy_consistency(model, loader, device):
    model.eval()
    image, _ = unpack(next(iter(loader)))
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
    meter = ConfuseMatrixMeter(n_class=2)
    for batch in loader:
        image, target = unpack(batch)
        out = model(image[:, :3].to(device, non_blocking=True), image[:, 3:6].to(device, non_blocking=True))[0]
        meter.update_cm((out > 0.5).long().cpu().numpy(), target.cpu().numpy())
    infer_params = sum(p.numel() for p in model.parameters())
    flops = None
    if profile is not None:
        d1 = torch.randn(1, 3, args.inHeight, args.inWidth, device=device)
        flops, _ = profile(model, inputs=(d1, d1), verbose=False)
    return meter.get_scores(), infer_params, flops


def append_test(logger, args, scores, param_stats, infer_params, flops, swap_err, deploy_err, elapsed):
    logger.log_message("=" * 100)
    logger.log_message("=== TEST RESULTS ===")
    logger.log_message("Metric Split: test")
    logger.log_message(f"Dataset: {args.dataset_name}")
    logger.log_message(f"Experiment: {args.experiment_id}")
    logger.log_message(f"Implementation: {args.implementation_version}")
    logger.log_message(f"DCA Mode: {args.dca_mode}")
    logger.log_message(f"Teacher Package: {args.teacher_package}")
    logger.log_message(f"Aux Task: {args.aux_task}")
    logger.log_message(f"Seed: {args.seed}")
    logger.log_message(f"Max Steps: {args.max_steps}")
    logger.log_message(f"Batch Size: {args.batch_size}")
    # Parameter accounting (review P0): the KD aux head is training-only and was
    # previously folded into "Train Params"; report student / aux / total / deploy.
    logger.log_message(f"Student Params: {param_stats['student'] / 1e6:.4f}M")
    logger.log_message(f"Aux Params: {param_stats['aux'] / 1e6:.4f}M")
    logger.log_message(f"Train Params: {param_stats['training_total'] / 1e6:.4f}M")
    logger.log_message(f"Infer Params: {infer_params / 1e6:.4f}M")
    logger.log_message(f"FLOPs: {flops / 1e9:.4f}G" if flops is not None else "FLOPs: unavailable")
    logger.log_message(f"Temporal swap max error: {swap_err:.8e}")
    logger.log_message(f"Deploy max error: {deploy_err:.8e}")
    for k, lab in {"recall": "Recall", "precision": "Precision", "OA": "OA", "F1": "F1", "IoU": "IoU", "Kappa": "Kappa"}.items():
        logger.log_message(f"{lab}: {scores[k]:.6f}")
    logger.log_message(f"Total time: {elapsed}")
    logger.log_message("=== END TEST RESULTS ===")


def main():
    args = parse_args()
    if args.device == "cuda":
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu")
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args.data_fingerprint = _fingerprints(args.data_root)

    model = build_model(args).to(device)
    student_params = sum(p.numel() for p in model.parameters())
    # C0/C1 have frozen expected counts so any accidental architecture drift is
    # caught. The S-PCG gate variants (S1/S2) legitimately change the gate head, so
    # for those we enforce only the deploy budget and report the measured count.
    expected = {"none": EXPECTED_C0_PARAMS, "moe128": EXPECTED_C1_PARAMS}.get(args.dca_mode)
    if expected is not None and student_params != expected:
        raise RuntimeError(f"student params {student_params:,} != expected {expected:,}")
    if student_params >= PARAM_CAP:
        raise RuntimeError(f"student params {student_params:,} >= cap {PARAM_CAP:,}")

    # Auxiliary head: legacy 128ch evidence head for dense KD, or the 1ch (65-param)
    # task-reliable head for the J1 arms. Both are training-only.
    if args.use_dense_kd:
        aux = ChangeEvidenceHead(64, 128, scale_index=2).to(device)
    elif args.use_taskkd:
        aux = TaskReliableAuxHead(64, scale_index=0).to(device)
    else:
        aux = None
    aux_params = sum(p.numel() for p in aux.parameters()) if aux is not None else 0
    taskcfg = None
    if args.aux_task in ("gate", "taskkd"):
        taskcfg = TaskReliableConfig.from_json(args.taskkd_calibration)
        if taskcfg.dataset.upper() != args.dataset_name.upper():
            raise ValueError(f"calibration dataset {taskcfg.dataset} != {args.dataset_name}")
        if taskcfg.teacher_package != args.teacher_package:
            raise ValueError(f"calibration teacher {taskcfg.teacher_package} != {args.teacher_package}")
    param_stats = {
        "student": student_params,
        "aux": aux_params,
        "training_total": student_params + aux_params,
    }
    params = list(model.parameters()) + (list(aux.parameters()) if aux is not None else [])
    optimizer = build_optimizer(args, params)
    criterion = build_loss(args.dice_reduction)

    if args.load_cache:
        reader = TeacherCacheReaderV2(args.teacher_cache_root, args.dataset_name)
        base = get_loader(args.data_root, os.path.join(args.data_root, "list", "train.txt"),
                          batchsize=1, trainsize=args.inWidth, num_workers=args.num_workers,
                          return_state=True, seed=args.seed).dataset
        train_dataset = CataCacheDataset(base, reader, "train")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                                   generator=torch.Generator().manual_seed(args.seed), persistent_workers=False)
    else:
        train_loader = get_loader(args.data_root, os.path.join(args.data_root, "list", "train.txt"),
                                  batchsize=args.batch_size, trainsize=args.inWidth, num_workers=args.num_workers, seed=args.seed)

    val_loader = get_test_loader(args.data_root, os.path.join(args.data_root, "list", "val.txt"), batchsize=args.batch_size, testsize=args.inWidth, num_workers=args.num_workers)
    test_loader = get_test_loader(args.data_root, os.path.join(args.data_root, "list", "test.txt"), batchsize=args.batch_size, testsize=args.inWidth, num_workers=args.num_workers)

    args.max_epochs = int(np.ceil(args.max_steps / len(train_loader)))
    start_epoch, gs, best_val_f1 = 0, 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("format mismatch")
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        if aux is not None and ck.get("aux"):
            aux.load_state_dict(ck["aux"])
        start_epoch = int(ck["epoch"]) + 1
        gs = int(ck["global_step"])
        best_val_f1 = float(ck["best_val_f1"])
        restore_rng(ck.get("rng"))

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = save_dir / log_path
    last_path = save_dir / "last_checkpoint.pth"
    if not args.resume and (log_path.exists() or last_path.exists()):
        raise FileExistsError("existing run")
    logger = TrainingLogger(str(log_path), {**vars(args),
                                            "student_params": f"{param_stats['student']/1e6:.4f}M",
                                            "aux_params": f"{param_stats['aux']/1e6:.4f}M",
                                            "training_total_params": f"{param_stats['training_total']/1e6:.4f}M",
                                            "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset), "n_test": len(test_loader.dataset)}, append=bool(args.resume))

    best_path = None
    started = datetime.datetime.now()
    for epoch in range(start_epoch, args.max_epochs):
        if gs >= args.max_steps:
            break
        train_losses, _, lr, gs, aux_info = train_epoch(args, train_loader, model, aux,
                                                        criterion, optimizer, epoch, gs,
                                                        device, taskcfg)
        val_loss, val = evaluate(val_loader, model, criterion, args.main_loss_weights, device)
        is_best = val["F1"] > best_val_f1
        if is_best:
            best_val_f1 = float(val["F1"])
        ck = build_ckpt(model, aux, optimizer, epoch, gs, best_val_f1, args)
        save_ckpt_atomic(ck, last_path)
        if is_best:
            nb = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
            save_ckpt_atomic(ck, nb)
            if best_path is not None and best_path != nb and best_path.name.startswith("best_model_F1="):
                best_path.unlink(missing_ok=True)
            best_path = nb
        logger.log_epoch(epoch, args.max_epochs, train_losses, {"f1": val["F1"], "iou": val["IoU"], "kappa": val["Kappa"], "recall": val["recall"], "precision": val["precision"], "oa": val["OA"]}, lr, torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0, is_best)
        logger.log_message(f"Val loss: {val_loss:.6f}; global_step: {gs}")
        if args.use_taskkd:
            logger.log_message(
                "Aux region: task={} skipped={}/{} kept={} ".format(
                    args.aux_task, aux_info["aux_skipped"], aux_info["n_batches"],
                    aux_info["region_batches"])
                + " ".join(f"{k}={aux_info[k]:.5f}" for k in
                           ("n_pos", "n_neg", "g_pos_frac", "g_neg_frac", "band_frac")
                           if k in aux_info))
        if gs >= args.max_steps:
            break

    if best_path is None or not best_path.is_file():
        raise RuntimeError("no best checkpoint")
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)

    swap_err = swap_consistency(model, test_loader, device)
    deploy_err = deploy_consistency(model, test_loader, device)
    scores, infer_params, flops = test_deployed(args, test_loader, model, device)
    if swap_err >= SWAP_ATOL:
        raise RuntimeError(f"swap error {swap_err:.2e}")
    if deploy_err >= DEPLOY_ATOL:
        raise RuntimeError(f"deploy error {deploy_err:.2e}")
    if expected is not None and infer_params != expected:
        raise RuntimeError(f"deploy params {infer_params:,} != expected {expected:,}")
    if infer_params >= PARAM_CAP:
        raise RuntimeError(f"deploy params {infer_params:,} >= cap {PARAM_CAP:,}")
    append_test(logger, args, scores, param_stats, infer_params, flops, swap_err, deploy_err, datetime.datetime.now() - started)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
