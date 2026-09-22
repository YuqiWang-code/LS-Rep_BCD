#!/usr/bin/env python3
"""Fine-tune the FA-SCRD teacher (DINOv3 ViT-B/16 + LoRA + change head) on CD.

This produces a CD-adapted, fixed teacher. Only the LoRA adapter and the
lightweight change head are trainable; the DINOv3 backbone stays frozen.

Usage:
    python -m models.tools.train_teacher \
        --weight_path <dinov3_vitb16.pth> \
        --data_root /share_datasets/CD/WHU-CD-256 \
        --dataset_name WHU \
        --save_dir <teacher_ckpt_dir> \
        --max_steps 5000 --batch_size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.fa_scrd_teacher import FASCRDTeacher  # noqa: E402
from models.losses.combined_loss import BCEDiceLoss  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FA-SCRD teacher on CD")
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2333)
    return parser.parse_args()


def _read_rgb(path: Path, size: int = 256) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Failed to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


def _imagenet_norm(x: np.ndarray) -> np.ndarray:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def load_sample(data_root: Path, sample_id: str, size: int = 256):
    # sample_id already carries its extension (e.g. "00000.png").
    pre = _read_rgb(data_root / "A" / sample_id, size)
    post = _read_rgb(data_root / "B" / sample_id, size)
    label = cv2.imread(str(data_root / "label" / sample_id), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise OSError(f"Failed to read label for {sample_id}")
    label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
    label = (label >= 128).astype(np.float32)

    xa = torch.from_numpy(_imagenet_norm(pre).transpose(2, 0, 1)).float()
    xb = torch.from_numpy(_imagenet_norm(post).transpose(2, 0, 1)).float()
    y = torch.from_numpy(label).float().unsqueeze(0)
    return xa, xb, y


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    data_root = Path(args.data_root)
    sample_ids = [
        line.strip()
        for line in (data_root / "list" / "train.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    teacher = FASCRDTeacher(args.weight_path).to(device)
    print(f"[teacher] trainable params: {teacher.trainable_param_count}")

    optimizer = torch.optim.AdamW(
        [p for p in teacher.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    criterion = BCEDiceLoss(dice_reduction="batch")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    teacher.train()
    global_step = 0
    sample_idx = 0
    while global_step < args.max_steps:
        batch_xa, batch_xb, batch_y = [], [], []
        for _ in range(args.batch_size):
            xa, xb, y = load_sample(data_root, sample_ids[sample_idx % len(sample_ids)])
            batch_xa.append(xa)
            batch_xb.append(xb)
            batch_y.append(y)
            sample_idx += 1

        xa = torch.stack(batch_xa).to(device)
        xb = torch.stack(batch_xb).to(device)
        y = torch.stack(batch_y).to(device)

        optimizer.zero_grad(set_to_none=True)
        logit = teacher.forward_change_logit(xa, xb)
        pred = torch.sigmoid(logit)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        global_step += 1
        if global_step % 50 == 0 or global_step == 1:
            print(f"step {global_step}/{args.max_steps} loss={loss.item():.4f}", flush=True)

    checkpoint = {
        "format_version": 1,
        "model": teacher.state_dict(),
        "weight_path": args.weight_path,
        "dataset_name": args.dataset_name,
    }
    out = save_dir / "teacher_checkpoint.pth"
    torch.save(checkpoint, out)
    print(f"[teacher] saved to {out}")


if __name__ == "__main__":
    main()
