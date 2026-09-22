#!/usr/bin/env python3
"""Generate offline FA-SCRD teacher cache (bi-temporal mid/deep features + change logit).

For each training sample the CD-adapted teacher produces:
    t1_mid, t2_mid   (block 6)   [768, 16, 16]  fp16
    t1_deep, t2_deep (block 11)  [768, 16, 16]  fp16
    teacher_change_logit          [1, 256, 256]  fp16 (sigmoid probability)
plus a small meta block. Everything is stored under
    <cache_root>/<DATASET>/<split>/<sample_id>.pt

Usage:
    python -m models.tools.generate_teacher_cache \
        --teacher_ckpt <dir>/teacher_checkpoint.pth \
        --data_root /share_datasets/CD/WHU-CD-256 \
        --dataset_name WHU \
        --cache_root /share_datasets/CD_teacher_cache/FA_SCRD_DINOv3_CD \
        --gpu_id 0
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.fa_scrd_teacher import FASCRDTeacher  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FA-SCRD teacher cache")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--split", default="train")
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


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")

    data_root = Path(args.data_root)
    dataset = args.dataset_name.upper()
    split = args.split
    sample_ids = [
        line.strip()
        for line in (data_root / "list" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher = FASCRDTeacher(teacher_ckpt["weight_path"]).to(device)
    teacher.load_state_dict(teacher_ckpt["model"], strict=False)
    teacher.eval()

    out_root = Path(args.cache_root) / dataset / split
    out_root.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = hashlib.sha256(
        str(args.teacher_ckpt).encode()
    ).hexdigest()[:16]

    with torch.no_grad():
        for i, sample_id in enumerate(sample_ids):
            # sample_id already carries its extension (e.g. "00000.png").
            pre = _read_rgb(data_root / "A" / sample_id)
            post = _read_rgb(data_root / "B" / sample_id)

            xa = torch.from_numpy(_imagenet_norm(pre).transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            xb = torch.from_numpy(_imagenet_norm(post).transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            mid_a, deep_a = teacher.extract_features(xa)
            mid_b, deep_b = teacher.extract_features(xb)
            logit = teacher.forward_change_logit(xa, xb)
            change_prob = torch.sigmoid(logit)

            entry = {
                "meta": {
                    "dataset": dataset,
                    "checkpoint_hash": checkpoint_hash,
                    "resolution": 256,
                    "normalization": "imagenet",
                },
                "t1_mid": mid_a.squeeze(0).cpu().half(),
                "t2_mid": mid_b.squeeze(0).cpu().half(),
                "t1_deep": deep_a.squeeze(0).cpu().half(),
                "t2_deep": deep_b.squeeze(0).cpu().half(),
                "teacher_change_logit": change_prob.squeeze(0).cpu().half(),
            }
            torch.save(entry, out_root / f"{sample_id}.pt")

            if (i + 1) % 100 == 0 or i == 0:
                print(f"[cache] {dataset}/{split} {i + 1}/{len(sample_ids)}", flush=True)

    print(f"[cache] done: {len(sample_ids)} samples -> {out_root}")


if __name__ == "__main__":
    main()
