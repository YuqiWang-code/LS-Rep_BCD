#!/usr/bin/env python3
"""Generate full-resolution SAGE-CD DINO semantic teacher cache.

Unlike FA-SCRD (which cached 16x16 Foundation features for relation KD), SAGE-CD
only caches the full-res *task target*:
    dino_change_logit : [1,256,256] fp16  (raw change logit z_D)
    dino_change_prob  : [1,256,256] fp16  (sigmoid probability p_D)
plus a small meta block. This reuses the existing FA-SCRD DINOv3 teacher.

Usage:
    python -m models.tools.generate_sage_cache \
        --teacher_ckpt <dir>/teacher_checkpoint.pth \
        --data_root /share_datasets/CD/SYSU-CD-256 \
        --dataset_name SYSU \
        --cache_root /share_datasets/CD_teacher_cache/SAGE_DINO3_CD \
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
    p = argparse.ArgumentParser(description="Generate SAGE-CD DINO semantic cache")
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--split", default="train")
    return p.parse_args()


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
    ckpt_hash = hashlib.sha256(str(args.teacher_ckpt).encode()).hexdigest()[:16]

    with torch.no_grad():
        for i, sample_id in enumerate(sample_ids):
            pre = _read_rgb(data_root / "A" / sample_id)
            post = _read_rgb(data_root / "B" / sample_id)
            xa = torch.from_numpy(_imagenet_norm(pre).transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            xb = torch.from_numpy(_imagenet_norm(post).transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            logit = teacher.forward_change_logit(xa, xb)  # [1,1,256,256] raw logit
            prob = torch.sigmoid(logit)

            entry = {
                "meta": {
                    "dataset": dataset,
                    "checkpoint_hash": ckpt_hash,
                    "resolution": 256,
                    "normalization": "imagenet",
                    "teacher": "dinov3_vitb16",
                },
                "dino_change_logit": logit.squeeze(0).cpu().half(),
                "dino_change_prob": prob.squeeze(0).cpu().half(),
            }
            torch.save(entry, out_root / f"{sample_id}.pt")

            if (i + 1) % 200 == 0 or i == 0:
                print(f"[sage-cache] {dataset}/{split} {i + 1}/{len(sample_ids)}", flush=True)

    print(f"[sage-cache] done: {len(sample_ids)} samples -> {out_root}")


if __name__ == "__main__":
    main()
