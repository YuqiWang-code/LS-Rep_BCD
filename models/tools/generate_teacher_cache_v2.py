#!/usr/bin/env python3
"""Generate Compact Teacher Cache v2 for one (teacher, dataset, split).

Reads the canonical 256x256 source image (same as the student's Scale step),
resizes to the teacher's native input size, runs the frozen teacher encoder in
batches, and stores the projected symmetric change evidence per sample.

Usage:
    python -m models.tools.generate_teacher_cache_v2 \
        --teacher_package dinov3_sat --weight_dir pre-trained_weights \
        --data_root /share_datasets/CD/SYSU-CD-256 --dataset_name SYSU \
        --cache_root /share_datasets/CD_teacher_cache/CATA_CD_v2 \
        --split train --gpu_id 0 --batch_size 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.cache_v2 import CACHE_SCHEMA_VERSION  # noqa: E402
from models.distill.teacher_package import TEACHER_REGISTRY, build_teacher_package  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Compact Teacher Cache v2")
    p.add_argument("--teacher_package", required=True)
    p.add_argument("--weight_dir", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--projection_seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def _read_rgb_01(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Failed to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


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
    if args.max_samples is not None:
        sample_ids = sample_ids[: args.max_samples]

    package, meta, cin = build_teacher_package(
        args.teacher_package, args.weight_dir, projection_seed=args.projection_seed, device=device
    )
    package = package.to(device)
    package.eval()
    teacher_input = meta.input_size

    out_root = Path(args.cache_root) / args.teacher_package / dataset / split
    out_root.mkdir(parents=True, exist_ok=True)

    weight_file = Path(args.weight_dir) / TEACHER_REGISTRY[args.teacher_package].weight
    weight_hash = hashlib.sha256(weight_file.read_bytes()).hexdigest() if weight_file.is_file() else "n/a"

    bs = args.batch_size
    n_done = 0
    with torch.no_grad():
        for start in range(0, len(sample_ids), bs):
            batch_ids = sample_ids[start:start + bs]
            imgs_a, imgs_b = [], []
            for sid in batch_ids:
                pre = _read_rgb_01(data_root / "A" / sid, 256)
                post = _read_rgb_01(data_root / "B" / sid, 256)
                if tuple(teacher_input) != (256, 256):
                    pre = cv2.resize(pre, (teacher_input[1], teacher_input[0]), interpolation=cv2.INTER_LINEAR)
                    post = cv2.resize(post, (teacher_input[1], teacher_input[0]), interpolation=cv2.INTER_LINEAR)
                imgs_a.append(pre)
                imgs_b.append(post)

            xa = torch.from_numpy(np.stack(imgs_a, axis=0).transpose(0, 3, 1, 2)).float().to(device)
            xb = torch.from_numpy(np.stack(imgs_b, axis=0).transpose(0, 3, 1, 2)).float().to(device)

            out = package.encode_pair(xa, xb)
            lc = out["local_change"].cpu().half()
            conf = out["confidence"].cpu().half()
            gd = out["global_desc"].cpu().half()

            for j, sid in enumerate(batch_ids):
                entry = {
                    "teacher_id": args.teacher_package,
                    "sample_id": sid,
                    "local_change": lc[j],
                    "confidence": conf[j],
                    "global_desc": gd[j],
                    "meta": {
                        "weight_hash": weight_hash,
                        "normalization": {"mean": list(meta.mean), "std": list(meta.std)},
                        "color_order": meta.color_order,
                        "teacher_input_size": list(meta.input_size),
                        "source_size": 256,
                        "cache_schema_version": CACHE_SCHEMA_VERSION,
                        "projection_seed": args.projection_seed,
                        "projection_in": int(cin),
                    },
                }
                torch.save(entry, out_root / f"{sid}.pt")
            n_done += len(batch_ids)
            if start == 0 or n_done % 1000 < bs:
                print(f"[cache-v2] {args.teacher_package}/{dataset}/{split} {n_done}/{len(sample_ids)}", flush=True)

    manifest = {
        "teacher_id": args.teacher_package,
        "dataset": dataset,
        "split": split,
        "weight_file": weight_file.name,
        "weight_hash": weight_hash,
        "normalization": {"mean": list(meta.mean), "std": list(meta.std)},
        "color_order": meta.color_order,
        "teacher_input_size": list(meta.input_size),
        "source_size": 256,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "projection_seed": args.projection_seed,
        "projection_in": int(cin),
        "n_samples": len(sample_ids),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[cache-v2] done: {len(sample_ids)} samples -> {out_root}")


if __name__ == "__main__":
    main()
