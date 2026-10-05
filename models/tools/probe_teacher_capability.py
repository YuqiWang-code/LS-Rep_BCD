#!/usr/bin/env python3
"""Compute train-only teacher capability probes from a compact cache and labels.

Usage:
    python -m models.tools.probe_teacher_capability \
        --cache_root /share_datasets/CD_teacher_cache/CATA_CD_v2/dinov3_sat \
        --data_root /share_datasets/CD/SYSU-CD-256 --dataset_name SYSU \
        --teacher_package dinov3_sat --output probes/SYSU_dinov3_sat.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.capability_probe import confidence_calibration, separability_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute teacher capability probes")
    p.add_argument("--cache_root", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--teacher_package", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=256)
    return p.parse_args()


def _read_label(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"Failed to read {path}")
    return (img >= 128).astype(np.float32)


def main() -> None:
    args = parse_args()
    root = Path(args.cache_root) / args.dataset_name.upper() / args.split
    data_root = Path(args.data_root)

    ids = [
        line.strip()
        for line in (data_root / "list" / f"{args.split}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.max_samples]

    sep = {"change_mean": [], "unchanged_mean": [], "ratio": []}
    cal = {"conf_change": [], "conf_unchanged": []}

    for sid in ids:
        d = torch.load(root / f"{sid}.pt", map_location="cpu", weights_only=False)
        lc = d["local_change"].float()
        conf = d["confidence"].float()
        lab = torch.from_numpy(_read_label(data_root / "label" / sid))

        s = separability_scores(lc, lab)
        c = confidence_calibration(conf, lab)
        for k in sep:
            sep[k].append(s[k])
        for k in cal:
            cal[k].append(c[k])

    probe = {
        "teacher_id": args.teacher_package,
        "dataset": args.dataset_name.upper(),
        "split": args.split,
        "n_samples": len(ids),
        "separability": {k: float(np.mean(v)) for k, v in sep.items()},
        "calibration": {k: float(np.mean(v)) for k, v in cal.items()},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(f"[probe] {args.teacher_package}/{args.dataset_name}: ratio={probe['separability']['ratio']:.3f}")


if __name__ == "__main__":
    main()
