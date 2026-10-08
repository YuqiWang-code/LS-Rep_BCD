#!/usr/bin/env python3
"""Compute train-only teacher capability probes from a compact cache and labels.

Review fixes (2026-10-08):
  - stratified train-only sampling with a fixed seed (no "first N" order bias);
  - A/B/label resolved with the CDDataset fallback rules (CDD ``test_`` prefix,
    png<->jpg), never a bare path join;
  - records the sampled index SHA and the per-sample count for traceability;
  - emits the P1-P5 extended probes alongside the legacy separability/calibration.

Usage:
    python -m models.tools.probe_teacher_capability \
        --cache_root .../CATA_CD_v2/dinov3_sat --data_root /share_datasets/CD/SYSU-CD-256 \
        --dataset_name SYSU --teacher_package dinov3_sat --output probes_v2/SYSU_dinov3_sat.json
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

from models.agent.dataset_signature import _resolve  # noqa: E402
from models.distill.capability_probe import (  # noqa: E402
    confidence_calibration,
    extended_probes,
    separability_scores,
)

EXT_KEYS = ("balanced_auroc", "hard_negative_separation", "boundary_contrast",
            "size_conditioned_auroc_gap", "unchanged_leakage_ratio")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute teacher capability probes")
    p.add_argument("--cache_root", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--teacher_package", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=2333)
    p.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _read_label(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"Failed to read {path}")
    return (img >= 128).astype(np.float32)


def _stratified_ids(data_root: Path, ids: list[str], dataset: str, n: int, seed: int) -> list[str]:
    """Split by change ratio into 3 quantile bins and sample evenly."""
    ratios = []
    for sid in ids:
        lab = _read_label(_resolve(data_root, "label", sid, dataset))
        ratios.append(float(lab.mean()))
    ratios = np.asarray(ratios)
    order = np.argsort(ratios, kind="mergesort")
    bins = np.array_split(order, 3)
    rng = np.random.default_rng(seed)
    per = max(1, n // 3)
    picked: list[int] = []
    for b in bins:
        take = min(per, len(b))
        picked.extend(rng.choice(b, take, replace=False).tolist())
    picked = sorted(set(picked))
    return [ids[i] for i in picked]


def main() -> None:
    args = parse_args()
    dataset = args.dataset_name.upper()
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_root) / dataset / args.split

    ids = [
        line.strip()
        for line in (data_root / "list" / f"{args.split}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.stratify and args.max_samples < len(ids):
        selected = _stratified_ids(data_root, ids, dataset, args.max_samples, args.seed)
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(ids))[: args.max_samples]
        selected = [ids[i] for i in sorted(idx.tolist())]

    idx_sha = hashlib.sha256(",".join(selected).encode()).hexdigest()[:16]

    sep = {"change_mean": [], "unchanged_mean": [], "ratio": []}
    cal = {"conf_change": [], "conf_unchanged": []}
    ext = {k: [] for k in EXT_KEYS}
    empty_pairs = 0

    for sid in selected:
        path = cache_dir / f"{sid}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"cache missing: {path}")
        d = torch.load(path, map_location="cpu", weights_only=False)
        lc = d["local_change"].float()
        conf = d["confidence"].float()
        lab = torch.from_numpy(_read_label(_resolve(data_root, "label", sid, dataset)))

        s = separability_scores(lc, lab)
        c = confidence_calibration(conf, lab)
        e = extended_probes(conf, lab, seed=args.seed)
        for k in sep:
            sep[k].append(s[k])
        for k in cal:
            cal[k].append(c[k])
        for k in EXT_KEYS:
            if e.get(k) is not None:
                ext[k].append(float(e[k]))
        empty_pairs += int(e.get("_empty_pair", 0.0))

    probe = {
        "teacher_id": args.teacher_package,
        "dataset": dataset,
        "split": args.split,
        "n_samples": len(selected),
        "sampled_index_sha256": idx_sha,
        "stratified": bool(args.stratify),
        "empty_pair_ratio": empty_pairs / max(len(selected), 1),
        "separability": {k: float(np.mean(v)) for k, v in sep.items()},
        "calibration": {k: float(np.mean(v)) for k, v in cal.items()},
    }
    for k in EXT_KEYS:
        probe[k] = (float(np.mean(ext[k])) if ext[k] else None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(f"[probe] {args.teacher_package}/{dataset}: n={len(selected)} "
          f"ratio={probe['separability']['ratio']:.3f} auroc={probe['balanced_auroc']} -> {out.name}")


if __name__ == "__main__":
    main()
