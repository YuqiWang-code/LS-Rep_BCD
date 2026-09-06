#!/usr/bin/env python3
"""Validate SAM structure cache coverage/schema and create audit previews.

The schema check runs over every cached sample (in parallel); the aggregate
statistics (instances / coverage / high-quality / boundary-near-GT) are
estimated on a configurable random subsample, since they are audit signals
rather than correctness gates. ``number_of_samples`` in the summary is always
the full count, so downstream ``require_validation=True`` gating is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets.cd_dataset import CDDataset
from models.distill.teacher_cache import TeacherCache


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a Run4 SAM structure cache")
    parser.add_argument("--dataset_name", required=True, choices=["LEVIR", "SYSU", "WHU", "CDD"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--visualize_dir", default=None)
    parser.add_argument("--num_visualize", type=int, default=100)
    parser.add_argument("--stats_sample_limit", type=int, default=2000,
                        help="Number of random samples used to estimate aggregate "
                             "statistics; schema is always checked on every sample.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Parallel processes for the per-sample cache check.")
    parser.add_argument("--seed", type=int, default=2333)
    args = parser.parse_args()
    if args.num_visualize < 0:
        raise ValueError("num_visualize must be non-negative")
    if args.stats_sample_limit < 0:
        raise ValueError("stats_sample_limit must be non-negative")
    if args.num_workers < 1:
        raise ValueError("num_workers must be positive")
    return args


def validate_tensor(name, value, dtype, height, width, bounded=False):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype is {value.dtype}; expected {dtype}")
    if tuple(value.shape) != (1, height, width):
        raise ValueError(f"{name} shape is {tuple(value.shape)}; expected {(1, height, width)}")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if bounded and (float(value.min()) < 0.0 or float(value.max()) > 1.0):
        raise ValueError(f"{name} is outside [0, 1]")


def gt_boundary_band(label, kernel=7):
    binary = (label > 127).astype(np.uint8)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return cv2.dilate(gradient, np.ones((kernel, kernel), np.uint8)) > 0


def colorize_instances(instance):
    instance = np.asarray(instance, dtype=np.int64)
    color = np.zeros((*instance.shape, 3), dtype=np.uint8)
    foreground = instance > 0
    color[..., 0][foreground] = (instance[foreground] * 37 % 211 + 44).astype(np.uint8)
    color[..., 1][foreground] = (instance[foreground] * 67 % 211 + 44).astype(np.uint8)
    color[..., 2][foreground] = (instance[foreground] * 97 % 211 + 44).astype(np.uint8)
    return color


def labeled(panel, title):
    output = panel.copy()
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 220), 24), (0, 0, 0), -1)
    cv2.putText(output, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output


def write_preview(dataset, sample_id, pack, destination):
    t1 = dataset._read(dataset._resolve_path("A", sample_id), cv2.IMREAD_COLOR, "T1 image")
    t2 = dataset._read(dataset._resolve_path("B", sample_id), cv2.IMREAD_COLOR, "T2 image")
    label = dataset._read(dataset._resolve_path("label", sample_id), cv2.IMREAD_GRAYSCALE, "label")
    gt = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    i1 = colorize_instances(pack["t1"]["instance_id"][0].numpy())
    i2 = colorize_instances(pack["t2"]["instance_id"][0].numpy())
    boundary = torch.maximum(pack["t1"]["boundary"], pack["t2"]["boundary"])[0].float().numpy()
    quality = torch.maximum(pack["t1"]["quality"], pack["t2"]["quality"])[0].float().numpy()
    audit = cv2.applyColorMap(np.clip(boundary * quality * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    panels = [
        labeled(t1, "T1"), labeled(t2, "T2"), labeled(gt, "GT change"),
        labeled(i1, "SAM instances T1"), labeled(i2, "SAM instances T2"),
        labeled(audit, "boundary x quality"),
    ]
    canvas = np.concatenate(
        (np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)), axis=0
    )
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Failed to write SAM preview: {destination}")


def atomic_json(data, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _worker_check(payload):
    """Schema-check one sample (and compute stats when flagged).

    Runs in a worker process; only receives paths so nothing but the tensors
    is pickled. The height/width is read from the cached instance map, avoiding
    a per-sample cv2.imread that the old loop needed just for the shape.
    """
    cache_root, label_path, sample_id, do_stats = payload
    cache = TeacherCache(cache_root)
    pack = cache.load(sample_id)
    stats = None

    for time_key in ("t1", "t2"):
        if time_key not in pack or not isinstance(pack[time_key], dict):
            raise KeyError(f"{sample_id}: missing nested key '{time_key}'")
        struct = pack[time_key]
        instance = struct["instance_id"]
        height, width = int(instance.shape[1]), int(instance.shape[2])
        validate_tensor(
            f"{sample_id}/{time_key}/instance_id", struct.get("instance_id"),
            torch.int32, height, width,
        )
        validate_tensor(
            f"{sample_id}/{time_key}/boundary", struct.get("boundary"),
            torch.float16, height, width, True,
        )
        validate_tensor(
            f"{sample_id}/{time_key}/quality", struct.get("quality"),
            torch.float16, height, width, True,
        )
        if not do_stats:
            continue
        quality = struct["quality"].float()
        foreground = instance > 0
        ids = torch.unique(instance[foreground]) if foreground.any() else torch.empty(0)
        stats = stats or {}
        stats[f"instances_{time_key}"] = float(ids.numel())
        stats[f"coverage_{time_key}"] = float(foreground.float().mean())
        denominator = foreground.sum().clamp_min(1)
        stats[f"high_quality_{time_key}"] = float(
            ((quality >= 0.8) & foreground).sum() / denominator
        )

    if do_stats:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise OSError(f"OpenCV failed to read label: {label_path}")
        gt_band = gt_boundary_band(label)
        sam_boundary = torch.maximum(
            pack["t1"]["boundary"], pack["t2"]["boundary"],
        )[0].float().numpy() > 0.1
        stats["boundary_near_gt_ratio"] = float(
            (sam_boundary & gt_band).sum() / max(sam_boundary.sum(), 1)
        )
    return sample_id, stats


def main():
    args = parse_args()
    cache = TeacherCache(args.cache_root)
    manifest = cache.manifest
    if manifest.get("teacher_type") != "sam2_struct_v2":
        raise ValueError(f"Unexpected teacher_type: {manifest.get('teacher_type')}")
    if manifest.get("source_dataset") != args.dataset_name:
        raise ValueError(
            f"Cache dataset is {manifest.get('source_dataset')}; requested {args.dataset_name}"
        )
    dataset = CDDataset("train", args.data_root, dataset_name=args.dataset_name)
    expected = set(dataset.file_list)
    actual = set(cache.entries)
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise ValueError(f"Cache coverage mismatch; missing={missing}, extra={extra}")

    rng = random.Random(args.seed)
    stats_limit = min(args.stats_sample_limit, len(dataset.file_list))
    stats_ids = set(rng.sample(dataset.file_list, stats_limit)) if stats_limit > 0 else set()
    visual_ids = set(rng.sample(dataset.file_list, min(args.num_visualize, len(dataset))))
    visualize_dir = Path(args.visualize_dir) if args.visualize_dir else None
    if visualize_dir is not None:
        visualize_dir.mkdir(parents=True, exist_ok=True)

    payloads = [
        (
            args.cache_root,
            str(dataset._resolve_path("label", sample_id)),
            sample_id,
            sample_id in stats_ids,
        )
        for sample_id in dataset.file_list
    ]

    totals = {
        "instances_t1": 0.0, "instances_t2": 0.0,
        "coverage_t1": 0.0, "coverage_t2": 0.0,
        "high_quality_t1": 0.0, "high_quality_t2": 0.0,
        "boundary_near_gt_ratio": 0.0,
    }
    chunksize = max(1, len(payloads) // (args.num_workers * 4))
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = executor.map(_worker_check, payloads, chunksize=chunksize)
        for _sample_id, stats in tqdm(
            results, total=len(payloads), desc=f"Validate SAM cache {args.dataset_name}"
        ):
            if stats is not None:
                for key, value in stats.items():
                    totals[key] += value

    if visualize_dir is not None:
        for sample_id in visual_ids:
            pack = cache.load(sample_id)
            safe_name = sample_id.replace("/", "__").replace("\\", "__")
            write_preview(dataset, sample_id, pack, visualize_dir / f"{Path(safe_name).stem}.jpg")

    count = max(stats_limit, 1)
    summary = {
        "dataset": args.dataset_name,
        "number_of_samples": len(dataset),
        "stats_sample_count": stats_limit,
        "config_hash": manifest.get("config_hash"),
        **{key: value / count for key, value in totals.items()},
        "visualized_samples": len(visual_ids) if visualize_dir is not None else 0,
    }
    atomic_json(summary, Path(args.cache_root) / "validation_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Teacher cache validation passed: {len(dataset)} samples")


if __name__ == "__main__":
    main()
