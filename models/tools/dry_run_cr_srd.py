#!/usr/bin/env python3
"""Real-cache CUDA timing, memory and routing diagnostics for Run4 CR-SRD."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_loss
from models.datasets.cd_dataset import get_loader
from models.distill import TeacherCache
from models.scripts.train import EXPERIMENTS, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="CR-SRD real-cache dry run")
    parser.add_argument("--experiment", default="J3", choices=("J2", "J3"))
    parser.add_argument("--dataset_name", required=True, choices=("SYSU", "WHU"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_cache_root", required=True)
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--hsd_lambda", type=float, default=0.06)
    parser.add_argument("--hsd_max_ratio", type=float, default=0.12)
    args = parser.parse_args()
    if args.batch_size <= 0 or not 1 <= args.steps <= 100 or args.num_workers < 0:
        raise ValueError("batch_size/workers invalid or steps outside [1,100]")
    recipe = EXPERIMENTS[args.experiment]
    args.pretrained = True
    args.auxiliary_mode = recipe["auxiliary_mode"]
    args.joint_temporal_bn = recipe["joint_temporal_bn"]
    args.cr_use_coherence_gate = recipe["cr_use_coherence_gate"]
    args.cr_use_magnitude_fallback = recipe["cr_use_magnitude_fallback"]
    return args


def nested_to(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: nested_to(item, device) for key, item in value.items()}
    return value


def mean(values):
    return statistics.fmean(values) if values else 0.0


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    checkpoint = Path(args.pretrained_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LWGANet checkpoint missing: {checkpoint}")
    cache = TeacherCache(args.teacher_cache_root, require_validation=True)
    if cache.manifest.get("source_dataset") != args.dataset_name:
        raise ValueError(
            f"Teacher cache dataset={cache.manifest.get('source_dataset')} "
            f"but dry-run dataset={args.dataset_name}"
        )
    if cache.manifest.get("teacher_type") != "sam2_struct_v2":
        raise ValueError("CR-SRD requires a sam2_struct_v2 cache")
    loader = get_loader(
        args.data_root, str(Path(args.data_root) / "list" / "train.txt"),
        batchsize=args.batch_size, trainsize=256,
        num_workers=args.num_workers, teacher_cache=cache,
        derive_sam_hsd=True,
    )
    if set(cache.entries) != set(loader.dataset.file_list):
        raise ValueError("Teacher cache coverage does not match the training list")

    model = build_model(args).to(device).train()
    criterion = build_loss("batch")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-4, betas=(0.9, 0.99),
        eps=1e-8, weight_decay=1e-4,
    )
    torch.cuda.reset_peak_memory_stats(device)
    data_times, step_times, cap_scales, cap_hits = [], [], [], []
    tracked_keys = (
        "cr_direction", "cr_magnitude", "cr_stable", "magnitude_mean",
        "coherence_mean", "coherence_p25", "coherence_p50", "coherence_p75",
        "coherence_low_frac", "coherence_mid_frac", "coherence_high_frac",
        "trust_change_mean", "trust_stable_mean", "coverage_change_mean",
        "coverage_stable_mean", "structural_uncertainty_mean",
    )
    tracked = {key: [] for key in tracked_keys}

    iterator = iter(loader)
    fetch_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            image, target, _, teacher_pack = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image, target, _, teacher_pack = next(iterator)
        data_times.append(time.perf_counter() - fetch_started)
        step_started = time.perf_counter()
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        teacher_pack = nested_to(teacher_pack, device)
        optimizer.zero_grad(set_to_none=True)
        predictions, auxiliary = model(
            image[:, :3], image[:, 3:6], target=target,
            teacher_pack=teacher_pack, compute_auxiliary=True,
        )
        main_loss = sum(
            criterion(prediction, target) for prediction in predictions
        )
        details = auxiliary["cr_srd"]
        raw_weighted = args.hsd_lambda * details["total"]
        cap = args.hsd_max_ratio * main_loss.detach()
        scale = torch.clamp(
            cap / (raw_weighted.detach() + 1e-8), max=1.0,
        )
        loss = main_loss + raw_weighted * scale.detach()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_started)
        cap_scales.append(float(scale.detach()))
        cap_hits.append(float(scale.detach() < 1.0 - 1e-7))
        for key in tracked_keys:
            tracked[key].append(float(details[key].detach()))
        print(
            f"step={step}/{args.steps} loss={float(loss.detach()):.4f} "
            f"cap_scale={cap_scales[-1]:.4f} "
            f"coherence={tracked['coherence_mean'][-1]:.4f}"
        )
        fetch_started = time.perf_counter()

    summary = {
        "experiment": args.experiment,
        "dataset": args.dataset_name,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "train_params": sum(parameter.numel() for parameter in model.parameters()),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "mean_data_time_s": mean(data_times),
        "mean_step_time_s": mean(step_times),
        "mean_cap_scale": mean(cap_scales),
        "cap_hit_rate": mean(cap_hits),
        **{key: mean(values) for key, values in tracked.items()},
    }
    print("CR-SRD REAL CACHE DRY RUN PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
