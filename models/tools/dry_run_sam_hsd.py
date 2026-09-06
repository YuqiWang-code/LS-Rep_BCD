#!/usr/bin/env python3
"""Real-cache CUDA dry run for SAM-HSD throughput and memory acceptance."""

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

from models import A2Net_LWGANet_L0, build_loss
from models.datasets.cd_dataset import get_loader
from models.distill import TeacherCache


def parse_args():
    parser = argparse.ArgumentParser(description="SAM-HSD real-cache dry run")
    parser.add_argument("--dataset_name", required=True, choices=["SYSU", "WHU"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--teacher_cache_root", required=True)
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or not 1 <= args.steps <= 100 or args.num_workers < 0:
        raise ValueError("batch_size/workers invalid or steps outside [1,100]")
    return args


def nested_to(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: nested_to(item, device) for key, item in value.items()}
    return value


def capped_auxiliary(raw_loss, main_loss, coefficient=0.06, max_ratio=0.12):
    weighted = coefficient * raw_loss
    cap = max_ratio * main_loss.detach()
    scale = torch.clamp(cap / (weighted.detach() + 1e-8), max=1.0)
    return weighted * scale.detach()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    checkpoint = Path(args.pretrained_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LWGANet checkpoint missing: {checkpoint}")
    cache = TeacherCache(args.teacher_cache_root, require_validation=True)
    loader = get_loader(
        args.data_root, str(Path(args.data_root) / "list" / "train.txt"),
        batchsize=args.batch_size, trainsize=256, num_workers=args.num_workers,
        teacher_cache=cache, derive_sam_hsd=True,
    )
    if set(cache.entries) != set(loader.dataset.file_list):
        raise ValueError("Teacher cache coverage does not match the training list")

    model = A2Net_LWGANet_L0(
        pretrained=True, pretrained_path=str(checkpoint),
        auxiliary_mode="sam_hsd",
        sam_hsd_cfg={
            "mode": "full", "evidence_mode": "directional",
            "use_scgr": True, "robust_filter": True, "boundary_band": 7,
        },
    ).to(device).train()
    criterion = build_loss("batch")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=5e-4, betas=(0.9, 0.99),
        eps=1e-8, weight_decay=1e-4,
    )
    torch.cuda.reset_peak_memory_stats(device)
    data_times, step_times = [], []
    tracked = {
        key: [] for key in (
            "e_plus_mean", "e_minus_mean", "e_stable_mean",
            "e_uncertain_mean", "reliable_positive_ratio",
            "reliable_negative_ratio", "hard_negative_ratio",
            "hard_positive_ratio",
        )
    }

    iterator = iter(loader)
    fetch_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            image, target, _, teacher_pack = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image, target, _, teacher_pack = next(iterator)
        data_time = time.perf_counter() - fetch_started
        step_started = time.perf_counter()
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        teacher_pack = nested_to(teacher_pack, device)
        optimizer.zero_grad(set_to_none=True)
        predictions, auxiliary = model(
            image[:, :3], image[:, 3:6], target=target,
            teacher_pack=teacher_pack, compute_auxiliary=True,
        )
        main_loss = sum(criterion(prediction, target) for prediction in predictions)
        details = auxiliary["sam_hsd"]
        auxiliary_loss = capped_auxiliary(details["total"], main_loss)
        loss = main_loss + auxiliary_loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        step_time = time.perf_counter() - step_started
        data_times.append(data_time)
        step_times.append(step_time)
        for key in tracked:
            tracked[key].append(float(details[key].detach()))
        print(
            f"dry step [{step}/{args.steps}] loss={float(loss.detach()):.4f} "
            f"data={data_time:.3f}s step={step_time:.3f}s "
            f"peak={torch.cuda.max_memory_allocated(device)/(1024**3):.2f}GiB"
        )
        fetch_started = time.perf_counter()

    summary = {
        "dataset": args.dataset_name,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "workers": args.num_workers,
        "mean_data_time_s": statistics.mean(data_times),
        "mean_step_time_s": statistics.mean(step_times),
        "p95_data_time_s": sorted(data_times)[max(0, int(0.95 * len(data_times)) - 1)],
        "p95_step_time_s": sorted(step_times)[max(0, int(0.95 * len(step_times)) - 1)],
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        **{key: statistics.mean(values) for key, values in tracked.items()},
    }
    print("SAM-HSD REAL CACHE DRY RUN PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
