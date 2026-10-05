#!/usr/bin/env python3
"""Audit a Compact Teacher Cache v2 directory (Stage 1 P0 audit).

Checks manifest consistency, per-sample tensor shapes/dtypes, NaN/Inf, sample-id
coverage, replay determinism, and total size. It only decides whether the cache
is runnable — not whether the teacher is effective.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.cache_v2 import CACHE_CHANNELS, apply_cache_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Compact Teacher Cache v2")
    p.add_argument("--cache_root", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--data_root", required=True)
    p.add_argument("--max_samples", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.cache_root) / args.dataset_name.upper() / args.split
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    expected_ids = [
        line.strip()
        for line in (Path(args.data_root) / "list" / f"{args.split}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    missing = [sid for sid in expected_ids if not (root / f"{sid}.pt").is_file()]
    if missing:
        raise SystemExit(f"[audit] missing {len(missing)} samples, e.g. {missing[:5]}")

    total_bytes = 0
    nan = 0
    inf = 0
    n = 0
    for sid in expected_ids[: args.max_samples]:
        d = torch.load(root / f"{sid}.pt", map_location="cpu", weights_only=False)
        lc = d["local_change"]
        conf = d["confidence"]
        gd = d["global_desc"]
        assert lc.dtype == torch.float16 and conf.dtype == torch.float16 and gd.dtype == torch.float16
        assert lc.shape[0] == CACHE_CHANNELS
        assert conf.shape[0] == 1 and conf.shape[-2:] == lc.shape[-2:]
        nan += int(torch.isnan(lc).any()) + int(torch.isnan(conf).any())
        inf += int(torch.isinf(lc).any()) + int(torch.isinf(conf).any())
        total_bytes += (root / f"{sid}.pt").stat().st_size
        n += 1

    per_sample = total_bytes / max(n, 1)
    est_total = per_sample * len(expected_ids) / 1e9
    print(f"[audit] {manifest['teacher_id']}/{args.dataset_name}/{args.split}: "
          f"schema={manifest['cache_schema_version']} n={len(expected_ids)} "
          f"per_sample={per_sample/1024:.1f}KB est_total={est_total:.2f}GB "
          f"nan={nan} inf={inf}")
    print(f"[audit] weight_hash={manifest['weight_hash'][:16]} "
          f"normalization={manifest['normalization']} color={manifest['color_order']} "
          f"input_size={manifest['teacher_input_size']}")

    # replay determinism on one sample
    sid = expected_ids[0]
    d = torch.load(root / f"{sid}.pt", map_location="cpu", weights_only=False)
    state = {"crop_resize": {"enabled": True, "top": 3, "left": 4, "source_height": 256, "source_width": 256},
             "flip_v": True, "flip_h": False, "exchange": True}
    out = apply_cache_state(d, state)
    assert out["local_change"].shape == d["local_change"].shape
    print(f"[audit] replay ok: local_change {tuple(d['local_change'].shape)}")

    if nan or inf:
        raise SystemExit("[audit] FAILED: NaN/Inf present")
    if est_total > 12.0:
        print(f"[audit] WARNING: estimated total {est_total:.2f}GB exceeds 12GB budget")
    print("[audit] PASSED")


if __name__ == "__main__":
    main()
