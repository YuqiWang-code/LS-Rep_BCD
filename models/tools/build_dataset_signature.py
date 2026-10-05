#!/usr/bin/env python3
"""Compute a dataset signature vector from the train split and save it as JSON.

Usage:
    python -m models.tools.build_dataset_signature \
        --data_root /share_datasets/CD/SYSU-CD-256 --dataset_name SYSU \
        --output /share_datasets/CD_teacher_cache/CATA_CD_v2/signatures/SYSU.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.agent.dataset_signature import SIGNATURE_NAMES, compute_dataset_signature  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute dataset signature")
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    vector, details = compute_dataset_signature(
        args.data_root, args.dataset_name, split=args.split, max_samples=args.max_samples
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset_name.upper(),
                "split": args.split,
                "names": SIGNATURE_NAMES,
                "vector": [float(v) for v in vector],
                "details": {k: round(v, 6) for k, v in details.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[signature] {args.dataset_name}: saved {len(vector)}-dim signature to {out}")


if __name__ == "__main__":
    main()
