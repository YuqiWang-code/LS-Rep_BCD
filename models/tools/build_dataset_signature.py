#!/usr/bin/env python3
"""Compute a dataset signature vector from the train split and save it as JSON.

``--schema_version 1`` keeps the historical 18-dim signature (never overwritten);
``--schema_version 2`` (default) writes the corrected 20-dim signature with global
object-area quantiles, fixed n_components / Sobel magnitude, and CDD-safe path
resolution.

Usage:
    python -m models.tools.build_dataset_signature \
        --data_root /share_datasets/CD/SYSU-CD-256 --dataset_name SYSU \
        --schema_version 2 --output .../signatures_v2/SYSU.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.agent.dataset_signature import (  # noqa: E402
    SIGNATURE_NAMES,
    SIGNATURE_V2_NAMES,
    SIGNATURE_V2_VERSION,
    compute_dataset_signature,
    compute_dataset_signature_v2,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute dataset signature")
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--schema_version", type=int, default=SIGNATURE_V2_VERSION, choices=(1, 2))
    p.add_argument("--pilot", type=int, default=200, help="samples used to calibrate the pseudo-change threshold (v2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.schema_version == 1:
        vector, details = compute_dataset_signature(
            args.data_root, args.dataset_name, split=args.split, max_samples=args.max_samples
        )
        names = SIGNATURE_NAMES
    else:
        vector, details = compute_dataset_signature_v2(
            args.data_root, args.dataset_name, split=args.split,
            max_samples=args.max_samples, pilot=args.pilot,
        )
        names = SIGNATURE_V2_NAMES

    if len(vector) != len(names):
        raise AssertionError(f"signature dim {len(vector)} != names {len(names)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset_name.upper(),
                "split": args.split,
                "schema_version": args.schema_version,
                "names": names,
                "vector": [float(v) for v in vector],
                "details": {k: round(v, 6) for k, v in details.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[signature] {args.dataset_name} (schema v{args.schema_version}): "
          f"saved {len(vector)}-dim signature to {out}")


if __name__ == "__main__":
    main()
