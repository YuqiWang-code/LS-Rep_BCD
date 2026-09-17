#!/usr/bin/env python3
"""Read-only validation for paired Run3 SAMStruct + OVCDistill Teacher Cache.

This tool validates the CURRENT cache contract used by Run3 BT-SAM-RDT.

It never:
    - rebuilds Teacher Cache;
    - modifies Teacher Cache;
    - rewrites manifests;
    - writes validation summaries;
    - changes dataset files.

Validation scope
----------------
1. Dataset train split
    - list/train.txt exists;
    - non-empty;
    - no duplicate sample IDs.

2. Paired Teacher Cache construction
    - SAM manifest exists and is valid;
    - OV manifest exists and is valid;
    - source_dataset matches the requested dataset;
    - SAM teacher_type == sam2_struct_v2;
    - config_hash is present;
    - SAM/OV entries exactly cover the train split;
    - every manifest-referenced cache file exists.

3. Every sample
    SAMStruct:
        t1/t2:
            instance_id : [1,H,W], int32/int64, non-negative
            boundary    : [1,H,W], finite, [0,1]
            quality     : [1,H,W], finite, [0,1]

        All six SAM fields must share one spatial shape.

    OVCDistill:
        soft_change:
            l1/l2 : [1,H,W], finite, [0,1]

        confidence:
            l1/l2 : [1,H,W], finite, [0,1]

        relation:
            l1/l2 : [8,H,W], finite

        Within each OV level, soft_change/confidence/relation must share
        the same H,W.

4. Pair-level checks
    - every requested sample loads from BOTH teachers;
    - sample order is driven only by list/train.txt;
    - cache fingerprint is reported for reproducibility.

Important
---------
Passing this script proves cache coverage/schema/integrity only.

It does NOT prove:
    - Teacher accuracy;
    - SAM/OV usefulness;
    - Run3 effectiveness;
    - augmentation replay correctness during training;
    - that a Foundation prior improves Student performance.

Run from project root, for example:

    python models/tools/validate_teacher_cache.py \
        --data_root /share_datasets/CD/SYSU-CD-256 \
        --dataset_name SYSU \
        --sam_cache_root /share_datasets/CD_teacher_cache/SAMStruct/SYSU \
        --ov_cache_root /share_datasets/CD_teacher_cache/OVCDistill/SYSU

The exact cache-root layout should follow the current server manifests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from models.distill.teacher_cache import (  # noqa: E402
    PairedTeacherCache,
    canonical_dataset,
    validate_pack,
)


SUPPORTED_DATASETS = {
    "CDD",
    "LEVIR",
    "SYSU",
    "WHU",
}


# ============================================================================
# Arguments
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only full validation of paired Run3 "
            "SAMStruct + OVCDistill Teacher Cache"
        )
    )

    parser.add_argument(
        "--data_root",
        required=True,
        help=(
            "Dataset root containing list/train.txt, "
            "for example /share_datasets/CD/SYSU-CD-256"
        ),
    )

    parser.add_argument(
        "--dataset_name",
        required=True,
        help="One of CDD / LEVIR / SYSU / WHU",
    )

    parser.add_argument(
        "--sam_cache_root",
        required=True,
        help="SAMStruct cache root containing manifest.json",
    )

    parser.add_argument(
        "--ov_cache_root",
        required=True,
        help="OVCDistill cache root containing manifest.json",
    )

    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
        help=(
            "Print progress every N samples. "
            "Use 0 to disable periodic progress output."
        ),
    )

    args = parser.parse_args()

    dataset_name = canonical_dataset(
        args.dataset_name
    )

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            "dataset_name must resolve to one of "
            f"{sorted(SUPPORTED_DATASETS)}, got {args.dataset_name!r}"
        )

    args.dataset_name = dataset_name

    if args.progress_every < 0:
        raise ValueError(
            "--progress_every must be >= 0"
        )

    return args


# ============================================================================
# Dataset split validation
# ============================================================================


def read_train_ids(
    data_root: Path,
) -> List[str]:
    split_path = (
        data_root
        / "list"
        / "train.txt"
    )

    if not split_path.is_file():
        raise FileNotFoundError(
            split_path
        )

    entries = [
        value.strip()
        for value in (
            split_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if value.strip()
    ]

    if not entries:
        raise ValueError(
            f"{split_path}: empty train split"
        )

    unique = set(
        entries
    )

    if len(entries) != len(unique):
        duplicates = []
        seen = set()

        for sample_id in entries:
            if sample_id in seen:
                duplicates.append(
                    sample_id
                )
            else:
                seen.add(
                    sample_id
                )

        preview = ", ".join(
            duplicates[:10]
        )

        raise ValueError(
            f"{split_path}: duplicate sample IDs found; "
            f"count={len(entries) - len(unique)}; "
            f"examples=[{preview}]"
        )

    return entries


# ============================================================================
# Extra defensive checks around loaded packs
# ============================================================================


def _check_tensor_device(
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    TeacherCache.load() is expected to be CPU-side.

    This prevents accidental cache validation from consuming GPU memory.
    """
    if tensor.device.type != "cpu":
        raise ValueError(
            f"{name}: cache tensor unexpectedly loaded on "
            f"{tensor.device}; expected CPU"
        )


def validate_loaded_pair(
    pair: Dict,
    sample_id: str,
) -> Dict[str, object]:
    """
    Re-run the public schema checks and collect lightweight diagnostics.

    PairedTeacherCache.load() already validates each cache. This additional
    layer makes failures explicit at the paired Run3 interface and verifies
    the expected nested dictionary contract.
    """
    if not isinstance(
        pair,
        dict,
    ):
        raise ValueError(
            f"{sample_id}: paired cache load must return a dict"
        )

    if set(
        pair.keys()
    ) != {
        "sam",
        "ov",
    }:
        raise ValueError(
            f"{sample_id}: paired cache must contain exactly "
            "'sam' and 'ov'"
        )

    sam = pair[
        "sam"
    ]

    ov = pair[
        "ov"
    ]

    # Re-run the shared authoritative schema checks.
    validate_pack(
        sam,
        "sam",
        source=f"{sample_id}:sam",
    )

    validate_pack(
        ov,
        "ov",
        source=f"{sample_id}:ov",
    )

    # ------------------------------------------------------------------
    # CPU-only read contract
    # ------------------------------------------------------------------

    for temporal in (
        "t1",
        "t2",
    ):
        for key in (
            "instance_id",
            "boundary",
            "quality",
        ):
            _check_tensor_device(
                sam[
                    temporal
                ][
                    key
                ],
                f"{sample_id}:sam:{temporal}:{key}",
            )

    for level in (
        "l1",
        "l2",
    ):
        for key in (
            "soft_change",
            "confidence",
            "relation",
        ):
            _check_tensor_device(
                ov[
                    key
                ][
                    level
                ],
                f"{sample_id}:ov:{key}:{level}",
            )

    # ------------------------------------------------------------------
    # Lightweight per-sample diagnostics
    # ------------------------------------------------------------------

    sam_shape = tuple(
        int(v)
        for v in sam[
            "t1"
        ][
            "instance_id"
        ].shape
    )

    ov_l1_shape = tuple(
        int(v)
        for v in ov[
            "soft_change"
        ][
            "l1"
        ].shape
    )

    ov_l2_shape = tuple(
        int(v)
        for v in ov[
            "soft_change"
        ][
            "l2"
        ].shape
    )

    max_t1_instance = int(
        sam[
            "t1"
        ][
            "instance_id"
        ].max()
    )

    max_t2_instance = int(
        sam[
            "t2"
        ][
            "instance_id"
        ].max()
    )

    return {
        "sam_shape": sam_shape,
        "ov_l1_shape": ov_l1_shape,
        "ov_l2_shape": ov_l2_shape,
        "max_t1_instance": max_t1_instance,
        "max_t2_instance": max_t2_instance,
    }


# ============================================================================
# Main validation
# ============================================================================


def main() -> None:
    args = parse_args()

    data_root = Path(
        args.data_root
    ).expanduser()

    sam_cache_root = Path(
        args.sam_cache_root
    ).expanduser()

    ov_cache_root = Path(
        args.ov_cache_root
    ).expanduser()

    if not data_root.is_dir():
        raise NotADirectoryError(
            data_root
        )

    if not sam_cache_root.is_dir():
        raise NotADirectoryError(
            sam_cache_root
        )

    if not ov_cache_root.is_dir():
        raise NotADirectoryError(
            ov_cache_root
        )

    entries = read_train_ids(
        data_root
    )

    started = time.perf_counter()

    # Construction itself validates:
    #   - manifests;
    #   - dataset identity;
    #   - SAM teacher type;
    #   - config hashes;
    #   - exact train coverage;
    #   - referenced file existence.
    cache = PairedTeacherCache(
        sam_cache_root,
        ov_cache_root,
        args.dataset_name,
        entries,
    )

    sam_shapes = set()
    ov_l1_shapes = set()
    ov_l2_shapes = set()

    max_t1_instance_id = 0
    max_t2_instance_id = 0

    for index, sample_id in enumerate(
        entries,
        start=1,
    ):
        pair = cache.load(
            sample_id
        )

        stats = validate_loaded_pair(
            pair,
            sample_id,
        )

        sam_shapes.add(
            stats[
                "sam_shape"
            ]
        )

        ov_l1_shapes.add(
            stats[
                "ov_l1_shape"
            ]
        )

        ov_l2_shapes.add(
            stats[
                "ov_l2_shape"
            ]
        )

        max_t1_instance_id = max(
            max_t1_instance_id,
            int(
                stats[
                    "max_t1_instance"
                ]
            ),
        )

        max_t2_instance_id = max(
            max_t2_instance_id,
            int(
                stats[
                    "max_t2_instance"
                ]
            ),
        )

        if (
            args.progress_every > 0
            and (
                index
                % args.progress_every
                == 0
                or index
                == len(entries)
            )
        ):
            elapsed = (
                time.perf_counter()
                - started
            )

            rate = (
                index
                / max(
                    elapsed,
                    1e-9,
                )
            )

            print(
                f"Validated {index}/{len(entries)} samples "
                f"({rate:.2f} samples/s)",
                flush=True,
            )

    elapsed = (
        time.perf_counter()
        - started
    )

    result = {
        "status": "passed",
        "dataset_name": (
            args.dataset_name
        ),
        "n_samples": (
            len(
                entries
            )
        ),
        "fingerprint": (
            cache.fingerprint
        ),
        "sam_manifest_config_hash": (
            cache
            .sam
            .manifest[
                "config_hash"
            ]
        ),
        "ov_manifest_config_hash": (
            cache
            .ov
            .manifest[
                "config_hash"
            ]
        ),
        "sam_teacher_type": (
            cache
            .sam
            .manifest
            .get(
                "teacher_type"
            )
        ),
        "sam_shapes_seen": [
            list(
                shape
            )
            for shape in sorted(
                sam_shapes
            )
        ],
        "ov_l1_shapes_seen": [
            list(
                shape
            )
            for shape in sorted(
                ov_l1_shapes
            )
        ],
        "ov_l2_shapes_seen": [
            list(
                shape
            )
            for shape in sorted(
                ov_l2_shapes
            )
        ],
        "max_t1_instance_id_seen": (
            max_t1_instance_id
        ),
        "max_t2_instance_id_seen": (
            max_t2_instance_id
        ),
        "elapsed_seconds": (
            round(
                elapsed,
                3,
            )
        ),
        "scope": (
            "read-only validation of train coverage, manifest identity, "
            "sample IDs, referenced files, config hashes, tensor schema, "
            "tensor finiteness/ranges, and paired Run3 load contract; "
            "does not validate teacher accuracy, prior quality, "
            "augmentation replay, or downstream F1 gain"
        ),
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback

        traceback.print_exc()

        sys.exit(
            1
        )
