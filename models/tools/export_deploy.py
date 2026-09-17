#!/usr/bin/env python3
"""Export the unchanged deployable A2Net student from a Run3 v3 checkpoint.

The Run3 training checkpoint contains both:
    - the deployable A2Net-LWGANet-L0 student; and
    - the complete training-only BT-SAM-RDT auxiliary
      (Fast Teacher + EMA Target Teacher).

This tool exports ONLY the student graph.

Safety / correctness contract
-----------------------------
1. Accept only the current training checkpoint format v3.
2. Never load training_auxiliary.* into the deploy model.
3. Require strict Student state loading.
4. Physically call switch_to_deploy().
5. Require exactly 2,913,094 deploy parameters.
6. Require no training_auxiliary.* keys in the exported state.
7. Refuse to overwrite an existing output file.
8. Save atomically through models.utils.checkpoint.save_checkpoint_atomic.

Run from the project root:

    python models/tools/export_deploy.py \
        --checkpoint /path/to/best_model_F1=xxxxxx.pth \
        --output /path/to/deploy_student.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from models import A2Net_LWGANet_L0  # noqa: E402
from models.utils.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    save_checkpoint_atomic,
)


EXPECTED_DEPLOY_PARAMS = 2_913_094

DEPLOY_ARTIFACT_TYPE = "a2net_lwganet_l0_deploy_student"
DEPLOY_FORMAT_VERSION = 1


# ============================================================================
# Arguments
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the unchanged A2Net-LWGANet-L0 student "
            "from a Run3 format-v3 training checkpoint"
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Run3 training checkpoint (.pth)",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="New output path for the deploy-only checkpoint",
    )

    return parser.parse_args()


# ============================================================================
# Validation helpers
# ============================================================================


def _require_mapping(
    value: Any,
    name: str,
) -> Mapping:
    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(
            f"{name} must be a mapping"
        )

    return value


def validate_training_checkpoint(
    checkpoint: Any,
    source_path: Path,
) -> Dict[str, Any]:
    """Validate the minimum Run3 v3 checkpoint contract."""
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            f"{source_path}: expected checkpoint dict"
        )

    version = checkpoint.get(
        "format_version"
    )

    if (
        version
        != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            f"{source_path}: expected current Run3 training checkpoint "
            f"format_version={CHECKPOINT_FORMAT_VERSION}, "
            f"got {version!r}. "
            "Historical Direction-C/v2 checkpoints are intentionally rejected."
        )

    if "model" not in checkpoint:
        raise KeyError(
            f"{source_path}: checkpoint missing 'model'"
        )

    model_state = _require_mapping(
        checkpoint[
            "model"
        ],
        "checkpoint['model']",
    )

    if not model_state:
        raise ValueError(
            f"{source_path}: checkpoint model state is empty"
        )

    # Run3 B0 checkpoints legitimately have no training auxiliary.
    #
    # R3A/R3 checkpoints should have one. We infer the expected behavior from
    # the saved experiment configuration and reject an inconsistent package.
    saved_args = checkpoint.get(
        "args",
        {},
    )

    if saved_args is None:
        saved_args = {}

    saved_args = _require_mapping(
        saved_args,
        "checkpoint['args']",
    )

    experiment = saved_args.get(
        "experiment"
    )

    mechanism = saved_args.get(
        "mechanism"
    )

    teacher_update = bool(
        saved_args.get(
            "teacher_update",
            False,
        )
    )

    has_auxiliary_state = any(
        str(key).startswith(
            "training_auxiliary."
        )
        for key in model_state.keys()
    )

    expects_auxiliary = (
        teacher_update
        or mechanism == "bt_sam_rdt"
        or experiment in {
            "R3A",
            "R3",
        }
    )

    if (
        expects_auxiliary
        and not has_auxiliary_state
    ):
        raise ValueError(
            f"{source_path}: saved configuration indicates a Run3 "
            "BT-SAM-RDT experiment, but model state contains no "
            "training_auxiliary.* parameters"
        )

    required_progress = (
        "global_step",
        "epoch",
        "best_val_f1",
    )

    for key in required_progress:
        if key not in checkpoint:
            raise KeyError(
                f"{source_path}: checkpoint missing {key!r}"
            )

    return {
        "model_state": dict(
            model_state
        ),
        "saved_args": dict(
            saved_args
        ),
        "has_auxiliary_state": (
            has_auxiliary_state
        ),
    }


def extract_student_state(
    model_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remove every training-only auxiliary parameter/buffer."""
    student_state = {
        key: value
        for key, value in model_state.items()
        if not str(
            key
        ).startswith(
            "training_auxiliary."
        )
    }

    if not student_state:
        raise ValueError(
            "No Student parameters remain after removing "
            "training_auxiliary.*"
        )

    leaked_keys = [
        key
        for key in student_state
        if str(
            key
        ).startswith(
            "training_auxiliary."
        )
    ]

    if leaked_keys:
        raise RuntimeError(
            "Internal export error: auxiliary keys survived filtering"
        )

    return student_state


def build_deploy_model(
    student_state: Mapping[str, torch.Tensor],
) -> A2Net_LWGANet_L0:
    """Strictly reconstruct and validate the deployable Student."""
    model = A2Net_LWGANet_L0(
        pretrained=False,
        pretrained_path=None,
        auxiliary_mode="none",
        auxiliary_cfg=None,
    )

    incompatible = model.load_state_dict(
        student_state,
        strict=True,
    )

    # strict=True already raises on mismatch. Keep explicit assertions so a
    # future PyTorch/API change cannot silently weaken this export contract.
    if incompatible.missing_keys:
        raise RuntimeError(
            "Deploy Student load has missing keys: "
            + ", ".join(
                incompatible.missing_keys
            )
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Deploy Student load has unexpected keys: "
            + ", ".join(
                incompatible.unexpected_keys
            )
        )

    model.switch_to_deploy()
    model.eval()

    deploy_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if (
        deploy_params
        != EXPECTED_DEPLOY_PARAMS
    ):
        raise RuntimeError(
            "Unexpected deploy parameter count: "
            f"{deploy_params:,}; "
            f"expected {EXPECTED_DEPLOY_PARAMS:,}"
        )

    if hasattr(
        model,
        "training_auxiliary",
    ):
        raise RuntimeError(
            "switch_to_deploy() failed to remove training_auxiliary"
        )

    if (
        model.auxiliary_mode
        != "none"
    ):
        raise RuntimeError(
            "Deploy model auxiliary_mode must be 'none'"
        )

    if (
        model.training_mechanism
        != "none"
    ):
        raise RuntimeError(
            "Deploy model training_mechanism must be 'none'"
        )

    leaked_state_keys = [
        key
        for key in model.state_dict().keys()
        if key.startswith(
            "training_auxiliary."
        )
    ]

    if leaked_state_keys:
        raise RuntimeError(
            "Training-only state leaked into deploy model: "
            + ", ".join(
                leaked_state_keys[
                    :5
                ]
            )
        )

    return model


# ============================================================================
# Export
# ============================================================================


def build_deploy_package(
    checkpoint: Mapping[str, Any],
    model: A2Net_LWGANet_L0,
    source_path: Path,
    had_auxiliary_state: bool,
) -> Dict[str, Any]:
    """Build a minimal deploy artifact with provenance metadata."""
    saved_args = checkpoint.get(
        "args",
        {},
    )

    if not isinstance(
        saved_args,
        Mapping,
    ):
        saved_args = {}

    # Keep provenance compact and non-training.
    source_metadata = {
        "checkpoint": str(
            source_path
        ),
        "training_format_version": int(
            checkpoint[
                "format_version"
            ]
        ),
        "global_step": int(
            checkpoint[
                "global_step"
            ]
        ),
        "epoch": int(
            checkpoint[
                "epoch"
            ]
        ),
        "best_val_f1": float(
            checkpoint[
                "best_val_f1"
            ]
        ),
        "experiment": saved_args.get(
            "experiment"
        ),
        "experiment_name": saved_args.get(
            "experiment_name"
        ),
        "implementation_version": saved_args.get(
            "implementation_version"
        ),
        "dataset_name": saved_args.get(
            "dataset_name"
        ),
        "seed": saved_args.get(
            "seed"
        ),
        "had_training_auxiliary": bool(
            had_auxiliary_state
        ),
    }

    return {
        "artifact_type": (
            DEPLOY_ARTIFACT_TYPE
        ),
        "format_version": (
            DEPLOY_FORMAT_VERSION
        ),
        "model": (
            model.state_dict()
        ),
        "deploy_parameters": (
            EXPECTED_DEPLOY_PARAMS
        ),
        "input_size": (
            256
        ),
        "source": (
            source_metadata
        ),
    }


def verify_deploy_package(
    package: Mapping[str, Any],
) -> None:
    """Final in-memory check immediately before disk write."""
    if (
        package.get(
            "artifact_type"
        )
        != DEPLOY_ARTIFACT_TYPE
    ):
        raise RuntimeError(
            "Invalid deploy artifact_type"
        )

    if (
        package.get(
            "format_version"
        )
        != DEPLOY_FORMAT_VERSION
    ):
        raise RuntimeError(
            "Invalid deploy format_version"
        )

    if (
        package.get(
            "deploy_parameters"
        )
        != EXPECTED_DEPLOY_PARAMS
    ):
        raise RuntimeError(
            "Invalid deploy parameter metadata"
        )

    state = _require_mapping(
        package.get(
            "model"
        ),
        "deploy package model state",
    )

    leaked = [
        key
        for key in state.keys()
        if str(
            key
        ).startswith(
            "training_auxiliary."
        )
    ]

    if leaked:
        raise RuntimeError(
            "Deploy package contains training-only auxiliary state"
        )


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()

    source_path = Path(
        args.checkpoint
    ).expanduser()

    output_path = Path(
        args.output
    ).expanduser()

    if not source_path.is_file():
        raise FileNotFoundError(
            source_path
        )

    # Export is intentionally non-destructive.
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists; "
            "choose a new output path"
        )

    if (
        source_path.resolve()
        == output_path.resolve()
    ):
        raise ValueError(
            "Input training checkpoint and output deploy checkpoint "
            "must be different files"
        )

    checkpoint = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
    )

    validated = validate_training_checkpoint(
        checkpoint,
        source_path,
    )

    student_state = extract_student_state(
        validated[
            "model_state"
        ]
    )

    model = build_deploy_model(
        student_state
    )

    package = build_deploy_package(
        checkpoint=checkpoint,
        model=model,
        source_path=source_path,
        had_auxiliary_state=validated[
            "has_auxiliary_state"
        ],
    )

    verify_deploy_package(
        package
    )

    save_checkpoint_atomic(
        package,
        output_path,
    )

    print(
        "=" * 80
    )

    print(
        "Run3 deploy export passed"
    )

    print(
        f"Source: {source_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Source training format: "
        f"v{checkpoint['format_version']}"
    )

    print(
        f"Source global step: "
        f"{checkpoint['global_step']}"
    )

    print(
        f"Training auxiliary removed: "
        f"{validated['has_auxiliary_state']}"
    )

    print(
        f"Deploy parameters: "
        f"{EXPECTED_DEPLOY_PARAMS:,}"
    )

    print(
        "=" * 80
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
