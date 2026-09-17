"""Run3 BT-SAM-RDT checkpoint utilities.

This module defines the single checkpoint format used by the current Run3
codebase.

Run3 checkpoint contract
------------------------
format_version = 3

A training checkpoint stores:

    model
        Complete training model state.

        For R3A / R3 this includes:
            training_auxiliary.fast.*
            training_auxiliary.target.*

        Therefore Fast Teacher and EMA Target Teacher weights are already
        included in model.state_dict().

    optimizer
        Student optimizer state.

    teacher_optimizer
        Fast Teacher optimizer state for R3A / R3.
        None for B0.

    epoch
    global_step
    best_val_f1

    args
        Exact experiment configuration required by train.py resume checks.

    rng
        Python / NumPy / PyTorch CPU / all visible CUDA RNG states.

Historical Direction-C / router / format-v2 compatibility is intentionally
not implemented here. Current Run3 training must not silently resume an old
experiment package.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 3


# ============================================================================
# RNG state
# ============================================================================


def rng_state() -> Dict[str, Any]:
    """Capture stochastic state required for exact Run3 resume.

    CUDA RNG state is captured for every currently visible CUDA device.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(
    state: Optional[Dict[str, Any]],
) -> None:
    """Restore a Run3 RNG snapshot.

    Parameters
    ----------
    state:
        Dictionary produced by :func:`rng_state`.

        ``None`` is accepted only as a convenience for callers that explicitly
        have no RNG state to restore. Current Run3 checkpoints are expected to
        contain a non-None RNG dictionary.
    """
    if state is None:
        return

    if not isinstance(
        state,
        dict,
    ):
        raise TypeError(
            "RNG state must be a dict or None"
        )

    required = {
        "python",
        "numpy",
        "torch",
        "cuda",
    }

    missing = (
        required
        - set(
            state.keys()
        )
    )

    if missing:
        raise ValueError(
            "RNG state is missing fields: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    python_state = state[
        "python"
    ]

    numpy_state = state[
        "numpy"
    ]

    torch_state = state[
        "torch"
    ]

    cuda_state = state[
        "cuda"
    ]

    if python_state is None:
        raise ValueError(
            "Run3 RNG state is missing Python RNG state"
        )

    if numpy_state is None:
        raise ValueError(
            "Run3 RNG state is missing NumPy RNG state"
        )

    if torch_state is None:
        raise ValueError(
            "Run3 RNG state is missing PyTorch CPU RNG state"
        )

    random.setstate(
        python_state
    )

    np.random.set_state(
        numpy_state
    )

    torch.set_rng_state(
        torch_state
    )

    # A checkpoint produced without CUDA can be restored on CPU.
    if cuda_state is None:
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Checkpoint contains CUDA RNG state but CUDA is unavailable; "
            "exact Run3 resume cannot be guaranteed"
        )

    if not isinstance(
        cuda_state,
        (list, tuple),
    ):
        raise TypeError(
            "CUDA RNG state must be a list/tuple of per-device states"
        )

    visible_devices = (
        torch.cuda.device_count()
    )

    if (
        len(
            cuda_state
        )
        != visible_devices
    ):
        raise RuntimeError(
            "CUDA device-count mismatch during exact resume: "
            f"checkpoint={len(cuda_state)}, "
            f"visible={visible_devices}"
        )

    torch.cuda.set_rng_state_all(
        cuda_state
    )


# ============================================================================
# Training checkpoint
# ============================================================================


def build_checkpoint(
    model,
    optimizer,
    epoch,
    global_step,
    best_val_f1,
    args,
    teacher_optimizer=None,
):
    """Build one complete Run3 format-v3 training checkpoint.

    Parameters
    ----------
    model:
        Full training model.

        For R3A/R3, ``model.state_dict()`` already contains both:
            training_auxiliary.fast.*
            training_auxiliary.target.*

    optimizer:
        Student optimizer.

    epoch:
        Current zero-based epoch index.

    global_step:
        Number of completed optimization steps.

    best_val_f1:
        Best validation F1 observed so far.

    args:
        argparse Namespace or Namespace-like object accepted by ``vars()``.

    teacher_optimizer:
        Fast Teacher optimizer for R3A/R3.
        ``None`` for B0.

    Returns
    -------
    dict
        Run3 checkpoint with ``format_version == 3``.
    """
    if model is None:
        raise ValueError(
            "model must not be None"
        )

    if optimizer is None:
        raise ValueError(
            "Student optimizer must not be None"
        )

    if args is None:
        raise ValueError(
            "args must not be None"
        )

    try:
        args_dict = dict(
            vars(
                args
            )
        )

    except TypeError as exc:
        raise TypeError(
            "args must support vars(args)"
        ) from exc

    if not args_dict:
        raise ValueError(
            "args must contain the Run3 experiment configuration"
        )

    experiment = args_dict.get(
        "experiment"
    )

    teacher_update = bool(
        args_dict.get(
            "teacher_update",
            False,
        )
    )

    mechanism = args_dict.get(
        "mechanism"
    )

    # R3A/R3 must carry a Fast Teacher optimizer for exact resume.
    expects_teacher_optimizer = (
        teacher_update
        or mechanism == "bt_sam_rdt"
        or experiment in {
            "R3A",
            "R3",
        }
    )

    if (
        expects_teacher_optimizer
        and teacher_optimizer is None
    ):
        raise ValueError(
            "Run3 BT-SAM-RDT checkpoint requires teacher_optimizer "
            "for exact resume"
        )

    # B0 should not accidentally contain a Teacher optimizer.
    if (
        not expects_teacher_optimizer
        and teacher_optimizer is not None
    ):
        raise ValueError(
            "Clean B0 checkpoint must not contain teacher_optimizer"
        )

    model_state = (
        model.state_dict()
    )

    if not model_state:
        raise ValueError(
            "model.state_dict() is empty"
        )

    has_auxiliary_state = any(
        key.startswith(
            "training_auxiliary."
        )
        for key in model_state.keys()
    )

    if (
        expects_teacher_optimizer
        and not has_auxiliary_state
    ):
        raise ValueError(
            "Run3 BT-SAM-RDT configuration expects training_auxiliary "
            "state, but model.state_dict() contains none"
        )

    if (
        not expects_teacher_optimizer
        and has_auxiliary_state
    ):
        raise ValueError(
            "Clean B0 configuration unexpectedly contains "
            "training_auxiliary state"
        )

    epoch = int(
        epoch
    )

    global_step = int(
        global_step
    )

    best_val_f1 = float(
        best_val_f1
    )

    if epoch < 0:
        raise ValueError(
            "epoch must be non-negative"
        )

    if global_step < 0:
        raise ValueError(
            "global_step must be non-negative"
        )

    if not np.isfinite(
        best_val_f1
    ):
        raise ValueError(
            "best_val_f1 must be finite"
        )

    return {
        "format_version": (
            CHECKPOINT_FORMAT_VERSION
        ),

        # Complete training state.
        "model": (
            model_state
        ),

        # Student optimizer.
        "optimizer": (
            optimizer.state_dict()
        ),

        # Fast Teacher optimizer for R3A/R3; None for B0.
        "teacher_optimizer": (
            teacher_optimizer.state_dict()
            if teacher_optimizer is not None
            else None
        ),

        # Training progress.
        "epoch": (
            epoch
        ),

        "global_step": (
            global_step
        ),

        "best_val_f1": (
            best_val_f1
        ),

        # Exact experiment configuration.
        "args": (
            args_dict
        ),

        # Exact stochastic state.
        "rng": (
            rng_state()
        ),
    }


# ============================================================================
# Atomic persistence
# ============================================================================


def save_checkpoint_atomic(
    checkpoint,
    path,
) -> None:
    """Atomically save one checkpoint.

    The payload is first written to:

        <destination>.tmp

    and is moved over the destination with ``os.replace`` only after
    ``torch.save`` succeeds.

    This means:
        - an interrupted temporary write does not destroy the previous valid
          destination;
        - replacing ``last_checkpoint.pth`` remains atomic;
        - best-checkpoint retention/deletion policy stays outside this utility
          and is controlled by train.py.
    """
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "checkpoint must be a dict"
        )

    destination = Path(
        path
    )

    if destination.name in {
        "",
        ".",
        "..",
    }:
        raise ValueError(
            f"Invalid checkpoint destination: {destination}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(
        destination.name
        + ".tmp"
    )

    # Remove only a stale temporary file from a previous interrupted write.
    #
    # Never remove the valid destination here.
    if temporary.exists():
        if temporary.is_dir():
            raise IsADirectoryError(
                temporary
            )

        temporary.unlink()

    try:
        torch.save(
            checkpoint,
            temporary,
        )

        if not temporary.is_file():
            raise RuntimeError(
                "torch.save returned without creating the temporary checkpoint"
            )

        os.replace(
            temporary,
            destination,
        )

    except Exception:
        # Best-effort cleanup of the incomplete temporary file only.
        # The previous valid destination is untouched unless os.replace()
        # already succeeded.
        try:
            if temporary.exists():
                temporary.unlink()

        except OSError:
            pass

        raise


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "rng_state",
    "restore_rng_state",
    "build_checkpoint",
    "save_checkpoint_atomic",
]
