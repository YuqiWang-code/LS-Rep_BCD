"""Atomic baseline / Direction-C / dynamic-teacher checkpoints.

Checkpoint format
-----------------
v2:
    Historical B0 / C0 / C1-C5 checkpoints.
    Stores:
        model
        main optimizer
        optional router optimizer
        training progress
        args
        RNG state

v3:
    Adds:
        optional teacher optimizer

For RDT-CD, both fast teachers and EMA target teachers are normal registered
submodules of model.training_auxiliary, so their weights are already included
in model.state_dict().  Exact resume therefore additionally requires only the
separate teacher optimizer state, which is stored here in v3.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 3


def rng_state() -> Dict[str, Any]:
    """
    Capture all RNG states needed for exact training resume.

    CUDA RNG state is saved for every currently visible CUDA device.
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
    """
    Restore RNG state saved by rng_state().

    Historical checkpoints may pass None; in that case this function is a
    no-op.
    """
    if not state:
        return

    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_state = state.get("torch")
    cuda_state = state.get("cuda")

    if python_state is not None:
        random.setstate(python_state)

    if numpy_state is not None:
        np.random.set_state(numpy_state)

    if torch_state is not None:
        torch.set_rng_state(torch_state)

    if (
        cuda_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(cuda_state)


def build_checkpoint(
    model,
    optimizer,
    epoch,
    global_step,
    best_val_f1,
    args,
    router_optimizer=None,
    teacher_optimizer=None,
):
    """
    Build one complete training checkpoint.

    Parameters
    ----------
    model:
        Full model.

        For dynamic_teacher runs, model.state_dict() already contains:
            training_auxiliary.fast.*
            training_auxiliary.target.*

        Therefore fast-teacher and EMA-target weights require no special
        checkpoint field.

    optimizer:
        Main student optimizer.

    epoch:
        Current epoch index.

    global_step:
        Current global optimization-step count.

    best_val_f1:
        Best validation F1 observed so far.

    args:
        argparse Namespace or Namespace-like object accepted by vars().

    router_optimizer:
        Optional legacy Direction-C router optimizer.

    teacher_optimizer:
        Optional RDT-CD fast-teacher optimizer.

        This MUST be saved for exact dynamic-teacher resume because its Adam
        moments are not part of model.state_dict().

    Returns
    -------
    dict
        Checkpoint format version 3.
    """
    if model is None:
        raise ValueError("model must not be None")

    if optimizer is None:
        raise ValueError("optimizer must not be None")

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,

        # Model state includes the complete training auxiliary whenever it
        # still exists at save time.
        "model": model.state_dict(),

        # Optimizer states are intentionally separate because the student,
        # legacy router, and dynamic teachers have disjoint update graphs.
        "optimizer": optimizer.state_dict(),

        "router_optimizer": (
            router_optimizer.state_dict()
            if router_optimizer is not None
            else None
        ),

        "teacher_optimizer": (
            teacher_optimizer.state_dict()
            if teacher_optimizer is not None
            else None
        ),

        # Training progress.
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_f1": float(best_val_f1),

        # Exact experiment configuration.
        "args": vars(args),

        # Exact stochastic state.
        "rng": rng_state(),
    }


def save_checkpoint_atomic(
    checkpoint,
    path,
) -> None:
    """
    Atomically save a checkpoint.

    The checkpoint is first written to:
        <destination>.tmp

    and only after torch.save() succeeds is it atomically moved over the
    destination with os.replace().  This avoids leaving a partially written
    final checkpoint after interruption.

    Existing valid checkpoints are not removed before the replacement
    succeeds.
    """
    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    try:
        torch.save(
            checkpoint,
            temporary,
        )

        os.replace(
            temporary,
            destination,
        )

    except Exception:
        # Best-effort cleanup of an incomplete temporary file only.
        # Never delete/overwrite the existing destination on failure.
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
