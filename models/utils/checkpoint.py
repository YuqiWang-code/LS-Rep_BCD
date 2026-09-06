"""Atomic SAM-HSD checkpoints with optimizer and RNG state."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_checkpoint(model, optimizer, epoch, global_step, best_val_f1, args):
    """
    Build checkpoint dict.

    Args:
        model: Model state dict
        optimizer: Main optimizer state dict
        epoch: Current epoch
        global_step: Global step count
        best_val_f1: Best validation F1 so far
        args: Training arguments

    Returns:
        Checkpoint dict
    """
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_f1": best_val_f1,
        "args": vars(args),
        "rng": rng_state(),
    }


def save_checkpoint_atomic(checkpoint, path):
    """Avoid leaving a partially written checkpoint after interruption."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)
