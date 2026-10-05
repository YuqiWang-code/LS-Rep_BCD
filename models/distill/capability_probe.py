"""Train-only teacher capability probes (used by the selection Agent, not for test).

These are cheap, offline statistics computed on the TRAIN split from cached
teacher evidence + labels. They do not decide whether a teacher works; the full
40K student run does. They only provide the Agent an interpretable input state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def separability_scores(local_change: torch.Tensor, label: torch.Tensor):
    """Change vs unchanged separability of cached teacher evidence.

    ``local_change``: [N, C, H, W] (or [C,H,W]); ``label``: [N,1,H,W] binary.
    Returns a dict with change_mean, unchanged_mean, ratio, and pooled AUC proxy.
    """
    if local_change.dim() == 3:
        local_change = local_change.unsqueeze(0)
    if label.dim() == 3:
        label = label.unsqueeze(0)
    label = F.interpolate(label.float(), size=local_change.shape[-2:], mode="nearest")

    mag = local_change.abs().mean(dim=1)  # [N,H,W]
    pos = label.bool()
    neg = ~pos
    change_mean = mag[pos].mean().item() if pos.any() else 0.0
    unchanged_mean = mag[neg].mean().item() if neg.any() else 0.0
    ratio = (change_mean + 1e-6) / (unchanged_mean + 1e-6)
    return {
        "change_mean": change_mean,
        "unchanged_mean": unchanged_mean,
        "ratio": ratio,
    }


@torch.no_grad()
def confidence_calibration(confidence: torch.Tensor, label: torch.Tensor):
    """Mean confidence on changed vs unchanged pixels."""
    if confidence.dim() == 3:
        confidence = confidence.unsqueeze(0)
    if label.dim() == 3:
        label = label.unsqueeze(0)
    label = F.interpolate(label.float(), size=confidence.shape[-2:], mode="nearest")
    pos = label.bool()
    neg = ~pos
    return {
        "conf_change": confidence[pos].mean().item() if pos.any() else 0.0,
        "conf_unchanged": confidence[neg].mean().item() if neg.any() else 0.0,
    }


__all__ = ["separability_scores", "confidence_calibration"]
