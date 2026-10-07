"""Train-only teacher capability probes (used by the selection Agent, not for test).

These are cheap, offline statistics computed on the TRAIN split from cached
teacher evidence + labels. They do not decide whether a teacher works; the full
40K student run does. They only provide the Agent an interpretable input state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_4d(x: torch.Tensor) -> torch.Tensor:
    """Normalize [H,W] / [1,H,W] / [N,1,H,W] to 4D."""
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x


@torch.no_grad()
def separability_scores(local_change: torch.Tensor, label: torch.Tensor):
    """Change vs unchanged separability of cached teacher evidence.

    ``local_change``: [N, C, H, W] (or [C,H,W]); ``label``: [H,W] / [N,1,H,W] binary.
    """
    if local_change.dim() == 3:
        local_change = local_change.unsqueeze(0)
    label = _to_4d(label)
    label = F.interpolate(label.float(), size=local_change.shape[-2:], mode="nearest").squeeze(1)

    mag = local_change.abs().mean(dim=1)  # [N,H,W]
    pos = label.bool()                     # [N,H,W]
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
    label = _to_4d(label)
    label = F.interpolate(label.float(), size=confidence.shape[-2:], mode="nearest").squeeze(1)
    conf = confidence.mean(dim=1)  # [N,H,W]
    pos = label.bool()
    neg = ~pos
    return {
        "conf_change": conf[pos].mean().item() if pos.any() else 0.0,
        "conf_unchanged": conf[neg].mean().item() if neg.any() else 0.0,
    }


__all__ = ["separability_scores", "confidence_calibration"]
