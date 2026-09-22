"""Failure-aware soft region weighting (FA-SCRD router).

Replaces the historical hard Brier-gain audit (which rejected ~99.9% of pixels)
with a continuous, detached soft advantage. Regions the student already gets
right, or where the teacher is unreliable, receive a low KD weight; regions the
student still struggles with and the teacher handles well receive a high weight.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_failure_weight(
    pred_student: torch.Tensor,
    pred_teacher: torch.Tensor,
    label: torch.Tensor,
    tau_a: float = 0.5,
) -> torch.Tensor:
    """Compute per-pixel soft KD weight (fully detached).

    Args:
        pred_student: [B, 1, H, W] student sigmoid probabilities.
        pred_teacher: [B, 1, H, W] teacher sigmoid probabilities (change logit).
        label:        [B, 1, H, W] binary labels (0/1).
        tau_a:        advantage temperature.

    Returns:
        weight [B, 1, H, W], detached.
    """
    p_s = pred_student.detach().float()
    p_t = pred_teacher.detach().float()
    y = label.detach().float()

    difficulty = (p_s - y).abs()  # student difficulty d_r
    e_s = F.binary_cross_entropy(p_s.clamp(1e-5, 1 - 1e-5), y, reduction="none")
    e_t = F.binary_cross_entropy(p_t.clamp(1e-5, 1 - 1e-5), y, reduction="none")

    advantage = torch.sigmoid((e_s - e_t) / tau_a)  # soft advantage a_r
    weight = (difficulty * advantage).detach()
    return weight
