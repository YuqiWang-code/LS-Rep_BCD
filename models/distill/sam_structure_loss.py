"""SAM structure loss functions for boundary and affinity supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_mean(loss, weight, eps=1e-6):
    """Compute weighted mean with numerical stability."""
    return (loss * weight).sum() / weight.sum().clamp_min(eps)


def soft_boundary(prob, kernel=3):
    """
    Compute soft boundary from probability map using morphological gradient.

    Args:
        prob: Probability map [B, 1, H, W]
        kernel: Kernel size for dilation/erosion

    Returns:
        Boundary map [B, 1, H, W] in range [0, 1]
    """
    pad = kernel // 2

    dilated = F.max_pool2d(
        prob,
        kernel,
        stride=1,
        padding=pad,
    )

    eroded = -F.max_pool2d(
        -prob,
        kernel,
        stride=1,
        padding=pad,
    )

    return (dilated - eroded).clamp(0, 1)


def sam_boundary_loss(
    prediction,
    target,
    teacher_pack,
    boundary_band=7,
):
    """
    SAM boundary structural loss.

    Only supervises predicted boundaries near GT change boundaries using
    SAM's object boundary maps weighted by mask quality.

    Args:
        prediction: Model prediction [B, 1, H, W]
        target: GT change mask [B, 1, H, W]
        teacher_pack: Dict with 't1' and 't2', each containing 'boundary' and 'quality'
        boundary_band: Dilation kernel for GT boundary band

    Returns:
        Scalar loss
    """
    b1 = teacher_pack["t1"]["boundary"].float()
    b2 = teacher_pack["t2"]["boundary"].float()

    q1 = teacher_pack["t1"]["quality"].float()
    q2 = teacher_pack["t2"]["quality"].float()

    # Union of T1/T2 boundaries (max captures either time's structure)
    sam_boundary = torch.maximum(b1, b2)
    quality = torch.maximum(q1, q2)

    # Predicted boundary
    pred_boundary = soft_boundary(prediction.float())

    # GT boundary
    gt_boundary = soft_boundary(target.float())

    # GT boundary band (only supervise near actual change boundaries)
    gt_band = F.max_pool2d(
        gt_boundary,
        boundary_band,
        stride=1,
        padding=boundary_band // 2,
    )

    # Weight: SAM boundary * quality * near GT change boundary
    weight = (sam_boundary * quality * gt_band).detach()

    # Loss: encourage predicted boundary to be 1 where SAM says there's a boundary
    loss_map = F.smooth_l1_loss(
        pred_boundary,
        torch.ones_like(pred_boundary),
        reduction="none",
    )

    return weighted_mean(loss_map, weight)


def sam_affinity_loss(
    prediction,
    target,
    teacher_pack,
):
    """
    Intra-object affinity loss.

    Encourages pixels within the same SAM object and same GT change region
    to have similar predictions.

    Args:
        prediction: Model prediction [B, 1, H, W]
        target: GT change mask [B, 1, H, W]
        teacher_pack: Dict with 't1' and 't2', each containing 'instance_id' and 'quality'

    Returns:
        Scalar loss
    """
    # Only consider GT change pixels
    change_mask = (target > 0.5).float()

    # 4-neighborhood shifts
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    total_loss = 0.0
    total_weight = 0.0

    # T1 and T2 partitions are independent. Accumulate valid pairs from both
    # rather than discarding structures that only exist in one time phase.
    for time_key in ("t1", "t2"):
        instance_id = teacher_pack[time_key]["instance_id"].long()
        quality = teacher_pack[time_key]["quality"].float()
        for dy, dx in shifts:
            shifted_id = torch.roll(instance_id, shifts=(dy, dx), dims=(-2, -1))
            shifted_q = torch.roll(quality, shifts=(dy, dx), dims=(-2, -1))
            same_object = (instance_id == shifted_id) & (instance_id > 0)
            both_change = change_mask * torch.roll(change_mask, shifts=(dy, dx), dims=(-2, -1))
            high_quality = torch.minimum(quality, shifted_q) > 0.5
            valid = (same_object * both_change * high_quality).float()
            if dy == -1:
                valid[:, :, -1, :] = 0
            elif dy == 1:
                valid[:, :, 0, :] = 0
            if dx == -1:
                valid[:, :, :, -1] = 0
            elif dx == 1:
                valid[:, :, :, 0] = 0
            pred_diff = torch.abs(
                prediction - torch.roll(prediction, shifts=(dy, dx), dims=(-2, -1))
            )
            weight = valid * torch.minimum(quality, shifted_q)
            total_loss += (pred_diff * weight).sum()
            total_weight += weight.sum()

    return total_loss / total_weight.clamp_min(1e-6)
