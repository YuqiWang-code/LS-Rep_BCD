"""SAM structure adapter for A2Net training."""

from __future__ import annotations

import torch.nn as nn

from .sam_structure_loss import sam_affinity_loss, sam_boundary_loss


class SAMStructureAdapter(nn.Module):
    """
    Training-only SAM structure adapter.

    Applies boundary and affinity losses using cached SAM structure information.
    No learnable parameters; deleted at deployment.

    Args:
        boundary_weight: Weight for boundary loss
        affinity_weight: Weight for affinity loss
        boundary_band: Dilation kernel for GT boundary band
    """

    def __init__(
        self,
        boundary_weight=0.7,
        affinity_weight=0.3,
        boundary_band=7,
    ):
        super().__init__()

        self.boundary_weight = boundary_weight
        self.affinity_weight = affinity_weight
        self.boundary_band = boundary_band

    def forward(self, masks, target, teacher_pack):
        """
        Compute SAM structure loss.

        Args:
            masks: Tuple of predictions at multiple scales, use masks[0] (finest)
            target: GT change mask [B, 1, H, W]
            teacher_pack: Dict with 't1' and 't2' structure info

        Returns:
            Dict with 'total', 'boundary', 'affinity' losses
        """
        prediction = masks[0]

        loss_boundary = sam_boundary_loss(
            prediction,
            target,
            teacher_pack,
            self.boundary_band,
        )

        loss_affinity = sam_affinity_loss(
            prediction,
            target,
            teacher_pack,
        )

        total = (
            self.boundary_weight * loss_boundary
            + self.affinity_weight * loss_affinity
        )

        return {
            "total": total,
            "boundary": loss_boundary,
            "affinity": loss_affinity,
        }
