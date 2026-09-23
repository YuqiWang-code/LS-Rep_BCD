"""Training-only auxiliary heads for SAGE-CD.

These heads are NOT part of the deployed student. They consume the TFM change
features (returned via ``return_change_features=True``) and produce full-res
logits that are distilled against teacher task targets, keeping the main
decoder's GT supervision isolated (CrossKD idea).

    c4 (1/16) -> AuxSemanticHead  -> semantic change logit [B,1,256,256]
    c2 (1/4)  -> AuxBoundaryHead  -> boundary change logit  [B,1,256,256]  (SAM stage)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxSemanticHead(nn.Module):
    """1x1 conv on the 1/16 change feature, upsampled to full resolution."""

    def __init__(self, in_ch: int = 64, out_size: int = 256):
        super().__init__()
        self.out_size = out_size
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, c4: torch.Tensor) -> torch.Tensor:
        z = self.conv(c4)  # [B,1,H/16,W/16]
        return F.interpolate(z, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


class AuxBoundaryHead(nn.Module):
    """Boundary change head on the 1/4 change feature (used in the SAM stage)."""

    def __init__(self, in_ch: int = 64, out_size: int = 256):
        super().__init__()
        self.out_size = out_size
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, c2: torch.Tensor) -> torch.Tensor:
        z = self.conv(c2)
        return F.interpolate(z, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


__all__ = ["AuxSemanticHead", "AuxBoundaryHead"]
