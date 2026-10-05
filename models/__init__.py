"""CATA-CD v2 public API.

Deployable model: A2Net-LWGANet-L0 (+ optional Deployable Change Adapter).
Teachers / cache / translators / agent are training-only and are never part of
the deploy graph.
"""

from .a2net import A2Net_LWGANet_L0
from .losses.combined_loss import BCEDiceLoss, build_loss


__all__ = [
    "A2Net_LWGANet_L0",
    "BCEDiceLoss",
    "build_loss",
]
