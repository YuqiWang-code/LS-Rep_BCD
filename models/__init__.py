"""Public model API for LS-Rep_BCD Run3 BT-SAM-RDT.

The deployable model remains:
    A2Net-LWGANet-L0

Run3 BT-SAM-RDT is a training-only auxiliary mechanism. It is configured
through ``A2Net_LWGANet_L0(auxiliary_mode="bt_sam_rdt", ...)`` and is removed
by ``switch_to_deploy()`` before deployment.

This package-level namespace intentionally exposes only the stable public
training/deployment API.
"""

from .a2net import A2Net_LWGANet_L0
from .losses.combined_loss import BCEDiceLoss, build_loss


__all__ = [
    "A2Net_LWGANet_L0",
    "BCEDiceLoss",
    "build_loss",
]
