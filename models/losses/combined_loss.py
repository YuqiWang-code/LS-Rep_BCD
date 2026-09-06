"""BCE + Dice loss for probability outputs used by A2Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_reduction: str = "sample"):
        super().__init__()
        if dice_reduction not in {"sample", "batch"}:
            raise ValueError("dice_reduction must be 'sample' or 'batch'")
        self.dice_reduction = dice_reduction

    def forward(self, inputs, targets):
        # PyTorch 2.6 rejects probability-form BCE inside CUDA autocast. Keep
        # the numerically sensitive loss in FP32 while the network remains AMP.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            inputs = inputs.float()
            targets = targets.float()
            bce = F.binary_cross_entropy(inputs, targets)
            eps = 1e-5
            if self.dice_reduction == "sample":
                dims = tuple(range(1, inputs.ndim))
                inter = (inputs * targets).sum(dim=dims)
                denom = inputs.sum(dim=dims) + targets.sum(dim=dims)
                dice = (2.0 * inter + eps) / (denom + eps)
                dice_loss = 1.0 - dice.mean()
            else:
                inter = (inputs * targets).sum()
                dice = (2.0 * inter + eps) / (inputs.sum() + targets.sum() + eps)
                dice_loss = 1.0 - dice
        return bce + dice_loss


def build_loss(dice_reduction: str = "sample"):
    return BCEDiceLoss(dice_reduction=dice_reduction)
