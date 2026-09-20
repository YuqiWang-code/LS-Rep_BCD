"""Parameter-free cross-temporal calibration for binary change detection.

This file implements the SCTC family used by A2Net-LWGANet-L0:

    S0: ``mode='none'``
        Identity baseline. No temporal calibration.

    S1: ``mode='symmetric'``
        Full-pixel symmetric pair calibration. Statistics are estimated from
        all spatial locations with equal weight.

    S2: ``mode='sctc'``
        Unchanged-Aware Symmetric Cross-Temporal Calibration (SCTC). A
        detached cosine-agreement map gives higher statistical weight to
        cross-temporally stable locations. Both temporal features are then
        mapped to one shared pair-specific feature domain before A2Net's
        original absolute-difference temporal fusion.

Design contracts
----------------
1. Zero learnable parameters and zero buffers.
2. No teacher, cache, GT, auxiliary branch, or training-only state.
3. The same operation is used during training, validation, and deployment.
4. Strict temporal symmetry: swapping T1/T2 swaps the calibrated outputs.
5. ``mode='none'`` is an exact identity path.
6. Agreement weights are detached so the backbone cannot improve the loss by
   directly manipulating the routing/statistical weights.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_CALIBRATION_MODES = (
    "none",
    "symmetric",
    "sctc",
)

# Numerical constant only. It is deliberately not exposed as a training
# hyper-parameter in the formal SCTC experiments.
SCTC_EPS = 1e-5


class SymmetricTemporalCalibration(nn.Module):
    """Calibrate one pair of same-shape temporal feature maps.

    Parameters
    ----------
    mode:
        ``none``
            Identity baseline.

        ``symmetric``
            Estimate per-sample/per-channel mean and variance from all pixels
            and map both temporal features to their shared midpoint domain.

        ``sctc``
            Estimate the same statistics with a detached cosine-agreement map
            as spatial weights. High-agreement locations contribute more to
            the nuisance/domain statistics.

    eps:
        Numerical stability constant. It does not create learnable state.
    """

    def __init__(
        self,
        mode: str = "none",
        eps: float = SCTC_EPS,
    ) -> None:
        super().__init__()

        mode = str(mode).lower()
        if mode not in SUPPORTED_CALIBRATION_MODES:
            raise ValueError(
                f"Unsupported temporal calibration mode {mode!r}; "
                f"expected one of {SUPPORTED_CALIBRATION_MODES}"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.mode = mode
        self.eps = float(eps)

    @staticmethod
    def _validate_pair(
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(x1) or not torch.is_tensor(x2):
            raise TypeError("Temporal features must be torch tensors")
        if x1.ndim != 4 or x2.ndim != 4:
            raise ValueError("Temporal features must be [B,C,H,W]")
        if x1.shape != x2.shape:
            raise ValueError(
                "Temporal features must have identical shapes, got "
                f"{tuple(x1.shape)} vs {tuple(x2.shape)}"
            )
        if x1.shape[1] < 1:
            raise ValueError("Temporal feature channel count must be positive")

    def _agreement_weight(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Return detached symmetric cosine-agreement weights in [0,1]."""
        x1_norm = F.normalize(
            x1,
            p=2,
            dim=1,
            eps=self.eps,
        )
        x2_norm = F.normalize(
            x2,
            p=2,
            dim=1,
            eps=self.eps,
        )
        cosine = (x1_norm * x2_norm).sum(dim=1, keepdim=True)
        agreement = (cosine + 1.0) * 0.5
        return agreement.clamp_(0.0, 1.0).detach()

    def _weighted_stats(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return weighted per-sample/per-channel mean and variance."""
        if weight.ndim != 4 or weight.shape[1] != 1:
            raise ValueError("weight must be [B,1,H,W]")
        if weight.shape[0] != x.shape[0] or weight.shape[-2:] != x.shape[-2:]:
            raise ValueError("weight and feature spatial/batch shapes must match")

        denom = weight.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        mean = (x * weight).sum(dim=(2, 3), keepdim=True) / denom
        variance = (
            ((x - mean).square() * weight).sum(dim=(2, 3), keepdim=True)
            / denom
        )
        return mean, variance

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        self._validate_pair(x1, x2)

        if self.mode == "none":
            if not return_diagnostics:
                return x1, x2
            zero = x1.detach().float().new_zeros(())
            one = x1.detach().float().new_ones(())
            diagnostics = {
                "agreement_mean": one,
                "weight_mean": one,
                "pre_abs_diff": (x1.detach().float() - x2.detach().float())
                .abs()
                .mean(),
                "post_abs_diff": (x1.detach().float() - x2.detach().float())
                .abs()
                .mean(),
                "calibration_shift_t1": zero,
                "calibration_shift_t2": zero,
            }
            return x1, x2, diagnostics

        original_dtype = x1.dtype

        # Keep the statistics numerically stable under AMP while preserving
        # the surrounding network's original mixed-precision policy.
        with torch.autocast(device_type=x1.device.type, enabled=False):
            f1 = x1.float()
            f2 = x2.float()

            agreement = self._agreement_weight(f1, f2)
            if self.mode == "symmetric":
                weight = torch.ones_like(agreement)
            else:
                weight = agreement

            mean1, var1 = self._weighted_stats(f1, weight)
            mean2, var2 = self._weighted_stats(f2, weight)

            std1 = torch.sqrt(var1 + self.eps)
            std2 = torch.sqrt(var2 + self.eps)

            # One common pair-specific domain. Using midpoint statistics keeps
            # the operation completely direction-free.
            shared_mean = 0.5 * (mean1 + mean2)
            shared_var = 0.5 * (var1 + var2)
            shared_std = torch.sqrt(shared_var + self.eps)

            calibrated1 = shared_std * ((f1 - mean1) / std1) + shared_mean
            calibrated2 = shared_std * ((f2 - mean2) / std2) + shared_mean

        calibrated1 = calibrated1.to(dtype=original_dtype)
        calibrated2 = calibrated2.to(dtype=original_dtype)

        if not return_diagnostics:
            return calibrated1, calibrated2

        diagnostics = {
            "agreement_mean": agreement.mean().detach(),
            "weight_mean": weight.mean().detach(),
            "pre_abs_diff": (f1.detach() - f2.detach()).abs().mean(),
            "post_abs_diff": (
                calibrated1.detach().float() - calibrated2.detach().float()
            )
            .abs()
            .mean(),
            "calibration_shift_t1": (
                calibrated1.detach().float() - f1.detach()
            )
            .abs()
            .mean(),
            "calibration_shift_t2": (
                calibrated2.detach().float() - f2.detach()
            )
            .abs()
            .mean(),
        }
        return calibrated1, calibrated2, diagnostics


class MultiScaleTemporalCalibration(nn.Module):
    """Apply the same zero-parameter temporal calibration to multiple scales."""

    def __init__(
        self,
        mode: str = "none",
        num_scales: int = 4,
        eps: float = SCTC_EPS,
    ) -> None:
        super().__init__()
        if num_scales < 1:
            raise ValueError("num_scales must be positive")

        self.mode = str(mode).lower()
        self.num_scales = int(num_scales)
        self.calibrator = SymmetricTemporalCalibration(
            mode=self.mode,
            eps=eps,
        )

    def forward(
        self,
        features1: Sequence[torch.Tensor],
        features2: Sequence[torch.Tensor],
        return_diagnostics: bool = False,
    ):
        if len(features1) != self.num_scales or len(features2) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} feature scales, got "
                f"{len(features1)} and {len(features2)}"
            )

        out1 = []
        out2 = []
        diagnostics: Dict[str, torch.Tensor] = {}

        for index, (x1, x2) in enumerate(zip(features1, features2), start=2):
            if return_diagnostics:
                y1, y2, detail = self.calibrator(
                    x1,
                    x2,
                    return_diagnostics=True,
                )
                for key, value in detail.items():
                    diagnostics[f"s{index}_{key}"] = value
            else:
                y1, y2 = self.calibrator(x1, x2)

            out1.append(y1)
            out2.append(y2)

        result1 = tuple(out1)
        result2 = tuple(out2)

        if return_diagnostics:
            return result1, result2, diagnostics
        return result1, result2


__all__ = [
    "SUPPORTED_CALIBRATION_MODES",
    "SCTC_EPS",
    "SymmetricTemporalCalibration",
    "MultiScaleTemporalCalibration",
]
