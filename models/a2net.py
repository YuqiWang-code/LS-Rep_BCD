"""A2Net-LWGANet-L0 with parameter-free SCTC temporal calibration."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.lwganet import LWGANet_L0_1242_e32_k11_GELU
from .decoder.a2net_decoder import (
    Decoder,
    NeighborFeatureAggregation,
    TemporalFusionModule,
)
from .decoder.temporal_calibration import SUPPORTED_CALIBRATION_MODES


class A2Net_LWGANet_L0(nn.Module):
    """A2Net-LWGANet-L0 binary change detector.

    Deploy graph
    ------------

        T1/T2
          -> shared LWGANet-L0
          -> independent NeighborFeatureAggregation
          -> optional zero-parameter symmetric temporal calibration
          -> original A2Net TemporalFusionModule
          -> original Decoder

    Formal experiment modes
    -----------------------
    ``none``
        S0 clean baseline.

    ``symmetric``
        S1 full-pixel symmetric pair calibration.

    ``sctc``
        S2 Unchanged-Aware Symmetric Cross-Temporal Calibration.

    SCTC is part of the deployable student graph. It introduces no learnable
    parameters, no teacher/cache dependency, and no training-only branch.
    """

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        temporal_calibration_mode: str = "none",
        joint_temporal_bn: bool = False,
    ) -> None:
        super().__init__()

        temporal_calibration_mode = str(temporal_calibration_mode).lower()
        if temporal_calibration_mode not in SUPPORTED_CALIBRATION_MODES:
            raise ValueError(
                f"temporal_calibration_mode must be one of "
                f"{SUPPORTED_CALIBRATION_MODES}, got "
                f"{temporal_calibration_mode!r}"
            )

        self.temporal_calibration_mode = temporal_calibration_mode
        self.joint_temporal_bn = bool(joint_temporal_bn)

        self.backbone = LWGANet_L0_1242_e32_k11_GELU(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )

        self.mid_d = 64

        self.swa = NeighborFeatureAggregation(
            [32, 32, 64, 128, 256],
            self.mid_d,
        )

        self.tfm = TemporalFusionModule(
            self.mid_d,
            self.mid_d,
            calibration_mode=self.temporal_calibration_mode,
        )

        self.decoder = Decoder(self.mid_d)

    def extract_pair_features(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ):
        """Extract shared-backbone features for T1 and T2.

        With ``joint_temporal_bn=True`` the two temporal views are processed as
        one joint batch so the shared backbone's BatchNorm statistics are
        computed jointly over T1+T2 (a normalization control).
        """
        if self.joint_temporal_bn:
            x = torch.cat([x1, x2], dim=0)
            feats = tuple(self.backbone(x))
            b = x1.shape[0]
            return tuple(f[:b] for f in feats), tuple(f[b:] for f in feats)
        return tuple(self.backbone(x1)), tuple(self.backbone(x2))

    def _forward_main_path(
        self,
        features1,
        features2,
        output_size,
        return_decoder_features: bool = False,
        return_change_features: bool = False,
    ):
        aggregated1 = self.swa(*features1)
        aggregated2 = self.swa(*features2)

        # Temporal calibration is implemented inside self.tfm immediately
        # before the original absolute-difference operation.
        change = self.tfm(
            *aggregated1,
            *aggregated2,
        )

        decoder_output = self.decoder(*change)
        decoder_features = tuple(decoder_output[:4])
        raw_logits = decoder_output[4:]

        predictions = tuple(
            torch.sigmoid(
                F.interpolate(
                    logits,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            for logits in raw_logits
        )

        if return_change_features:
            # change = (c2, c3, c4, c5): the TFM change features used for KD.
            return predictions, change
        if return_decoder_features:
            return predictions, decoder_features
        return predictions

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_decoder_features: bool = False,
        return_change_features: bool = False,
    ):
        if x1.ndim != 4 or x2.ndim != 4:
            raise ValueError("x1 and x2 must be [B,3,H,W]")
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1/x2 must have identical shapes, got "
                f"{tuple(x1.shape)} vs {tuple(x2.shape)}"
            )
        if x1.shape[1] != 3:
            raise ValueError("A2Net expects RGB temporal inputs with 3 channels")

        features1, features2 = self.extract_pair_features(x1, x2)
        return self._forward_main_path(
            features1,
            features2,
            x1.shape[-2:],
            return_decoder_features=return_decoder_features,
            return_change_features=return_change_features,
        )

    def switch_to_deploy(self):
        """Return the deployable model without changing its predictions.

        SCTC is not a training auxiliary and therefore must remain in the
        deploy graph. The method is intentionally a no-op to keep the existing
        project deployment API stable.
        """
        return self


__all__ = ["A2Net_LWGANet_L0"]
