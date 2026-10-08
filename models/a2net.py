"""A2Net-LWGANet-L0 binary change detector (CATA-CD clean baseline).

Deploy graph
------------
    T1/T2
      -> shared LWGANet-L0
      -> NeighborFeatureAggregation
      -> TemporalFusionModule (absolute difference)
      -> [DeployableChangeAdapter, optional]
      -> Decoder
      -> change map

``dca_mode``:
    ``none``         C0: clean A2Net-LWGANet-L0 (deploy params 2,913,094).
    ``moe128``       C1: + DeployableChangeAdapter(width=128) with the original
                     global gate over change features (S-PCG S0).
    ``moe128_stats`` C1 + S-PCG **S1**: per-scale gate on the symmetric difference
                     statistic ``GAP(|a_s - b_s|)`` only.
    ``moe128_sympcg`` C1 + S-PCG **S2**: per-scale gate that adds the symmetric
                     common context ``GAP((a_s+b_s)/2)`` and the pooled phase
                     difference ``|GAP(a_s) - GAP(b_s)|``.

The DCA is part of the deploy graph. Teachers / cache / translators / agent are
training-only and are never registered inside this module.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.lwganet import LWGANet_L0_1242_e32_k11_GELU
from .decoder.a2net_decoder import Decoder, NeighborFeatureAggregation, TemporalFusionModule
from .decoder.deployable_change_adapter import DeployableChangeAdapter

# dca_mode -> gate_mode
SUPPORTED_DCA_MODES = ("none", "moe128", "moe128_stats", "moe128_sympcg")
DCA_GATE_MODES = {"moe128": "legacy", "moe128_stats": "stats", "moe128_sympcg": "sympcg"}


class A2Net_LWGANet_L0(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        dca_mode: str = "none",
    ) -> None:
        super().__init__()

        dca_mode = str(dca_mode).lower()
        if dca_mode not in SUPPORTED_DCA_MODES:
            raise ValueError(f"dca_mode must be one of {SUPPORTED_DCA_MODES}, got {dca_mode!r}")
        self.dca_mode = dca_mode

        self.backbone = LWGANet_L0_1242_e32_k11_GELU(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )

        self.mid_d = 64

        self.swa = NeighborFeatureAggregation([32, 32, 64, 128, 256], self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.mid_d)
        self.dca = (
            DeployableChangeAdapter(self.mid_d, width=128,
                                    gate_mode=DCA_GATE_MODES[self.dca_mode])
            if self.dca_mode != "none"
            else None
        )
        self.decoder = Decoder(self.mid_d)

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
            raise ValueError(f"x1/x2 shapes differ: {tuple(x1.shape)} vs {tuple(x2.shape)}")
        if x1.shape[1] != 3:
            raise ValueError("A2Net expects RGB temporal inputs with 3 channels")

        features1 = tuple(self.backbone(x1))
        features2 = tuple(self.backbone(x2))

        aggregated1 = self.swa(*features1)
        aggregated2 = self.swa(*features2)

        change = self.tfm(*aggregated1, *aggregated2)  # (c2, c3, c4, c5)

        if self.dca is not None:
            # S-PCG gate modes need the pre-fusion temporal pair (a_s, b_s); the
            # legacy S0 gate path ignores ``pair`` and stays bit-identical.
            change = self.dca(*change, pair=(aggregated1, aggregated2))

        decoder_output = self.decoder(*change)
        decoder_features = tuple(decoder_output[:4])
        raw_logits = decoder_output[4:]

        output_size = x1.shape[-2:]
        predictions = tuple(
            torch.sigmoid(F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False))
            for logits in raw_logits
        )

        if return_change_features:
            return predictions, change
        if return_decoder_features:
            return predictions, decoder_features
        return predictions

    def switch_to_deploy(self):
        """Return the deployable model. The DCA is deployable, so this is a no-op."""
        return self


__all__ = ["A2Net_LWGANet_L0", "SUPPORTED_DCA_MODES"]
