"""Unchanged student with a removable, loss-only direction-C auxiliary."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.lwganet import LWGANet_L0_1242_e32_k11_GELU
from .decoder.a2net_decoder import Decoder, NeighborFeatureAggregation, TemporalFusionModule


class A2Net_LWGANet_L0(nn.Module):
    """The deploy graph is always backbone -> SWA -> TFM -> Decoder."""

    def __init__(self, pretrained=True, pretrained_path=None,
                 auxiliary_mode="none", routing_cfg=None):
        super().__init__()
        if auxiliary_mode not in {"none", "direction_c"}:
            raise ValueError("Only none / direction_c are supported; old auxiliary modes were removed")
        self.backbone = LWGANet_L0_1242_e32_k11_GELU(
            pretrained=pretrained, pretrained_path=pretrained_path,
        )
        self.mid_d = 64
        self.swa = NeighborFeatureAggregation([32, 32, 64, 128, 256], self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.mid_d)
        self.decoder = Decoder(self.mid_d)
        self.auxiliary_mode = auxiliary_mode
        if auxiliary_mode == "direction_c":
            from .distill.routing import DirectionC
            # Keep baseline and C identical under the same seed, including the
            # next random input/augmentation. Auxiliary initialization is isolated.
            with torch.random.fork_rng(devices=[]):
                self.training_auxiliary = DirectionC(self.mid_d, **(routing_cfg or {}))

    @property
    def use_training_auxiliary(self):
        return self.auxiliary_mode != "none" and hasattr(self, "training_auxiliary")

    def extract_pair_features(self, x1, x2):
        return tuple(self.backbone(x1)), tuple(self.backbone(x2))

    def _forward_main_path(self, features1, features2, output_size,
                           return_decoder_features=False):
        aggregated1 = self.swa(*features1)
        aggregated2 = self.swa(*features2)
        change = self.tfm(*aggregated1, *aggregated2)
        decoder_features_and_logits = self.decoder(*change)
        decoder_features = tuple(decoder_features_and_logits[:4])
        raw_logits = decoder_features_and_logits[4:]
        predictions = tuple(
            torch.sigmoid(F.interpolate(
                logits, size=output_size, mode="bilinear", align_corners=False,
            ))
            for logits in raw_logits
        )
        if return_decoder_features:
            return predictions, decoder_features
        return predictions

    def forward(self, x1, x2, target=None, teacher_pack=None, compute_auxiliary=True,
                force_action=None):
        features1, features2 = self.extract_pair_features(x1, x2)
        need_aux = self.training and compute_auxiliary and self.use_training_auxiliary
        output = self._forward_main_path(features1, features2, x1.shape[-2:],
                                         return_decoder_features=need_aux)
        predictions, decoder_features = output if need_aux else (output, None)
        if not self.training:
            return predictions
        auxiliary = {}
        if need_aux:
            if target is None or teacher_pack is None:
                raise ValueError("Direction C requires training GT and both teacher caches")
            auxiliary["direction_c"] = self.training_auxiliary(
                decoder_features[0], predictions[0], target, teacher_pack,
                force_action=force_action,
            )
        return predictions, auxiliary

    def switch_to_deploy(self):
        """Idempotently remove all training-only structure computation."""
        if hasattr(self, "training_auxiliary"):
            del self.training_auxiliary
        self.auxiliary_mode = "none"
        return self
