"""A2Net-LWGANet-L0 with detachable legacy SAMStruct or SAM-HSD losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.lwganet import LWGANet_L0_1242_e32_k11_GELU
from .decoder.a2net_decoder import Decoder, NeighborFeatureAggregation, TemporalFusionModule
from .distill import EIRHSDAdapter, SAMHSDAdapter, SAMStructureAdapter


class A2Net_LWGANet_L0(nn.Module):
    """The deploy graph is always backbone -> SWA -> TFM -> Decoder."""

    def __init__(self, pretrained=True, pretrained_path=None,
                 auxiliary_mode="none", sam_hsd_cfg=None, eir_hsd_cfg=None,
                 legacy_sam_cfg=None):
        super().__init__()
        if auxiliary_mode not in {"none", "legacy_sam", "sam_hsd", "eir_hsd"}:
            raise ValueError(f"Unsupported auxiliary_mode: {auxiliary_mode}")
        self.backbone = LWGANet_L0_1242_e32_k11_GELU(
            pretrained=pretrained, pretrained_path=pretrained_path,
        )
        self.mid_d = 64
        self.swa = NeighborFeatureAggregation([32, 32, 64, 128, 256], self.mid_d)
        self.tfm = TemporalFusionModule(self.mid_d, self.mid_d)
        self.decoder = Decoder(self.mid_d)
        self.auxiliary_mode = auxiliary_mode
        if auxiliary_mode == "legacy_sam":
            self.training_auxiliary = SAMStructureAdapter(**(legacy_sam_cfg or {}))
        elif auxiliary_mode == "sam_hsd":
            self.training_auxiliary = SAMHSDAdapter(**(sam_hsd_cfg or {}))
        elif auxiliary_mode == "eir_hsd":
            self.training_auxiliary = EIRHSDAdapter(**(eir_hsd_cfg or {}))

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

    def forward(self, x1, x2, target=None, teacher_pack=None, compute_auxiliary=True):
        features1, features2 = self.extract_pair_features(x1, x2)
        need_decoder_features = (
            self.training and compute_auxiliary
            and self.auxiliary_mode in {"sam_hsd", "eir_hsd"}
            and self.use_training_auxiliary
        )
        main_output = self._forward_main_path(
            features1, features2, x1.shape[-2:],
            return_decoder_features=need_decoder_features,
        )
        if need_decoder_features:
            predictions, decoder_features = main_output
        else:
            predictions, decoder_features = main_output, None
        if not self.training:
            return predictions
        auxiliary = {}
        if compute_auxiliary and self.use_training_auxiliary:
            if target is None or teacher_pack is None:
                raise ValueError("target and teacher_pack are required by the training auxiliary")
            if self.auxiliary_mode == "legacy_sam":
                auxiliary["legacy_sam"] = self.training_auxiliary(
                    predictions, target, teacher_pack,
                )
            elif self.auxiliary_mode == "sam_hsd":
                auxiliary["sam_hsd"] = self.training_auxiliary(
                    features1, features2, decoder_features,
                    predictions, target, teacher_pack,
                )
            else:
                auxiliary["eir_hsd"] = self.training_auxiliary(
                    features1, features2, decoder_features,
                    predictions, target, teacher_pack,
                )
        return predictions, auxiliary

    def switch_to_deploy(self):
        """Idempotently remove all training-only structure computation."""
        if hasattr(self, "training_auxiliary"):
            del self.training_auxiliary
        self.auxiliary_mode = "none"
        return self
