# Copyright (c) Meta Platforms, Inc. and affiliates.
# Minimal DINOv3 encoder wrapper for CATA-CD teacher cache generation.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .models.vision_transformer import vit_base, vit_large

__version__ = "0.0.1"


@dataclass
class EncoderMeta:
    name: str
    mean: tuple
    std: tuple
    input_size: tuple
    color_order: str
    feature_stride: int
    note: str


META_B16 = EncoderMeta(
    "DINOv3 ViT-B/16 LVD-1689M (neutral)",
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
    (256, 256), "RGB", 16, "neutral frozen DINOv3 LVD",
)
META_L16 = EncoderMeta(
    "DINOv3 ViT-L/16 SAT-493M",
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
    (256, 256), "RGB", 16, "satellite-pretrained DINOv3",
)
META = META_B16  # fallback


class _Encoder(nn.Module):
    def __init__(self, model: nn.Module, meta: EncoderMeta):
        super().__init__()
        self.model = model
        self.meta = meta

    def load_pretrained(self, weight_path: str) -> None:
        sd = torch.load(weight_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[dinov3] loaded {weight_path}: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("[dinov3]   missing:", missing[:8])
        if unexpected:
            print("[dinov3]   unexpected:", unexpected[:8])

    def forward_features(self, x: torch.Tensor) -> dict:
        # x: [B,3,H,W] already normalized by (META.mean, META.std) in RGB.
        dense = self.model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
        return {"dense": dense}


def _build(vit_fn, weight_path, meta, untie_local_norm: bool = False) -> _Encoder:
    model = vit_fn(
        patch_size=16,
        n_storage_tokens=4,
        mask_k_bias=True,
        layerscale_init=1e-5,
        untie_global_and_local_cls_norm=untie_local_norm,
    )
    encoder = _Encoder(model, meta)
    if weight_path:
        encoder.load_pretrained(weight_path)
    encoder.eval()
    return encoder


def build_encoder_vitb(weight_path=None) -> _Encoder:
    return _build(vit_base, weight_path, META_B16, untie_local_norm=False)


def build_encoder_vitl(weight_path=None) -> _Encoder:
    return _build(vit_large, weight_path, META_L16, untie_local_norm=True)


def build_encoder(weight_path=None) -> _Encoder:
    return build_encoder_vitb(weight_path)
