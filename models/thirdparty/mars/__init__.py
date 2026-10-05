"""Minimal importable MaRS-Base RGB SwinV2 encoder (self-contained, no timm)."""

from .model import Encoder, EncoderMeta, META, build_encoder

__all__ = ["Encoder", "EncoderMeta", "META", "build_encoder"]
