"""Minimal importable RemoteCLIP ViT-L/14 visual encoder (frozen teacher).

Exposes the same ``visual.*`` state-dict layout as the RemoteCLIP checkpoint
so it can be loaded with ``load_state_dict(..., strict=False)``.
"""

from .model import Encoder, EncoderMeta, META, build_encoder

__all__ = ["build_encoder", "META", "Encoder", "EncoderMeta"]
