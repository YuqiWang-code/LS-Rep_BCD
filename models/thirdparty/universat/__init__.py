"""UniverSat Base frozen-teacher encoder (minimal, self-contained).

Exports:
    build_encoder(weight_path=None) -> Encoder
    META                            -> EncoderMeta
    Encoder                         -> nn.Module
    EncoderMeta                     -> dataclass
"""

from .model import Encoder, EncoderMeta, META, build_encoder

__all__ = ["build_encoder", "META", "Encoder", "EncoderMeta"]
