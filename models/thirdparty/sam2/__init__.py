"""SAM 2.1 Hiera image encoder (frozen teacher backbone).

Re-exports the minimal, self-contained SAM 2 image encoder implemented in
:mod:`models.thirdparty.sam2.model`.
"""

from .model import Encoder, EncoderMeta, META, build_encoder

__all__ = ["Encoder", "EncoderMeta", "META", "build_encoder"]
