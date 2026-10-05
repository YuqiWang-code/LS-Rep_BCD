"""Minimal, self-contained PyTorch encoder for NVIDIA RADIO "C-RADIOv3-B".

This module re-implements just enough of the official RADIO repo
(https://github.com/NVlabs/RADIO) to rebuild and load the C-RADIOv3-B
checkpoint (``c-radio_v3-b_half.pth``) and use it as a FROZEN feature
extractor for binary change detection.

Only ``torch`` is required (no timm / transformers / huggingface / einops).

----------------------------------------------------------------------
Architecture (reconstructed from the checkpoint's ``arch`` string)
----------------------------------------------------------------------
The checkpoint stores ``arch == 'vit_base_patch16_v2_224'``, which in the
official repo is a ``timm`` VisionTransformer built from the
``vit_base_patch14_reg4_dinov2`` config with the following overrides, plus
NVIDIA's Conditional Position Embedding (CPE) patch-generator:

    patch_size      = 16
    embed_dim       = 768
    depth           = 12
    num_heads       = 12
    mlp hidden      = 3072 (mlp_ratio 4)
    LayerScale init = 1e-5   (blocks use ``ls1.gamma`` / ``ls2.gamma``)
    registers       = 4      (timm ``reg_token``, kept for state-dict parity)
    cls tokens      = 4      (one per distilled teacher: clip/siglip2/dino_v2/sam)
    CPE registers   = 4      (``cls_token.token`` = 4 cls + 4 reg = 8 prefix tokens)
    CPE max size    = 2048   (``pos_embed`` covers 128x128 = 16384 patches)
    final norm      = Identity  (``args.model_norm=False``)

The CPE patch generator replaces timm's ``patch_embed``/``pos_embed``/
``cls_token``: it patchifies the image with a linear projection, adds a
bilinearly-interpolated 2D absolute position embedding (so any resolution
that is a multiple of the patch size is accepted), and prepends the 8 prefix
tokens.

----------------------------------------------------------------------
Input / output contract
----------------------------------------------------------------------
* Input  : ``[B, 3, H, W]`` float tensor of RAW images in ``[0, 1]``.
  Normalization is INTERNAL: an ``InputConditioner`` subtracts the CLIP mean
  and divides by the CLIP std restored from the checkpoint's
  ``input_conditioner.norm_mean`` / ``input_conditioner.norm_std``. Hence
  ``META.mean=(0,0,0)``, ``META.std=(1,1,1)``.

* Output : ``{"dense": [B, 768, H//16, W//16]}`` in float32.
  The 8 prefix tokens (4 cls + 4 registers) are dropped; the remaining tokens
  are the per-patch features reshaped to NCHW. Dense feature stride = 16.

Default input size is 224x224 -> dense map 14x14, stride 16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Architecture constants for C-RADIOv3-B (vit_base_patch16_v2_224)
# ---------------------------------------------------------------------------
_PATCH_SIZE = 16
_EMBED_DIM = 768
_DEPTH = 12
_NUM_HEADS = 12
_MLP_RATIO = 4.0
_INIT_VALUES = 1e-5
_LAYER_NORM_EPS = 1e-6
_NUM_CLS_TOKENS = 4
_NUM_REGISTERS = 4            # CPE registers (see cls_token.token shape (8, 768))
_NUM_SKIP = _NUM_CLS_TOKENS + _NUM_REGISTERS
_CPE_MAX_RES = 2048           # cpe_max_size in the training args
_POS_ROWS = _POS_COLS = _CPE_MAX_RES // _PATCH_SIZE  # 128
_NUM_PATCHES = _POS_ROWS * _POS_COLS                  # 16384


@dataclass
class EncoderMeta:
    name: str
    mean: tuple
    std: tuple
    input_size: tuple
    color_order: str
    feature_stride: int
    note: str


META = EncoderMeta(
    name="RADIO-C-RADIOv3-B (vit_base_patch16_v2_224)",
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    input_size=(224, 224),
    color_order="RGB",
    feature_stride=_PATCH_SIZE,
    note=(
        "Normalization is INTERNAL (InputConditioner applies CLIP mean/std restored "
        "from the checkpoint); feed raw [0,1] images. Dense output is [B,768,H/16,W/16]."
    ),
)


# ---------------------------------------------------------------------------
# Vendored ViT pieces (equivalent to timm's VisionTransformer + CPE)
# ---------------------------------------------------------------------------

class InputConditioner(nn.Module):
    """Normalizes raw [0,1] images: ``(x - norm_mean) / norm_std``."""

    def __init__(self):
        super().__init__()
        self.register_buffer("norm_mean", torch.zeros(3, 1, 1))
        self.register_buffer("norm_std", torch.ones(3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.norm_mean) / self.norm_std


class ClsToken(nn.Module):
    """Prependable cls+register tokens. Parameter name is ``token`` to match the ckpt."""

    def __init__(self, ndim: int, num_tokens: int, num_registers: int):
        super().__init__()
        scale = ndim ** -0.5
        self.token = nn.Parameter(
            torch.randn(num_tokens + num_registers, ndim) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([token, x], dim=1)


class ViTPatchGenerator(nn.Module):
    """CPE patch generator: linear embed -> interpolated 2D abs pos -> cls tokens."""

    def __init__(
        self,
        patch_size: int = _PATCH_SIZE,
        embed_dim: int = _EMBED_DIM,
        num_rows: int = _POS_ROWS,
        num_cols: int = _POS_COLS,
        num_cls_tokens: int = _NUM_CLS_TOKENS,
        num_registers: int = _NUM_REGISTERS,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.embedder = nn.Linear(3 * patch_size * patch_size, embed_dim, bias=False)
        scale = embed_dim ** -0.5
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_rows * num_cols, embed_dim) * scale
        )
        self.cls_token = ClsToken(embed_dim, num_cls_tokens, num_registers)

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        py, px = H // p, W // p
        # (B, C, H, W) -> (B, py*px, C*p*p) in row-major patch order
        patches = x.reshape(B, C, py, p, px, p).permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, py * px, C * p * p).contiguous()
        return self.embedder(patches)

    def get_pos_enc(self, batch_size: int, input_size: tuple) -> torch.Tensor:
        """2D absolute pos embedding, bilinearly interpolated to the input grid.

        Mirrors the official eval-mode (``self.training=False``) CPE path: the
        (128,128) learned grid is first interpolated to ``max(h,w)`` and then
        cropped to ``(h,w)``.
        """
        input_dims = tuple(d // self.patch_size for d in input_size)  # (h, w)

        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1)
        pos_embed = pos_embed.permute(0, 3, 1, 2).contiguous()  # (1, C, 128, 128)

        max_dim = max(input_dims)
        pos_embed = F.interpolate(
            pos_embed.float(),
            size=(max_dim, max_dim),
            mode="bilinear",
            align_corners=False,
        ).to(pos_embed.dtype)

        # crop to the actual grid (no-op for square inputs)
        if input_dims[0] < pos_embed.shape[-2]:
            pos_embed = pos_embed[..., : input_dims[0], :]
        if input_dims[1] < pos_embed.shape[-1]:
            pos_embed = pos_embed[..., :, : input_dims[1]]

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(
                pos_embed.float(),
                size=input_dims,
                mode="bilinear",
                align_corners=False,
            ).to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)  # (1, h*w, C)
        return pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.embed_patches(x)                       # (B, N, C)
        pos_enc = self.get_pos_enc(patches.shape[0], input_size=x.shape[2:])
        patches = patches + pos_enc                           # (B, N, C)
        patches = self.cls_token(patches)                     # (B, 8+N, C)
        return patches


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LayerScale(nn.Module):
    """LayerScale whose parameter is named ``gamma`` (matches ckpt ``ls*.gamma``)."""

    def __init__(self, dim: int, init_values: float = _INIT_VALUES):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = _MLP_RATIO):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=_LAYER_NORM_EPS)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, proj_bias=True)
        self.ls1 = LayerScale(dim, _INIT_VALUES)
        self.norm2 = nn.LayerNorm(dim, eps=_LAYER_NORM_EPS)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, _INIT_VALUES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class RadioBackbone(nn.Module):
    """The ``base_model`` sub-network (state-dict keys match the ckpt's
    ``base_model.*`` prefix)."""

    def __init__(self):
        super().__init__()
        # timm ViT ``reg_tokens=4`` parameter; kept for exact state-dict parity.
        # NOTE: it is NOT used by the CPE forward path (a leftover from timm).
        self.reg_token = nn.Parameter(torch.zeros(1, _NUM_REGISTERS, _EMBED_DIM))
        self.patch_generator = ViTPatchGenerator()
        self.blocks = nn.Sequential(
            *[Block(_EMBED_DIM, _NUM_HEADS) for _ in range(_DEPTH)]
        )
        # final LayerNorm is Identity (args.model_norm=False), so it is omitted.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_generator(x)   # (B, 8+N, C)
        x = self.blocks(x)
        return x


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Frozen RADIO C-RADIOv3-B teacher for binary change detection.

    Usage::

        from models.thirdparty.radio import build_encoder
        m = build_encoder('pre-trained_weights/c-radio_v3-b_half.pth').eval()
        out = m.forward_features(x)   # x in [0,1], out['dense']: [B,768,h,w]
    """

    def __init__(self):
        super().__init__()
        self.input_conditioner = InputConditioner()
        self.base_model = RadioBackbone()
        self.patch_size = _PATCH_SIZE
        self.num_skip = _NUM_SKIP

    def load_pretrained(self, weight_path: str) -> None:
        """Load weights, skipping training-only keys and casting fp16 -> fp32."""
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]

        def _strip(prefix: str):
            return {
                k[len(prefix):]: v.float()
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

        backbone_sd = _strip("base_model.")
        cond_sd = _strip("input_conditioner.")

        skipped = [
            k for k in state_dict.keys()
            if not k.startswith(("base_model.", "input_conditioner."))
        ]

        missing, unexpected = self.base_model.load_state_dict(backbone_sd, strict=False)
        self.input_conditioner.load_state_dict(cond_sd)

        print(
            f"[RADIO] loaded {len(backbone_sd)} backbone tensors + "
            f"{len(cond_sd)} conditioner buffers from {weight_path}"
        )
        print(f"[RADIO] skipped {len(skipped)} training-only keys "
              f"(e.g. {sorted(skipped)[:3]})")
        print(f"[RADIO] backbone missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[RADIO]   missing examples: {missing[:5]}")
        if unexpected:
            print(f"[RADIO]   unexpected examples: {unexpected[:5]}")

    def forward_features(self, x: torch.Tensor) -> dict:
        """Run the backbone and return the dense per-patch feature map.

        Args:
            x: ``[B, 3, H, W]`` raw image in ``[0, 1]`` (H, W multiples of 16).

        Returns:
            ``{"dense": [B, 768, H//16, W//16]}`` in float32.
        """
        x = self.input_conditioner(x)
        y = self.base_model(x)                       # (B, 8+N, 768)
        y = y[:, self.num_skip:]                     # drop cls+register tokens
        B, N, C = y.shape
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        y = y.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
        return {"dense": y.float()}


def build_encoder(weight_path: str = None) -> Encoder:
    """Build the RADIO C-RADIOv3-B encoder, optionally loading pretrained weights."""
    m = Encoder()
    if weight_path:
        m.load_pretrained(weight_path)
    return m
