"""Minimal, self-contained CLIP visual transformer (ViT-L/14) for RemoteCLIP.

This module reproduces ONLY the visual tower of the RemoteCLIP checkpoint
(``pre-trained_weights/RemoteCLIP-ViT-L-14.pt``), which is stored in the
OpenCLIP / OpenAI-CLIP state-dict layout.  The text tower is intentionally
excluded.

The submodule is named ``visual`` so that its ``state_dict`` keys are exactly
the ``visual.*`` keys present in the checkpoint (``visual.conv1.weight``,
``visual.ln_pre.*``, ``visual.transformer.resblocks.*``, ``visual.ln_post.*``,
``visual.proj``, ``visual.class_embedding``, ``visual.positional_embedding``).

Only ``torch`` is required; there are no ``timm`` / ``transformers`` / repo
imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class EncoderMeta:
    name: str
    mean: tuple
    std: tuple
    input_size: tuple
    color_order: str
    feature_stride: int
    note: str


# RemoteCLIP uses the OpenCLIP ``ViT-L-14`` (OpenAI) preprocessing, i.e. the
# standard CLIP ImageNet normalization in RGB with input resolution 224.
META = EncoderMeta(
    name="RemoteCLIP-ViT-L-14",
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    input_size=(224, 224),
    color_order="RGB",
    feature_stride=14,
    note=(
        "RemoteCLIP ViT-L/14 visual encoder (OpenCLIP/OpenAI-CLIP layout, "
        "standard CLIP normalization). patch 14 -> 16x16 grid, feature dim 1024."
    ),
)


class QuickGELU(nn.Module):
    """The activation used by OpenAI CLIP (and RemoteCLIP) MLPs."""

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class MultiheadAttention(nn.Module):
    """Fused-QKV attention matching OpenAI CLIP's state-dict layout.

    Keys produced: ``in_proj_weight``, ``in_proj_bias``, ``out_proj.weight``,
    ``out_proj.bias``.
    """

    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * width))
        self.out_proj = nn.Linear(width, width)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # t: [L, B, width] -> [B, heads, L, head_dim]
        L, B, _ = t.shape
        return t.reshape(L, B, self.heads, self.head_dim).permute(1, 2, 0, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [L, B, width] (query == key == value)
        L, B, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)  # [L, B, 3*width]
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (self._split_heads(t) for t in (q, k, v))  # [B, heads, L, head_dim]

        scale = self.head_dim ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)  # [B, heads, L, L]
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, heads, L, head_dim]
        out = out.permute(2, 0, 1, 3).reshape(L, B, self.width)  # [L, B, width]
        return self.out_proj(out)


class ResidualAttentionBlock(nn.Module):
    """Pre-norm residual block (ln_1 + attn, ln_2 + mlp).

    MLP keys: ``mlp.c_fc.*``, ``mlp.c_proj.*`` (the QuickGELU has no params).
    """

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = MultiheadAttention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class CLIPVisionTransformer(nn.Module):
    """ViT-L/14 visual tower (OpenAI CLIP layout).

    embed/width 1024, depth 24, heads 16, patch 14, pre-norm (ln_pre).
    ``proj`` is the CLIP projection head (1024 -> 768) and is loaded for
    key-completeness but NOT applied by :meth:`forward_features`.
    """

    def __init__(
        self,
        input_size: int = 224,
        patch_size: int = 14,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        output_dim: int = 768,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.width = width
        self.output_dim = output_dim
        self.grid_size = input_size // patch_size  # 16 for 224 / 14

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size ** 2 + 1, width)  # 257 positions (CLS + 256)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward_features(self, x: torch.Tensor):
        """Return (dense, global).

        - ``dense``:  [B, width, H/p, W/p]  post-transformer patch tokens.
        - ``global``: [B, width]             CLS token after ``ln_post``.

        Following OpenAI CLIP semantics, ``ln_post`` is applied only to the
        pooled CLS token; the patch tokens are the raw transformer outputs.
        """
        x = self.conv1(x)  # [B, width, H/p, W/p]
        B, C, Hp, Wp = x.shape
        x = x.reshape(B, C, Hp * Wp).permute(0, 2, 1)  # [B, N, width]

        cls = self.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, N + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [S, B, width]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, S, width]

        patches = x[:, 1:, :]  # [B, N, width]
        dense = patches.permute(0, 2, 1).reshape(B, C, Hp, Wp).contiguous()
        global_feat = self.ln_post(x[:, 0, :])  # [B, width]

        return dense, global_feat


class Encoder(nn.Module):
    """Frozen RemoteCLIP visual backbone (dense patch features)."""

    def __init__(self):
        super().__init__()
        self.visual = CLIPVisionTransformer()

    def load_pretrained(self, weight_path: str):
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        visual_state = {k: v for k, v in state.items() if k.startswith("visual.")}
        dropped = [k for k in state.keys() if not k.startswith("visual.")]

        missing, unexpected = self.load_state_dict(visual_state, strict=False)

        print(
            f"[RemoteCLIP] checkpoint keys: {len(state)}, visual keys loaded: "
            f"{len(visual_state)}, non-visual keys dropped: {len(dropped)}"
        )
        if dropped:
            print(f"[RemoteCLIP] dropped key examples: {dropped[:5]}")
        print(f"[RemoteCLIP] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            print(f"[RemoteCLIP] missing examples: {missing[:5]}")
        if unexpected:
            print(f"[RemoteCLIP] unexpected examples: {unexpected[:5]}")

        del state, visual_state
        return self

    def forward_features(self, x: torch.Tensor):
        # x: [B, 3, H, W] already normalized per META.
        dense, global_feat = self.visual.forward_features(x)
        return {"dense": dense, "global": global_feat}


def build_encoder(weight_path: str = None) -> Encoder:
    m = Encoder()
    if weight_path:
        m.load_pretrained(weight_path)
    return m
