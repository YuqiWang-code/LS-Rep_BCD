"""Minimal, self-contained UniverSat Base encoder (frozen-teacher backbone).

This module re-implements the *encoder* part of UniverSat
(https://github.com/gastruc/UniverSat, "Resolution- and Modality-Agnostic
Transformers for Earth Observation", NeurIPS 2026) so that it can be used as a
FROZEN teacher for binary change detection.

Only ``torch`` (and, for loading, ``safetensors``) is required -- no
``others/``, timm, transformers, huggingface or einops imports.  The released
checkpoint (``universat_base.safetensors``) stores the encoder under a
``model.`` prefix (the HF ``_UniverSatHub`` wrapper put the raw ``UniverSat``
at ``.model``); :meth:`Encoder.load_pretrained` strips that prefix.

Architecture facts (Base, embed 768):
  * trunk blocks = [Bi_ACA_in (ACABlock), SA x12 (gated RoPE ViT blocks),
    Bilinear_out, CA_Sub (CrossBlock)]; 4 register tokens; qk-norm + RoPE +
    LayerScale (``gamma_1``/``gamma_2``) + RMSNorm-with-``scale``.
  * ``UniversalPatchEncoder`` (UPE, embed_dim=96) with axial cross-attention
    order [S1, C, T, S], expand_dim [2,2,2,2], gating, MP-Fourier spectral /
    pixel embeddings, and a sub-patch skip connection (``out['spatial']``).

Hard-coded single-modality configuration (spot = SPOT-6/7 1 m RGB):
  * wavelengths [0.665, 0.56, 0.49] um (R, G, B), input_res = 1 m/px,
    subpatch = 10, patch_size = 40 m  ->  scale = 4.0, patch_px = 40 px.
  * dense output at ``feature_stride = 4`` (= patch_px / subpatch), so
    ``forward_features`` returns ``[B, 768, H/4, W/4]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


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
    name="UniverSat Base (spot-RGB)",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    input_size=(256, 256),
    color_order="RGB",
    feature_stride=4,
    note=(
        "UniverSat Base frozen encoder, single-modality 'spot' (SPOT-6/7 1 m RGB, "
        "3 bands R=0.665um G=0.56um B=0.49um). patch_size=40m -> scale=4.0, "
        "patch_px=40, subpatch=10 -> dense feature_stride=4 (output = H/4). Input "
        "must be square and divisible by 4; the 40px patch unfold crops the last "
        "H%40 px (256 uses 240 px). Input is expected already normalized. "
        "Upstream UniverSat normalizes per-dataset with channel z-scores computed "
        "on raw 0-255 values (NORM_spot_patch.json, NOT shipped in the repo); the "
        "META mean/std above are ImageNet-style [0,1]-scale defaults matching this "
        "repo's CDD pipeline -- swap in PASTIS-HD SPOT6 stats for exact parity. "
        "forward_features applies NO normalization."
    ),
)

# Fixed single-modality configuration (spot).
_SPOT_WAVELENGTHS = [0.665, 0.56, 0.49]
_SPOT_INPUT_RES = 1.0  # meters / pixel
_SPOT_SUBPATCH = 10
_PATCH_SIZE_M = 40.0  # meters  -> scale = patch_size / 10
_SCALE = _PATCH_SIZE_M / 10.0  # 4.0

# Per-(dataset, modality) SSL projector heads present in the checkpoint. They
# are unused at inference (dataset="") but must exist so load_state_dict matches
# every key.
_DEFAULT_MODALITIES_DICT: Dict[str, List[str]] = {
    "flair": ["spotRGBN", "aerialflair", "s2flair", "s1flair", "dem"],
    "pastishd": ["spot", "s2", "s1"],
    "planted": ["s2", "s1", "l7", "alos", "modis"],
    "tsaits": ["aerial", "s2", "s1"],
    "s2naip": ["naip", "l8", "s2", "s1"],
    "hyperglobal": ["EO1"],
    "earthview": ["rgbneon", "ndemneon", "neon"],
    "spectralearth": ["enmap"],
}


# ---------------------------------------------------------------------------
# Building blocks (ported verbatim from UniverSat, masking/training stripped)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """RMS LayerNorm that stores a ``scale`` parameter (matches checkpoint)."""

    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def rotate_half_2d(x):
    x1, x2, x3, x4 = x.chunk(4, dim=-1)
    return torch.cat((-x2, x1, -x4, x3), dim=-1)


class Rope2D(nn.Module):
    """2D rotary position embedding; ``freqs`` is a persistent parameter."""

    def __init__(self, dim: int, max_freq=7, min_freq=7e-4):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.freqs = nn.Parameter(torch.empty(2, self.dim))
        self._init_freqs()

    def _init_freqs(self):
        freqs_1d = self.max_freq * (self.max_freq / self.min_freq) ** torch.linspace(
            0, -1, self.dim // 4
        )
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        with torch.no_grad():
            self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x, coords, norm_coords=False):
        if norm_coords:
            coords = (coords - coords.amin(dim=(1, 2), keepdim=True)) / (
                coords.amax(dim=(1, 2), keepdim=True) - coords.amin(dim=(1, 2), keepdim=True)
            )
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half_2d(x) * angle.sin()


class Rope1D(nn.Module):
    """1D rotary position embedding (buffers are non-persistent -> no keys)."""

    def __init__(self, dim: int, max_seq_len: int = 367, base: int = 100_000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        seq_idx = torch.arange(self.max_seq_len, dtype=theta.dtype, device=theta.device)
        idx_theta = torch.einsum("i,j->ij", seq_idx, theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        if input_pos is None:
            rope_cache = rope_cache.unsqueeze(0).unsqueeze(2)
        else:
            rope_cache = rope_cache.unsqueeze(2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        return x_out.flatten(3).type_as(x)


class MPFourier(nn.Module):
    """Multiplicative-Fourier feature embedding (EDM2-style) with a learned projection."""

    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * math.pi * torch.randn(num_channels // 4) * bandwidth)
        self.register_buffer("phases", 2 * math.pi * torch.rand(num_channels // 4))
        self.proj = nn.Linear(num_channels // 4, num_channels)

    def forward(self, x):
        original_dtype = x.dtype
        y = x.to(torch.float32) * self.freqs.to(torch.float32) + self.phases.to(torch.float32)
        y = y.cos() * math.sqrt(2)
        return self.proj(y.to(original_dtype))


def get_edge_coordinates(n: int, dtype: torch.dtype, device: torch.device):
    side = n // 4
    reg_coords = torch.zeros(1, n, 2, dtype=dtype, device=device)
    c = torch.arange(side, dtype=dtype, device=device) / side
    reg_coords[:, 0 * side : 1 * side, 0] = c
    reg_coords[:, 0 * side : 1 * side, 1] = 0
    reg_coords[:, 1 * side : 2 * side, 0] = 1
    reg_coords[:, 1 * side : 2 * side, 1] = c
    reg_coords[:, 2 * side : 3 * side, 0] = 1 - c
    reg_coords[:, 2 * side : 3 * side, 1] = 1
    reg_coords[:, 3 * side : 4 * side, 0] = 0
    reg_coords[:, 3 * side : 4 * side, 1] = 1 - c
    return reg_coords


def get_coords(x, grid_size, n_modalities, register, res=1, modis=False):
    b = x.shape[0]
    coord_x = torch.linspace(0, 1, grid_size, device=x.device, dtype=torch.float32) * res
    coord_y = torch.linspace(0, 1, grid_size, device=x.device, dtype=torch.float32) * res
    coords_all = torch.cartesian_prod(coord_x, coord_y)
    coords_all = coords_all.repeat(n_modalities, 1)
    coords_all = coords_all[None].expand(b, -1, -1)
    if modis:
        coords_all = torch.cat(
            [torch.zeros(b, 1, 2, device=x.device, dtype=torch.float32), coords_all], dim=1
        )
    if register > 3:
        reg_coords = get_edge_coordinates(register, torch.float32, x.device).expand(b, -1, -1)
    else:
        reg_coords = torch.zeros(b, register, 2, device=x.device, dtype=torch.float32)
    coords = torch.cat([reg_coords, coords_all], dim=1)
    return coords


class Attention(nn.Module):
    """Gated self-attention with qk-norm and 2D RoPE (used by the SA trunk blocks)."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        gating=False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.gating = gating
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_norm = norm_layer(dim // num_heads)
        self.k_norm = norm_layer(dim // num_heads)
        if self.gating:
            self.wq = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = Rope2D(self.head_dim)

    def forward(self, x, coords):
        B, N, C = x.shape
        if self.gating:
            q = self.wq(x).view(B, N, self.num_heads, -1)
            q, gate_score = torch.split(q, [C // self.num_heads, C // self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B, N, -1, C // self.num_heads)
            q = q.reshape(B, N, -1, C // self.num_heads)
        else:
            q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads))
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.rope(k.permute(0, 2, 1, 3), coords[:, None, :, :])
        q = self.rope(q.permute(0, 2, 1, 3), coords[:, None, :, :])
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).contiguous()
        if self.gating:
            x = x * torch.sigmoid(gate_score)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """SA trunk block: RMSNorm + gated attention + LayerScale + MLP."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        gating=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            gating=gating,
        )
        self.drop_path1 = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = nn.Identity()
        self.gamma_1 = nn.Parameter(1e-6 * torch.ones(dim))
        self.gamma_2 = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x, coords=None):
        x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), coords))
        x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class ACAAttention(nn.Module):
    """Axial cross-attention with a pooled single query (used by the UPE)."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        n_queries=1,
        expand_dim=1,
        max_seq_len=750,
        RoPe=None,
        gating=False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expand_dim = expand_dim
        self.scale = (self.head_dim * expand_dim) ** -0.5
        self.n_queries = n_queries
        self.gating = gating
        self.q_norm = norm_layer((dim * expand_dim) // num_heads)
        self.k_norm = norm_layer((dim * expand_dim) // num_heads)
        if self.gating:
            self.wq = nn.Linear(dim * 2, dim * expand_dim * 2, bias=True)
        else:
            self.wq = nn.Linear(dim * 2, dim * expand_dim, bias=True)
        self.wk = nn.Linear(dim, dim * expand_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim * expand_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if RoPe == "2D":
            self.rope = Rope2D(self.head_dim * expand_dim)
        elif RoPe == "1D":
            self.rope = Rope1D(self.head_dim * expand_dim, max_seq_len=max_seq_len)
        else:
            self.rope = None
        self.RoPe = RoPe

    def get_qkv(self, x):
        B, N, C = x.shape
        C = C * self.expand_dim
        if N == 1:
            return self.wv(x)
        q = torch.cat(
            [torch.mean(x, dim=1).unsqueeze(1), torch.max(x, dim=1).values.unsqueeze(1)], dim=2
        )
        if self.gating:
            q = self.wq(q).view(B, self.n_queries, self.num_heads, -1)
            q, gate_score = torch.split(q, [C // self.num_heads, C // self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B, self.n_queries, self.num_heads, C // self.num_heads)
            q = q.reshape(B, self.n_queries, self.num_heads, C // self.num_heads)
        else:
            q = self.wq(q).view(B, self.n_queries, self.num_heads, C // self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads))
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.gating:
            return q, k, v, gate_score
        return q, k, v

    def apply_rope(self, q, k, coords):
        if self.RoPe == "2D":
            k = self.rope(k.permute(0, 2, 1, 3), coords[:, None, :, :])
            q = q.permute(0, 2, 1, 3)
        elif self.RoPe == "1D":
            k = self.rope(k, input_pos=coords).permute(0, 2, 1, 3)
            q = self.rope(q).permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
        return q, k

    def forward(self, x, coords=None):
        B, N, C = x.shape
        C = C * self.expand_dim
        if N == 1:  # a single token: no attention needed
            return self.wv(x)
        if self.gating:
            q, k, v, gate_score = self.get_qkv(x)
        else:
            q, k, v = self.get_qkv(x)
        q, k = self.apply_rope(q, k, coords)
        if q.shape[-2] == 1 and not self.training:
            attn = (q * k).sum(dim=-1).unsqueeze(-2) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn.transpose(-2, -1) * v).sum(dim=-2, keepdim=True)
        else:
            x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).contiguous()
        if self.gating:
            x = x * torch.sigmoid(gate_score)
        x = x.reshape(B, self.n_queries, C)
        return x


class ACABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        n_queries=1,
        expand_dim=1,
        RoPe=None,
        max_seq_len=750,
        gating=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ACAAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            n_queries=n_queries,
            expand_dim=expand_dim,
            RoPe=RoPe,
            max_seq_len=max_seq_len,
            gating=gating,
        )
        self.drop_path1 = nn.Identity()
        self.expand_dim = expand_dim
        dim = dim * expand_dim
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = nn.Identity()
        self.gamma_1 = nn.Parameter(1e-6 * torch.ones(dim))
        self.gamma_2 = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x, coords=None):
        if self.expand_dim == 2:
            res = torch.cat([x.mean(dim=1).unsqueeze(1), x.max(dim=1).values.unsqueeze(1)], dim=2)
        elif self.expand_dim == 4:
            res = torch.cat(
                [
                    x.mean(dim=1).unsqueeze(1),
                    x.min(dim=1).values.unsqueeze(1),
                    x.max(dim=1).values.unsqueeze(1),
                    x.std(dim=1).unsqueeze(1),
                ],
                dim=2,
            )
        elif self.expand_dim == 3:
            res = torch.cat(
                [x.mean(dim=1).unsqueeze(1), x.min(dim=1).values.unsqueeze(1), x.max(dim=1).values.unsqueeze(1)],
                dim=2,
            )
        else:
            res = x.max(dim=1).values.unsqueeze(1)
        x = res + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), coords))
        x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    """Cross-attention; q_norm/k_norm use ``nn.LayerNorm`` (matches checkpoint)."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=None,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        gating=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.gating = gating
        head_dim = dim // num_heads
        self.scale = qk_scale or (head_dim) ** -0.5
        self.q_norm = norm_layer(dim // num_heads)
        self.k_norm = norm_layer(dim // num_heads)
        if self.gating:
            self.wq = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.rope = Rope2D(dim // num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, coords_q, kv, coords_kv, n_registers, n_registers_q, n_modalities, masking=False):
        B, Nkv, C = kv.shape
        B, Nq, C = q.shape
        if self.gating:
            q = self.wq(q).view(B, Nq, self.num_heads, -1)
            q, gate_score = torch.split(q, [C // self.num_heads, C // self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B, Nq, self.num_heads, C // self.num_heads)
            q = q.reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            q = self.wq(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(self.wk(kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        v = self.wv(kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.rope(q, coords_q[:, None, :, :])
        k = self.rope(k, coords_kv[:, None, :, :])
        x = F.scaled_dot_product_attention(q, k, v)  # masking=False at inference
        x = x.transpose(1, 2).contiguous()
        if self.gating:
            x = x * torch.sigmoid(gate_score)
        x = x.reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=None,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        gating=False,
    ):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            gating=gating,
        )
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, q, coords_q, kv, coords_kv, n_registers, n_registers_q, n_modalities):
        x = q + self.drop_path(
            self.attn(self.norm_q(q), coords_q, self.norm_kv(kv), coords_kv, n_registers, n_registers_q, n_modalities)
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def to_scale(x, scale, res, subpatch=1):
    """Tile a per-pixel feature map into (patch, sub-patch, pixel) grids.

    Ported verbatim; the final layout is ``(B*S, T, C, s, sub^2, D)`` where
    ``S`` is the number of patches, ``s`` the number of sub-patches per patch
    and ``sub^2`` the number of pixels per sub-patch.
    """
    grid_size = max(int(scale * 10 / res), 1)
    B, T, C, H, W, D = x.shape
    x = x.permute(0, 1, 2, 5, 3, 4)  # B, T, C, D, H, W
    x = x.unfold(4, grid_size, grid_size).unfold(5, grid_size, grid_size)
    x = x.unfold(6, subpatch, subpatch).unfold(7, subpatch, subpatch)
    x = x.flatten(4, 5).flatten(5, 6).flatten(6, 7)
    x = x.permute(0, 4, 1, 2, 5, 6, 3).flatten(0, 1)
    return x


class UniversalPatchEncoder(nn.Module):
    """Per-modality patch encoder (UPE) with axial cross-attention."""

    def __init__(
        self,
        embed_dim=64,
        final_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        norm_layer=RMSNorm,
        n_queries: Optional[List[int]] = None,
        order: Optional[List[str]] = None,
        expand_dim: Optional[List[int]] = None,
        gating=False,
    ):
        super().__init__()
        n_queries = n_queries or []
        order = order or []
        expand_dim = expand_dim or []
        assert len(n_queries) == len(order) == len(expand_dim)

        self.spectral_embed = MPFourier(embed_dim)
        self.embed = MPFourier(embed_dim, bandwidth=2.5)

        for modality in ["VV", "VH", "Ratio_VV_VH", "HH", "HV", "Ratio_HH_HV", "nDEM", "DSM"]:
            setattr(self, "_".join(["Encoding", modality]), nn.Parameter(torch.randn(embed_dim), requires_grad=True))

        blocks = []
        spectral, temporal, spatial = True, True, True
        max_seq_len = 12
        for i in range(len(order)):
            num_heads_i = num_heads
            if order[i] == "C" and spectral:
                RoPe = None
                max_seq_len = 250
                spectral = False
                num_heads_i = num_heads_i // 4
            elif order[i] == "T" and temporal:
                RoPe = "1D"
                max_seq_len = 750
                temporal = False
                self.registers_temporal = nn.Parameter(torch.empty(1, 2, embed_dim))
                nn.init.normal_(self.registers_temporal, std=0.02)
                num_heads_i = num_heads_i // 2
            elif order[i] == "S1":
                RoPe = "2D"
                num_heads_i = 1 if num_heads_i < 3 else num_heads_i // 6
            elif order[i] == "S" and spatial:
                RoPe = "2D"
                spatial = False
                self.registers_spatial = nn.Parameter(torch.empty(1, 4, embed_dim))
                self.spatial_proj = nn.Linear(embed_dim, final_dim, bias=True)
                nn.init.normal_(self.registers_spatial, std=0.02)
            else:
                RoPe = "1D"
                max_seq_len = max(n_queries)
            blocks.append(
                ACABlock(
                    dim=embed_dim,
                    num_heads=num_heads_i,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop_rate,
                    n_queries=n_queries[i],
                    expand_dim=expand_dim[i],
                    max_seq_len=max_seq_len,
                    RoPe=RoPe,
                    gating=gating,
                )
            )
            embed_dim = embed_dim * expand_dim[i]

        self.predictor_blocks = nn.ModuleList(blocks)
        self.order = order
        self.n_queries = n_queries
        self.expand_dim = expand_dim
        self.predictor_norm = norm_layer(embed_dim)
        self.final_proj = nn.Linear(embed_dim, final_dim, bias=True)

    def forward(self, x, modality, wavelengths, scale, dates, subpatch=1, res=1):
        B_o = x.shape[0]
        x = x.unsqueeze(-1)

        coords_spectral_tensors = []
        for wv in wavelengths:
            if isinstance(wv, str):
                coords_spectral_tensors.append(getattr(self, f"Encoding_{wv}").view(1, 1, -1))
            else:
                coords_spectral_tensors.append(
                    self.spectral_embed(torch.tensor(wv, device=x.device, dtype=x.dtype).view(1, 1, 1))
                )
        coords_spectral = torch.cat(coords_spectral_tensors, dim=1)
        coords_spectral = coords_spectral.repeat(x.shape[0], 1, 1)

        x = self.embed(x)
        x = to_scale(x, scale, res, subpatch)

        B, T, C, S, s, D = x.shape
        P = B // B_o

        coords_spectral = coords_spectral.repeat_interleave(P, dim=0)
        coords_spatial = get_coords(x, int(S ** 0.5), 1, 4, res=1) * (scale * int(S ** 0.5) / 100)
        coords_sub_spatial = get_coords(x, int(s ** 0.5), 1, 0, res=1) * (subpatch * res) / 10

        B, T, C, S, s, D = x.shape

        coords_temporal = torch.cat(
            [
                torch.zeros(x.shape[0], self.registers_temporal.shape[1], device=x.device),
                dates.repeat_interleave(x.shape[0] // dates.shape[0], dim=0),
            ],
            dim=1,
        ).int()

        out = {"coords_spatial": coords_spatial, "N_masked": B // B_o}
        D_orig = D
        for i in range(len(self.order)):
            blk = self.predictor_blocks[i]
            if self.order[i] == "C":
                if coords_spectral is not None:
                    D_expand = D // D_orig
                    cs = coords_spectral.repeat(1, 1, D_expand) if D_expand > 1 else coords_spectral
                    x = x + cs.unsqueeze(1).unsqueeze(3)
                x = x.permute(0, 1, 3, 2, 4)
                x = x.flatten(0, 2)
                x = blk(x)
                coords_spectral = None
                C = self.n_queries[i]
                D = D * self.expand_dim[i]
                x = x.view(B, T, S, C, D).permute(0, 1, 3, 2, 4)
            elif self.order[i] == "T":
                out["temporal"] = x
                x = x.permute(0, 2, 3, 1, 4)
                x = x.flatten(0, 2)
                if T > 1:
                    x = torch.cat([self.registers_temporal.repeat(B * C * S, 1, 1), x], dim=1)
                else:
                    coords_temporal = coords_temporal[:, self.registers_temporal.shape[1] :]
                if coords_temporal is not None:
                    x = blk(x, coords=coords_temporal.repeat_interleave(C * S, dim=0))
                else:
                    x = blk(x)
                coords_temporal = None
                T = self.n_queries[i]
                D = D * self.expand_dim[i]
                x = x.view(B, C, S, T, D).permute(0, 3, 1, 2, 4)
            elif self.order[i] == "S":
                out["spatial"] = self.spatial_proj(x.flatten(1, 3))
                x = x.flatten(0, 2)
                x = torch.cat([self.registers_spatial.repeat(B * T * C, 1, 1), x], dim=1)
                if coords_spatial is not None:
                    x = blk(x, coords=coords_spatial.repeat_interleave(T * C, dim=0))
                else:
                    x = blk(x)
                coords_spatial = None
                S = self.n_queries[i]
                D = D * self.expand_dim[i]
                x = x.view(B, T, C, S, D)
            elif self.order[i] == "S1":
                x = x.flatten(0, 3)
                x = blk(x, coords=coords_sub_spatial.repeat_interleave(S * T * C, dim=0))
                D = D * self.expand_dim[i]
                x = x.view(B, T, C, S, 1, D).squeeze(4)

        x = self.predictor_norm(x)
        x = self.final_proj(x)
        out["tokens"] = x
        return out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def _unroll_block_list(blocks: List[str]) -> List[str]:
    unrolled = []
    for b in blocks:
        if "x" in b:
            n = int(b.split("x")[1])
            unrolled.extend([b.split("x")[0]] * n)
        else:
            unrolled.append(b)
    return unrolled


class Encoder(nn.Module):
    """UniverSat Base encoder (embed 768) restricted to the spot-RGB modality."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_registers = 4
        self.feature_stride = META.feature_stride

        gating = True
        block_type = _unroll_block_list(["Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"])
        self.block_type = block_type

        # --- Universal Patch Encoder (spot) ---
        self.spatial_encoder = UniversalPatchEncoder(
            embed_dim=embed_dim // 8,  # 96
            final_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,  # UPE default; not overridden in the Base config
            attn_drop_rate=0.0,
            norm_layer=RMSNorm,
            n_queries=[1, 1, 1, 1],
            expand_dim=[2, 2, 2, 2],
            order=["S1", "C", "T", "S"],
            gating=gating,
        )

        # --- registers ---
        self.registers = nn.Parameter(torch.empty(1, self.n_registers, embed_dim))
        nn.init.normal_(self.registers, std=0.02)

        # --- SSL projector heads (unused at inference, kept for key matching) ---
        for dataset in sorted(_DEFAULT_MODALITIES_DICT.keys()):
            for modality in _DEFAULT_MODALITIES_DICT[dataset]:
                if modality != "modis":
                    setattr(
                        self,
                        f"projector__{dataset}_{modality}",
                        nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 2),
                            nn.GELU(),
                            RMSNorm(embed_dim * 2),
                            nn.Linear(embed_dim * 2, embed_dim),
                        ),
                    )

        # --- trunk blocks ---
        blocks = []
        for bt in block_type:
            if bt == "SA":
                blocks.append(
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=False,
                        norm_layer=RMSNorm,
                        attn_drop=0.0,
                        drop_path=0.0,
                        gating=gating,
                    )
                )
            elif bt == "Bi_ACA_in":
                blocks.append(
                    ACABlock(
                        dim=embed_dim,
                        num_heads=8,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=False,
                        expand_dim=1,
                        attn_drop=0.0,
                        norm_layer=RMSNorm,
                        n_queries=1,
                        RoPe=None,
                        gating=gating,
                    )
                )
            elif bt == "Bilinear_out":
                blocks.append(nn.Identity())
            elif bt == "CA_Sub":
                blocks.append(
                    CrossBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=False,
                        norm_layer=RMSNorm,
                        attn_drop=0.0,
                        drop_path=0.0,
                        gating=gating,
                    )
                )
            else:
                raise ValueError(f"Unknown block type: {bt}")
        self.blocks = nn.ModuleList(blocks)

    # ------------------------------------------------------------------
    # single-modality (spot) forward
    # ------------------------------------------------------------------
    def _upe_forward(self, x: torch.Tensor):
        B0 = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        patch_px = max(int(_SCALE * 10 / _SPOT_INPUT_RES), 1)  # 40
        size = H // patch_px  # patches per side

        token = x.unsqueeze(1)  # [B, 1, 3, H, W]  (T=1)
        dates = torch.zeros(B0, 1, device=x.device).int()

        out = self.spatial_encoder(
            token, "spot", _SPOT_WAVELENGTHS, _SCALE, dates, subpatch=_SPOT_SUBPATCH, res=_SPOT_INPUT_RES
        )

        N = out["N_masked"]  # number of patches per sample
        tokens = out["tokens"].view(B0, N, self.embed_dim)
        coords_in = get_coords(tokens, size, 1, 0, 1)

        sub = out["spatial"].view(B0, N, -1, self.embed_dim)
        _, _, S_sub, _ = sub.shape
        S_sub = int(S_sub ** (1 / 2))
        full_size = size * S_sub
        sub = sub.flatten(1, 2)
        coord_spatial = (
            get_coords(sub, full_size, 1, 0, 1)
            .reshape(B0, size, S_sub, size, S_sub, 2)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B0, size ** 2, S_sub ** 2, 2)
        )

        return (
            [tokens],
            [coords_in],
            [size],
            [sub],
            [coord_spatial.flatten(1, 2)],
        )

    def _trunk_forward(self, tokens, coords_in, token_sizes, spatial, coords_spatial, latent_grid, output_grid):
        B = tokens[0].shape[0]
        n_modalities = len(spatial)

        spatial = torch.cat(spatial, dim=1).detach()
        coords_spatial = torch.cat(coords_spatial, dim=1)
        coords_in = torch.cat(coords_in, dim=1)

        coords = get_coords(tokens[0], int(latent_grid ** 0.5), 1, self.n_registers, 1)
        coords_out = get_coords(tokens[0], int(output_grid ** 0.5), 1, self.n_registers, res=1)
        tokens_in = torch.cat(tokens, dim=1)  # kept for parity; unused in the Base path

        for i, blk in enumerate(self.blocks):
            bt = self.block_type[i]
            if bt == "SA":
                tokens = blk(tokens, coords)
            elif bt == "CA_Sub":
                tokens = blk(
                    tokens, coords_out, spatial, coords_spatial,
                    n_registers=0, n_registers_q=self.n_registers, n_modalities=n_modalities,
                )
            elif bt == "Bilinear_out":
                registers, t = tokens[:, : self.n_registers], tokens[:, self.n_registers :]
                if latent_grid != output_grid:
                    t = t.view(B, int(latent_grid ** 0.5), int(latent_grid ** 0.5), self.embed_dim).permute(0, 3, 1, 2)
                    out_size = (int(output_grid ** 0.5), int(output_grid ** 0.5))
                    t = F.interpolate(t, size=out_size, mode="bilinear", align_corners=True)
                    t = t.permute(0, 2, 3, 1).flatten(1, 2)
                tokens = torch.cat((registers, t), dim=1)
            elif bt == "Bi_ACA_in":
                tokens_out = []
                for t, s in zip(tokens, token_sizes):
                    tokens_out.append(t)
                tokens = torch.stack(tokens_out, dim=2)  # [B, N, M, D]
                Bn, Nn, Mn, Dn = tokens.shape
                tokens = tokens.flatten(0, 1)
                tokens = blk(tokens)
                tokens = tokens.view(Bn, Nn, Dn)
                if self.registers is not None:
                    registers = self.registers.expand(tokens.shape[0], -1, -1)
                    tokens = torch.cat((registers, tokens), dim=1)
            else:
                raise ValueError(f"Unknown block type: {bt}")
        return tokens

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode an already-normalized spot-RGB image.

        Args:
            x: ``[B, 3, H, W]`` fp32, already normalized per ``META``
               (channel z-score). Square input (H == W), divisible by 4.

        Returns:
            ``{"dense": [B, embed_dim, H//feature_stride, W//feature_stride]}``.
        """
        B0, C, H, W = x.shape
        assert C == 3, f"spot expects 3 RGB channels, got {C}"
        assert H == W, f"square input required, got {H}x{W}"

        patch_px = max(int(_SCALE * 10 / _SPOT_INPUT_RES), 1)  # 40
        latent_side = max(H // patch_px, 1)
        latent_grid = latent_side * latent_side
        out_side = max(H // self.feature_stride, 1)
        output_grid = out_side * out_side

        tokens, coords_in, token_sizes, spatial, coords_spatial = self._upe_forward(x)
        tokens = self._trunk_forward(tokens, coords_in, token_sizes, spatial, coords_spatial, latent_grid, output_grid)

        tokens = tokens[:, self.n_registers :]  # drop register tokens
        dense = tokens.view(B0, out_side, out_side, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        return {"dense": dense}

    def load_pretrained(self, weight_path: str) -> None:
        """Load the released safetensors checkpoint (stripping the ``model.`` prefix)."""
        from safetensors.torch import load_file

        sd = load_file(weight_path)
        stripped = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(stripped, strict=False)
        print(f"[universat] load_pretrained: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing:
            print("[universat] missing examples:", missing[:5])
        if unexpected:
            print("[universat] unexpected examples:", unexpected[:5])


def build_encoder(weight_path: Optional[str] = None) -> Encoder:
    """Build the UniverSat Base encoder, optionally loading pretrained weights."""
    m = Encoder()
    if weight_path:
        m.load_pretrained(weight_path)
    return m
