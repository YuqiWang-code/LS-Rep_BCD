"""Minimal, self-contained AnySat encoder (spot RGB path only).

Vendored reconstruction of the AnySat "base" encoder (CVPR 2025 Highlight),
restricted to the single-date 3-channel RGB ``spot`` modality at 1 m GSD.
It reproduces the *official* AnySat architecture (``Any_multi.AnyModule`` in
``release`` mode + ``PatchMLPMulti`` + ``TransformerMulti`` + iRPE) so that the
provided ``pre-trained_weights/AnySat.pth`` state_dict can be loaded (strict=False)
and ``forward_features`` returns the dense feature map used by the reference
``output='dense'`` path.

Only ``torch`` is required (no einops / timm / transformers / flash-attn; the
vanilla scaled-dot-product attention path is used, which is numerically
equivalent to the flash-attn path for inference).

The dense token grid is produced exactly like the official
``forward_release(x, scale=1, output='dense', output_modality='spot')`` call:

* input  [B, 3, 256, 256]  ->  patch embedding (patch_size=10, stride=10)
* token grid 25x25 (``floor((256-10)/10)+1``)
* per-patch "sub-patch" detail token (768) concatenated with the cross-attention
  context token (768)  ->  dense channels = 1536
* output returned as [B, 1536, 25, 25] (NCHW, fp32)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# EncoderMeta / META
# --------------------------------------------------------------------------- #
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
    name="AnySat (spot RGB path)",
    # The AnySat repo does NOT define a fixed spot-RGB mean/std (the demo only
    # normalizes aerial/S1/S2 with dataset stats; Pastis spot tiles are loaded
    # raw 0-255 with optional per-fold stats). ImageNet RGB is used here as the
    # documented fallback per the extraction brief.
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    input_size=(256, 256),
    color_order="RGB",
    feature_stride=10,
    note=(
        "Assumed ImageNet RGB normalization (repo defines no spot mean/std). "
        "Hard-codes modality='spot', GSD=1.0 m, patch_size=10, scale=1 inside "
        "forward_features. Dense grid is floor((H-10)/10)+1 = 25x25 for a 256 "
        "input (the last ~6 px are ignored); channels = 1536 = 768 cross-token "
        "+ 768 sub-patch token."
    ),
)


# --------------------------------------------------------------------------- #
# Small building blocks (from utils.py / pos_embed.py / irpe.py, trimmed)
# --------------------------------------------------------------------------- #
def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    with torch.no_grad():
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _trunc_normal_(tensor, mean, std, a, b)


class Mlp(nn.Module):
    """ViT-style MLP (GELU, no norm, dropout 0) — matches checkpoint keys fc1/fc2."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# --------------------------------------------------------------------------- #
# Positional embeddings (from pos_embed.py)
# --------------------------------------------------------------------------- #
def _get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb


def _get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def get_2d_sincos_pos_embed_with_scale(embed_dim, grid_size, scale, cls_token=False, modis=False):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)  # 2 x h x w
    grid = torch.einsum("chw,n->cnhw", grid, torch.tensor([scale]))
    _, n, h, w = grid.shape
    pos_embed = _get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([n, 1, embed_dim], dtype=torch.float32), pos_embed], dim=1
        )
    if modis:
        pos_embed = torch.cat(
            [torch.zeros([n, 1, embed_dim], dtype=torch.float32), pos_embed], dim=1
        )
    return pos_embed


def get_2d_sincos_pos_embed_with_resolution(embed_dim, grid_size, res, cls_token=False, modalities=()):
    pos_embed_final = {}
    for modality in modalities:
        grid_size_aug = max(1, int(grid_size * 10 / res[modality]))
        grid_h = torch.arange(grid_size_aug, dtype=torch.float32)
        grid_w = torch.arange(grid_size_aug, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = torch.einsum("chw,n->cnhw", grid, torch.tensor([res[modality]]))
        _, n, h, w = grid.shape
        pos_embed = _get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
        pos_embed = pos_embed.reshape(n, h * w, embed_dim)
        if cls_token:
            pos_embed = torch.cat(
                [torch.zeros([n, 1, embed_dim], dtype=torch.float32), pos_embed], dim=1
            )
        pos_embed_final[modality] = pos_embed
    return pos_embed_final


# --------------------------------------------------------------------------- #
# Contextual iRPE on keys (from irpe.py, euclidean method, trimmed)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _piecewise_index(relative_position, alpha, beta, gamma, dtype):
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (
        torch.sign(rp_out)
        * (
            alpha
            + torch.log(rp_abs_out / alpha) / math.log(gamma / alpha) * (beta - alpha)
        ).round().clip(max=beta)
    ).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        idx = idx.round().to(dtype)
    idx[not_mask] = y_out
    return idx


class ContextualRPE(nn.Module):
    """Contextual iRPE on keys (euclidean method, shared head, skip=1).

    Exactly matches the repo's ``get_rpe_config(ratio=1.9, method='euc',
    mode='ctx', shared_head=True, skip=1, rpe_on='k')`` which yields
    ``lookup_table_weight`` of shape ``(1, head_dim, 8)``.
    """

    def __init__(self, head_dim, ratio=1.9):
        super().__init__()
        self.head_dim = head_dim
        self.alpha = 1.0 * ratio
        self.beta = 2.0 * ratio
        self.gamma = 8.0 * ratio
        self.beta_int = int(self.beta)
        # 2*beta_int + 1 buckets for the spatial grid + 1 extra bucket for the cls token
        self.num_buckets = 2 * self.beta_int + 1 + 1
        # shared_head=True -> lookup table has a single head, broadcast over all heads
        self.lookup_table_weight = nn.Parameter(
            torch.zeros(1, head_dim, self.num_buckets)
        )
        trunc_normal_(self.lookup_table_weight, std=0.02)

    def _bucket_ids(self, height, width, device, dtype):
        rows = torch.arange(height, dtype=dtype, device=device).view(height, 1).repeat(1, width)
        cols = torch.arange(width, dtype=dtype, device=device).view(1, width).repeat(height, 1)
        pos = torch.stack([rows, cols], 2)  # [h, w, 2]
        L = height * width
        pos1 = pos.view(L, 1, 2)
        pos2 = pos.view(1, L, 2)
        diff = pos1 - pos2  # [L, L, 2]
        # euclidean method
        dis = diff.square().sum(2).float().sqrt().round()
        bucket = _piecewise_index(dis, self.alpha, self.beta, self.gamma, dtype)
        bucket = bucket + self.beta_int
        bucket = bucket.view(height, width, height, width)
        bucket = bucket[:height, :width, :height, :width].reshape(L, L)
        # add the cls skip token bucket
        num_buckets_no_skip = 2 * self.beta_int + 1
        new_bids = bucket.new_empty(size=(1 + L, 1 + L))
        new_bids[:1] = num_buckets_no_skip
        new_bids[:, :1] = num_buckets_no_skip
        new_bids[1:, 1:] = bucket
        return new_bids.contiguous()

    def forward(self, x, height=None, width=None, pos=None, modis=False):
        # x: [B, H, L, head_dim]
        B, H, L, D = x.shape
        if height is None:
            E = int(math.sqrt(L))
            height = width = E
        device = x.device
        dtype = torch.long
        rp_bucket = self._bucket_ids(height, width, device, dtype)  # [1 + h*w, 1 + h*w]
        rp_bucket = rp_bucket.unsqueeze(0).repeat(B, 1, 1)  # [B, L, L]

        # contextual + transposed lookup (shared head broadcast)
        offset = torch.arange(
            0, L * self.num_buckets, self.num_buckets, dtype=rp_bucket.dtype, device=device
        ).view(-1, 1).unsqueeze(0).repeat(B, 1, 1)
        ctx_flatten = (rp_bucket + offset).flatten(1, 2)  # [B, L*L]

        lookup_table = torch.matmul(
            x.transpose(0, 1).reshape(-1, B * L, self.head_dim),
            self.lookup_table_weight,
        ).view(-1, B, L, self.num_buckets).transpose(0, 1)  # [B, H, L, num_buckets]

        look = lookup_table.flatten(2)  # [B, H, L*num_buckets]
        out = torch.gather(look, dim=2, index=ctx_flatten.unsqueeze(1)).view(B, -1, L, L)
        return out


# --------------------------------------------------------------------------- #
# Attention variants (from utils_ViT.py)
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    """Standard scaled-dot-product attention used by the main transformer blocks."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RPEAttention(nn.Module):
    """Attention with image relative position encoding (spatial-encoder blocks)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe_k = ContextualRPE(head_dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        if mask is None:
            height = int((N) ** 0.5)
        else:
            height = mask.shape[-1]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pos=mask, height=height, width=height)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Main transformer block (no RPE, no LayerScale, no drop-path)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BlockTransformer(nn.Module):
    """Spatial-encoder block (RPE attention)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, attn_drop=0.0, drop=0.0):
        super().__init__()
        # NOTE: TransformerMulti builds these with norm_layer=nn.LayerNorm (default eps=1e-5)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                 attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossRPEAttentionMulti(nn.Module):
    """Cross-attention (query = learned token) used by the final cross block (release mode)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))
        self.rpe_k = ContextualRPE(head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward_release(self, x, mask=None, n_modalities=1, modis=False, scale=1):
        B, N, C = x.shape
        num_patches = N // n_modalities + int((N - int(modis)) % n_modalities > 0) + int(modis)
        pos_embed = get_2d_sincos_pos_embed_with_scale(
            C, int(num_patches ** 0.5), scale, cls_token=True, modis=modis
        ).to(x.device)
        if mask is None:
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed.expand(B, -1, -1)
        else:
            num_patches = mask.shape[-1] + 1 + modis
            mask_pos = mask.unsqueeze(-1).repeat(1, 1, pos_embed.shape[-1])
            pos_embed_e = pos_embed.expand(B, -1, -1)
            masked_pos_embed = torch.gather(pos_embed.expand(B, -1, -1)[:, 1:], dim=1, index=mask_pos)
            pos_embed = torch.cat([pos_embed_e[:, :(1 + modis)], masked_pos_embed], dim=1)
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed
        q = q_.reshape(B, num_patches, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rpe_k is not None:
            height = int(num_patches ** 0.5)
            rpe = self.rpe_k(q, height=height, width=height, pos=mask, modis=modis)
            attn += torch.cat(
                [rpe[:, :, :, :(1 + modis)], rpe[:, :, :, (1 + modis):].repeat(1, 1, 1, n_modalities)],
                dim=-1,
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, num_patches, C)
        x = torch.cat([x[:, :1], x[:, (1 + modis):]], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossBlockMulti(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CrossRPEAttentionMulti(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward_release(self, x, n_modalities=1, modis=False, scale=1):
        x = self.attn.forward_release(self.norm1(x), n_modalities=n_modalities, modis=modis, scale=scale)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Projector (from patch_embeddings.py)
# --------------------------------------------------------------------------- #
class PatchMLPMulti(nn.Module):
    def __init__(self, in_chans=3, resolution=1.0, embed_dim=768, patch_size=10,
                 bias=False, mlp=()):
        super().__init__()
        self.patch_size = patch_size
        self.res = int(10 / resolution)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                     stride=patch_size, bias=bias)
        layers = []
        for i in range(len(mlp) - 1):
            layers.extend([
                nn.Linear(mlp[i], mlp[i + 1]),
                nn.LayerNorm(mlp[i + 1]),
                nn.ReLU(),
            ])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, scale):
        x = self.patch_embed(x)
        grid_size = (self.res // self.patch_size, self.res // self.patch_size)
        x = x.unfold(2, grid_size[0], grid_size[0]).unfold(3, grid_size[1], grid_size[1])
        x = x.flatten(4, 5)
        x = x.unfold(2, scale, scale).unfold(3, scale, scale)
        x = x.flatten(2, 3).permute(0, 1, 2, 4, 5, 3).flatten(3, 5)
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.flatten(0, 1)
        x = self.mlp(x)
        return x


# --------------------------------------------------------------------------- #
# Spatial encoder (from Transformer.py, TransformerMulti, release mode)
# --------------------------------------------------------------------------- #
class TransformerMulti(nn.Module):
    def __init__(self, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, attn_drop_rate=0.0, drop_path_rate=0.0,
                 input_res=None):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.input_res = input_res if input_res is not None else {"spot": 10}
        self.predictor_blocks = nn.ModuleList([
            BlockTransformer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, attn_drop=attn_drop_rate, drop=0.0)
            for _ in range(depth)
        ])
        self.predictor_norm = nn.LayerNorm(embed_dim)  # default eps=1e-5 (matches repo)

    def forward_release(self, x, modality, scale, keep_subpatch=False):
        B, N, C = x.shape
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x += get_2d_sincos_pos_embed_with_resolution(
            C, scale, self.input_res, cls_token=True, modalities=[modality]
        )[modality].to(x.device)
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)
        if keep_subpatch:
            return x[:, 0], x[:, 1:]
        return x[:, 0]


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, embed_dim=768, depth=6, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality = "spot"
        self.scale = 1  # patch_size(10 m) // 10
        self.gsd = 1.0  # meters / pixel

        # main cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # spot RGB projector (patch_size=10, in_chans=3, resolution=1.0 m)
        self.projector_spot = PatchMLPMulti(
            in_chans=3, resolution=self.gsd, embed_dim=embed_dim, patch_size=10,
            bias=False, mlp=[embed_dim, embed_dim * 2, embed_dim],
        )

        # per-modality spatial encoder
        self.spatial_encoder = TransformerMulti(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4.0,
            qkv_bias=True, attn_drop_rate=0.0, drop_path_rate=0.0,
            input_res={"spot": 10},
        )

        # shared transformer: 6 vanilla blocks + 1 cross block
        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
                   attn_drop=0.0, proj_drop=0.0) for _ in range(depth)]
            + [CrossBlockMulti(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                               qkv_bias=True, attn_drop=0.0, drop=0.0)]
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.spatial_encoder.cls_token, std=0.02)

    def load_pretrained(self, weight_path):
        ck = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "state_dict" in ck:
            state_dict = ck["state_dict"]
        else:
            state_dict = ck
        model_keys = set(self.state_dict().keys())
        ck_keys = set(state_dict.keys())
        missing = sorted(model_keys - ck_keys)
        unexpected = sorted(ck_keys - model_keys)
        print(f"[AnySat] load_pretrained: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[AnySat]   missing examples: {missing[:5]}")
        if unexpected:
            print(f"[AnySat]   unexpected examples: {unexpected[:5]}")
        self.load_state_dict(state_dict, strict=False)
        return self

    def forward_features(self, x):
        # x: [B, 3, H, W] already normalized per META
        B, _, H, W = x.shape
        scale = self.scale
        modality = self.modality
        n_modalities = 1
        modis = False
        batch_size = B
        device = x.device

        # 1) patch projection -> token [B*N, 1, C]
        token = self.projector_spot(x, scale)
        Bt, _, Ct = token.shape
        N = Bt // batch_size
        num_patches = int(N ** 0.5)

        # 2) spatial encoder (keep sub-patch tokens)
        patch_cls, subs = self.spatial_encoder.forward_release(
            token, modality, scale, keep_subpatch=True
        )
        subpatches = subs.view(-1, N, subs.shape[1], subs.shape[2])  # [B, N, P, C]

        # 3) build main-transformer tokens (cls + patch tokens)
        pos_embed = get_2d_sincos_pos_embed_with_scale(
            Ct, num_patches, scale, cls_token=True
        ).to(device)
        patch_cls = patch_cls.view(-1, N, self.embed_dim)  # [B, N, C]
        tokens = patch_cls + pos_embed[:, 1:, :]  # [B, N, C]
        cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(patch_cls.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # [B, N+1, C]

        # 4) main transformer blocks
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1].forward_release(
            tokens, n_modalities=n_modalities, modis=modis, scale=scale
        )

        # 5) dense assembly (identical to reference forward_release dense path)
        tokens = tokens[:, 1:].unsqueeze(2).repeat(1, 1, subpatches.shape[2], 1)  # [B, N, P, C]
        dense_tokens = torch.cat([tokens, subpatches], dim=3)  # [B, N, P, 2C]
        Bn, Nn, P, D = dense_tokens.shape
        patch_size_sub = int(P ** 0.5)
        size = num_patches * patch_size_sub
        dense_tokens = dense_tokens.unsqueeze(2).permute(0, 2, 4, 1, 3)
        dense_tokens = dense_tokens.view(Bn, 1, D, Nn, patch_size_sub, patch_size_sub)
        dense_tokens = dense_tokens.view(
            Bn, 1, D, num_patches, num_patches, patch_size_sub, patch_size_sub
        ).permute(0, 1, 2, 3, 5, 4, 6)
        dense_tokens = dense_tokens.reshape(Bn, 1, D, size, size).flatten(0, 1).permute(0, 2, 3, 1)
        dense = dense_tokens.permute(0, 3, 1, 2).contiguous()  # [B, 2C, size, size]

        return {"dense": dense}


def build_encoder(weight_path=None):
    m = Encoder()
    if weight_path:
        m.load_pretrained(weight_path)
    return m
