"""Self-contained Swin Transformer V2 (Base) encoder for the MaRS RGB backbone.

This module vendors the SwinV2 architecture so that the released
``mars_base_rgb_encoder_only.pth`` checkpoint loads directly by parameter name,
with **no ``timm`` dependency and no imports from ``others/``**.

It reproduces ``timm``'s ``SwinTransformerV2`` (NOT the ``SwinTransformerV2Cr``
variant) with the exact ``swinv2_base_window8_256`` configuration used by MaRS:

    embed_dim = 128
    depths    = (2, 2, 18, 2)
    num_heads = (4, 8, 16, 32)
    window_size = 8
    patch_size  = 4

This is the Microsoft Swin V2 formulation: scaled cosine attention, log-spaced
continuous relative position bias (``cpb_mlp`` + ``logit_scale`` + ``q_bias`` /
``v_bias``) and post-norm residual blocks.  Parameter names are emitted exactly
as ``patch_embed.*``, ``layers_0.*`` ... ``layers_3.*`` and ``norm.*`` so that
``load_state_dict(..., strict=False)`` matches every checkpoint key.

Reference: timm 1.0.15 ``models/swin_transformer_v2.py`` (SwinTransformerV2),
https://github.com/huggingface/pytorch-image-models  (Apache-2.0).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderMeta:
    name: str
    mean: tuple
    std: tuple
    input_size: tuple
    color_order: str
    feature_stride: int   # stride of the "dense" (1/16) feature
    note: str


META = EncoderMeta(
    name="MaRS-Base RGB SwinV2 encoder (swinv2_base_window8_256)",
    mean=(0.34122, 0.35890, 0.32749),
    std=(0.24573, 0.21404, 0.20827),
    input_size=(256, 256),
    color_order="RGB",
    feature_stride=16,
    note=(
        "[0,1]-scale mean/std: MaRS normalizes RGB with (x-mean)/std on 0-255 values "
        "(RGB_MEAN=[87.01,91.52,83.51], RGB_STD=[62.66,54.58,53.11]); dividing both by 255 gives these "
        "[0,1] equivalents, so normalize_rgb(x/255) matches MaRS exactly. "
        "Window size is 8 (model name swinv2_base_window8_256). "
        "The released encoder-only checkpoint strips the final LayerNorm ('norm'); that module is "
        "freshly initialized and only affects the optional 's32' feature."
    ),
)


def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))


def _window_partition(x, window_size):
    """x: (B, H, W, C) -> windows: (nW*B, wh, ww, C)."""
    B, H, W, C = x.shape
    wh, ww = window_size
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh, ww, C)
    return windows


def _window_reverse(windows, window_size, img_size):
    """windows: (nW*B, wh, ww, C) -> x: (B, H, W, C)."""
    H, W = img_size
    wh, ww = window_size
    C = windows.shape[-1]
    x = windows.view(-1, H // wh, W // ww, wh, ww, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class _PatchEmbed(nn.Module):
    """2D image -> patch embedding (conv proj + LayerNorm, NHWC output)."""

    def __init__(self, in_chans=3, embed_dim=128, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                    # (B, C, H/4, W/4)
        x = x.permute(0, 2, 3, 1)           # (B, H/4, W/4, C)
        x = self.norm(x)
        return x


class _PatchMerging(nn.Module):
    """Patch merging: concat 2x2 neighbourhood -> Linear(4C->2C) -> LayerNorm(2C).

    NOTE: the concatenation order follows timm's reshape/permute/flatten, which
    differs from Swin-V1's x0/x1/x2/x3 indexing.  The learned ``reduction``
    weights assume this exact order, so it must not be changed.
    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        _, H, W, _ = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _WindowAttention(nn.Module):
    """SwinV2 window attention: scaled cosine attention + continuous relative
    position bias (CPB) via a small MLP over log-spaced relative coordinates."""

    def __init__(self, dim, window_size, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = _to_2tuple(window_size)  # Wh, Ww
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("k_bias", torch.zeros(dim), persistent=False)
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self):
        wh, ww = self.window_size
        # log-spaced relative coordinate table
        relative_coords_h = torch.arange(-(wh - 1), wh, dtype=torch.float32)
        relative_coords_w = torch.arange(-(ww - 1), ww, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid(relative_coords_h, relative_coords_w, indexing="ij")
        )  # (2, 2*Wh-1, 2*Ww-1)
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # (1, 2Wh-1, 2Ww-1, 2)
        relative_coords_table[:, :, :, 0] /= (wh - 1)
        relative_coords_table[:, :, :, 1] /= (ww - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # pair-wise relative position index for each token inside the window
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += wh - 1
        relative_coords[:, :, 1] += ww - 1
        relative_coords[:, :, 0] *= 2 * ww - 1
        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self, x, mask=None):
        # x: (num_windows*B, N, C), N = window_area
        B_, N, C = x.shape

        qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # scaled cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        # continuous relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)  # (Wh*Ww, Wh*Ww, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, N, N)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _SwinBlock(nn.Module):
    """SwinV2 block with post-norm branches and shifted-window attention."""

    def __init__(self, dim, num_heads, window_size, shift_size=0, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = _to_2tuple(window_size)
        # target shift (window//2 for odd-index blocks, 0 for even-index blocks);
        # effective shift is zeroed when the feature map is no larger than the window.
        self.target_shift_size = _to_2tuple(shift_size)

        self.attn = _WindowAttention(dim, self.window_size, num_heads,
                                     attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def _shifted_window_attn(self, x):
        B, H, W, C = x.shape
        wh, ww = self.window_size
        sh = 0 if H <= wh else self.target_shift_size[0]
        sw = 0 if W <= ww else self.target_shift_size[1]
        has_shift = sh > 0 or sw > 0

        if has_shift:
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))

        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        if has_shift:
            img_mask = torch.zeros((1, Hp, Wp, 1), dtype=x.dtype, device=x.device)
            cnt = 0
            for h in ((0, -wh), (-wh, -sh), (-sh, None)):
                for w in ((0, -ww), (-ww, -sw), (-sw, None)):
                    img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                    cnt += 1
            mask_windows = _window_partition(img_mask, (wh, ww))  # (nW, wh, ww, 1)
            mask_windows = mask_windows.view(-1, wh * ww)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        x_windows = _window_partition(x, (wh, ww))  # (nW*B, wh, ww, C)
        x_windows = x_windows.view(-1, wh * ww, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, wh, ww, C)
        x = _window_reverse(attn_windows, (wh, ww), (Hp, Wp))
        x = x[:, :H, :W, :].contiguous()

        if has_shift:
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
        return x

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        x = x + self.norm1(self._shifted_window_attn(x))
        x = x.reshape(B, -1, C)
        x = x + self.norm2(self.mlp(x))
        x = x.reshape(B, H, W, C)
        return x


class _BasicLayer(nn.Module):
    """One SwinV2 stage: optional PatchMerging downsample + a stack of blocks."""

    def __init__(self, dim, out_dim, depth, num_heads, window_size, downsample=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            _SwinBlock(
                dim=out_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
            )
            for i in range(depth)
        ])
        if downsample:
            self.downsample = _PatchMerging(dim=dim, out_dim=out_dim)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class Encoder(nn.Module):
    """Frozen SwinV2-Base RGB encoder exposing multi-scale features.

    Stage outputs (for a 256x256 input):
        s4 (stride 4,  128 ch) -> not returned
        s8 (stride 8,  256 ch)
        dense (stride 16, 512 ch)
        s32 (stride 32, 1024 ch, final LayerNorm applied)
    """

    def __init__(self):
        super().__init__()
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
        window_size = 8
        in_chans = 3
        patch_size = 4

        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 1024

        self.patch_embed = _PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)

        dims = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]  # 128, 256, 512, 1024
        self.layers_0 = _BasicLayer(dims[0], dims[0], depths[0], num_heads[0], window_size, downsample=False)
        self.layers_1 = _BasicLayer(dims[0], dims[1], depths[1], num_heads[1], window_size, downsample=True)
        self.layers_2 = _BasicLayer(dims[1], dims[2], depths[2], num_heads[2], window_size, downsample=True)
        self.layers_3 = _BasicLayer(dims[2], dims[3], depths[3], num_heads[3], window_size, downsample=True)
        self._stages = [self.layers_0, self.layers_1, self.layers_2, self.layers_3]

        # Final LayerNorm. NOTE: the released encoder-only checkpoint does NOT
        # contain 'norm.weight'/'norm.bias', so this stays at its default init.
        self.norm = nn.LayerNorm(self.num_features)

    def load_pretrained(self, weight_path):
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(ckpt.keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)

        print(f"[MaRS encoder] loading {weight_path}")
        print(f"  checkpoint keys: {len(ckpt_keys)}   model keys: {len(model_keys)}")
        print(f"  missing   (in model, not in ckpt): {len(missing)}")
        for k in missing[:8]:
            print(f"    - {k}")
        print(f"  unexpected (in ckpt, not in model): {len(unexpected)}")
        for k in unexpected[:8]:
            print(f"    - {k}")

        res = self.load_state_dict(ckpt, strict=False)
        return res

    def forward_features(self, x):
        """x: (B, 3, H, W) already normalized per META.
        Returns {"dense", "s8", "s32"} in fp32 NCHW."""
        x = self.patch_embed(x)  # (B, H/4, W/4, C)
        feats = {}
        for i, stage in enumerate(self._stages):
            x = stage(x)
            if i == 1:
                feats["s8"] = x.permute(0, 3, 1, 2).contiguous()    # (B, 256, H/8, W/8)
            elif i == 2:
                feats["dense"] = x.permute(0, 3, 1, 2).contiguous()  # (B, 512, H/16, W/16)
        x = self.norm(x)
        feats["s32"] = x.permute(0, 3, 1, 2).contiguous()  # (B, 1024, H/32, W/32)
        return feats

    def forward(self, x):
        return self.forward_features(x)


def build_encoder(weight_path=None):
    m = Encoder()
    if weight_path:
        m.load_pretrained(weight_path)
    return m
