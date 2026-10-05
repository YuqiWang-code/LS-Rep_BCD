"""Minimal, self-contained SAM 2.1 Hiera image encoder.

This module vendors only the SAM 2 *image encoder* (Hiera trunk + FPN neck),
faithfully adapted from the facebookresearch/sam2 reference implementation:

    sam2/modeling/backbones/hieradet.py
    sam2/modeling/backbones/image_encoder.py
    sam2/modeling/backbones/utils.py
    sam2/modeling/position_encoding.py
    sam2/modeling/sam2_utils.py   (DropPath / MLP only)

The mask decoder, memory encoder/attention and prompt machinery are *not*
included. The resulting module is a frozen feature extractor intended to be
used as a teacher backbone for binary change detection.

Dependencies: torch only (no timm, transformers, einops, iopath, numpy).

Licensed under the same license as the SAM 2 repository (see the LICENSE file
that is preserved under ``others/sam2-main/``).
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderMeta:
    name: str
    mean: tuple          # (R,G,B) means the model's normalization expects
    std: tuple           # (R,G,B) stds
    input_size: tuple    # (H, W) verified to work
    color_order: str     # "RGB" or "BGR"
    feature_stride: int  # stride of the returned "dense" feature
    note: str


META = EncoderMeta(
    name="SAM 2.1 Hiera Large (image encoder only)",
    # SAM2 does NOT normalize internally: it expects inputs that have already
    # been normalized with the ImageNet statistics below (see
    # sam2/utils/transforms.py -> torchvision Normalize(mean, std) applied to
    # a float tensor in [0, 1]).
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    input_size=(256, 256),
    color_order="RGB",
    feature_stride=16,
    note=(
        "Fully convolutional Hiera trunk (patch stride 4, q_pool stride 2 at "
        "stage 2/3/4) + FPN neck (d_model=256), scalp=1 as in SAM 2.1 config. "
        "forward_features(x) returns: 'dense' (stride 16, the SAM2 "
        "vision_features), 's8' (stride 8) and 's4' (stride 4), all 256 "
        "channels. Inputs must be RGB float32 in [0,1] already normalized by "
        "mean/std above."
    ),
)


# ---------------------------------------------------------------------------
# Shared building blocks (DropPath / MLP), adapted from sam2_utils.py
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """Stochastic depth (adapted from timm). Kept for architectural fidelity;
    drop_prob is 0.0 for the pretrained Hiera config so it is a no-op."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# Windowing helpers + patch embedding, adapted from backbones/utils.py
# ---------------------------------------------------------------------------
def window_partition(x: torch.Tensor, window_size: int):
    """Partition [B, H, W, C] into non-overlapping windows with padding."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw, hw):
    """Reverse window_partition and remove padding."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 144,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


# ---------------------------------------------------------------------------
# Hiera trunk, adapted from backbones/hieradet.py
# ---------------------------------------------------------------------------
def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)  # (B, C, H', W') -> (B, H', W', C)
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)

        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)

        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(dim, dim_out, num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """Reference: https://arxiv.org/abs/2306.00989"""

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        weights_path=None,
        return_interm_layers: bool = True,
    ):
        super().__init__()
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks

        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )
            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        return outputs


# ---------------------------------------------------------------------------
# Sine position embedding, adapted from position_encoding.py (no numpy needed)
# ---------------------------------------------------------------------------
class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.cache = {}

    @torch.no_grad()
    def _pe(self, B, device, *cache_key):
        H, W = cache_key
        if cache_key in self.cache:
            return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)

        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .view(1, -1, 1)
            .repeat(B, 1, W)
        )
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .view(1, 1, -1)
            .repeat(B, H, 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        cache_key = (x.shape[-2], x.shape[-1])
        return self._pe(B, x.device, *cache_key)


# ---------------------------------------------------------------------------
# FPN neck + image encoder, adapted from backbones/image_encoder.py
# ---------------------------------------------------------------------------
class FpnNeck(nn.Module):
    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )
            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)
        return out, pos


class ImageEncoder(nn.Module):
    def __init__(self, trunk: nn.Module, neck: nn.Module, scalp: int = 0):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features, pos = features[:-self.scalp], pos[:-self.scalp]
        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output


# ---------------------------------------------------------------------------
# Public Encoder + builder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """SAM 2.1 Hiera Large image encoder (frozen teacher backbone)."""

    def __init__(self, scalp: int = 1):
        super().__init__()
        trunk = Hiera(
            embed_dim=144,
            num_heads=2,
            stages=(2, 6, 36, 4),
            global_att_blocks=(23, 33, 43),
            window_pos_embed_bkg_spatial_size=(7, 7),
            window_spec=(8, 4, 16, 8),
        )
        neck = FpnNeck(
            position_encoding=PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                scale=None,
                temperature=10000,
            ),
            d_model=256,
            backbone_channel_list=[1152, 576, 288, 144],
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        )
        self.image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=scalp)

    def load_pretrained(self, weight_path: str):
        state = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            # SAM 2 checkpoints nest everything under a single 'model' key.
            state = state["model"]

        missing, unexpected = self.load_state_dict(state, strict=False)

        n_total = len(state)
        n_enc = len(self.state_dict())
        print(f"[sam2 encoder] checkpoint keys: {n_total}")
        print(f"[sam2 encoder] image_encoder parameters: {n_enc}")
        print(f"[sam2 encoder] missing keys: {len(missing)}")
        if missing:
            for k in list(missing)[:5]:
                print(f"    MISSING  {k}")
        print(f"[sam2 encoder] unexpected keys (non-image_encoder, ignored): {len(unexpected)}")
        if unexpected:
            for k in list(unexpected)[:5]:
                print(f"    UNEXPECTED  {k}")
        return self

    def forward_features(self, x: torch.Tensor):
        """x: [B, 3, H, W] float32, RGB, already normalized by META.mean/std.

        Returns a dict of fp32 feature maps:
            "dense": stride-16 feature (SAM2 'vision_features'), 256 channels
            "s8":    stride-8  feature, 256 channels
            "s4":    stride-4  feature, 256 channels
        """
        out = self.image_encoder(x)
        fpn = out["backbone_fpn"]  # [s4, s8, s16] after scalp=1
        return {
            "dense": fpn[-1],
            "s8": fpn[1],
            "s4": fpn[0],
        }


def build_encoder(weight_path: str = None) -> Encoder:
    m = Encoder()
    if weight_path:
        m.load_pretrained(weight_path)
    return m
