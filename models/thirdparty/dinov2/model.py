"""Self-contained DINOv2 ViT-B/14 encoder (dense patch features).

Loads the official ``dinov2_vitb14_pretrain.pth`` (plain state_dict). The
checkpoint pos_embed has 1370 tokens (trained at 518px / 37x37 grid) and is
bicubically interpolated to the 224px / 16x16 grid on load, matching the official
DINOv2 ``interpolate_pos_encoding``. ``forward_features`` returns the normalized
patch-token grid as "dense" [B,768,16,16] and the CLS token as "global" [B,768].
"""

from __future__ import annotations

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
    feature_stride: int
    note: str


META = EncoderMeta(
    "DINOv2 ViT-B/14",
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
    (224, 224), "RGB", 14, "generic self-supervised dense features",
)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1.0):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=14, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid = img_size // patch_size
        self.meta = META

        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def _interpolate_pos_embed(self, pe: torch.Tensor) -> torch.Tensor:
        target = self.grid * self.grid + 1
        if pe.shape[1] == target:
            return pe
        cls = pe[:, :1]
        spat = pe[:, 1:]
        n = int(round(spat.shape[1] ** 0.5))
        spat = spat.reshape(1, n, n, -1).permute(0, 3, 1, 2)
        spat = F.interpolate(spat, size=(self.grid, self.grid), mode="bicubic", align_corners=False)
        spat = spat.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)
        return torch.cat([cls, spat], dim=1)

    def load_pretrained(self, weight_path: str) -> None:
        sd = torch.load(weight_path, map_location="cpu", weights_only=False)
        if "pos_embed" in sd:
            sd["pos_embed"] = self._interpolate_pos_embed(sd["pos_embed"])
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[dinov2] loaded {weight_path}: missing={len(missing)} unexpected={len(unexpected)}")

    def forward_features(self, x: torch.Tensor) -> dict:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        patch = x[:, 1:].transpose(1, 2).reshape(B, self.embed_dim, self.grid, self.grid)
        return {"dense": patch, "global": x[:, 0]}


def build_encoder(weight_path=None) -> Encoder:
    encoder = Encoder()
    if weight_path:
        encoder.load_pretrained(weight_path)
    encoder.eval()
    return encoder
