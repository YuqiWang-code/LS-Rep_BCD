"""Teacher packages: frozen foundation encoder + fixed projection -> compact change evidence.

A Teacher Package is ``{Teacher_j + CacheSchema_j + MinimalTranslator_j}``. The first
iteration uses one uniform translator for every teacher: dense features are
normalized, differenced (``|F_A - F_B|``, already exchange-invariant), then pushed
through a fixed (seeded) random projection to 128 channels. This keeps the
capability-validation matrix as fair as possible across heterogeneous teachers.

All teacher encoders live in ``models.thirdparty.<name>`` and expose
``build_encoder(weight_path)`` returning an ``nn.Module`` whose
``forward_features(x) -> {"dense": [B,C,h,w], ...}``. Normalization facts are
exposed via the module-level ``META`` dataclass.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache_v2 import CACHE_CHANNELS


@dataclass
class TeacherSpec:
    module: str          # import path of models.thirdparty.<name>
    weight: str          # weight filename under pre-trained_weights/
    builder: str = "build_encoder"  # function name to call
    feature_key: str = "dense"
    projection_seed: int = 0
    note: str = ""


TEACHER_REGISTRY: Dict[str, TeacherSpec] = {
    "sam2":       TeacherSpec("models.thirdparty.sam2", "sam2.1_hiera_large.pt", note="SAM2.1 Hiera-Large image encoder"),
    "dinov2":     TeacherSpec("models.thirdparty.dinov2", "dinov2_vitb14_pretrain.pth", note="DINOv2 ViT-B/14"),
    "dinov3_lvd": TeacherSpec("models.thirdparty.dinov3", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", builder="build_encoder_vitb", note="DINOv3 ViT-B/16 LVD-1689M (neutral)"),
    "dinov3_sat": TeacherSpec("models.thirdparty.dinov3", "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", builder="build_encoder_vitl", note="DINOv3 ViT-L/16 SAT-493M"),
    "remoteclip": TeacherSpec("models.thirdparty.remoteclip", "RemoteCLIP-ViT-L-14.pt", note="RemoteCLIP ViT-L/14 visual"),
    "mars":       TeacherSpec("models.thirdparty.mars", "mars_base_rgb_encoder_only.pth", note="MaRS-Base RGB SwinV2"),
    "anysat":     TeacherSpec("models.thirdparty.anysat", "AnySat.pth", note="AnySat (spot RGB)"),
    "universat":  TeacherSpec("models.thirdparty.universat", "universat_base.safetensors", note="UniverSat Base"),
    "radio":      TeacherSpec("models.thirdparty.radio", "c-radio_v3-b_half.pth", note="C-RADIOv3-B"),
}

WAVE_A = ["sam2", "dinov2", "dinov3_lvd", "dinov3_sat", "remoteclip", "mars"]
WAVE_B = ["anysat", "universat", "radio"]


def normalize_rgb(x: torch.Tensor, mean, std) -> torch.Tensor:
    """Normalize a [B,3,H,W] tensor in [0,1] RGB order."""
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def _fixed_projection(cin: int, cout: int, seed: int) -> nn.Conv2d:
    """Deterministic fixed random projection (scaled Gaussian), frozen."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(cout, cin, generator=g) / (cin ** 0.5)
    conv = nn.Conv2d(cin, cout, 1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(w.view(cout, cin, 1, 1))
    conv.requires_grad_(False)
    return conv


class DenseChangeTeacherPackage(nn.Module):
    """Uniform dense-change translator shared by all teachers in the first iteration."""

    def __init__(self, teacher_id: str, encoder: nn.Module, meta, projection_seed: int = 0, feature_key: str = "dense"):
        super().__init__()
        self.teacher_id = teacher_id
        self.encoder = encoder
        self.meta = meta
        self.projection_seed = projection_seed
        self.feature_key = feature_key
        self.proj: nn.Conv2d | None = None

    def _ensure_proj(self, cin: int, device) -> nn.Conv2d:
        if self.proj is None:
            self.proj = _fixed_projection(cin, CACHE_CHANNELS, self.projection_seed).to(device)
        return self.proj

    @torch.no_grad()
    def encode_pair(self, xa: torch.Tensor, xb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a T1/T2 pair into compact symmetric change evidence.

        ``xa``, ``xb``: [B,3,H,W] RGB in [0,1].
        """
        xa = normalize_rgb(xa, self.meta.mean, self.meta.std)
        xb = normalize_rgb(xb, self.meta.mean, self.meta.std)
        fa = self.encoder.forward_features(xa)[self.feature_key].float()
        fb = self.encoder.forward_features(xb)[self.feature_key].float()
        fa = F.normalize(fa, dim=1)
        fb = F.normalize(fb, dim=1)
        evidence = (fa - fb).abs()  # [B, C, h, w], exchange-invariant

        cin = evidence.shape[1]
        proj = self._ensure_proj(cin, evidence.device)
        local_change = proj(evidence)                       # [B, 128, h, w]
        confidence = evidence.max(dim=1, keepdim=True)[0]    # [B, 1, h, w]
        gap = F.normalize(evidence.mean(dim=(2, 3)), dim=1)  # [B, C]
        global_desc = F.linear(gap, proj.weight.view(CACHE_CHANNELS, -1))  # [B, 128]

        return {
            "local_change": local_change,
            "confidence": confidence,
            "global_desc": global_desc,
        }


def build_teacher_package(
    teacher_id: str,
    weight_dir: str | Path,
    projection_seed: int = 0,
    device: str = "cpu",
):
    """Build a Teacher Package (encoder + fixed projection) from the registry."""
    spec = TEACHER_REGISTRY[teacher_id]
    module = importlib.import_module(spec.module)
    builder = getattr(module, spec.builder)
    encoder = builder(str(Path(weight_dir) / spec.weight))
    encoder.eval()
    meta = getattr(encoder, "meta", None) or getattr(module, "META", None)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, meta.input_size[0], meta.input_size[1])
        out = encoder.forward_features(dummy)
        cin = out[spec.feature_key].shape[1]

    package = DenseChangeTeacherPackage(
        teacher_id, encoder, meta, projection_seed=projection_seed, feature_key=spec.feature_key
    )
    return package, meta, cin


__all__ = [
    "TEACHER_REGISTRY",
    "WAVE_A",
    "WAVE_B",
    "TeacherSpec",
    "DenseChangeTeacherPackage",
    "build_teacher_package",
    "normalize_rgb",
    "CACHE_CHANNELS",
]
