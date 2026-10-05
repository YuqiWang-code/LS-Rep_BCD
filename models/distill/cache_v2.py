"""Compact Teacher Cache v2.

Uniform per-sample schema (see plan §9):

    {
        "teacher_id": str,
        "sample_id": str,
        "local_change": Tensor[128, Ht, Wt],   # FP16, symmetric change evidence
        "confidence":    Tensor[1, Ht, Wt],    # FP16
        "global_desc":   Tensor[128],          # FP16
        "meta": {...}                          # weight hash, normalization, color order, ...
    }

The cached evidence is already T1/T2 exchange-invariant, so geometric replay only
needs crop and flip (no temporal swap).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

CACHE_SCHEMA_VERSION = 2
CACHE_CHANNELS = 128


def _crop_resize(x: torch.Tensor, top: int, left: int, src_h: int, src_w: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    t = max(0, int(round(top * H / src_h)))
    l = max(0, int(round(left * W / src_w)))
    b = max(t + 1, H - t)
    r = max(l + 1, W - l)
    cropped = x[..., t:b, l:r]
    if cropped.shape[-2:] != (H, W):
        if cropped.dim() == 3:
            cropped = cropped.unsqueeze(0)
            cropped = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
            cropped = cropped.squeeze(0)
        else:
            cropped = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
    return cropped


def apply_cache_state(
    cache: Dict[str, torch.Tensor],
    state: Dict,
    src_h: int = 256,
    src_w: int = 256,
) -> Dict[str, torch.Tensor]:
    """Replay crop/flip onto cached (already exchange-invariant) tensors."""
    out = {k: v for k, v in cache.items() if isinstance(v, torch.Tensor)}

    crop = state.get("crop_resize")
    if crop and crop.get("enabled"):
        top, left = crop["top"], crop["left"]
        for key in list(out.keys()):
            out[key] = _crop_resize(out[key], top, left, src_h, src_w)

    if state.get("flip_v"):
        for key in list(out.keys()):
            out[key] = torch.flip(out[key], dims=[-2])
    if state.get("flip_h"):
        for key in list(out.keys()):
            out[key] = torch.flip(out[key], dims=[-1])

    return out


class TeacherCacheReaderV2:
    """Read per-sample compact cache files produced by generate_teacher_cache_v2."""

    def __init__(self, cache_root: str | Path, dataset_name: str):
        self.cache_root = Path(cache_root)
        self.dataset_name = dataset_name.upper()

    def _path(self, split: str, sample_id: str) -> Path:
        return self.cache_root / self.dataset_name / split / f"{sample_id}.pt"

    def load(self, split: str, sample_id: str) -> Dict[str, torch.Tensor]:
        path = self._path(split, sample_id)
        if not path.is_file():
            raise FileNotFoundError(f"Teacher cache missing: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        return {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CACHE_CHANNELS",
    "apply_cache_state",
    "TeacherCacheReaderV2",
]
