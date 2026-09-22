"""Teacher-cache loading and synchronized geometric replay for FA-SCRD.

The teacher cache is generated offline on full-resolution 256x256 images. At
training time the exact same geometric augmentation (crop / flip / temporal
exchange) that is applied to A/B/label must be replayed onto the cached
teacher features so that the distilled change relation stays spatially and
temporally aligned with the student's inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F


def _crop_resize(x: torch.Tensor, top: int, left: int, src_h: int, src_w: int) -> torch.Tensor:
    """Crop [top:src_h-top, left:src_w-left] and resize back to the original shape."""
    H, W = x.shape[-2:]
    t = max(0, int(round(top * H / src_h)))
    l = max(0, int(round(left * W / src_w)))
    b = max(t + 1, H - t)
    r = max(l + 1, W - l)
    cropped = x[..., t:b, l:r]
    if cropped.shape[-2:] != (H, W):
        # Cache tensors are 3D [C, H, W]; F.interpolate needs a batch dim.
        if cropped.dim() == 3:
            cropped = cropped.unsqueeze(0)
            cropped = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
            cropped = cropped.squeeze(0)
        else:
            cropped = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
    return cropped


def apply_cache_state(cache: Dict[str, torch.Tensor], state: Dict, src_h: int = 256, src_w: int = 256) -> Dict[str, torch.Tensor]:
    """Replay crop / flip / temporal-exchange onto cached teacher tensors.

    ``cache`` keys are feature names (e.g. ``t1_mid``, ``t2_mid``, ``t1_deep``,
    ``t2_deep``, ``teacher_change_logit``). Only the t1/t2 features are swapped
    on exchange; the change logit is exchange-invariant.
    """
    out = {k: v for k, v in cache.items()}

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

    if state.get("exchange"):
        for suffix in ("mid", "deep"):
            a = f"t1_{suffix}"
            b = f"t2_{suffix}"
            if a in out and b in out:
                out[a], out[b] = out[b], out[a]

    return out


class TeacherCacheReader:
    """Read per-sample teacher cache files produced by generate_teacher_cache."""

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
