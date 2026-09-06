"""Replay student geometry on cached teacher spatial maps."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def _replay_tensor(value: torch.Tensor, state: Dict[str, Any], key: str = None) -> torch.Tensor:
    """
    Replay geometric transformations on a cached teacher tensor.

    Args:
        value: Cached tensor [C, H, W] or [H, W]
        state: Transform state dict
        key: Optional key name to determine interpolation mode

    Returns:
        Transformed tensor
    """
    original_dtype = value.dtype
    squeeze_batch = value.ndim == 3

    if value.ndim == 2:
        value = value.unsqueeze(0)
        squeeze_batch = True
    if value.ndim != 3:
        raise ValueError(f"Teacher map must be [C,H,W] or [H,W], got {tuple(value.shape)}")

    # Determine interpolation mode
    # instance_id must use nearest to preserve integer IDs
    nearest = (key == "instance_id")
    mode = "nearest" if nearest else "bilinear"

    x = value.float().unsqueeze(0)

    scale = state.get("scale", {})
    if scale:
        target_size = (int(scale["height"]), int(scale["width"]))
        if x.shape[-2:] != target_size:
            if mode == "nearest":
                x = F.interpolate(x, size=target_size, mode="nearest")
            else:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    crop = state.get("crop_resize", {})
    if crop.get("enabled", False):
        h, w = x.shape[-2:]
        source_h = int(crop["source_height"])
        source_w = int(crop["source_width"])
        top = int(round(int(crop["top"]) * h / source_h))
        left = int(round(int(crop["left"]) * w / source_w))
        x = x[..., top:h - top, left:w - left]

        if mode == "nearest":
            x = F.interpolate(x, size=(h, w), mode="nearest")
        else:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    if state.get("flip_v", False):
        x = torch.flip(x, dims=(-2,))
    if state.get("flip_h", False):
        x = torch.flip(x, dims=(-1,))

    x = x.squeeze(0)

    # Round instance_id back to integers
    if nearest:
        x = x.round().to(original_dtype)
    else:
        x = x.to(original_dtype)

    return x if not (squeeze_batch and x.shape[0] == 1) else x


def replay_teacher_pack(pack: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply geometry recursively to teacher pack; metadata is intentionally not returned.

    Handles T1/T2 exchange for RandomExchange transform.

    Args:
        pack: Teacher cache pack (may contain 't1', 't2' nested dicts)
        state: Transform state dict

    Returns:
        Transformed teacher pack
    """
    result: Dict[str, Any] = {}

    for key, value in pack.items():
        if isinstance(value, dict):
            result[key] = replay_teacher_pack(value, state)
        elif torch.is_tensor(value):
            result[key] = _replay_tensor(value, state, key)

    # Handle RandomExchange: swap t1 and t2
    if state.get("exchange", False) and "t1" in result and "t2" in result:
        result["t1"], result["t2"] = result["t2"], result["t1"]

    return result
