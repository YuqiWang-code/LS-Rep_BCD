"""Build permutation-invariant structural fields after augmentation replay."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


DERIVED_KEYS = (
    "occupancy", "affinity_r1", "affinity_r2", "affinity_r4",
    "interior_depth", "object_scale", "compactness",
)


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    """OpenCV L2 distance with a dependency-free erosion fallback for tests."""
    try:
        import cv2
        return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    except ImportError:
        current = mask.astype(bool)
        distance = np.zeros(mask.shape, dtype=np.float32)
        while current.any():
            distance[current] += 1.0
            padded = np.pad(current, 1, constant_values=False)
            current = (
                padded[1:-1, 1:-1]
                & padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:]
            )
        return distance


def _to_numpy_2d(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected [1,H,W] or [H,W], got {tuple(value.shape)}")
    return np.ascontiguousarray(array)


def _same_instance_affinity(instance: np.ndarray, radius: int) -> np.ndarray:
    """Average horizontal, vertical and diagonal same-instance relations."""
    height, width = instance.shape
    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for dy, dx in ((0, radius), (radius, 0), (radius, radius), (radius, -radius)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        if y1 <= y0 or x1 <= x0:
            continue
        source = instance[y0:y1, x0:x1]
        neighbor = instance[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        same = ((source == neighbor) & (source > 0)).astype(np.float32)
        total[y0:y1, x0:x1] += same
        count[y0:y1, x0:x1] += 1.0
    return total / np.maximum(count, 1.0)


def _derived_fields(structure: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    instance = _to_numpy_2d(structure["instance_id"]).astype(np.int32, copy=False)
    boundary = _to_numpy_2d(structure["boundary"]).astype(np.float32, copy=False)
    occupancy = (instance > 0).astype(np.float32)

    # One distance transform is sufficient because the cached boundary separates
    # instances.  Per-instance maxima then provide the requested normalization.
    traversable = ((instance > 0) & (boundary < 0.1)).astype(np.uint8)
    distance = _distance_transform(traversable)
    max_id = int(instance.max(initial=0))
    maxima = np.zeros(max_id + 1, dtype=np.float32)
    if max_id > 0:
        np.maximum.at(maxima, instance.reshape(-1), distance.reshape(-1))
    interior_depth = np.zeros_like(distance)
    foreground = instance > 0
    if foreground.any():
        interior_depth[foreground] = distance[foreground] / np.maximum(
            maxima[instance[foreground]], 1e-6
        )

    area = np.bincount(instance.reshape(-1), minlength=max_id + 1).astype(np.float32)
    scale_lut = np.log1p(area) / np.log1p(float(instance.size))
    scale_lut[0] = 0.0
    object_scale = scale_lut[instance]

    perimeter = np.bincount(
        instance.reshape(-1),
        weights=((boundary > 0.1) & foreground).reshape(-1).astype(np.float32),
        minlength=max_id + 1,
    ).astype(np.float32)
    compact_lut = np.zeros(max_id + 1, dtype=np.float32)
    if max_id > 0:
        compact_lut[1:] = np.clip(
            4.0 * np.pi * area[1:] / (perimeter[1:] ** 2 + 1e-6), 0.0, 1.0
        )
    compactness = compact_lut[instance]

    arrays = {
        "occupancy": occupancy,
        "affinity_r1": _same_instance_affinity(instance, 1),
        "affinity_r2": _same_instance_affinity(instance, 2),
        "affinity_r4": _same_instance_affinity(instance, 4),
        "interior_depth": interior_depth,
        "object_scale": object_scale,
        "compactness": compactness,
    }
    return {
        key: torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(torch.float16)
        for key, value in arrays.items()
    }


def prepare_relational_pack(pack: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Derive all PI-DTRS fields from an already replayed T1/T2 cache pack."""
    if not all(key in pack for key in ("t1", "t2")):
        raise KeyError("SAM cache pack must contain t1 and t2")
    result = {}
    for time_key in ("t1", "t2"):
        structure = dict(pack[time_key])
        for required in ("instance_id", "boundary", "quality"):
            if required not in structure:
                raise KeyError(f"SAM cache {time_key} is missing {required}")
        structure.update(_derived_fields(structure))
        result[time_key] = structure
    return result


class RelationalStructureBuilder(nn.Module):
    """Validate and expose replay-derived fields without duplicating the pack.

    Keeping instance IDs int32 and cached/derived maps float16 is important for
    the large-batch memory budget. Individual loss terms promote only the maps
    they currently consume to float32.
    """

    def forward(self, pack):
        result = {}
        for time_key in ("t1", "t2"):
            structure = pack[time_key]
            missing = [key for key in DERIVED_KEYS if key not in structure]
            if missing:
                raise KeyError(
                    f"Relational fields must be built after replay; {time_key} missing {missing}"
                )
            result[time_key] = {
                "instance_id": structure["instance_id"],
                "boundary": structure["boundary"],
                "quality": structure["quality"],
                **{key: structure[key] for key in DERIVED_KEYS},
            }
        return result
