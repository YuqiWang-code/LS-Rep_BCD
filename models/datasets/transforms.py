"""Paired geometric and photometric transforms for binary change detection.

The replayable state is deliberately limited to geometry and temporal exchange.
It can therefore be applied to cached teacher maps without applying the student's
RGB normalization to teacher data.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable

import cv2
import numpy as np
import torch


AugState = Dict[str, Any]


class Scale:
    def __init__(self, wi: int, he: int):
        self.w = wi
        self.h = he

    def __call__(self, img, label, state=None):
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if state is not None:
            state["scale"] = {"height": self.h, "width": self.w}
        return img, label


class RandomCropResize:
    """Symmetrically crop random margins, then resize to the original size."""

    def __init__(self, crop_area: int):
        self.max_margin = crop_area

    def __call__(self, img, label, state=None):
        enabled = random.random() < 0.5
        top = random.randint(0, self.max_margin) if enabled else 0
        left = random.randint(0, self.max_margin) if enabled else 0
        h, w = img.shape[:2]
        if top * 2 >= h or left * 2 >= w:
            raise ValueError(f"Crop margin ({top}, {left}) is invalid for {(h, w)}")
        if enabled:
            img = cv2.resize(
                img[top:h - top, left:w - left], (w, h), interpolation=cv2.INTER_LINEAR
            )
            label = cv2.resize(
                label[top:h - top, left:w - left], (w, h), interpolation=cv2.INTER_NEAREST
            )
        if state is not None:
            state["crop_resize"] = {
                "enabled": enabled,
                "top": top,
                "left": left,
                "source_height": h,
                "source_width": w,
            }
        return img, label


class RandomFlip:
    def __call__(self, image, label, state=None):
        flip_v = random.random() < 0.5
        flip_h = random.random() < 0.5
        if flip_v:
            image = cv2.flip(image, 0)  # vertical (top-bottom)
            label = cv2.flip(label, 0)
        if flip_h:
            image = cv2.flip(image, 1)  # horizontal (left-right)
            label = cv2.flip(label, 1)
        if state is not None:
            state["flip_v"] = flip_v
            state["flip_h"] = flip_h
        return image, label


class RandomExchange:
    def __call__(self, image, label, state=None):
        exchange = random.random() < 0.5
        if exchange:
            image = np.concatenate((image[:, :, 3:6], image[:, :, 0:3]), axis=2)
        if state is not None:
            state["exchange"] = exchange
        return image, label


class Normalize:
    def __init__(self, mean: Iterable[float], std: Iterable[float]):
        self.mean = np.asarray(tuple(mean), dtype=np.float32).reshape(1, 1, 6)
        self.std = np.asarray(tuple(std), dtype=np.float32).reshape(1, 1, 6)

    def __call__(self, image, label, state=None):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        # CDD labels are JPEG and contain compression-induced intermediate
        # grayscale values. A shared >=128 rule is valid for all four datasets.
        label = (label >= 128).astype(np.float32)
        return image, label


class ToTensor:
    def __init__(self, scale: int = 1):
        self.scale = scale

    def __call__(self, image, label, state=None):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(
                label, (w // self.scale, h // self.scale), interpolation=cv2.INTER_NEAREST
            )

        # OpenCV gives [B1,G1,R1,B2,G2,R2]. Convert each temporal RGB
        # triplet independently. Reversing all six channels would also swap T1/T2.
        t1 = image[:, :, 0:3][:, :, ::-1]
        t2 = image[:, :, 3:6][:, :, ::-1]
        image = np.concatenate((t1, t2), axis=2).copy().transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(np.asarray(label, dtype=np.float32)).unsqueeze(0)
        return image_tensor, label_tensor


class Compose:
    """Compose transforms while optionally returning replayable augmentation state."""

    def __init__(self, transforms, return_state: bool = False):
        self.transforms = transforms
        self.return_state = return_state

    def __call__(self, img, label):
        state: AugState | None = {} if self.return_state else None
        for transform in self.transforms:
            img, label = transform(img, label, state=state)
        if state is None:
            return img, label
        return img, label, state
