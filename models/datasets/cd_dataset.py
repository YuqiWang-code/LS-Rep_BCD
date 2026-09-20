"""Datasets and dataloaders for paired binary change detection.

Clean SCTC data pipeline
------------------------
This module serves only the deployable A2Net-LWGANet-L0 + SCTC student.

It contains no:
    - Teacher Cache
    - SAM / OV data
    - cache replay
    - distillation metadata
    - training-only auxiliary input

Training samples are returned as:
    image, label

When ``return_meta=True``:
    image, label, sample_id

The paired T1/T2 image and binary label always receive identical geometric
augmentation. ``RandomExchange`` swaps T1/T2 jointly inside the six-channel
image and leaves the binary change label unchanged.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch.utils.data

from .transforms import (
    Compose,
    Normalize,
    RandomCropResize,
    RandomExchange,
    RandomFlip,
    Scale,
    ToTensor,
)


class CDDataset(torch.utils.data.Dataset):
    """Paired binary change-detection dataset."""

    def __init__(
        self,
        dataset: str,
        file_root: str,
        transform=None,
        dataset_name: str = "LEVIR",
        return_meta: bool = False,
        seed: int = 2333,
    ) -> None:
        self.split = str(dataset)
        self.root = Path(file_root)
        self.transform = transform
        self.dataset_name = str(dataset_name).upper()
        self.return_meta = bool(return_meta)
        self.seed = int(seed)
        self.epoch = 0

        list_path = self.root / "list" / f"{self.split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"Dataset list not found: {list_path}")

        self.file_list = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not self.file_list:
            raise ValueError(f"Dataset split is empty: {list_path}")

        if len(self.file_list) != len(set(self.file_list)):
            raise ValueError(f"Duplicate sample IDs found in: {list_path}")

    def __len__(self) -> int:
        return len(self.file_list)

    def _resolve_path(self, folder: str, entry: str) -> Path:
        relative = Path(entry)
        candidates = [self.root / folder / relative]

        if self.dataset_name == "CDD" and not relative.name.startswith("test_"):
            candidates.append(
                self.root / folder / relative.with_name(f"test_{relative.name}")
            )

        for base in tuple(candidates):
            suffix = base.suffix.lower()
            if suffix == ".png":
                candidates.append(base.with_suffix(".jpg"))
            elif suffix in {".jpg", ".jpeg"}:
                candidates.append(base.with_suffix(".png"))

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        tried = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Unable to resolve {folder} file for {entry!r}. Tried:\n  - {tried}"
        )

    @staticmethod
    def _read(path: Path, flags: int, kind: str):
        value = cv2.imread(str(path), flags)
        if value is None:
            raise OSError(f"OpenCV failed to read {kind}: {path}")
        return value

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx < 0 or idx >= len(self.file_list):
            raise IndexError(idx)

        state = random.getstate()
        random.seed(self.seed + self.epoch * 1_000_003 + idx)
        try:
            return self._get_sample(idx)
        finally:
            random.setstate(state)

    def _get_sample(self, idx: int):
        sample_id = self.file_list[idx]

        pre_path = self._resolve_path("A", sample_id)
        post_path = self._resolve_path("B", sample_id)
        label_path = self._resolve_path("label", sample_id)

        pre = self._read(pre_path, cv2.IMREAD_COLOR, "T1 image")
        post = self._read(post_path, cv2.IMREAD_COLOR, "T2 image")
        label = self._read(label_path, cv2.IMREAD_GRAYSCALE, "label")

        if pre.shape != post.shape:
            raise ValueError(
                f"T1/T2 shape mismatch for {sample_id!r}: "
                f"T1={pre.shape}, T2={post.shape}"
            )

        if pre.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Image/label shape mismatch for {sample_id!r}: "
                f"image={pre.shape[:2]}, label={label.shape[:2]}"
            )

        image = np.concatenate((pre, post), axis=2)

        if self.transform is not None:
            transformed = self.transform(image, label)
            if not isinstance(transformed, (tuple, list)) or len(transformed) != 2:
                raise RuntimeError(
                    "Clean SCTC transform pipeline must return exactly (image, label)"
                )
            image, label = transformed

        if self.return_meta:
            return image, label, sample_id

        return image, label

    def get_img_info(self, idx: int):
        path = self._resolve_path("A", self.file_list[int(idx)])
        image = self._read(path, cv2.IMREAD_COLOR, "T1 image")
        return {
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        }


def _dataset_name(file_root: str) -> str:
    upper = str(file_root).upper()
    if "SYSU" in upper:
        return "SYSU"
    if "WHU" in upper:
        return "WHU"
    if "CDD" in upper:
        return "CDD"
    if "LEVIR" in upper:
        return "LEVIR"
    return "LEVIR"


def _normalization():
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    return mean, std


def get_loader(
    file_root,
    list_file,
    batchsize=32,
    trainsize=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    return_meta=False,
    seed=2333,
):
    """Build the clean SCTC training DataLoader."""
    mean, std = _normalization()

    transform = Compose(
        [
            Normalize(mean=mean, std=std),
            Scale(trainsize, trainsize),
            RandomCropResize(int(7.0 / 224.0 * trainsize)),
            RandomFlip(),
            RandomExchange(),
            ToTensor(),
        ],
        return_state=False,
    )

    split_name = os.path.basename(str(list_file)).lower()
    split = "train" if "train" in split_name else "val"

    dataset = CDDataset(
        split,
        file_root=file_root,
        transform=transform,
        dataset_name=_dataset_name(file_root),
        return_meta=return_meta,
        seed=seed,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batchsize),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=split == "train",
        generator=torch.Generator().manual_seed(int(seed)),
        persistent_workers=False,
    )


def get_test_loader(
    file_root,
    list_file,
    batchsize=32,
    testsize=256,
    num_workers=4,
    pin_memory=True,
    return_meta=False,
):
    """Build deterministic validation/test DataLoader."""
    mean, std = _normalization()

    transform = Compose(
        [
            Normalize(mean=mean, std=std),
            Scale(testsize, testsize),
            ToTensor(),
        ],
        return_state=False,
    )

    split_name = os.path.basename(str(list_file)).lower()
    split = "val" if "val" in split_name else "test"

    dataset = CDDataset(
        split,
        file_root=file_root,
        transform=transform,
        dataset_name=_dataset_name(file_root),
        return_meta=return_meta,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batchsize),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
    )


__all__ = [
    "CDDataset",
    "get_loader",
    "get_test_loader",
]
