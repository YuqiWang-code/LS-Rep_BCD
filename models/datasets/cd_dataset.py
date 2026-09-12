"""Datasets and dataloaders for paired binary change detection."""

from __future__ import annotations

import os
import random
from pathlib import Path
import cv2
import numpy as np
import torch.utils.data

from .cache_transforms import replay_teacher_pack
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
    def __init__(
        self,
        dataset: str,
        file_root: str,
        transform=None,
        dataset_name: str = "LEVIR",
        return_meta: bool = False,
        teacher_cache=None,
        seed: int = 2333,
    ):
        self.split = dataset
        self.root = Path(file_root)
        list_path = self.root / "list" / f"{dataset}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"Dataset list not found: {list_path}")
        self.file_list = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.transform = transform
        self.dataset_name = dataset_name
        self.return_meta = return_meta or teacher_cache is not None
        self.teacher_cache = teacher_cache
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.file_list)

    def _resolve_path(self, folder: str, entry: str) -> Path:
        relative = Path(entry)
        candidates = [self.root / folder / relative]
        if self.dataset_name == "CDD" and not relative.name.startswith("test_"):
            candidates.append(self.root / folder / relative.with_name(f"test_{relative.name}"))
        for base in tuple(candidates):
            if base.suffix.lower() == ".png":
                candidates.append(base.with_suffix(".jpg"))
            elif base.suffix.lower() == ".jpg":
                candidates.append(base.with_suffix(".png"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        tried = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Unable to resolve {folder} file for '{entry}'. Tried:\n  - {tried}")

    @staticmethod
    def _read(path: Path, flags: int, kind: str):
        value = cv2.imread(str(path), flags)
        if value is None:
            raise OSError(f"OpenCV failed to read {kind}: {path}")
        return value

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        # Per-sample augmentation makes epoch-boundary resume independent of
        # worker scheduling and auxiliary RNG consumption.
        state = random.getstate()
        random.seed(self.seed + self.epoch * 1_000_003 + idx)
        try:
            return self._get_sample(idx)
        finally:
            random.setstate(state)

    def _get_sample(self, idx):
        sample_id = self.file_list[idx]
        pre_path = self._resolve_path("A", sample_id)
        post_path = self._resolve_path("B", sample_id)
        label_path = self._resolve_path("label", sample_id)

        pre = self._read(pre_path, cv2.IMREAD_COLOR, "T1 image")
        post = self._read(post_path, cv2.IMREAD_COLOR, "T2 image")
        label = self._read(label_path, cv2.IMREAD_GRAYSCALE, "label")
        if pre.shape != post.shape or pre.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Shape mismatch for '{sample_id}': T1={pre.shape}, T2={post.shape}, label={label.shape}"
            )
        image = np.concatenate((pre, post), axis=2)

        aug_state = {}
        if self.transform:
            transformed = self.transform(image, label)
            if len(transformed) == 3:
                image, label, aug_state = transformed
            else:
                image, label = transformed

        if self.teacher_cache is not None:
            pack = self.teacher_cache.load(sample_id)
            pack = replay_teacher_pack(pack, aug_state)
            return image, label, sample_id, pack
        if self.return_meta:
            return image, label, sample_id
        return image, label

    def get_img_info(self, idx):
        path = self._resolve_path("A", self.file_list[idx])
        image = self._read(path, cv2.IMREAD_COLOR, "T1 image")
        return {"height": image.shape[0], "width": image.shape[1]}


def _dataset_name(file_root: str) -> str:
    upper = file_root.upper()
    for name in ("SYSU", "WHU", "CDD"):
        if name in upper:
            return name
    return "LEVIR"


def get_loader(
    file_root,
    list_file,
    img_ext=".png",
    file_prefix="",
    batchsize=32,
    trainsize=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    return_meta=False,
    teacher_cache=None,
    seed=2333,
):
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    transform = Compose(
        [
            Normalize(mean=mean, std=std),
            Scale(trainsize, trainsize),
            RandomCropResize(int(7.0 / 224.0 * trainsize)),
            RandomFlip(),
            RandomExchange(),
            ToTensor(),
        ],
        return_state=teacher_cache is not None,
    )
    split = "train" if "train" in os.path.basename(list_file) else "val"
    dataset = CDDataset(
        split,
        file_root=file_root,
        transform=transform,
        dataset_name=_dataset_name(file_root),
        return_meta=return_meta,
        teacher_cache=teacher_cache,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=split == "train",
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=False,
    )


def get_test_loader(
    file_root,
    list_file,
    img_ext=".png",
    file_prefix="",
    batchsize=32,
    testsize=256,
    num_workers=4,
    pin_memory=True,
    return_meta=False,
):
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    transform = Compose([Normalize(mean=mean, std=std), Scale(testsize, testsize), ToTensor()])
    split = "val" if "val" in os.path.basename(list_file) else "test"
    dataset = CDDataset(
        split,
        file_root=file_root,
        transform=transform,
        dataset_name=_dataset_name(file_root),
        return_meta=return_meta,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
