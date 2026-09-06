"""Manifest-backed, read-only access to offline VFM teacher caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


CACHE_VERSION = 1


def load_manifest(cache_root: str | Path) -> Dict[str, Any]:
    path = Path(cache_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Teacher cache manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"Unsupported teacher cache version {manifest.get('cache_version')}; "
            f"expected {CACHE_VERSION}"
        )
    return manifest


class TeacherCache:
    def __init__(self, cache_root: str | Path, split: str = "train",
                 require_validation: bool = False):
        self.root = Path(cache_root)
        self.split = split
        self.manifest = load_manifest(self.root)
        self.entries = self.manifest.get("entries", {})
        if not isinstance(self.entries, dict):
            raise ValueError("Teacher cache manifest 'entries' must be a mapping")
        self.split_root = self.root / split
        if not self.split_root.is_dir():
            raise FileNotFoundError(f"Teacher cache split directory not found: {self.split_root}")
        self.validation_summary = None
        if require_validation:
            summary_path = self.root / "validation_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"Validated teacher cache summary not found: {summary_path}"
                )
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            if summary.get("config_hash") != self.manifest.get("config_hash"):
                raise ValueError(
                    "Teacher cache validation_summary config_hash does not match manifest"
                )
            if summary.get("dataset") != self.manifest.get("source_dataset"):
                raise ValueError(
                    "Teacher cache validation_summary dataset does not match manifest"
                )
            if int(summary.get("number_of_samples", -1)) != len(self.entries):
                raise ValueError(
                    "Teacher cache validation_summary sample count does not match manifest"
                )
            self.validation_summary = summary

    def __len__(self) -> int:
        return len(self.entries)

    def path_for(self, sample_id: str) -> Path:
        if sample_id not in self.entries:
            raise KeyError(f"Sample '{sample_id}' is absent from the teacher cache manifest")
        relative = self.entries[sample_id]
        return self.split_root / relative

    def load(self, sample_id: str) -> Dict[str, Any]:
        path = self.path_for(sample_id)
        if not path.is_file():
            raise FileNotFoundError(f"Teacher cache missing for '{sample_id}': {path}")
        pack = torch.load(path, map_location="cpu", weights_only=False)
        if pack.get("sample_id") != sample_id:
            raise ValueError(
                f"Teacher cache sample mismatch: requested '{sample_id}', "
                f"file contains '{pack.get('sample_id')}'"
            )
        if pack.get("meta", {}).get("config_hash") != self.manifest.get("config_hash"):
            raise ValueError(f"Teacher cache metadata hash mismatch: {path}")
        tensors = {key: value for key, value in pack.items() if key not in {"sample_id", "meta"}}
        if not tensors:
            raise ValueError(f"Teacher cache contains no tensors: {path}")
        return tensors
