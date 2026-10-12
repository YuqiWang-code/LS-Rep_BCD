#!/usr/bin/env python3
"""Compact Teacher Cache v3: multi-channel symmetric pair evidence for CTP-Critic.

Third-round schema (report §5.1/§5.3). Where v2 stored only a *fixed random
projection* of the difference, v3 keeps the two exchange-symmetric fields the
privileged critic actually consumes:

    E = |LN(F_A) - LN(F_B)|          change evidence        (exchange-symmetric)
    M = 0.5 * (LN(F_A) + LN(F_B))    symmetric scene context (exchange-symmetric)

with ``LN`` the channel-wise L2 normalization already used in v2. **No projection
and no max-channel collapse**: the training-time translator ``P`` sees the full
native channel structure.

Why both fields are symmetric matters for the replay contract (report P0-A):

* because ``E`` and ``M`` are invariant under ``A <-> B``, geometric replay needs
  only crop/flip -- there is no temporal swap to apply;
* if a future variant stores raw ``F_A``/``F_B`` instead, the swap MUST exchange
  them. :func:`apply_cache_state_v3` therefore refuses to run on a cache that
  advertises raw temporal taps, rather than silently producing misaligned data.

Storage is deliberately **incremental**: a (teacher, dataset) pair is generated
only when an experiment needs it. Full-native-grid cost varies hugely between
teachers (14x14x768 RADIO ~ 20 GiB for four datasets, but 64x64x768 UniverSat
would be ~411 GiB), so the tool never assumes the whole pool is materialized.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from .cache_v2 import _crop_resize, _is_spatial

CACHE_SCHEMA_VERSION_V3 = 3
# Taps that are already exchange-symmetric; replay needs crop/flip only.
SYMMETRIC_TAPS = ("evidence", "context")
# Taps that would require an explicit A<->B exchange during replay.
RAW_TEMPORAL_TAPS = ("feature_a", "feature_b")


def symmetric_evidence(fa: torch.Tensor, fb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(E, M)`` from a pair of dense teacher feature maps.

    ``fa``/``fb``: [B,C,h,w] (or [C,h,w]). Normalization matches v2 exactly
    (channel-wise L2) so the change evidence is numerically comparable.
    """
    squeeze = fa.dim() == 3
    if squeeze:
        fa, fb = fa[None], fb[None]
    na = F.normalize(fa.float(), dim=1)
    nb = F.normalize(fb.float(), dim=1)
    evidence = (na - nb).abs()
    context = 0.5 * (na + nb)
    if squeeze:
        return evidence[0], context[0]
    return evidence, context


@dataclass
class CacheV3Meta:
    """Manifest payload for one (teacher, dataset, split) triple."""

    teacher_id: str
    dataset: str
    split: str
    weight_file: str
    weight_hash: str
    color_order: str
    source_size: int
    native_h: int
    native_w: int
    channels: int
    taps: tuple[str, ...] = SYMMETRIC_TAPS
    normalize: str = "channel_l2"
    dtype: str = "float16"
    n_samples: int = 0
    split_fingerprint: str = ""
    augmentation_contract: str = "crop_resize+flip_v+flip_h replayed; E/M exchange-symmetric"
    schema_version: int = CACHE_SCHEMA_VERSION_V3
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["taps"] = list(self.taps)
        return json.dumps(d, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> "CacheV3Meta":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(d.get("schema_version", -1)) != CACHE_SCHEMA_VERSION_V3:
            raise ValueError(
                f"{path}: schema_version={d.get('schema_version')} != {CACHE_SCHEMA_VERSION_V3}")
        d["taps"] = tuple(d.get("taps", SYMMETRIC_TAPS))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known}
        d = {k: v for k, v in d.items() if k in known}
        m = cls(**d)
        m.extra = extra
        return m


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_cache_state_v3(cache: Dict[str, torch.Tensor], state: Dict,
                         src_h: int = 256, src_w: int = 256) -> Dict[str, torch.Tensor]:
    """Replay crop/flip onto the symmetric v3 fields.

    Raises if the cache carries raw temporal taps, because those would need the
    ``A <-> B`` exchange applied in lock-step and silently skipping it would
    produce spatially correct but semantically swapped supervision.
    """
    present_raw = [k for k in RAW_TEMPORAL_TAPS if k in cache]
    if present_raw:
        raise ValueError(
            f"cache carries raw temporal taps {present_raw}; exchange them explicitly "
            f"during replay instead of using the symmetric v3 path")
    out = {k: v for k, v in cache.items() if isinstance(v, torch.Tensor)}
    crop = state.get("crop_resize")
    if crop and crop.get("enabled"):
        top, left = crop["top"], crop["left"]
        for key in list(out.keys()):
            if _is_spatial(out[key]):
                out[key] = _crop_resize(out[key], top, left, src_h, src_w)
    if state.get("flip_v"):
        for key in list(out.keys()):
            if _is_spatial(out[key]):
                out[key] = torch.flip(out[key], dims=[-2])
    if state.get("flip_h"):
        for key in list(out.keys()):
            if _is_spatial(out[key]):
                out[key] = torch.flip(out[key], dims=[-1])
    return out


class TeacherCacheV3Reader:
    """Read a v3 cache tree, with guards that the report's P0 audit asks for."""

    #: Training must never consume val/test teacher cache.
    TRAIN_ONLY_SPLITS = ("train",)

    def __init__(self, cache_root: str | Path, dataset_name: str, teacher_id: str):
        self.cache_root = Path(cache_root)
        self.dataset_name = str(dataset_name).upper()
        self.teacher_id = str(teacher_id)
        self._meta: CacheV3Meta | None = None

    @property
    def dir(self) -> Path:
        return self.cache_root / self.teacher_id / self.dataset_name / "train"

    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def load_meta(self) -> CacheV3Meta:
        if self._meta is None:
            p = self.manifest_path()
            if not p.is_file():
                raise FileNotFoundError(f"v3 manifest missing: {p}")
            self._meta = CacheV3Meta.from_json(p)
        return self._meta

    def _path(self, split: str, sample_id: str) -> Path:
        return self.cache_root / self.teacher_id / self.dataset_name / split / f"{sample_id}.pt"

    def load(self, split: str, sample_id: str) -> Dict[str, torch.Tensor]:
        if split not in self.TRAIN_ONLY_SPLITS:
            raise PermissionError(
                f"refusing to read teacher cache for split={split!r}; training may only "
                f"consume {self.TRAIN_ONLY_SPLITS}")
        path = self._path(split, sample_id)
        if not path.is_file():
            raise FileNotFoundError(f"Teacher v3 cache missing: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        return {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}

    def verify(self, sample_ids: list[str]) -> dict:
        """Check that every requested sample exists and shapes match the manifest."""
        meta = self.load_meta()
        missing = [s for s in sample_ids if not self._path("train", s).is_file()]
        report = {"teacher": self.teacher_id, "dataset": self.dataset_name,
                  "n_requested": len(sample_ids), "n_missing": len(missing),
                  "missing_examples": missing[:5]}
        if sample_ids:
            probe = self.load("train", sample_ids[0])
            report["tensors"] = {k: list(v.shape) for k, v in probe.items()}
            ok = all(k in probe for k in SYMMETRIC_TAPS)
            report["has_symmetric_taps"] = ok
            ch = probe[SYMMETRIC_TAPS[0]].shape[0]
            hw = tuple(probe[SYMMETRIC_TAPS[0]].shape[-2:])
            report["manifest_channels"] = meta.channels
            report["manifest_native_hw"] = [meta.native_h, meta.native_w]
            report["channels_match"] = (ch == meta.channels)
            report["native_hw_match"] = (hw == (meta.native_h, meta.native_w))
        return report


def save_sample_v3(out_dir: Path, sample_id: str, evidence: torch.Tensor,
                   context: torch.Tensor) -> None:
    """Atomically write one v3 cache sample (fp16)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"evidence": evidence.half().contiguous(),
               "context": context.half().contiguous()}
    path = out_dir / f"{sample_id}.pt"
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


__all__ = [
    "CACHE_SCHEMA_VERSION_V3",
    "SYMMETRIC_TAPS",
    "RAW_TEMPORAL_TAPS",
    "CacheV3Meta",
    "TeacherCacheV3Reader",
    "apply_cache_state_v3",
    "file_sha256",
    "save_sample_v3",
    "symmetric_evidence",
]
