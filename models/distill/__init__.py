"""CATA-CD distillation modules (training-only; never part of the deploy graph)."""

from .cache_v2 import CACHE_CHANNELS, CACHE_SCHEMA_VERSION, TeacherCacheReaderV2, apply_cache_state
from .kd import ChangeEvidenceHead, dense_change_kd, gradient_budget_lambda
from .teacher_package import (
    TEACHER_REGISTRY,
    WAVE_A,
    WAVE_B,
    DenseChangeTeacherPackage,
    build_teacher_package,
    normalize_rgb,
)

__all__ = [
    "CACHE_CHANNELS",
    "CACHE_SCHEMA_VERSION",
    "TeacherCacheReaderV2",
    "apply_cache_state",
    "ChangeEvidenceHead",
    "dense_change_kd",
    "gradient_budget_lambda",
    "TEACHER_REGISTRY",
    "WAVE_A",
    "WAVE_B",
    "DenseChangeTeacherPackage",
    "build_teacher_package",
    "normalize_rgb",
]
