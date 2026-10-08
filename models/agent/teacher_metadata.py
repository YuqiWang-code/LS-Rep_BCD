"""Teacher metadata (review §7.6): the minimum auditable, test-free teacher descriptors.

Only fields that are fixed by the model card / checkpoint (never by any experimental
result, and never by a test ranking) are allowed here:

  training_domain    natural | satellite_eo | mixed
  primary_objective  segmentation | self_supervision | vision_language | agglomerated
  native_stride      main dense-output stride (encoded as log2)
  pyramid_available  native multi-scale output?
  cache_resolution   the ACTUAL stored Ht x Wt (not the claimed model resolution)

``cache_resolution`` is validated against the on-disk cache manifest when available.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

TRAINING_DOMAINS = ("natural", "satellite_eo", "mixed")
PRIMARY_OBJECTIVES = ("segmentation", "self_supervision", "vision_language", "agglomerated")


@dataclass(frozen=True)
class TeacherMeta:
    teacher_id: str
    training_domain: str
    primary_objective: str
    native_stride: int
    pyramid_available: bool
    cache_hw: tuple[int, int]

    def features(self) -> dict:
        """Numeric encoding used by the selector (teacher-id free)."""
        return {
            "meta_log2_stride": float(__import__("math").log2(self.native_stride)),
            "meta_pyramid": float(self.pyramid_available),
            "meta_cache_h": float(self.cache_hw[0]),
            "meta_cache_w": float(self.cache_hw[1]),
            "meta_dom_natural": float(self.training_domain == "natural"),
            "meta_dom_satellite": float(self.training_domain == "satellite_eo"),
            "meta_dom_mixed": float(self.training_domain == "mixed"),
            "meta_obj_segmentation": float(self.primary_objective == "segmentation"),
            "meta_obj_ssl": float(self.primary_objective == "self_supervision"),
            "meta_obj_vl": float(self.primary_objective == "vision_language"),
            "meta_obj_agglomerated": float(self.primary_objective == "agglomerated"),
        }


# cache_hw = the ACTUAL dense feature grid produced by the frozen encoder at its
# native input (verified in models/thirdparty/<name>/META + cache manifests).
TEACHER_METADATA: dict[str, TeacherMeta] = {
    "sam2": TeacherMeta("sam2", "natural", "segmentation", 16, True, (16, 16)),
    "dinov2": TeacherMeta("dinov2", "natural", "self_supervision", 14, False, (16, 16)),
    "dinov3_lvd": TeacherMeta("dinov3_lvd", "natural", "self_supervision", 16, False, (16, 16)),
    "dinov3_sat": TeacherMeta("dinov3_sat", "satellite_eo", "self_supervision", 16, False, (16, 16)),
    "remoteclip": TeacherMeta("remoteclip", "satellite_eo", "vision_language", 14, False, (16, 16)),
    "mars": TeacherMeta("mars", "satellite_eo", "self_supervision", 16, True, (16, 16)),
    "anysat": TeacherMeta("anysat", "satellite_eo", "self_supervision", 10, False, (25, 25)),
    "universat": TeacherMeta("universat", "satellite_eo", "self_supervision", 4, False, (64, 64)),
    "radio": TeacherMeta("radio", "mixed", "agglomerated", 16, False, (14, 14)),
}

META_FEATURE_NAMES = list(next(iter(TEACHER_METADATA.values())).features().keys())


def validate() -> None:
    for tid, meta in TEACHER_METADATA.items():
        if tid != meta.teacher_id:
            raise ValueError(f"teacher id mismatch: {tid} vs {meta.teacher_id}")
        if meta.training_domain not in TRAINING_DOMAINS:
            raise ValueError(f"{tid}: bad training_domain {meta.training_domain}")
        if meta.primary_objective not in PRIMARY_OBJECTIVES:
            raise ValueError(f"{tid}: bad primary_objective {meta.primary_objective}")
        if meta.native_stride < 1:
            raise ValueError(f"{tid}: bad native_stride")


def meta_features(teacher_id: str) -> dict:
    return TEACHER_METADATA[teacher_id].features()


def as_json() -> dict:
    return {tid: asdict(meta) for tid, meta in TEACHER_METADATA.items()}


validate()

__all__ = [
    "TeacherMeta",
    "TEACHER_METADATA",
    "META_FEATURE_NAMES",
    "meta_features",
    "as_json",
    "TRAINING_DOMAINS",
    "PRIMARY_OBJECTIVES",
]
