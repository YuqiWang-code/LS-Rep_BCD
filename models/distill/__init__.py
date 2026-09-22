"""FA-SCRD distillation modules (training-only, removed at deploy time).

Contains the fixed DINOv3 teacher, the symmetric change-relation KD, the
failure-aware soft router, and the teacher-cache dataset.
"""

from .fa_scrd_teacher import FASCRDTeacher
from .failure_router import compute_failure_weight
from .relation_kd import symmetric_relation_kd

__all__ = [
    "FASCRDTeacher",
    "compute_failure_weight",
    "symmetric_relation_kd",
]
