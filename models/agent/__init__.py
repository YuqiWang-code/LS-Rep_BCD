"""CATA-CD selection Agent (Stage 3/4): dataset signature, registry, offline bandit."""

from .dataset_signature import compute_dataset_signature, SIGNATURE_NAMES
from .offline_bandit import TeacherAgentPolicy
from .teacher_registry import TeacherRegistry

__all__ = [
    "compute_dataset_signature",
    "SIGNATURE_NAMES",
    "TeacherAgentPolicy",
    "TeacherRegistry",
]
