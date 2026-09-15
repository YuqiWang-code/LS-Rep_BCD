"""Training-only teacher distillation utilities for RDT-CD + SCGR.

Public surface
--------------
- PairedTeacherCache:
    synchronized SAMStruct + OVCDistill cache access.
- DynamicTeacherDirectionC:
    current reciprocal dynamic-teacher auxiliary.
- SCGR task-space utilities:
    sparse-change, error-mass-aware, gradient-concordant routing.

Everything in this package is training-only.  None of these components is
required by the deployed A2Net-LWGANet-L0 graph.
"""

from .teacher_cache import PairedTeacherCache
from .task_space import (
    DEFAULT_REGION_SIZE,
    NUM_TEACHERS,
    OV_INDEX,
    REJECT_INDEX,
    SAM_INDEX,
    bernoulli_kl,
    build_error_mass,
    build_task_proposals,
    classifier_gradient_concordance,
    instance_transport,
    routed_bernoulli_kd,
    sparse_class_balance,
    task_space_route,
)
from .dynamic_teacher import DynamicTeacherDirectionC

__all__ = [
    "PairedTeacherCache",
    "DynamicTeacherDirectionC",
    "NUM_TEACHERS",
    "SAM_INDEX",
    "OV_INDEX",
    "REJECT_INDEX",
    "DEFAULT_REGION_SIZE",
    "instance_transport",
    "build_task_proposals",
    "bernoulli_kl",
    "sparse_class_balance",
    "build_error_mass",
    "classifier_gradient_concordance",
    "task_space_route",
    "routed_bernoulli_kd",
]
