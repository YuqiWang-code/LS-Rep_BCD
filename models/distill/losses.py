"""Run3 BT-SAM-RDT distillation-loss namespace.

Run3 intentionally contains no standalone trainable teacher-loss heads here.

Historical implementations removed from this module include:
    - TeacherHeads
    - SAM boundary/relation regression heads
    - OV response/relation regression heads
    - spatial Gram distillation
    - legacy SAM/OV auxiliary loss composition

Those mechanisms belong to earlier experiments and are not part of the
current Run3 BT-SAM-RDT method.

Current Run3 supervision is defined in:
    models/distill/task_space.py
        - BT-SAM structural prior
        - OV semantic prior
        - foundation-prior fusion
        - GT positive-Brier-gain audit
        - Bernoulli task-space KD

    models/distill/dynamic_teacher.py
        - one Fast Residual Teacher
        - one EMA Target Teacher
        - Fast Teacher fitting loss
        - Student KD objective

Keeping this module intentionally empty makes accidental imports of removed
legacy APIs fail immediately instead of silently reintroducing obsolete
training behavior.
"""

from __future__ import annotations


__all__: list[str] = []
