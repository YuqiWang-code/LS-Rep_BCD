"""
RDT-CD: Reciprocal Dynamic Teachers for Remote-Sensing Change Detection.

This module implements a TRAINING-ONLY reciprocal dynamic-teacher mechanism.

Core idea
---------
Existing DART-R-TS uses cache-derived task-space proposals whose knowledge
source is effectively static. Once the student becomes better than those
proposals, GT-audited positive-gain routing correctly rejects them and their
effective contribution collapses.

RDT-CD therefore does NOT treat SAMStruct / OVCDistill cache values as final
teacher targets. Instead, the cache values are used as teacher priors, while
small online residual teacher experts evolve during training.

The reciprocal loop is:

    fixed cache priors
           |
           v
    dynamic teacher experts <---- current detached student state
           |
           v
    dynamic proposals
           |
           v
    GT positive-gain audit
           |
           v
    teacher -> student distillation

Meanwhile:

    current student errors
           |
           v
    train fast teacher experts
           |
           v
    EMA update
           |
           v
    next-step target teachers

Important gradient contract
---------------------------
1. Student distillation loss:
       updates STUDENT only.
       Dynamic teacher proposals are produced by no-grad EMA teachers.

2. Teacher fitting loss:
       updates FAST TEACHERS only.
       Student feature / prediction are explicitly detached.

3. EMA target teachers:
       requires_grad=False and are updated explicitly by update_ema().

4. This module must never be inserted into the deploy feature path.
   A2Net.switch_to_deploy() should delete the entire training auxiliary.

5. No current-batch GT is used to construct the proposal seen by the student.
   GT is used only for:
       - positive-gain auditing;
       - teacher fitting for future teacher states;
       - difficulty/error weighting.

The existing task-space SAM/OV proposals remain useful, but only as priors:
    SAMStruct    -> structural seed
    OVCDistill   -> semantic soft-change seed

This file is intentionally self-contained on top of the existing:
    diagnostics.py
    task_space.py
"""

from __future__ import annotations

import copy
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diagnostics import build_cd_difficulty, masked_mean
from .task_space import (
    bernoulli_kl,
    build_task_proposals,
    task_space_route,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resize(
    x: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """FP32 bilinear resize used only inside the training auxiliary."""
    if x.shape[-2:] == tuple(size):
        return x.float()

    return F.interpolate(
        x.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def _safe_logit(
    p: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Stable logit for the probability-output A2Net student."""
    return torch.logit(
        p.float().clamp(eps, 1.0 - eps)
    )


# ---------------------------------------------------------------------------
# Online residual teacher
# ---------------------------------------------------------------------------

class ResidualTeacherExpert(nn.Module):
    """
    Small TRAINING-ONLY residual teacher expert.

    It does not predict an absolute change mask from scratch.

    Instead, given:
        - current detached decoder feature;
        - current detached student probability;
        - cache-derived teacher residual (seed - student);
        - cache reliability;

    it predicts a bounded correction in student-logit space.

    Input channels:
        student feature:     C
        student probability: 1
        cache residual:      1
        cache reliability:   1

    Total:
        C + 3

    The final layer is zero initialized so that at initialization:

        dynamic_teacher == current_student

    This is deliberate. The new teacher must earn useful corrections through
    online learning rather than injecting an arbitrary randomly initialized
    target into the student.
    """

    def __init__(
        self,
        channels: int = 64,
        hidden: int = 24,
    ):
        super().__init__()

        channels = int(channels)
        hidden = int(hidden)

        if channels <= 0:
            raise ValueError("channels must be positive")
        if hidden <= 0:
            raise ValueError("hidden must be positive")

        self.channels = channels
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Conv2d(
                channels + 3,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                1,
                kernel_size=1,
                bias=True,
            ),
        )

        # Critical initialization:
        # q_teacher == p_student at step 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        feature: torch.Tensor,
        probability: torch.Tensor,
        seed: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return low-resolution logit correction.

        All student/cache inputs are detached here as a second safety barrier,
        even if the caller already detached them.
        """
        if feature.ndim != 4:
            raise ValueError(
                "feature must be a [B,C,H,W] tensor"
            )

        if feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected feature with {self.channels} channels, "
                f"got {feature.shape[1]}"
            )

        if probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError(
                "probability must be [B,1,H,W]"
            )

        if seed.ndim != 4 or seed.shape[1] != 1:
            raise ValueError(
                "seed must be [B,1,H,W]"
            )

        if reliability.ndim != 4 or reliability.shape[1] != 1:
            raise ValueError(
                "reliability must be [B,1,H,W]"
            )

        if not (
            feature.shape[0]
            == probability.shape[0]
            == seed.shape[0]
            == reliability.shape[0]
        ):
            raise ValueError(
                "feature/probability/seed/reliability batch sizes must match"
            )

        spatial_size = feature.shape[-2:]

        f = feature.detach().float()

        p = _resize(
            probability.detach(),
            spatial_size,
        ).clamp(0.0, 1.0)

        s = _resize(
            seed.detach(),
            spatial_size,
        ).clamp(0.0, 1.0)

        r = _resize(
            reliability.detach(),
            spatial_size,
        ).clamp(0.0, 1.0)

        # Cache information is represented explicitly as what the static
        # teacher thinks differently from the current student.
        cache_residual = s - p

        x = torch.cat(
            (
                f,
                p,
                cache_residual,
                r,
            ),
            dim=1,
        )

        return self.net(x)


# ---------------------------------------------------------------------------
# Reciprocal dynamic-teacher mechanism
# ---------------------------------------------------------------------------

class ReciprocalDynamicTeacher(nn.Module):
    """
    RDT-CD training auxiliary.

    Teacher index:
        0 -> SAM dynamic teacher
        1 -> OV dynamic teacher

    Two versions of each teacher are maintained:

        fast[k]
            Optimized directly by teacher_total.

        target[k]
            EMA copy of fast[k].
            Used to construct the proposal distilled into the student.

    This separation prevents the current GT-supervised teacher fitting update
    from being immediately exposed to the same current student step.

    Expected external optimization
    ------------------------------
    Student optimizer:
        main_loss + lambda * output["total"]

    Teacher optimizer:
        output["teacher_total"]

    After teacher_optimizer.step():
        module.update_ema()

    Do NOT put parameters returned by teacher_parameters() into the student
    optimizer.
    """

    NUM_TEACHERS = 2
    SAM_INDEX = 0
    OV_INDEX = 1

    def __init__(
        self,
        channels: int = 64,
        hidden: int = 24,
        ema: float = 0.99,
        max_logit_delta: float = 2.0,
        boundary_radius: int = 2,
        small_area: int = 64,
        policy: str = "advantage",
        teacher: str = "both",
        difficulty: bool = True,
        cache_conditioning: bool = True,
    ):
        super().__init__()

        channels = int(channels)
        hidden = int(hidden)
        ema = float(ema)
        max_logit_delta = float(max_logit_delta)

        if channels <= 0:
            raise ValueError("channels must be positive")

        if hidden <= 0:
            raise ValueError("hidden must be positive")

        if not 0.0 <= ema < 1.0:
            raise ValueError(
                "ema must satisfy 0 <= ema < 1"
            )

        if max_logit_delta <= 0:
            raise ValueError(
                "max_logit_delta must be positive"
            )

        if policy not in {"advantage", "quality"}:
            raise ValueError(
                "policy must be advantage/quality"
            )

        if teacher not in {"both", "sam", "ov"}:
            raise ValueError(
                "teacher must be both/sam/ov"
            )

        self.channels = channels
        self.hidden = hidden

        self.ema = ema
        self.max_logit_delta = max_logit_delta

        self.boundary_radius = int(boundary_radius)
        self.small_area = int(small_area)

        self.policy = str(policy)
        self.teacher = str(teacher)
        self.difficulty = bool(difficulty)

        # Main ablation switch:
        #
        # True:
        #   SAM/OV cache priors condition the dynamic teachers.
        #
        # False:
        #   teacher seed = student prediction
        #   reliability = 1
        #
        # This produces a capacity-matched online auxiliary teacher control
        # without foundation-cache knowledge.
        self.cache_conditioning = bool(cache_conditioning)

        # Fast online teachers.
        self.fast = nn.ModuleList(
            [
                ResidualTeacherExpert(
                    channels=channels,
                    hidden=hidden,
                ),
                ResidualTeacherExpert(
                    channels=channels,
                    hidden=hidden,
                ),
            ]
        )

        # EMA teachers used for Student supervision.
        self.target = copy.deepcopy(self.fast)

        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

        self.target.eval()

    # ---------------------------------------------------------------------
    # Parameter / state API
    # ---------------------------------------------------------------------

    def teacher_parameters(self) -> Iterator[nn.Parameter]:
        """
        Parameters that MUST belong only to the teacher optimizer.
        """
        return self.fast.parameters()

    def fast_teacher_parameters(self) -> Iterator[nn.Parameter]:
        """Alias for explicit external use."""
        return self.fast.parameters()

    def target_teacher_parameters(self) -> Iterator[nn.Parameter]:
        """EMA teacher parameters; all should have requires_grad=False."""
        return self.target.parameters()

    @torch.no_grad()
    def copy_fast_to_target(self) -> None:
        """
        Hard-sync EMA target teachers from fast teachers.

        Normally needed only for explicit reinitialization/debugging.
        Initial construction is already synchronized.
        """
        for target_teacher, fast_teacher in zip(
            self.target,
            self.fast,
        ):
            target_teacher.load_state_dict(
                fast_teacher.state_dict()
            )

        self.target.eval()

        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_ema(
        self,
        decay: Optional[float] = None,
    ) -> float:
        """
        Update target teachers from fast teachers.

            target =
                decay * target
                + (1 - decay) * fast

        Returns
        -------
        float
            RMS magnitude of the actual EMA parameter update.

        This value is useful as a direct log signal proving that teacher state
        is evolving over training.
        """
        decay = self.ema if decay is None else float(decay)

        if not 0.0 <= decay < 1.0:
            raise ValueError(
                "EMA decay must satisfy 0 <= decay < 1"
            )

        square_update = 0.0
        parameter_count = 0

        for target_teacher, fast_teacher in zip(
            self.target,
            self.fast,
        ):
            for target_parameter, fast_parameter in zip(
                target_teacher.parameters(),
                fast_teacher.parameters(),
            ):
                update = (
                    fast_parameter.detach()
                    - target_parameter
                ) * (1.0 - decay)

                target_parameter.add_(update)

                square_update += (
                    update.float()
                    .square()
                    .sum()
                    .item()
                )

                parameter_count += update.numel()

            # The current experts contain no running-stat buffers, but copying
            # buffers keeps this function correct if a future non-trainable
            # buffer is introduced.
            for target_buffer, fast_buffer in zip(
                target_teacher.buffers(),
                fast_teacher.buffers(),
            ):
                target_buffer.copy_(fast_buffer)

        self.target.eval()

        return (
            square_update / max(parameter_count, 1)
        ) ** 0.5

    @torch.no_grad()
    def teacher_target_gap(self) -> float:
        """
        RMS parameter distance between fast and EMA teachers.

        A persistent exact zero after teacher optimization begins indicates
        that the dynamic teacher is not actually evolving.
        """
        square_gap = 0.0
        parameter_count = 0

        for target_teacher, fast_teacher in zip(
            self.target,
            self.fast,
        ):
            for target_parameter, fast_parameter in zip(
                target_teacher.parameters(),
                fast_teacher.parameters(),
            ):
                gap = (
                    fast_parameter.detach().float()
                    - target_parameter.detach().float()
                )

                square_gap += (
                    gap.square()
                    .sum()
                    .item()
                )

                parameter_count += gap.numel()

        return (
            square_gap / max(parameter_count, 1)
        ) ** 0.5

    # ---------------------------------------------------------------------
    # Internal teacher construction
    # ---------------------------------------------------------------------

    def _active_teacher_vector(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return [2] activation vector for teacher-training ablations.
        """
        if self.teacher == "both":
            values = (1.0, 1.0)

        elif self.teacher == "sam":
            values = (1.0, 0.0)

        elif self.teacher == "ov":
            values = (0.0, 1.0)

        else:
            raise RuntimeError(
                f"Unexpected teacher setting: {self.teacher}"
            )

        return torch.tensor(
            values,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def _build_seeds(
        self,
        prediction: torch.Tensor,
        teacher_pack: Optional[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build [B,2,H,W] seed proposals and reliability maps.

        cache_conditioning=True:
            reuse existing DART-R-TS SAM/OV proposal construction.

        cache_conditioning=False:
            capacity-matched no-cache control:
                seed = current student
                reliability = 1
        """
        p = prediction.detach().float()

        if self.cache_conditioning:
            if teacher_pack is None:
                raise ValueError(
                    "teacher_pack is required when "
                    "cache_conditioning=True"
                )

            seeds, reliability = build_task_proposals(
                prediction,
                teacher_pack,
            )

            seeds = (
                seeds.detach()
                .float()
                .clamp(0.0, 1.0)
            )

            reliability = (
                reliability.detach()
                .float()
                .clamp(0.0, 1.0)
            )

        else:
            seeds = (
                p.expand(
                    -1,
                    self.NUM_TEACHERS,
                    -1,
                    -1,
                )
                .clone()
            )

            reliability = torch.ones_like(seeds)

        expected = (
            prediction.shape[0],
            self.NUM_TEACHERS,
            prediction.shape[-2],
            prediction.shape[-1],
        )

        if tuple(seeds.shape) != expected:
            raise ValueError(
                "Teacher seeds must have shape "
                f"{expected}, got {tuple(seeds.shape)}"
            )

        if tuple(reliability.shape) != expected:
            raise ValueError(
                "Teacher reliability must have shape "
                f"{expected}, got {tuple(reliability.shape)}"
            )

        return seeds, reliability

    def _dynamic_proposal(
        self,
        expert: ResidualTeacherExpert,
        feature: torch.Tensor,
        prediction: torch.Tensor,
        seed: torch.Tensor,
        reliability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct one dynamic teacher proposal.

        Mathematical form:

            delta = G(
                stopgrad(F_s),
                stopgrad(p_s),
                cache_prior
            )

            q =
                sigmoid(
                    logit(stopgrad(p_s))
                    +
                    reliability
                    * Delta_max
                    * tanh(delta)
                )

        The correction therefore remains:
            - student-conditioned;
            - cache-conditioned;
            - bounded;
            - identity initialized.

        Returns
        -------
        proposal:
            [B,1,H,W]

        full_residual:
            bounded pre-reliability logit correction [B,1,H,W]
            for diagnostics.
        """
        p = prediction.detach().float()

        low_residual = expert(
            feature=feature.detach(),
            probability=p,
            seed=seed.detach(),
            reliability=reliability.detach(),
        )

        full_residual = _resize(
            low_residual,
            p.shape[-2:],
        )

        # Bound the expert correction before reliability modulation.
        bounded_residual = (
            self.max_logit_delta
            * torch.tanh(full_residual)
        )

        support = (
            reliability.detach()
            .float()
            .clamp(0.0, 1.0)
        )

        proposal = torch.sigmoid(
            _safe_logit(p)
            + support * bounded_residual
        )

        return (
            proposal.clamp(0.0, 1.0),
            bounded_residual,
        )

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(
        self,
        feature: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        teacher_pack: Optional[Dict],
        force_action: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one reciprocal dynamic-teacher training forward.

        Parameters
        ----------
        feature:
            Decoder feature from the unchanged student main path.
            Expected [B,C,Hf,Wf].

        prediction:
            Main student probability.
            Expected [B,1,H,W].

        target:
            Binary GT.
            Expected [B,1,H,W].

        teacher_pack:
            Synchronously replayed SAMStruct + OVCDistill cache pack.
            May be None only when cache_conditioning=False.

        force_action:
            Existing debug override:
                0 -> SAM
                1 -> OV
                2 -> Reject

            It affects Student routing through task_space_route().
            It does NOT alter teacher fitting; teacher fitting follows the
            configured teacher={"both","sam","ov"} setting.

        Returns
        -------
        dict

        Student-side differentiable key:
            "total"

        Teacher-side differentiable key:
            "teacher_total"

        These two losses intentionally belong to disjoint parameter graphs.
        """
        if feature.ndim != 4:
            raise ValueError(
                "feature must be [B,C,H,W]"
            )

        if feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected feature channels={self.channels}, "
                f"got {feature.shape[1]}"
            )

        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise ValueError(
                "prediction must be [B,1,H,W]"
            )

        if target.shape != prediction.shape:
            raise ValueError(
                "target and prediction must have identical shapes"
            )

        if feature.shape[0] != prediction.shape[0]:
            raise ValueError(
                "feature and prediction batch sizes must match"
            )

        # Difficulty diagnostics are detached inside build_cd_difficulty().
        diagnosis = build_cd_difficulty(
            prediction,
            target,
            self.boundary_radius,
            self.small_area,
        )

        # Keep the entire auxiliary numerically in FP32.
        # The student main path itself is untouched.
        device_type = prediction.device.type

        with torch.autocast(
            device_type=device_type,
            enabled=False,
        ):
            p = prediction.float()
            y = target.detach().float()

            # -------------------------------------------------------------
            # 1. Static cache information becomes PRIOR, not final target.
            # -------------------------------------------------------------
            seeds, reliability = self._build_seeds(
                prediction=p,
                teacher_pack=teacher_pack,
            )

            # -------------------------------------------------------------
            # 2. EMA target teachers create CURRENT dynamic proposals.
            #
            # No current GT is available to these proposal generators.
            # -------------------------------------------------------------
            dynamic_proposals = []
            target_residuals = []

            with torch.no_grad():
                for teacher_index in range(
                    self.NUM_TEACHERS
                ):
                    proposal, residual = (
                        self._dynamic_proposal(
                            expert=self.target[
                                teacher_index
                            ],
                            feature=feature,
                            prediction=p,
                            seed=seeds[
                                :,
                                teacher_index:
                                teacher_index + 1,
                            ],
                            reliability=reliability[
                                :,
                                teacher_index:
                                teacher_index + 1,
                            ],
                        )
                    )

                    dynamic_proposals.append(
                        proposal
                    )

                    target_residuals.append(
                        residual
                    )

            dynamic_proposals = torch.cat(
                dynamic_proposals,
                dim=1,
            )

            target_residuals = torch.cat(
                target_residuals,
                dim=1,
            )

            # -------------------------------------------------------------
            # 3. Student chooses useful teachers through existing exact
            #    task-space positive-gain auditing.
            # -------------------------------------------------------------
            route = task_space_route(
                prediction=p,
                target=y,
                proposals=dynamic_proposals,
                quality=reliability,
                policy=self.policy,
                teacher=self.teacher,
                force_action=force_action,
            )

            if self.difficulty:
                importance = (
                    diagnosis["importance"]
                    .detach()
                    .float()
                )
            else:
                importance = torch.ones_like(p)

            # -------------------------------------------------------------
            # 4. Teacher -> Student.
            #
            # dynamic_proposals are detached EMA outputs.
            # Therefore this path updates STUDENT only.
            # -------------------------------------------------------------
            kd_map = bernoulli_kl(
                prediction=p,
                target=dynamic_proposals,
            )

            effective = (
                route["effective_map"]
                .detach()
                .float()
            )

            denominator = (
                importance
                .sum((2, 3))
                .clamp_min(1.0)
            )

            per_teacher = (
                kd_map
                * effective
                * importance
            ).sum((2, 3)) / denominator

            student_kd = (
                per_teacher
                .sum(dim=1)
                .mean()
            )

            # -------------------------------------------------------------
            # 5. Fast teachers predict corrections from the SAME PRE-UPDATE
            #    student state.
            #
            # Student inputs are detached explicitly, therefore this graph
            # updates FAST TEACHERS only.
            # -------------------------------------------------------------
            fast_proposals = []
            fast_residuals = []

            for teacher_index in range(
                self.NUM_TEACHERS
            ):
                proposal, residual = (
                    self._dynamic_proposal(
                        expert=self.fast[
                            teacher_index
                        ],
                        feature=feature.detach(),
                        prediction=p.detach(),
                        seed=seeds[
                            :,
                            teacher_index:
                            teacher_index + 1,
                        ],
                        reliability=reliability[
                            :,
                            teacher_index:
                            teacher_index + 1,
                        ],
                    )
                )

                fast_proposals.append(
                    proposal
                )

                fast_residuals.append(
                    residual
                )

            fast_proposals = torch.cat(
                fast_proposals,
                dim=1,
            )

            fast_residuals = torch.cat(
                fast_residuals,
                dim=1,
            )

            # -------------------------------------------------------------
            # 6. Student -> Teacher.
            #
            # Teachers focus on where the CURRENT student still has error.
            #
            # reliability is already used to bound the teacher correction.
            # Here we use binary cache support instead of multiplying
            # reliability a second time, avoiding accidental confidence^2
            # suppression.
            # -------------------------------------------------------------
            student_abs_error = (
                p.detach() - y
            ).abs()

            cache_support = (
                reliability > 0
            ).float()

            teacher_active = (
                self._active_teacher_vector(
                    device=p.device,
                    dtype=p.dtype,
                )
                .view(1, self.NUM_TEACHERS, 1, 1)
            )

            # [B,1,H,W] -> broadcast to [B,2,H,W]
            fit_weight = (
                student_abs_error
                * importance
                * cache_support
                * teacher_active
            )

            y_two = y.expand_as(
                fast_proposals
            )

            teacher_bce_map = (
                F.binary_cross_entropy(
                    fast_proposals,
                    y_two,
                    reduction="none",
                )
            )

            fit_denominator = (
                fit_weight
                .sum((2, 3))
                .clamp_min(1e-6)
            )

            teacher_loss_per_sample = (
                teacher_bce_map
                * fit_weight
            ).sum((2, 3)) / fit_denominator

            # Do not average an inactive teacher as an artificial zero-loss
            # third party.
            active_vector = (
                teacher_active
                .view(self.NUM_TEACHERS)
            )

            teacher_total = (
                teacher_loss_per_sample
                .mean(dim=0)
                * active_vector
            ).sum() / active_vector.sum().clamp_min(1.0)

            # -------------------------------------------------------------
            # 7. Diagnostics.
            # -------------------------------------------------------------
            base_error = (
                p.detach() - y
            ).square()

            dynamic_brier_map = (
                dynamic_proposals.detach()
                - y_two
            ).square()

            fast_brier_map = (
                fast_proposals.detach()
                - y_two
            ).square()

            static_brier_map = (
                seeds.detach()
                - y_two
            ).square()

            dynamic_gain_map = (
                base_error
                - dynamic_brier_map
            )

            static_gain_map = (
                base_error
                - static_brier_map
            )

            support = (
                route["accepted_map"]
                .detach()
                .float()
            )

            def summary(
                x: torch.Tensor,
            ) -> torch.Tensor:
                """
                Difficulty-weighted full-image summary.

                Supports x shaped:
                    [B,1,H,W]
                    [B,2,H,W]

                Returns:
                    [B,1] or [B,2]
                """
                x = x.detach().float()

                num = (
                    x * importance
                ).sum((2, 3))

                den = (
                    importance
                    .sum((2, 3))
                    .clamp_min(1.0)
                )

                return num / den

            q_summary = summary(
                reliability
            )

            changed_support = masked_mean(
                support,
                y,
            ).mean()

            background_support = masked_mean(
                support,
                1.0 - y,
            ).mean()

            student_error_mass = (
                student_abs_error
                * importance
            ).mean()

            effective_mass = (
                effective
                .sum(dim=1)
                .mean()
            )

            # Important dynamic-health statistic:
            # absolute teacher mass is expected to decrease as the student
            # becomes good. What should NOT collapse prematurely is the
            # effective mass relative to the student's remaining error mass.
            effective_per_error_mass = (
                effective_mass
                / student_error_mass.clamp_min(1e-8)
            )

            dynamic_shift = summary(
                (
                    dynamic_proposals.detach()
                    - p.detach()
                ).abs()
            )

            static_shift = summary(
                (
                    seeds.detach()
                    - p.detach()
                ).abs()
            )

            target_residual_magnitude = summary(
                target_residuals.abs()
            )

            fast_residual_magnitude = summary(
                fast_residuals.detach().abs()
            )

            # Same external layout as TaskSpaceDirectionC:
            #
            # first two columns:
            #     SAM / OV mixture
            #
            # third column:
            #     reject
            mixture_summary = summary(
                route["mixture_map"]
            )

            reject_summary = summary(
                1.0 - support
            )

            weights = torch.cat(
                (
                    mixture_summary,
                    reject_summary,
                ),
                dim=1,
            )

            return {
                # =========================================================
                # Differentiable optimization outputs
                # =========================================================

                # Teacher -> Student:
                # include ONLY in the student optimization objective.
                "total": student_kd,

                # Student -> Teacher:
                # optimize ONLY through a separate teacher optimizer.
                "teacher_total": teacher_total,

                # Per-sample/per-teacher forms.
                "loss_per_teacher": per_teacher,

                "teacher_loss_per_teacher":
                    teacher_loss_per_sample,

                # =========================================================
                # Existing Direction-C compatible diagnostics
                # =========================================================
                "h": diagnosis["h"],
                "h_valid": diagnosis["valid"],

                "q": q_summary,

                "effective_weights":
                    summary(effective),

                "weights": weights,

                "action":
                    route["action"].detach(),

                # IMPORTANT:
                # proposals now means the DYNAMIC EMA teacher proposals.
                "proposals":
                    dynamic_proposals.detach(),

                # Static cache-derived priors are exposed separately.
                "static_proposals":
                    seeds.detach(),

                "pixel_reject_ratio":
                    (1.0 - support).mean(),

                "image_reject_ratio":
                    (
                        ~route["accepted_map"]
                        .flatten(1)
                        .any(1)
                    )
                    .float()
                    .mean(),

                "accepted_change_ratio":
                    changed_support,

                "accepted_bg_ratio":
                    background_support,

                "proposal_gain":
                    summary(
                        route["gain_map"]
                    ),

                "relative_gain":
                    summary(
                        route["relative_map"]
                    ),

                "eligible_ratio":
                    route["eligible_map"]
                    .detach()
                    .float()
                    .mean((2, 3)),

                "available_ratio":
                    route["available_map"]
                    .detach()
                    .float()
                    .mean((2, 3)),

                "student_brier":
                    base_error.mean(),

                "proposal_brier":
                    summary(
                        dynamic_brier_map
                    ),

                "effective_mass":
                    effective_mass,

                # Preserve historical names so existing logger code can be
                # adapted incrementally.
                "sam_transport":
                    per_teacher[:, self.SAM_INDEX]
                    .mean(),

                "ov_task":
                    per_teacher[:, self.OV_INDEX]
                    .mean(),

                # =========================================================
                # Dynamic-teacher-specific diagnostics
                # =========================================================

                # Dynamic EMA teacher error.
                "dynamic_teacher_brier":
                    summary(
                        dynamic_brier_map
                    ),

                # Fast teacher error before the optimizer step.
                "fast_teacher_brier":
                    summary(
                        fast_brier_map
                    ),

                # Original static cache-prior error.
                "static_teacher_brier":
                    summary(
                        static_brier_map
                    ),

                # Positive means the dynamic teacher is better than the
                # current student in Brier space.
                "dynamic_teacher_gain":
                    summary(
                        dynamic_gain_map
                    ),

                # Direct comparison against the unadapted static prior.
                "static_teacher_gain":
                    summary(
                        static_gain_map
                    ),

                # How far the evolved EMA teacher moved away from the
                # current student prediction.
                "dynamic_shift":
                    dynamic_shift,

                "dynamic_shift_sam":
                    dynamic_shift[
                        :, self.SAM_INDEX
                    ].mean(),

                "dynamic_shift_ov":
                    dynamic_shift[
                        :, self.OV_INDEX
                    ].mean(),

                # How far the original fixed cache prior is from the student.
                "static_shift":
                    static_shift,

                # Magnitude of learned logit correction.
                "target_residual_magnitude":
                    target_residual_magnitude,

                "fast_residual_magnitude":
                    fast_residual_magnitude,

                # Teacher fitting diagnostics.
                "teacher_fit_sam":
                    teacher_loss_per_sample[
                        :, self.SAM_INDEX
                    ].mean(),

                "teacher_fit_ov":
                    teacher_loss_per_sample[
                        :, self.OV_INDEX
                    ].mean(),

                # Remaining student difficulty.
                "student_abs_error":
                    student_abs_error.mean(),

                "student_error_mass":
                    student_error_mass,

                # Key health statistic:
                # teacher contribution relative to remaining student error.
                "effective_per_error_mass":
                    effective_per_error_mass,

                # Cache-support diagnostics.
                "cache_support_ratio":
                    cache_support.mean(
                        (0, 2, 3)
                    ),

                # Explicitly expose mode for checkpoint/log audit.
                "cache_conditioning":
                    p.new_tensor(
                        float(
                            self.cache_conditioning
                        )
                    ),
            }

    # ---------------------------------------------------------------------
    # Module behavior
    # ---------------------------------------------------------------------

    def train(
        self,
        mode: bool = True,
    ):
        """
        Fast teachers follow module train/eval mode.

        EMA target teachers are ALWAYS kept in eval mode. They currently have
        no dropout/batchnorm, but enforcing this invariant avoids accidental
        behavior changes if their architecture is extended later.
        """
        super().train(mode)

        self.fast.train(mode)
        self.target.eval()

        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

        return self

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, "
            f"hidden={self.hidden}, "
            f"ema={self.ema}, "
            f"max_logit_delta={self.max_logit_delta}, "
            f"policy={self.policy!r}, "
            f"teacher={self.teacher!r}, "
            f"difficulty={self.difficulty}, "
            f"cache_conditioning={self.cache_conditioning}"
        )


# ---------------------------------------------------------------------------
# Alias used by the A2Net auxiliary selector.
#
# The later a2net.py modification can simply do:
#
#     from .distill.dynamic_teacher import (
#         DynamicTeacherDirectionC as DirectionC
#     )
#
# This preserves the current Direction-C construction/call contract.
# ---------------------------------------------------------------------------

DynamicTeacherDirectionC = ReciprocalDynamicTeacher


__all__ = [
    "ResidualTeacherExpert",
    "ReciprocalDynamicTeacher",
    "DynamicTeacherDirectionC",
]