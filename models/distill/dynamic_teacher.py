"""RDT-CD + SCGR training-only dynamic teacher for binary change detection.

RDT-CD
------
Two small online residual experts (fast teachers) learn from the current
student error using GT.  EMA copies of those experts (target teachers) create
the proposals distilled into the student.

SCGR
----
Sparse-Change Gradient-Concordant Routing audits the EMA proposals before
student distillation.  A teacher is admitted only when its regional proposal:

1) is supported by the configured prior/reliability;
2) has positive GT-audited Brier utility;
3) has positive analytical classifier-gradient concordance with GT.

The admitted KD is normalized by the student's remaining balanced error mass
rather than by HxW image area, so sparse useful change regions are not drowned
by background pixels.

Critical gradient contract
--------------------------
Student loss:
    main_loss + lambda_kd * output["total"]
    -> updates STUDENT only.

Teacher loss:
    output["teacher_total"]
    -> updates FAST TEACHERS only.

EMA:
    update_ema() is called only after teacher_optimizer.step().
    EMA target teachers always have requires_grad=False.

No teacher/cache/router tensor is ever injected into the deploy feature path.
A2Net.switch_to_deploy() physically removes this complete auxiliary.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diagnostics import build_cd_difficulty, masked_mean
from .task_space import (
    DEFAULT_REGION_SIZE,
    NUM_TEACHERS,
    OV_INDEX,
    REJECT_INDEX,
    SAM_INDEX,
    build_task_proposals,
    routed_bernoulli_kd,
    task_space_route,
)


EPS = 1e-8


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _resize(
    x: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """FP32 bilinear resize used only in the training auxiliary."""
    if tuple(x.shape[-2:]) == tuple(size):
        return x.float()
    return F.interpolate(
        x.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def _safe_logit(
    probability: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Stable logit for the probability-output A2Net student."""
    return torch.logit(
        probability.float().clamp(eps, 1.0 - eps)
    )


def _finite_or_raise(
    name: str,
    tensor: torch.Tensor,
) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(
            f"{name} contains non-finite values"
        )


def _pair_region_mean(
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Summarize [B,2,Rh,Rw] evidence to [B,2].

    If ``mask`` is supplied, empty teacher/region support returns exactly zero.
    """
    if value.ndim != 4 or value.shape[1] != NUM_TEACHERS:
        raise ValueError(
            "Expected regional teacher tensor [B,2,Rh,Rw], got "
            f"{tuple(value.shape)}"
        )

    x = value.detach().float()
    if mask is None:
        return x.mean((2, 3))

    if mask.shape != value.shape:
        raise ValueError(
            "Regional summary mask must match value shape"
        )

    m = mask.detach().float()
    numerator = (x * m).sum((2, 3))
    denominator = m.sum((2, 3)).clamp_min(1.0)
    return numerator / denominator


def _masked_ratio(
    event: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Global detached ratio with empty support defined as zero."""
    event_f = event.detach().float()
    valid_f = valid.detach().float()
    numerator = (event_f * valid_f).sum()
    denominator = valid_f.sum()
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        numerator.new_zeros(()),
    )


# ---------------------------------------------------------------------------
# Online residual teacher
# ---------------------------------------------------------------------------

class ResidualTeacherExpert(nn.Module):
    """Small TRAINING-ONLY residual teacher expert.

    Inputs
    ------
    feature:
        current detached student decoder feature [B,C,Hf,Wf].
    probability:
        current detached student probability [B,1,H,W].
    seed:
        SAM/OV cache prior, or student identity prior in D2.
    reliability:
        cache reliability, or one in D2.

    Output
    ------
    A low-resolution unconstrained residual.  The caller applies tanh and the
    fixed ``max_logit_delta`` cap before constructing the dynamic proposal.

    The final convolution is exactly zero initialized so that at construction:
        q_dynamic == p_student.
    """

    def __init__(
        self,
        channels: int = 64,
        hidden: int = 24,
    ) -> None:
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

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        feature: torch.Tensor,
        probability: torch.Tensor,
        seed: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(
                "feature must be [B,C,H,W]"
            )
        if feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected feature channels={self.channels}, "
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

        spatial_size = tuple(feature.shape[-2:])

        # Explicit detach is a second barrier against teacher-fitting gradients
        # entering the Student.
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

        cache_residual = s - p
        teacher_input = torch.cat(
            (f, p, cache_residual, r),
            dim=1,
        )
        return self.net(teacher_input)


# ---------------------------------------------------------------------------
# Reciprocal dynamic teacher + SCGR
# ---------------------------------------------------------------------------

class ReciprocalDynamicTeacher(nn.Module):
    """RDT-CD auxiliary with parameter-free SCGR Student routing.

    Teacher index
    -------------
    0 = SAMStruct structural prior/expert
    1 = OVCDistill semantic prior/expert

    Fast teachers are optimized by ``teacher_total``.
    EMA target teachers create the proposals used by ``total``.
    """

    NUM_TEACHERS = NUM_TEACHERS
    SAM_INDEX = SAM_INDEX
    OV_INDEX = OV_INDEX
    REJECT_INDEX = REJECT_INDEX

    def __init__(
        self,
        channels: int = 64,
        hidden: int = 24,
        ema: float = 0.99,
        max_logit_delta: float = 2.0,
        boundary_radius: int = 2,
        small_area: int = 64,
        policy: str = "scgr",
        teacher: str = "both",
        difficulty: bool = True,
        cache_conditioning: bool = True,
        gradient_gate: bool = True,
    ) -> None:
        super().__init__()

        channels = int(channels)
        hidden = int(hidden)
        ema = float(ema)
        max_logit_delta = float(max_logit_delta)
        boundary_radius = int(boundary_radius)
        small_area = int(small_area)

        if channels <= 0:
            raise ValueError("channels must be positive")
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if not 0.0 <= ema < 1.0:
            raise ValueError(
                "ema must satisfy 0 <= ema < 1"
            )
        if max_logit_delta <= 0.0:
            raise ValueError(
                "max_logit_delta must be positive"
            )
        if boundary_radius < 0:
            raise ValueError(
                "boundary_radius must be nonnegative"
            )
        if small_area <= 0:
            raise ValueError(
                "small_area must be positive"
            )
        if policy != "scgr":
            raise ValueError(
                "Current implementation supports only policy='scgr'"
            )
        if teacher not in {"both", "sam", "ov"}:
            raise ValueError(
                "teacher must be one of: both/sam/ov"
            )

        self.channels = channels
        self.hidden = hidden
        self.ema = ema
        self.max_logit_delta = max_logit_delta
        self.boundary_radius = boundary_radius
        self.small_area = small_area
        self.policy = policy
        self.teacher = str(teacher)
        self.difficulty = bool(difficulty)
        self.cache_conditioning = bool(cache_conditioning)
        self.gradient_gate = bool(gradient_gate)
        self.region_size = DEFAULT_REGION_SIZE

        self.fast = nn.ModuleList(
            [
                ResidualTeacherExpert(
                    channels=channels,
                    hidden=hidden,
                )
                for _ in range(self.NUM_TEACHERS)
            ]
        )

        self.target = copy.deepcopy(self.fast)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.target.eval()

    # ------------------------------------------------------------------
    # Parameter / EMA API used by train.py and checkpointing.
    # ------------------------------------------------------------------

    def teacher_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters owned exclusively by the teacher optimizer."""
        return self.fast.parameters()

    def fast_teacher_parameters(self) -> Iterator[nn.Parameter]:
        return self.fast.parameters()

    def target_teacher_parameters(self) -> Iterator[nn.Parameter]:
        return self.target.parameters()

    @torch.no_grad()
    def copy_fast_to_target(self) -> None:
        """Hard synchronization for explicit initialization/debug use."""
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
        """EMA-update target teachers and return RMS update magnitude."""
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

            # Current experts have no running statistics, but keep future
            # non-trainable buffers synchronized safely.
            for target_buffer, fast_buffer in zip(
                target_teacher.buffers(),
                fast_teacher.buffers(),
            ):
                target_buffer.copy_(fast_buffer)

        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

        return (
            square_update / max(parameter_count, 1)
        ) ** 0.5

    @torch.no_grad()
    def teacher_target_gap(self) -> float:
        """RMS parameter distance between fast and EMA target teachers."""
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
                square_gap += gap.square().sum().item()
                parameter_count += gap.numel()

        return (
            square_gap / max(parameter_count, 1)
        ) ** 0.5

    # ------------------------------------------------------------------
    # Teacher construction.
    # ------------------------------------------------------------------

    def _active_teacher_vector(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.teacher == "both":
            values = (1.0, 1.0)
        elif self.teacher == "sam":
            values = (1.0, 0.0)
        elif self.teacher == "ov":
            values = (0.0, 1.0)
        else:
            raise RuntimeError(
                f"Unexpected teacher mode: {self.teacher!r}"
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
        """Build detached [SAM,OV] seeds and reliability.

        D1:
            real SAMStruct + OVCDistill cache priors.

        D2:
            seed = current student, reliability = 1.
            This preserves the same online-teacher capacity while removing
            foundation-cache information.
        """
        p = prediction.detach().float()

        if self.cache_conditioning:
            if teacher_pack is None:
                raise ValueError(
                    "teacher_pack is required when "
                    "cache_conditioning=True"
                )

            seeds, reliability = build_task_proposals(
                p,
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
            p.shape[0],
            self.NUM_TEACHERS,
            p.shape[-2],
            p.shape[-1],
        )

        if tuple(seeds.shape) != expected:
            raise ValueError(
                f"Teacher seeds must be {expected}, "
                f"got {tuple(seeds.shape)}"
            )
        if tuple(reliability.shape) != expected:
            raise ValueError(
                f"Teacher reliability must be {expected}, "
                f"got {tuple(reliability.shape)}"
            )

        _finite_or_raise("teacher seeds", seeds)
        _finite_or_raise("teacher reliability", reliability)

        return seeds, reliability

    def _dynamic_proposal(
        self,
        expert: ResidualTeacherExpert,
        feature: torch.Tensor,
        prediction: torch.Tensor,
        seed: torch.Tensor,
        reliability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct one online/EMA dynamic proposal.

        q = sigmoid(
                logit(stopgrad(p))
                + reliability * max_logit_delta * tanh(delta)
            )

        Returns
        -------
        proposal:
            [B,1,H,W]
        bounded_residual:
            bounded pre-reliability logit correction [B,1,H,W]
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
            tuple(p.shape[-2:]),
        )

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

    # ------------------------------------------------------------------
    # Main forward.
    # ------------------------------------------------------------------

    def forward(
        self,
        feature: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        teacher_pack: Optional[Dict],
        force_action: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Execute one RDT-CD + SCGR training forward."""
        if feature.ndim != 4:
            raise ValueError(
                "feature must be [B,C,Hf,Wf]"
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
        if not bool(torch.all((target == 0) | (target == 1))):
            raise ValueError(
                "target must be binary 0/1"
            )

        # Detached diagnostic/difficulty builder.  This is retained from the
        # previous RDT implementation so that this experiment changes routing,
        # not the teacher-fitting curriculum.
        diagnosis = build_cd_difficulty(
            prediction,
            target,
            self.boundary_radius,
            self.small_area,
        )

        device_type = prediction.device.type

        # Keep the auxiliary numerically in FP32.  The deploy path is outside
        # this context and remains untouched.
        with torch.autocast(
            device_type=device_type,
            enabled=False,
        ):
            # Student probability retains its graph here: only routed KD below
            # should use that graph.  Every teacher/SCGR evidence constructor
            # explicitly detaches its observations.
            p = prediction.float()
            y = target.detach().float()

            if self.difficulty:
                importance = (
                    diagnosis["importance"]
                    .detach()
                    .float()
                )
            else:
                importance = torch.ones_like(p)

            # --------------------------------------------------------------
            # 1. Fixed cache values are PRIORS, not student targets.
            # --------------------------------------------------------------
            seeds, reliability = self._build_seeds(
                prediction=p,
                teacher_pack=teacher_pack,
            )

            # --------------------------------------------------------------
            # 2. EMA target teachers make current Student-facing proposals.
            #    No current-batch GT enters this proposal generation.
            # --------------------------------------------------------------
            dynamic_proposals = []
            target_residuals = []

            with torch.no_grad():
                for teacher_index in range(self.NUM_TEACHERS):
                    proposal, residual = self._dynamic_proposal(
                        expert=self.target[teacher_index],
                        feature=feature,
                        prediction=p,
                        seed=seeds[
                            :,
                            teacher_index:teacher_index + 1,
                        ],
                        reliability=reliability[
                            :,
                            teacher_index:teacher_index + 1,
                        ],
                    )
                    dynamic_proposals.append(proposal)
                    target_residuals.append(residual)

            dynamic_proposals = torch.cat(
                dynamic_proposals,
                dim=1,
            )
            target_residuals = torch.cat(
                target_residuals,
                dim=1,
            )

            # --------------------------------------------------------------
            # 3. SCGR audits dynamic proposals.
            #
            # This is the ONLY method-level change relative to the preceding
            # RDT formulation:
            #   old: pixel-only positive Brier route + image-area-like KD
            #   new: regional utility + gradient concordance + error-mass KD
            # --------------------------------------------------------------
            route = task_space_route(
                prediction=p,
                target=y,
                proposals=dynamic_proposals,
                quality=reliability,
                feature=feature,
                importance=importance,
                policy=self.policy,
                teacher=self.teacher,
                force_action=force_action,
                gradient_gate=self.gradient_gate,
            )

            # --------------------------------------------------------------
            # 4. Teacher -> Student.
            #
            # EMA proposals and SCGR evidence are detached.  The differentiable
            # path is only from routed Bernoulli KD back to `p`, hence Student.
            # --------------------------------------------------------------
            kd = routed_bernoulli_kd(
                prediction=p,
                proposals=dynamic_proposals,
                route=route,
            )

            student_kd = kd["total"]
            per_teacher = kd["loss_per_teacher"]

            # --------------------------------------------------------------
            # 5. Fast teachers use the same PRE-UPDATE student state.
            #    Student/cache observations are detached inside the expert.
            # --------------------------------------------------------------
            fast_proposals = []
            fast_residuals = []

            for teacher_index in range(self.NUM_TEACHERS):
                proposal, residual = self._dynamic_proposal(
                    expert=self.fast[teacher_index],
                    feature=feature,
                    prediction=p,
                    seed=seeds[
                        :,
                        teacher_index:teacher_index + 1,
                    ],
                    reliability=reliability[
                        :,
                        teacher_index:teacher_index + 1,
                    ],
                )
                fast_proposals.append(proposal)
                fast_residuals.append(residual)

            fast_proposals = torch.cat(
                fast_proposals,
                dim=1,
            )
            fast_residuals = torch.cat(
                fast_residuals,
                dim=1,
            )

            # --------------------------------------------------------------
            # 6. Student -> Fast Teacher fitting.
            #
            # Intentionally preserved from previous RDT:
            # - focus on current student error;
            # - preserve existing difficulty importance;
            # - cache reliability already bounds the residual, therefore only
            #   binary cache support is used here to avoid confidence^2.
            #
            # SCGR does NOT gate teacher fitting.  Otherwise the routing change
            # would also alter the online teacher curriculum, confounding the
            # first experiment.
            # --------------------------------------------------------------
            student_abs_error = (
                p.detach() - y
            ).abs()

            cache_support = (
                reliability > 0.0
            ).float()

            teacher_active = (
                self._active_teacher_vector(
                    device=p.device,
                    dtype=p.dtype,
                )
                .view(
                    1,
                    self.NUM_TEACHERS,
                    1,
                    1,
                )
            )

            fit_weight = (
                student_abs_error
                * importance
                * cache_support
                * teacher_active
            )

            y_two = y.expand_as(fast_proposals)

            teacher_bce_map = F.binary_cross_entropy(
                fast_proposals,
                y_two,
                reduction="none",
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

            active_vector = teacher_active.view(
                self.NUM_TEACHERS
            )

            teacher_total = (
                teacher_loss_per_sample
                .mean(dim=0)
                * active_vector
            ).sum() / active_vector.sum().clamp_min(1.0)

            # --------------------------------------------------------------
            # 7. Diagnostics.
            # --------------------------------------------------------------
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

            accepted = (
                route["accepted_map"]
                .detach()
                .float()
            )

            mixture = (
                route["mixture_map"]
                .detach()
                .float()
            )

            error_mass_map = (
                route["error_mass_map"]
                .detach()
                .float()
            )

            kd_weight_map = (
                route["kd_weight_map"]
                .detach()
                .float()
            )

            def image_summary(
                x: torch.Tensor,
            ) -> torch.Tensor:
                """Difficulty-weighted [B,C,H,W] -> [B,C] summary."""
                x = x.detach().float()
                numerator = (
                    x * importance
                ).sum((2, 3))

                denominator = (
                    importance
                    .sum((2, 3))
                    .clamp_min(1.0)
                )
                return numerator / denominator

            q_summary = image_summary(reliability)

            changed_support = masked_mean(
                accepted,
                y,
            ).mean()

            background_support = masked_mean(
                accepted,
                1.0 - y,
            ).mean()

            dynamic_shift = image_summary(
                (
                    dynamic_proposals.detach()
                    - p.detach()
                ).abs()
            )

            static_shift = image_summary(
                (
                    seeds.detach()
                    - p.detach()
                ).abs()
            )

            target_residual_magnitude = image_summary(
                target_residuals.abs()
            )

            fast_residual_magnitude = image_summary(
                fast_residuals.detach().abs()
            )

            mixture_summary = image_summary(mixture)

            reject_summary = image_summary(
                1.0 - accepted
            )

            weights = torch.cat(
                (
                    mixture_summary,
                    reject_summary,
                ),
                dim=1,
            )

            # -------------------------
            # SCGR region diagnostics.
            # -------------------------
            region_available = (
                route["region_available"]
                .detach()
                .bool()
            )
            region_positive_utility = (
                route["region_positive_utility"]
                .detach()
                .bool()
            )
            region_positive_concordance = (
                route["region_positive_concordance"]
                .detach()
                .bool()
            )
            region_eligible = (
                route["region_eligible"]
                .detach()
                .bool()
            )
            region_accepted = (
                route["region_accepted"]
                .detach()
                .bool()
            )

            gradient_conflict_region = (
                region_available
                & region_positive_utility
                & (~region_positive_concordance)
            )

            positive_utility_available = (
                region_available
                & region_positive_utility
            )

            scgr_region_accept_ratio = (
                region_accepted.float().mean()
            )
            scgr_region_reject_ratio = (
                1.0 - scgr_region_accept_ratio
            )

            scgr_positive_utility_ratio = _masked_ratio(
                region_positive_utility,
                region_available,
            )

            scgr_positive_concordance_ratio = _masked_ratio(
                region_positive_concordance,
                region_available,
            )

            scgr_gradient_conflict_ratio = _masked_ratio(
                gradient_conflict_region,
                region_available,
            )

            # Of regions that are output-space beneficial, how many are
            # explicitly rejected by the gradient-concordance safety gate?
            if self.gradient_gate:
                scgr_negative_cosine_reject_ratio = _masked_ratio(
                    gradient_conflict_region,
                    positive_utility_available,
                )
            else:
                # In D1NG negative-cosine regions are diagnosed but are not
                # rejected by the disabled gradient gate.
                scgr_negative_cosine_reject_ratio = p.new_zeros(())

            scgr_region_utility = _pair_region_mean(
                route["region_utility"],
                region_available,
            )

            scgr_region_concordance = _pair_region_mean(
                route["region_concordance"],
                region_available,
            )

            scgr_region_quality = _pair_region_mean(
                route["region_quality"],
                region_available,
            )

            scgr_region_score = _pair_region_mean(
                route["region_score"],
                region_available,
            )

            # Per-image teacher share of the student's remaining error mass.
            total_error_mass_per_image = (
                error_mass_map
                .sum((1, 2, 3))
                .clamp_min(EPS)
            )

            teacher_error_mass_per_image = (
                kd_weight_map
                .sum((2, 3))
            )

            scgr_teacher_mass = (
                teacher_error_mass_per_image
                / total_error_mass_per_image.unsqueeze(1)
            )

            change_error_mass_map = (
                error_mass_map * y
            )
            background_error_mass_map = (
                error_mass_map * (1.0 - y)
            )

            scgr_change_error_mass = (
                change_error_mass_map.mean()
            )
            scgr_background_error_mass = (
                background_error_mass_map.mean()
            )

            scgr_effective_error_mass = (
                kd["accepted_error_mass"]
                / float(
                    p.shape[-2] * p.shape[-1]
                )
            ).mean()

            scgr_effective_per_error_mass = (
                kd[
                    "effective_per_error_mass_per_image"
                ]
                .mean()
            )

            # Historical names retained for before/after analysis.  They now
            # use the SCGR error-mass semantics rather than image-area mass.
            student_error_mass = (
                error_mass_map.mean()
            )
            effective_mass = (
                kd_weight_map
                .sum(dim=1)
                .mean()
            )
            effective_per_error_mass = (
                scgr_effective_per_error_mass
            )

            result = {
                # ==========================================================
                # Differentiable optimization outputs.
                # ==========================================================
                "total": student_kd,
                "teacher_total": teacher_total,
                "loss_per_teacher": per_teacher,
                "teacher_loss_per_teacher":
                    teacher_loss_per_sample,

                # ==========================================================
                # Existing Direction-C/RDT-compatible outputs.
                # ==========================================================
                "h": diagnosis["h"],
                "h_valid": diagnosis["valid"],
                "q": q_summary,

                "effective_weights":
                    image_summary(mixture),
                "weights": weights,
                "action":
                    route["action"].detach(),

                "proposals":
                    dynamic_proposals.detach(),
                "static_proposals":
                    seeds.detach(),

                "pixel_reject_ratio":
                    (1.0 - accepted).mean(),

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
                    image_summary(
                        route["gain_map"]
                    ),

                "relative_gain":
                    image_summary(
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
                    image_summary(
                        dynamic_brier_map
                    ),

                "effective_mass":
                    effective_mass,

                "sam_transport":
                    per_teacher[:, self.SAM_INDEX]
                    .mean(),

                "ov_task":
                    per_teacher[:, self.OV_INDEX]
                    .mean(),

                # ==========================================================
                # Dynamic-teacher diagnostics.
                # ==========================================================
                "dynamic_teacher_brier":
                    image_summary(
                        dynamic_brier_map
                    ),

                "fast_teacher_brier":
                    image_summary(
                        fast_brier_map
                    ),

                "static_teacher_brier":
                    image_summary(
                        static_brier_map
                    ),

                "dynamic_teacher_gain":
                    image_summary(
                        dynamic_gain_map
                    ),

                "static_teacher_gain":
                    image_summary(
                        static_gain_map
                    ),

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

                "static_shift":
                    static_shift,

                "target_residual_magnitude":
                    target_residual_magnitude,

                "fast_residual_magnitude":
                    fast_residual_magnitude,

                "teacher_fit_sam":
                    teacher_loss_per_sample[
                        :, self.SAM_INDEX
                    ].mean(),

                "teacher_fit_ov":
                    teacher_loss_per_sample[
                        :, self.OV_INDEX
                    ].mean(),

                "student_abs_error":
                    student_abs_error.mean(),

                "student_error_mass":
                    student_error_mass,

                "effective_per_error_mass":
                    effective_per_error_mass,

                "cache_support_ratio":
                    cache_support.mean(
                        (0, 2, 3)
                    ),

                "cache_conditioning":
                    p.new_tensor(
                        float(
                            self.cache_conditioning
                        )
                    ),

                "scgr_gradient_gate_enabled":
                    p.new_tensor(
                        float(self.gradient_gate)
                    ),

                "scgr_region_size":
                    p.new_tensor(
                        float(DEFAULT_REGION_SIZE)
                    ),

                # ==========================================================
                # SCGR-specific scalar diagnostics.
                # ==========================================================
                "scgr_region_accept_ratio":
                    scgr_region_accept_ratio,

                "scgr_region_reject_ratio":
                    scgr_region_reject_ratio,

                "scgr_change_accept_ratio":
                    changed_support,

                "scgr_background_accept_ratio":
                    background_support,

                "scgr_positive_utility_ratio":
                    scgr_positive_utility_ratio,

                "scgr_positive_concordance_ratio":
                    scgr_positive_concordance_ratio,

                "scgr_gradient_conflict_ratio":
                    scgr_gradient_conflict_ratio,

                "scgr_negative_cosine_reject_ratio":
                    scgr_negative_cosine_reject_ratio,

                "scgr_change_error_mass":
                    scgr_change_error_mass,

                "scgr_background_error_mass":
                    scgr_background_error_mass,

                "scgr_effective_error_mass":
                    scgr_effective_error_mass,

                "scgr_effective_per_error_mass":
                    scgr_effective_per_error_mass,

                # ==========================================================
                # SCGR teacher-pair summaries, [B,2].
                # Trainer converts these to *_sam / *_ov.
                # ==========================================================
                "scgr_region_utility":
                    scgr_region_utility,

                "scgr_region_concordance":
                    scgr_region_concordance,

                "scgr_region_quality":
                    scgr_region_quality,

                "scgr_region_score":
                    scgr_region_score,

                "scgr_teacher_mass":
                    scgr_teacher_mass,

                # Useful full maps for smoke/debug; trainer ignores them.
                "scgr_error_mass_map":
                    error_mass_map,

                "scgr_kd_weight_map":
                    kd_weight_map,

                "scgr_gradient_conflict_map":
                    route[
                        "gradient_conflict_map"
                    ].detach(),

                "scgr_region_eligible":
                    region_eligible,

                "scgr_region_action":
                    route[
                        "region_action"
                    ].detach(),
            }

            # Fail here rather than silently logging/training corrupt values.
            for name in (
                "total",
                "teacher_total",
                "scgr_region_accept_ratio",
                "scgr_positive_utility_ratio",
                "scgr_positive_concordance_ratio",
                "scgr_gradient_conflict_ratio",
                "scgr_effective_per_error_mass",
            ):
                _finite_or_raise(
                    name,
                    result[name],
                )

            return result

    # ------------------------------------------------------------------
    # Module behavior.
    # ------------------------------------------------------------------

    def train(
        self,
        mode: bool = True,
    ):
        """Fast teachers follow mode; EMA targets always stay in eval mode."""
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
            f"cache_conditioning={self.cache_conditioning}, "
            f"gradient_gate={self.gradient_gate}, "
            f"region_size={DEFAULT_REGION_SIZE}"
        )


# Construction alias used by models/a2net.py.
DynamicTeacherDirectionC = ReciprocalDynamicTeacher


__all__ = [
    "ResidualTeacherExpert",
    "ReciprocalDynamicTeacher",
    "DynamicTeacherDirectionC",
]
