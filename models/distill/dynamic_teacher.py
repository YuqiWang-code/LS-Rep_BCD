"""
Run3 BT-SAM-RDT:
Bi-Temporal Structural SAM Reciprocal Dynamic Teacher.

This module implements the TRAINING-ONLY dynamic-teacher component used by
Run3.

Run3 principle
--------------
Run1 maintained two independent dynamic teachers:

    SAM teacher
    OV teacher

and routed/competed them pixel-wise.

Run3 removes that design.

Instead:

    SAMStruct T1/T2
        -> explicit bi-temporal instance correspondence
        -> structural-change prior

    OVCDistill
        -> semantic soft-change prior

    structural prior + semantic prior
        -> one fused foundation prior
        -> one Fast Residual Teacher
        -> EMA
        -> one Target Teacher
        -> GT positive-Brier-gain safety audit
        -> Student KD

The main methodological change is therefore upstream of the dynamic teacher:
SAM no longer transports the Student prediction inside independent temporal
instances. It supplies genuinely bi-temporal structural information.

Gradient contract
-----------------
1. Student KD loss ``total``:
       updates STUDENT only.

   The proposal distilled into the Student is produced by the no-grad EMA
   Target Teacher.

2. Teacher fitting loss ``teacher_total``:
       updates FAST TEACHER only.

   Student feature and Student prediction are explicitly detached on this
   path.

3. EMA Target Teacher:
       requires_grad=False;
       updated only through ``update_ema()``.

4. Foundation priors:
       constructed without Student prediction and without GT.

5. GT:
       used only for:
           - positive-gain safety auditing;
           - Fast Teacher fitting;
           - detached Student-difficulty weighting.

6. Everything in this module is training-only.
   It must never enter the deploy feature path.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diagnostics import (
    build_cd_difficulty,
    masked_mean,
)
from .task_space import (
    bernoulli_kl,
    build_foundation_prior,
    task_space_audit,
)


# ============================================================================
# Utilities
# ============================================================================


def _resize(
    x: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """FP32 bilinear resize used only inside the training auxiliary."""
    size = tuple(
        int(v)
        for v in size
    )

    if (
        x.shape[-2:]
        == size
    ):
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
    """Stable logit for probability-output A2Net."""
    return torch.logit(
        p.float().clamp(
            eps,
            1.0 - eps,
        )
    )


def _validate_single_channel(
    x: torch.Tensor,
    name: str,
) -> None:
    if (
        not torch.is_tensor(x)
        or x.ndim != 4
        or x.shape[1] != 1
    ):
        raise ValueError(
            f"{name} must be [B,1,H,W]"
        )


# ============================================================================
# Run3 single residual teacher
# ============================================================================


class ResidualTeacherExpert(nn.Module):
    """
    One TRAINING-ONLY Run3 residual teacher.

    The teacher does not predict an absolute change map from scratch.

    Inputs
    ------
    1. Student decoder feature:
           F_s

    2. Student probability:
           p_s

    3. Fused-prior residual:
           q_fused - p_s

    4. BT-SAM structural prior:
           q_sam

    5. OV semantic prior:
           q_ov

    6. Fused foundation reliability:
           r_fused

    Input channels:
        C + 5

    Output
    ------
    One unconstrained low-resolution residual field.

    The caller converts this into a bounded logit correction.

    Initialization
    --------------
    The final convolution is zero initialized, therefore:

        dynamic teacher == Student

    at initialization.

    The dynamic teacher must learn useful corrections rather than inject a
    random target into the Student.
    """

    def __init__(
        self,
        channels: int = 64,
        hidden: int = 24,
    ) -> None:
        super().__init__()

        channels = int(
            channels
        )

        hidden = int(
            hidden
        )

        if channels <= 0:
            raise ValueError(
                "channels must be positive"
            )

        if hidden <= 0:
            raise ValueError(
                "hidden must be positive"
            )

        self.channels = channels
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Conv2d(
                channels + 5,
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

        # Identity initialization:
        # bounded correction = 0
        # -> q_teacher = p_student
        nn.init.zeros_(
            self.net[-1].weight
        )

        nn.init.zeros_(
            self.net[-1].bias
        )

    def forward(
        self,
        feature: torch.Tensor,
        probability: torch.Tensor,
        fused_prior: torch.Tensor,
        sam_prior: torch.Tensor,
        ov_prior: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict one low-resolution teacher logit correction.

        Every Student/Foundation input is detached here as an additional
        gradient-safety barrier.
        """
        if feature.ndim != 4:
            raise ValueError(
                "feature must be [B,C,H,W]"
            )

        if (
            feature.shape[1]
            != self.channels
        ):
            raise ValueError(
                f"Expected feature channels={self.channels}, "
                f"got {feature.shape[1]}"
            )

        for tensor, name in (
            (
                probability,
                "probability",
            ),
            (
                fused_prior,
                "fused_prior",
            ),
            (
                sam_prior,
                "sam_prior",
            ),
            (
                ov_prior,
                "ov_prior",
            ),
            (
                reliability,
                "reliability",
            ),
        ):
            _validate_single_channel(
                tensor,
                name,
            )

        batch_size = (
            feature.shape[0]
        )

        if any(
            tensor.shape[0]
            != batch_size
            for tensor in (
                probability,
                fused_prior,
                sam_prior,
                ov_prior,
                reliability,
            )
        ):
            raise ValueError(
                "Teacher inputs must share the same batch size"
            )

        spatial_size = (
            feature.shape[-2:]
        )

        f = (
            feature
            .detach()
            .float()
        )

        p = _resize(
            probability.detach(),
            spatial_size,
        ).clamp(
            0.0,
            1.0,
        )

        fused = _resize(
            fused_prior.detach(),
            spatial_size,
        ).clamp(
            0.0,
            1.0,
        )

        sam = _resize(
            sam_prior.detach(),
            spatial_size,
        ).clamp(
            0.0,
            1.0,
        )

        ov = _resize(
            ov_prior.detach(),
            spatial_size,
        ).clamp(
            0.0,
            1.0,
        )

        reliability = _resize(
            reliability.detach(),
            spatial_size,
        ).clamp(
            0.0,
            1.0,
        )

        foundation_residual = (
            fused
            - p
        )

        x = torch.cat(
            (
                f,
                p,
                foundation_residual,
                sam,
                ov,
                reliability,
            ),
            dim=1,
        )

        return self.net(
            x
        )


# ============================================================================
# BT-SAM-RDT
# ============================================================================


class BTSAMRDT(nn.Module):
    """
    Run3 BT-SAM-RDT training auxiliary.

    Only one dynamic teacher is maintained:

        Fast Teacher
            optimized by teacher_total

        EMA Target Teacher
            generated from Fast Teacher through EMA
            used to supervise the Student

    Expected optimization
    ---------------------
    Student optimizer:

        main_loss
        +
        kd_lambda * output["total"]

    Teacher optimizer:

        output["teacher_total"]

    After:

        teacher_optimizer.step()

    call:

        module.update_ema()

    Parameters returned by ``teacher_parameters()`` must never enter the
    Student optimizer.
    """

    def __init__(
        self,
        channels: int = 64,
        hidden: int = 24,
        ema: float = 0.99,
        max_logit_delta: float = 2.0,
        boundary_radius: int = 2,
        small_area: int = 64,
        policy: str = "advantage",
        difficulty: bool = True,
        use_ov: bool = True,
        use_sam: bool = True,
    ) -> None:
        super().__init__()

        channels = int(
            channels
        )

        hidden = int(
            hidden
        )

        ema = float(
            ema
        )

        max_logit_delta = float(
            max_logit_delta
        )

        boundary_radius = int(
            boundary_radius
        )

        small_area = int(
            small_area
        )

        if channels <= 0:
            raise ValueError(
                "channels must be positive"
            )

        if hidden <= 0:
            raise ValueError(
                "hidden must be positive"
            )

        if not (
            0.0
            <= ema
            < 1.0
        ):
            raise ValueError(
                "ema must satisfy 0 <= ema < 1"
            )

        if (
            max_logit_delta
            <= 0
        ):
            raise ValueError(
                "max_logit_delta must be positive"
            )

        if boundary_radius < 1:
            raise ValueError(
                "boundary_radius must be positive"
            )

        if small_area < 1:
            raise ValueError(
                "small_area must be positive"
            )

        if policy not in {
            "advantage",
            "quality",
        }:
            raise ValueError(
                "policy must be 'advantage' or 'quality'"
            )

        self.channels = channels
        self.hidden = hidden

        self.ema = ema
        self.max_logit_delta = (
            max_logit_delta
        )

        self.boundary_radius = (
            boundary_radius
        )

        self.small_area = (
            small_area
        )

        self.policy = str(
            policy
        )

        self.difficulty = bool(
            difficulty
        )

        # Main Run3:
        #     True
        #
        # R3-A SAM-only mechanism ablation:
        #     False
        #
        # This switch changes only how the Foundation prior is constructed.
        # The online teacher capacity remains identical.
        self.use_ov = bool(
            use_ov
        )

        self.use_sam = bool(
            use_sam
        )

        # ------------------------------------------------------------------
        # One Fast Teacher
        # ------------------------------------------------------------------
        self.fast = ResidualTeacherExpert(
            channels=channels,
            hidden=hidden,
        )

        # ------------------------------------------------------------------
        # One EMA Target Teacher
        # ------------------------------------------------------------------
        self.target = copy.deepcopy(
            self.fast
        )

        for parameter in (
            self.target.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        self.target.eval()

    # ======================================================================
    # Parameter/state API
    # ======================================================================

    def teacher_parameters(
        self,
    ) -> Iterator[nn.Parameter]:
        """
        Parameters belonging only to the Fast Teacher optimizer.
        """
        return self.fast.parameters()

    def fast_teacher_parameters(
        self,
    ) -> Iterator[nn.Parameter]:
        """Explicit alias for the Fast Teacher parameters."""
        return self.fast.parameters()

    def target_teacher_parameters(
        self,
    ) -> Iterator[nn.Parameter]:
        """EMA Target Teacher parameters; always requires_grad=False."""
        return self.target.parameters()

    @torch.no_grad()
    def copy_fast_to_target(
        self,
    ) -> None:
        """
        Hard-copy Fast Teacher parameters to the EMA Target Teacher.
        """
        self.target.load_state_dict(
            self.fast.state_dict()
        )

        self.target.eval()

        for parameter in (
            self.target.parameters()
        ):
            parameter.requires_grad_(
                False
            )

    @torch.no_grad()
    def update_ema(
        self,
        decay: Optional[float] = None,
    ) -> float:
        """
        EMA update:

            target =
                decay * target
                +
                (1 - decay) * fast

        Returns
        -------
        float
            RMS magnitude of the actual parameter update.
        """
        decay = (
            self.ema
            if decay is None
            else float(decay)
        )

        if not (
            0.0
            <= decay
            < 1.0
        ):
            raise ValueError(
                "EMA decay must satisfy 0 <= decay < 1"
            )

        square_update = 0.0
        parameter_count = 0

        for (
            target_parameter,
            fast_parameter,
        ) in zip(
            self.target.parameters(),
            self.fast.parameters(),
        ):
            update = (
                fast_parameter.detach()
                - target_parameter
            ) * (
                1.0
                - decay
            )

            target_parameter.add_(
                update
            )

            square_update += (
                update
                .float()
                .square()
                .sum()
                .item()
            )

            parameter_count += (
                update.numel()
            )

        # Future-proof buffer synchronization.
        for (
            target_buffer,
            fast_buffer,
        ) in zip(
            self.target.buffers(),
            self.fast.buffers(),
        ):
            target_buffer.copy_(
                fast_buffer
            )

        self.target.eval()

        for parameter in (
            self.target.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        return (
            square_update
            / max(
                parameter_count,
                1,
            )
        ) ** 0.5

    @torch.no_grad()
    def teacher_target_gap(
        self,
    ) -> float:
        """
        RMS parameter distance between Fast and EMA Target Teacher.

        A persistent exact zero after optimization begins indicates that the
        dynamic teacher is not actually evolving.
        """
        square_gap = 0.0
        parameter_count = 0

        for (
            target_parameter,
            fast_parameter,
        ) in zip(
            self.target.parameters(),
            self.fast.parameters(),
        ):
            gap = (
                fast_parameter
                .detach()
                .float()
                - target_parameter
                .detach()
                .float()
            )

            square_gap += (
                gap
                .square()
                .sum()
                .item()
            )

            parameter_count += (
                gap.numel()
            )

        return (
            square_gap
            / max(
                parameter_count,
                1,
            )
        ) ** 0.5

    # ======================================================================
    # Dynamic teacher construction
    # ======================================================================

    def _dynamic_proposal(
        self,
        expert: ResidualTeacherExpert,
        feature: torch.Tensor,
        prediction: torch.Tensor,
        fused_prior: torch.Tensor,
        sam_prior: torch.Tensor,
        ov_prior: torch.Tensor,
        reliability: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Construct one dynamic teacher proposal.

        Mathematical form
        -----------------
        delta = G(
            stopgrad(F_s),
            stopgrad(p_s),
            stopgrad(q_f - p_s),
            stopgrad(q_sam),
            stopgrad(q_ov),
            stopgrad(r_f)
        )

        q_T =
            sigmoid(
                logit(stopgrad(p_s))
                +
                r_f
                * Delta_max
                * tanh(delta)
            )

        The dynamic teacher is:
            - Student-aware;
            - explicitly Foundation-prior-aware;
            - structurally aware;
            - semantically aware;
            - reliability bounded;
            - identity initialized.
        """
        _validate_single_channel(
            prediction,
            "prediction",
        )

        _validate_single_channel(
            fused_prior,
            "fused_prior",
        )

        _validate_single_channel(
            sam_prior,
            "sam_prior",
        )

        _validate_single_channel(
            ov_prior,
            "ov_prior",
        )

        _validate_single_channel(
            reliability,
            "reliability",
        )

        p = (
            prediction
            .detach()
            .float()
        )

        low_residual = expert(
            feature=feature.detach(),
            probability=p,
            fused_prior=fused_prior.detach(),
            sam_prior=sam_prior.detach(),
            ov_prior=ov_prior.detach(),
            reliability=reliability.detach(),
        )

        full_residual = _resize(
            low_residual,
            p.shape[-2:],
        )

        bounded_residual = (
            self.max_logit_delta
            * torch.tanh(
                full_residual
            )
        )

        support = (
            reliability
            .detach()
            .float()
            .clamp(
                0.0,
                1.0,
            )
        )

        proposal = torch.sigmoid(
            _safe_logit(
                p
            )
            + support
            * bounded_residual
        )

        return (
            proposal.clamp(
                0.0,
                1.0,
            ),
            bounded_residual,
        )

    # ======================================================================
    # Forward
    # ======================================================================

    def forward(
        self,
        feature: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        teacher_pack: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one Run3 BT-SAM-RDT training forward.

        Inputs
        ------
        feature:
            Decoder feature from unchanged A2Net main path.
            [B,C,Hf,Wf]

        prediction:
            Student primary prediction.
            [B,1,H,W]

        target:
            Binary GT.
            [B,1,H,W]

        teacher_pack:
            Synchronously replayed SAMStruct + OVCDistill cache.

        Differentiable outputs
        ----------------------
        total:
            Student-only KD objective.

        teacher_total:
            Fast-Teacher-only fitting objective.
        """
        if feature.ndim != 4:
            raise ValueError(
                "feature must be [B,C,H,W]"
            )

        if (
            feature.shape[1]
            != self.channels
        ):
            raise ValueError(
                f"Expected feature channels={self.channels}, "
                f"got {feature.shape[1]}"
            )

        _validate_single_channel(
            prediction,
            "prediction",
        )

        _validate_single_channel(
            target,
            "target",
        )

        if (
            target.shape
            != prediction.shape
        ):
            raise ValueError(
                "target and prediction must have identical shapes"
            )

        if (
            feature.shape[0]
            != prediction.shape[0]
        ):
            raise ValueError(
                "feature and prediction batch sizes must match"
            )

        if teacher_pack is None:
            raise ValueError(
                "BT-SAM-RDT requires teacher_pack"
            )

        if not torch.all(
            (
                target == 0
            )
            | (
                target == 1
            )
        ):
            raise ValueError(
                "BT-SAM-RDT requires binary 0/1 GT"
            )

        # Detached Student difficulty diagnosis.
        diagnosis = build_cd_difficulty(
            prediction,
            target,
            self.boundary_radius,
            self.small_area,
        )

        device_type = (
            prediction.device.type
        )

        # Auxiliary calculations stay in FP32.
        with torch.autocast(
            device_type=device_type,
            enabled=False,
        ):
            p = prediction.float()

            y = (
                target
                .detach()
                .float()
            )

            # ==============================================================
            # 1. Construct FOUNDATION priors.
            #
            # IMPORTANT:
            # build_foundation_prior() receives neither Student prediction
            # nor GT. BT-SAM therefore remains an independent knowledge
            # source.
            # ==============================================================
            foundation = build_foundation_prior(
                teacher_pack=teacher_pack,
                output_size=p.shape[-2:],
                use_ov=self.use_ov,
                use_sam=self.use_sam,
            )

            sam_prior = (
                foundation[
                    "sam_prior"
                ]
                .detach()
                .float()
            )

            sam_reliability = (
                foundation[
                    "sam_reliability"
                ]
                .detach()
                .float()
            )

            ov_prior = (
                foundation[
                    "ov_prior"
                ]
                .detach()
                .float()
            )

            ov_reliability = (
                foundation[
                    "ov_reliability"
                ]
                .detach()
                .float()
            )

            fused_prior = (
                foundation[
                    "fused_prior"
                ]
                .detach()
                .float()
            )

            fused_reliability = (
                foundation[
                    "fused_reliability"
                ]
                .detach()
                .float()
            )

            sam_ov_conflict = (
                foundation[
                    "sam_ov_conflict"
                ]
                .detach()
                .float()
            )

            # ==============================================================
            # 2. EMA Target Teacher creates the proposal seen by Student.
            #
            # No current GT enters proposal generation.
            # ==============================================================
            with torch.no_grad():
                (
                    dynamic_proposal,
                    target_residual,
                ) = self._dynamic_proposal(
                    expert=self.target,
                    feature=feature,
                    prediction=p,
                    fused_prior=fused_prior,
                    sam_prior=sam_prior,
                    ov_prior=ov_prior,
                    reliability=fused_reliability,
                )

            # ==============================================================
            # 3. GT positive-gain SAFETY audit.
            #
            # No SAM-vs-OV routing remains.
            # One dynamic teacher is either admitted or rejected per pixel.
            # ==============================================================
            audit = task_space_audit(
                prediction=p,
                target=y,
                proposal=dynamic_proposal,
                reliability=fused_reliability,
                policy=self.policy,
            )

            if self.difficulty:
                importance = (
                    diagnosis[
                        "importance"
                    ]
                    .detach()
                    .float()
                )

            else:
                importance = (
                    torch.ones_like(
                        p
                    )
                )

            # ==============================================================
            # 4. Target Teacher -> Student.
            #
            # dynamic_proposal is detached.
            # Gradient therefore reaches Student only.
            # ==============================================================
            kd_map = bernoulli_kl(
                prediction=p,
                target=dynamic_proposal,
            )

            effective = (
                audit[
                    "effective_map"
                ]
                .detach()
                .float()
            )

            denominator = (
                importance
                .sum(
                    (
                        2,
                        3,
                    )
                )
                .clamp_min(
                    1.0
                )
            )

            student_kd_per_sample = (
                (
                    kd_map
                    * effective
                    * importance
                )
                .sum(
                    (
                        2,
                        3,
                    )
                )
                / denominator
            )

            student_kd = (
                student_kd_per_sample
                .mean()
            )

            # ==============================================================
            # 5. Fast Teacher proposal.
            #
            # Student and Foundation inputs are detached. This graph belongs
            # only to Fast Teacher parameters.
            # ==============================================================
            (
                fast_proposal,
                fast_residual,
            ) = self._dynamic_proposal(
                expert=self.fast,
                feature=feature.detach(),
                prediction=p.detach(),
                fused_prior=fused_prior,
                sam_prior=sam_prior,
                ov_prior=ov_prior,
                reliability=fused_reliability,
            )

            # ==============================================================
            # 6. Student -> Fast Teacher fitting.
            #
            # Fit where:
            #   - Student still has error;
            #   - Foundation prior has support;
            #   - detached difficulty weighting indicates importance.
            #
            # Reliability is already used inside the dynamic proposal to bound
            # the logit correction, so use binary support here to avoid an
            # accidental reliability^2 suppression.
            # ==============================================================
            student_abs_error = (
                p.detach()
                - y
            ).abs()

            foundation_support = (
                fused_reliability
                > 0
            ).float()

            fit_weight = (
                student_abs_error
                * importance
                * foundation_support
            )

            teacher_bce_map = (
                F.binary_cross_entropy(
                    fast_proposal,
                    y,
                    reduction="none",
                )
            )

            fit_denominator = (
                fit_weight
                .sum(
                    (
                        2,
                        3,
                    )
                )
                .clamp_min(
                    1e-6
                )
            )

            teacher_loss_per_sample = (
                (
                    teacher_bce_map
                    * fit_weight
                )
                .sum(
                    (
                        2,
                        3,
                    )
                )
                / fit_denominator
            )

            teacher_total = (
                teacher_loss_per_sample
                .mean()
            )

            # ==============================================================
            # 7. Diagnostics
            # ==============================================================
            student_brier_map = (
                p.detach()
                - y
            ).square()

            sam_brier_map = (
                sam_prior
                - y
            ).square()

            ov_brier_map = (
                ov_prior
                - y
            ).square()

            fused_brier_map = (
                fused_prior
                - y
            ).square()

            dynamic_brier_map = (
                dynamic_proposal.detach()
                - y
            ).square()

            fast_brier_map = (
                fast_proposal.detach()
                - y
            ).square()

            dynamic_gain_map = (
                student_brier_map
                - dynamic_brier_map
            )

            fused_gain_map = (
                student_brier_map
                - fused_brier_map
            )

            sam_gain_map = (
                student_brier_map
                - sam_brier_map
            )

            ov_gain_map = (
                student_brier_map
                - ov_brier_map
            )

            support = (
                audit[
                    "accepted_map"
                ]
                .detach()
                .float()
            )

            def summary(
                x: torch.Tensor,
            ) -> torch.Tensor:
                """
                Difficulty-weighted full-image per-sample summary.

                Input:
                    [B,1,H,W]

                Output:
                    [B,1]
                """
                x = (
                    x
                    .detach()
                    .float()
                )

                num = (
                    x
                    * importance
                ).sum(
                    (
                        2,
                        3,
                    )
                )

                den = (
                    importance
                    .sum(
                        (
                            2,
                            3,
                        )
                    )
                    .clamp_min(
                        1.0
                    )
                )

                return (
                    num
                    / den
                )

            changed_support = (
                masked_mean(
                    support,
                    y,
                )
                .mean()
            )

            background_support = (
                masked_mean(
                    support,
                    1.0 - y,
                )
                .mean()
            )

            student_error_mass = (
                student_abs_error
                * importance
            ).mean()

            effective_mass = (
                effective.mean()
            )

            effective_per_error_mass = (
                effective_mass
                / student_error_mass.clamp_min(
                    1e-8
                )
            )

            dynamic_shift = summary(
                (
                    dynamic_proposal.detach()
                    - p.detach()
                ).abs()
            )

            fused_shift = summary(
                (
                    fused_prior
                    - p.detach()
                ).abs()
            )

            sam_shift = summary(
                (
                    sam_prior
                    - p.detach()
                ).abs()
            )

            ov_shift = summary(
                (
                    ov_prior
                    - p.detach()
                ).abs()
            )

            target_residual_magnitude = (
                summary(
                    target_residual.abs()
                )
            )

            fast_residual_magnitude = (
                summary(
                    fast_residual
                    .detach()
                    .abs()
                )
            )

            conflict_available = (
                foundation[
                    "both_available"
                ]
                .detach()
                .float()
            )

            conflict_denominator = (
                conflict_available
                .sum()
                .clamp_min(
                    1.0
                )
            )

            mean_sam_ov_conflict = (
                (
                    sam_ov_conflict
                    * conflict_available
                ).sum()
                / conflict_denominator
            )

            return {
                # ==========================================================
                # Differentiable optimization outputs
                # ==========================================================
                "total": (
                    student_kd
                ),

                "teacher_total": (
                    teacher_total
                ),

                "student_kd_per_sample": (
                    student_kd_per_sample
                ),

                "teacher_loss_per_sample": (
                    teacher_loss_per_sample
                ),

                # ==========================================================
                # Difficulty diagnosis
                # ==========================================================
                "h": (
                    diagnosis[
                        "h"
                    ]
                ),

                "h_valid": (
                    diagnosis[
                        "valid"
                    ]
                ),

                # ==========================================================
                # Foundation-prior outputs
                # ==========================================================
                "sam_prior": (
                    sam_prior
                ),

                "sam_reliability": (
                    sam_reliability
                ),

                "ov_prior": (
                    ov_prior
                ),

                "ov_reliability": (
                    ov_reliability
                ),

                "fused_prior": (
                    fused_prior
                ),

                "fused_reliability": (
                    fused_reliability
                ),

                "sam_ov_conflict_map": (
                    sam_ov_conflict
                ),

                # ==========================================================
                # Bi-temporal SAM matching diagnostics
                # ==========================================================
                "pair_match_ratio": (
                    foundation[
                        "pair_match_ratio"
                    ].detach()
                ),

                "pair_match_iou": (
                    foundation[
                        "pair_match_iou"
                    ].detach()
                ),

                "pair_match_cov12": (
                    foundation[
                        "pair_match_cov12"
                    ].detach()
                ),

                "pair_match_cov21": (
                    foundation[
                        "pair_match_cov21"
                    ].detach()
                ),

                # ==========================================================
                # Dynamic teacher outputs
                # ==========================================================
                "proposal": (
                    dynamic_proposal.detach()
                ),

                "fast_proposal": (
                    fast_proposal.detach()
                ),

                "target_residual": (
                    target_residual.detach()
                ),

                "fast_residual": (
                    fast_residual.detach()
                ),

                # ==========================================================
                # GT safety audit
                # ==========================================================
                "accepted_map": (
                    audit[
                        "accepted_map"
                    ].detach()
                ),

                "effective_map": (
                    effective
                ),

                "gain_map": (
                    audit[
                        "gain_map"
                    ].detach()
                ),

                "relative_gain_map": (
                    audit[
                        "relative_gain_map"
                    ].detach()
                ),

                "available_map": (
                    audit[
                        "available_map"
                    ].detach()
                ),

                "eligible_map": (
                    audit[
                        "eligible_map"
                    ].detach()
                ),

                # ==========================================================
                # Acceptance statistics
                # ==========================================================
                "pixel_reject_ratio": (
                    1.0
                    - support
                ).mean(),

                "image_reject_ratio": (
                    (
                        ~audit[
                            "accepted_map"
                        ]
                        .flatten(
                            1
                        )
                        .any(
                            1
                        )
                    )
                    .float()
                    .mean()
                ),

                "accepted_change_ratio": (
                    changed_support
                ),

                "accepted_bg_ratio": (
                    background_support
                ),

                "available_ratio": (
                    audit[
                        "available_map"
                    ]
                    .detach()
                    .float()
                    .mean()
                ),

                "eligible_ratio": (
                    audit[
                        "eligible_map"
                    ]
                    .detach()
                    .float()
                    .mean()
                ),

                "effective_mass": (
                    effective_mass
                ),

                "effective_per_error_mass": (
                    effective_per_error_mass
                ),

                # ==========================================================
                # Prior quality / Brier diagnostics
                # ==========================================================
                "student_brier": (
                    student_brier_map
                    .mean()
                ),

                "sam_prior_brier": (
                    sam_brier_map
                    .mean()
                ),

                "ov_prior_brier": (
                    ov_brier_map
                    .mean()
                ),

                "fusion_prior_brier": (
                    fused_brier_map
                    .mean()
                ),

                "dynamic_teacher_brier": (
                    dynamic_brier_map
                    .mean()
                ),

                "fast_teacher_brier": (
                    fast_brier_map
                    .mean()
                ),

                "sam_prior_gain": (
                    sam_gain_map
                    .mean()
                ),

                "ov_prior_gain": (
                    ov_gain_map
                    .mean()
                ),

                "fusion_prior_gain": (
                    fused_gain_map
                    .mean()
                ),

                "dynamic_teacher_gain": (
                    dynamic_gain_map
                    .mean()
                ),

                # Difficulty-weighted versions useful for diagnosis.
                "sam_prior_brier_weighted": (
                    summary(
                        sam_brier_map
                    ).mean()
                ),

                "ov_prior_brier_weighted": (
                    summary(
                        ov_brier_map
                    ).mean()
                ),

                "fusion_prior_brier_weighted": (
                    summary(
                        fused_brier_map
                    ).mean()
                ),

                "dynamic_teacher_brier_weighted": (
                    summary(
                        dynamic_brier_map
                    ).mean()
                ),

                # ==========================================================
                # Teacher evolution diagnostics
                # ==========================================================
                "dynamic_shift": (
                    dynamic_shift.mean()
                ),

                "fused_prior_shift": (
                    fused_shift.mean()
                ),

                "sam_prior_shift": (
                    sam_shift.mean()
                ),

                "ov_prior_shift": (
                    ov_shift.mean()
                ),

                "target_residual_magnitude": (
                    target_residual_magnitude
                    .mean()
                ),

                "fast_residual_magnitude": (
                    fast_residual_magnitude
                    .mean()
                ),

                "student_abs_error": (
                    student_abs_error
                    .mean()
                ),

                "student_error_mass": (
                    student_error_mass
                ),

                "foundation_support_ratio": (
                    foundation_support
                    .mean()
                ),

                "sam_reliability_mean": (
                    sam_reliability
                    .mean()
                ),

                "ov_reliability_mean": (
                    ov_reliability
                    .mean()
                ),

                "fused_reliability_mean": (
                    fused_reliability
                    .mean()
                ),

                "sam_ov_conflict": (
                    mean_sam_ov_conflict
                ),

                # ==========================================================
                # Explicit Run3 configuration audit
                # ==========================================================
                "use_ov": (
                    p.new_tensor(
                        float(
                            self.use_ov
                        )
                    )
                ),

                "policy_advantage": (
                    p.new_tensor(
                        float(
                            self.policy
                            == "advantage"
                        )
                    )
                ),
            }

    # ======================================================================
    # Module behavior
    # ======================================================================

    def train(
        self,
        mode: bool = True,
    ):
        """
        Fast Teacher follows module train/eval state.

        EMA Target Teacher is always kept in eval mode and frozen.
        """
        super().train(
            mode
        )

        self.fast.train(
            mode
        )

        self.target.eval()

        for parameter in (
            self.target.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        return self

    def extra_repr(
        self,
    ) -> str:
        return (
            f"channels={self.channels}, "
            f"hidden={self.hidden}, "
            f"ema={self.ema}, "
            f"max_logit_delta={self.max_logit_delta}, "
            f"boundary_radius={self.boundary_radius}, "
            f"small_area={self.small_area}, "
            f"policy={self.policy!r}, "
            f"difficulty={self.difficulty}, "
            f"use_ov={self.use_ov}"
        )