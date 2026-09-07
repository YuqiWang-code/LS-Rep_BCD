"""Direction-aware, tolerance-aware PI-DTRS temporal evidence."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


FIELD_WEIGHTS = {
    "boundary": 0.25,
    "occupancy": 0.15,
    "affinity_r1": 0.12,
    "affinity_r2": 0.10,
    "affinity_r4": 0.08,
    "interior_depth": 0.10,
    "object_scale": 0.12,
    "compactness": 0.08,
}

# Fine spatial detail is deliberately strongest in shallow stages; coarse
# topology/shape progressively dominates deeper stages.
STAGE_COMPONENTS = (
    {"boundary": 0.45, "affinity_r1": 0.35, "interior_depth": 0.20},
    {"boundary": 0.30, "affinity_r1": 0.25,
     "affinity_r2": 0.25, "interior_depth": 0.20},
    {"affinity_r2": 0.25, "affinity_r4": 0.35,
     "interior_depth": 0.25, "object_scale": 0.15},
    {"affinity_r4": 0.40, "object_scale": 0.25,
     "compactness": 0.20, "occupancy": 0.15},
)

STRUCTURAL_CODE_NAMES = ("boundary", "local", "geometry", "stable")

# Channel routing from fine to coarse.  The target remains one shared
# four-channel code; only its semantic emphasis changes across stages.
EIR_STAGE_CHANNEL_WEIGHTS = (
    (0.50, 0.35, 0.05, 0.10),
    (0.30, 0.40, 0.15, 0.15),
    (0.10, 0.30, 0.40, 0.20),
    (0.05, 0.15, 0.50, 0.30),
)


class DirectionalTemporalEvidence(nn.Module):
    """Convert independent T1/T2 partitions into permutation-invariant evidence.

    Directional quality is asymmetric by construction: appearance uses Q2,
    disappearance uses Q1, while stable relations require sqrt(Q1*Q2).
    ``robust_filter=False`` disables both the two-pixel boundary tolerance and
    the 3x3 support filter, which gives H8 a precise no-robustification meaning.
    """

    def __init__(self, mode="directional", robust_filter=True,
                 tolerance=0.15, support_min=2, boundary_tolerance=2):
        super().__init__()
        if mode not in {"directional", "boundary_scalar", "unsigned"}:
            raise ValueError(f"Unsupported evidence mode: {mode}")
        self.mode = mode
        self.robust_filter = bool(robust_filter)
        self.tolerance = float(tolerance)
        self.support_min = int(support_min)
        self.boundary_tolerance = int(boundary_tolerance)
        if self.boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be non-negative")

    @staticmethod
    def _as_float(value):
        return value.float().clamp(0, 1)

    def _boundary_components(self, t1, t2):
        b1 = self._as_float(t1["boundary"])
        b2 = self._as_float(t2["boundary"])
        if self.robust_filter and self.boundary_tolerance > 0:
            kernel = 2 * self.boundary_tolerance + 1
            matched_b1 = F.max_pool2d(
                b1, kernel, stride=1, padding=self.boundary_tolerance,
            )
            matched_b2 = F.max_pool2d(
                b2, kernel, stride=1, padding=self.boundary_tolerance,
            )
            plus = b2 * (1.0 - matched_b1)
            minus = b1 * (1.0 - matched_b2)
            stable = torch.maximum(b1 * matched_b2, b2 * matched_b1)
        else:
            plus = F.relu(b2 - b1)
            minus = F.relu(b1 - b2)
            stable = torch.minimum(b1, b2)
        return plus.clamp(0, 1), minus.clamp(0, 1), stable.clamp(0, 1)

    def _field_components(self, key, t1, t2):
        if key == "boundary":
            return self._boundary_components(t1, t2)
        left = self._as_float(t1[key])
        right = self._as_float(t2[key])
        plus = F.relu(right - left)
        minus = F.relu(left - right)
        common = torch.minimum(
            self._as_float(t1["occupancy"]),
            self._as_float(t2["occupancy"]),
        )
        if key == "occupancy":
            stable = torch.minimum(left, right)
        else:
            # A/D/S/C are stable only when their values agree inside common
            # object support; min(left,right) would mark a large discrepancy
            # as both change and stable.
            stable = (1.0 - torch.abs(right - left)).clamp(0, 1) * common
        return plus, minus, stable

    def _support_filter(self, evidence):
        high = (evidence > self.tolerance).float()
        votes = F.avg_pool2d(high, 3, stride=1, padding=1) * 9.0
        return evidence * ((high > 0) & (votes > self.support_min)).float()

    def _aggregate(self, weights, t1, t2):
        reference = self._as_float(t1["boundary"])
        plus = torch.zeros_like(reference)
        minus = torch.zeros_like(reference)
        stable = torch.zeros_like(reference)
        for key, weight in weights.items():
            component_plus, component_minus, component_stable = \
                self._field_components(key, t1, t2)
            plus = plus + weight * component_plus
            minus = minus + weight * component_minus
            stable = stable + weight * component_stable
        return plus.clamp(0, 1), minus.clamp(0, 1), stable.clamp(0, 1)

    def _apply_mode_and_quality(self, plus, minus, stable,
                                plus_conf, minus_conf, stable_conf):
        plus = plus * plus_conf
        minus = minus * minus_conf
        stable = stable * stable_conf
        if self.mode == "unsigned":
            unsigned = torch.maximum(plus, minus)
            plus, minus = unsigned, unsigned
        if self.robust_filter:
            plus = self._support_filter(plus)
            minus = self._support_filter(minus)
        return plus.clamp(0, 1), minus.clamp(0, 1), stable.clamp(0, 1)

    def forward(self, structure):
        t1, t2 = structure["t1"], structure["t2"]
        instance_stride = int(torch.maximum(
            t1["instance_id"].max(), t2["instance_id"].max(),
        ).item()) + 1
        instance_stride = max(instance_stride, 1)
        q1 = self._as_float(t1["quality"])
        q2 = self._as_float(t2["quality"])
        plus_conf = q2
        minus_conf = q1
        stable_conf = torch.sqrt((q1 * q2).clamp_min(0))

        boundary_raw = self._boundary_components(t1, t2)
        if self.mode == "boundary_scalar":
            raw_plus, raw_minus, raw_stable = boundary_raw
        else:
            raw_plus, raw_minus, raw_stable = self._aggregate(
                FIELD_WEIGHTS, t1, t2,
            )
        plus, minus, stable = self._apply_mode_and_quality(
            raw_plus, raw_minus, raw_stable,
            plus_conf, minus_conf, stable_conf,
        )

        stage_targets = []
        for stage_weights in STAGE_COMPONENTS:
            if self.mode == "boundary_scalar":
                stage_raw = boundary_raw
            else:
                stage_raw = self._aggregate(stage_weights, t1, t2)
            stage_plus, stage_minus, stage_stable = self._apply_mode_and_quality(
                *stage_raw, plus_conf, minus_conf, stable_conf,
            )
            # Half precision keeps the four-stage teacher pyramid modest at
            # large batches; losses convert only the active stage to float32.
            stage_targets.append({
                "plus": stage_plus.to(torch.float16),
                "minus": stage_minus.to(torch.float16),
                "stable": stage_stable.to(torch.float16),
            })

        boundary_plus, boundary_minus, boundary_stable = \
            self._apply_mode_and_quality(
                *boundary_raw, plus_conf, minus_conf, stable_conf,
            )
        boundary_delta = torch.maximum(boundary_plus, boundary_minus)
        delta = torch.maximum(plus, minus)
        if self.mode == "unsigned":
            directional_loss_plus_conf = directional_loss_minus_conf = torch.maximum(q1, q2)
            change_conf = torch.maximum(q1, q2)
        else:
            directional_loss_plus_conf = plus_conf
            directional_loss_minus_conf = minus_conf
            dominant_plus = plus >= minus
            change_conf = torch.where(dominant_plus, plus_conf, minus_conf)
        change_conf = change_conf * (delta > 0).float()

        directional_conflict = (2.0 * torch.minimum(plus, minus)).clamp(0, 1)
        stable_change_conflict = torch.minimum(delta, stable)
        topology_conflict = F.avg_pool2d(
            torch.maximum(directional_conflict, stable_change_conflict),
            3, stride=1, padding=1,
        ).clamp(0, 1)
        # Symmetric logging/debug field only. Directional losses use their own
        # confidences and therefore do not suppress one-sided appearance.
        quality_uncertainty = 1.0 - torch.maximum(q1, q2)
        uncertainty = torch.maximum(quality_uncertainty, topology_conflict)

        scale1 = self._as_float(t1["object_scale"])
        scale2 = self._as_float(t2["object_scale"])
        small_object = (
            ((scale1 > 0) & (scale1 < 0.3))
            | ((scale2 > 0) & (scale2 < 0.3))
        ).float()
        return {
            "plus": plus,
            "minus": minus,
            "stable": stable,
            "delta": delta,
            "uncertainty": uncertainty,
            "topology_conflict": topology_conflict,
            "plus_conf": directional_loss_plus_conf,
            "minus_conf": directional_loss_minus_conf,
            "stable_conf": stable_conf,
            "change_conf": change_conf,
            "small_object": small_object,
            "encoder_unsigned": self.mode == "unsigned",
            "boundary_plus": boundary_plus,
            "boundary_minus": boundary_minus,
            "boundary_delta": boundary_delta,
            "boundary_stable": boundary_stable,
            "boundary_union": torch.maximum(
                self._as_float(t1["boundary"]),
                self._as_float(t2["boundary"]),
            ),
            "stage_targets": tuple(stage_targets),
            # Compute the batch-composite instance stride once instead of
            # forcing dozens of GPU synchronizations inside relation losses.
            "instance_stride": instance_stride,
            "structure": structure,
        }


class ExchangeInvariantStructuralEvidence(nn.Module):
    """Build the four-channel EIR-HSD teacher code and continuous trust.

    The default code is strictly invariant to swapping T1/T2.  The optional
    ``directional_restore`` mode is reserved for the R8 ablation: its first
    three channels encode signed residuals in [0, 1], while the stable channel
    and all trust/correction signals remain symmetric.  Missing SAM coverage
    always produces zero trust rather than being interpreted as stability.
    """

    def __init__(self, boundary_tolerance=2, trust_gamma=1.0,
                 directional_restore=False):
        super().__init__()
        self.boundary_tolerance = int(boundary_tolerance)
        self.trust_gamma = float(trust_gamma)
        self.directional_restore = bool(directional_restore)
        if self.boundary_tolerance < 0 or self.trust_gamma <= 0:
            raise ValueError("boundary_tolerance must be non-negative and trust_gamma positive")

    @staticmethod
    def _as_float(value):
        return value.float().clamp(0, 1)

    def _boundary_residual(self, t1, t2):
        b1 = self._as_float(t1["boundary"])
        b2 = self._as_float(t2["boundary"])
        if self.boundary_tolerance > 0:
            kernel = 2 * self.boundary_tolerance + 1
            matched_b1 = F.max_pool2d(
                b1, kernel, stride=1, padding=self.boundary_tolerance,
            )
            matched_b2 = F.max_pool2d(
                b2, kernel, stride=1, padding=self.boundary_tolerance,
            )
        else:
            matched_b1, matched_b2 = b1, b2
        plus = b2 * (1.0 - matched_b1)
        minus = b1 * (1.0 - matched_b2)
        delta = torch.maximum(plus, minus).clamp(0, 1)
        signed = (plus - minus).clamp(-1, 1)
        return delta, signed

    def forward(self, structure):
        t1, t2 = structure["t1"], structure["t2"]
        q1 = self._as_float(t1["quality"])
        q2 = self._as_float(t2["quality"])

        boundary, signed_boundary = self._boundary_residual(t1, t2)
        deltas = {"boundary": boundary}
        signed = {"boundary": signed_boundary}
        for key in (
            "occupancy", "affinity_r1", "affinity_r2", "affinity_r4",
            "interior_depth", "object_scale", "compactness",
        ):
            left = self._as_float(t1[key])
            right = self._as_float(t2[key])
            deltas[key] = torch.abs(right - left).clamp(0, 1)
            signed[key] = (right - left).clamp(-1, 1)

        local = (
            0.55 * deltas["affinity_r1"]
            + 0.45 * deltas["affinity_r2"]
        ).clamp(0, 1)
        geometry = (
            0.25 * deltas["affinity_r4"]
            + 0.25 * deltas["interior_depth"]
            + 0.30 * deltas["object_scale"]
            + 0.20 * deltas["compactness"]
        ).clamp(0, 1)
        common = torch.minimum(
            self._as_float(t1["occupancy"]),
            self._as_float(t2["occupancy"]),
        )
        mean_delta = torch.stack(tuple(deltas.values()), dim=1).mean(dim=1)
        stable = (common * (1.0 - mean_delta)).clamp(0, 1)

        symmetric_change = torch.cat((boundary, local, geometry), dim=1)
        # Semantic disagreement is a continuous confidence penalty, not a
        # hard pixel filter.  It is exchange invariant and bounded in [0, 1].
        semantic_mean = symmetric_change.mean(dim=1, keepdim=True)
        uncertainty = 1.5 * torch.abs(
            symmetric_change - semantic_mean,
        ).mean(dim=1, keepdim=True)
        uncertainty = F.avg_pool2d(
            uncertainty.clamp(0, 1), 3, stride=1, padding=1,
        ).clamp(0, 1)

        coverage_any = ((q1 > 0) | (q2 > 0)).float()
        coverage_both = ((q1 > 0) & (q2 > 0)).float()
        trust_change = (
            torch.maximum(q1, q2).pow(self.trust_gamma)
            * coverage_any * (1.0 - uncertainty)
        ).clamp(0, 1)
        trust_stable = (
            torch.sqrt((q1 * q2).clamp_min(0))
            * coverage_both * (1.0 - uncertainty)
        ).clamp(0, 1)

        signed_local = (
            0.55 * signed["affinity_r1"]
            + 0.45 * signed["affinity_r2"]
        ).clamp(-1, 1)
        signed_geometry = (
            0.25 * signed["affinity_r4"]
            + 0.25 * signed["interior_depth"]
            + 0.30 * signed["object_scale"]
            + 0.20 * signed["compactness"]
        ).clamp(-1, 1)
        # Expose the raw odd teacher for Run3 without changing the R2-R8
        # four-channel code.  Every channel is anti-equivariant under T1/T2
        # exchange and remains centered in [-1, 1].
        signed_structural_code = torch.cat(
            (signed_boundary, signed_local, signed_geometry), dim=1,
        )
        if self.directional_restore:
            change_code = (0.5 + 0.5 * signed_structural_code).clamp(0, 1)
        else:
            change_code = symmetric_change
        structural_code = torch.cat((change_code, stable), dim=1)

        instance_stride = int(torch.maximum(
            t1["instance_id"].max(), t2["instance_id"].max(),
        ).item()) + 1
        return {
            "structural_code": structural_code.to(torch.float16),
            "signed_structural_code": signed_structural_code.to(torch.float16),
            "boundary_residual": boundary,
            "local_residual": local,
            "geometry_residual": geometry,
            "stable_consensus": stable,
            "change_strength": symmetric_change.amax(dim=1, keepdim=True),
            "trust_change": trust_change,
            "trust_stable": trust_stable,
            "coverage_change": coverage_any,
            "coverage_stable": coverage_both,
            "uncertainty": uncertainty,
            "stage_channel_weights": EIR_STAGE_CHANNEL_WEIGHTS,
            "directional_restore": self.directional_restore,
            "instance_stride": max(instance_stride, 1),
            "structure": structure,
        }
