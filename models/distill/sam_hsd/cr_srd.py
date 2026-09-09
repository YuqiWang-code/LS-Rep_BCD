"""Coherence-routed structural residual distillation for Run4."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .encoder_hsd import SpatialResidualProbe
from .losses import coherence_routed_structural_loss
from .relational_structure import RelationalStructureBuilder
from .temporal_evidence import ExchangeInvariantStructuralEvidence


class CRStructuralEncoderHSD(nn.Module):
    """N2-identical shared mixed head with parameter-free loss routing."""

    def __init__(self, channels=(32, 64, 128, 256), hidden=16,
                 spatial_adapter=False,
                 stage_weights=(0.35, 0.30, 0.20, 0.15),
                 use_coherence_gate=True, use_magnitude_fallback=True):
        super().__init__()
        self.probes = nn.ModuleList(
            SpatialResidualProbe(channel, hidden, spatial=spatial_adapter)
            for channel in channels
        )
        # Names and shapes intentionally match the Run3 N2 encoder so a strict
        # state-dict copy can prove gate-off numerical equivalence.
        self.control_head = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 4, 1, bias=True),
        )
        self.stage_weights = tuple(float(weight) for weight in stage_weights)
        self.use_coherence_gate = bool(use_coherence_gate)
        self.use_magnitude_fallback = bool(use_magnitude_fallback)

    @staticmethod
    def _mixed_pair(left, right):
        return torch.cat((right - left, left * right), dim=1)

    def forward(self, features1, features2, evidence):
        if not (len(features1) == len(features2) == len(self.probes) == 4):
            raise ValueError("CR-SRD requires four matching feature stages")
        routes = evidence["stage_channel_weights"]
        if len(routes) != 4:
            raise ValueError("CR-SRD requires four stage channel routes")

        zero = features1[0].new_zeros(())
        totals = {"total": zero, "direction": zero, "magnitude": zero, "stable": zero}
        details = {}
        for index, (left, right, probe, stage_weight, route) in enumerate(
            zip(features1, features2, self.probes, self.stage_weights, routes),
            start=1,
        ):
            z1, z2 = probe(left), probe(right)
            prediction = torch.sigmoid(self.control_head(self._mixed_pair(z1, z2)))
            stage_loss, components = coherence_routed_structural_loss(
                prediction, evidence, channel_weights=route,
                use_coherence_gate=self.use_coherence_gate,
                use_magnitude_fallback=self.use_magnitude_fallback,
            )
            totals["total"] = totals["total"] + stage_weight * stage_loss
            totals["direction"] = (
                totals["direction"] + stage_weight * components["direction"]
            )
            totals["magnitude"] = (
                totals["magnitude"] + stage_weight * components["magnitude"]
            )
            totals["stable"] = (
                totals["stable"] + stage_weight * components["stable_routed"]
            )
            details[f"stage{index}"] = stage_loss
            details[f"stage{index}_direction"] = components["direction"]
            details[f"stage{index}_magnitude"] = components["magnitude"]
            details[f"stage{index}_stable"] = components["stable_routed"]
            for name in ("boundary", "local", "geometry"):
                details[f"stage{index}_{name}"] = components[name]
        return {**totals, **details}


def _coherence_diagnostics(evidence, max_quantile_samples=65536):
    """Summarize trusted nonzero-change support without a full-map sort."""
    coherence = evidence["direction_coherence"].float()
    magnitude = evidence["change_magnitude_code"].float()
    trust = evidence["trust_change"].float().expand_as(coherence)
    weight = trust * (magnitude > 1e-6).float()
    denominator = weight.sum().clamp_min(1e-6)
    mean = (coherence * weight).sum() / denominator
    low = ((coherence < 0.25).float() * weight).sum() / denominator
    high = ((coherence > 0.75).float() * weight).sum() / denominator
    mid = (1.0 - low - high).clamp(0, 1)

    flat_coherence = coherence.reshape(-1)
    flat_weight = weight.reshape(-1)
    stride = max(1, math.ceil(flat_coherence.numel() / max_quantile_samples))
    sampled = flat_coherence[::stride]
    sampled = sampled[flat_weight[::stride] > 0]
    if sampled.numel():
        quantiles = torch.quantile(
            sampled, sampled.new_tensor((0.25, 0.50, 0.75)),
        )
    else:
        quantiles = coherence.new_zeros(3, dtype=torch.float32)
    return {
        "coherence_mean": mean,
        "coherence_p25": quantiles[0],
        "coherence_p50": quantiles[1],
        "coherence_p75": quantiles[2],
        "coherence_low_frac": low,
        "coherence_mid_frac": mid,
        "coherence_high_frac": high,
    }


class CRSRDAdapter(nn.Module):
    """Training-only CR-SRD adapter; SAM2 cache remains frozen and offline."""

    def __init__(self, spatial_adapter=False, boundary_tolerance=2,
                 trust_gamma=1.0, use_coherence_gate=True,
                 use_magnitude_fallback=True):
        super().__init__()
        self.structure_builder = RelationalStructureBuilder()
        self.evidence_builder = ExchangeInvariantStructuralEvidence(
            boundary_tolerance=boundary_tolerance,
            trust_gamma=trust_gamma,
            directional_restore=True,
        )
        self.encoder = CRStructuralEncoderHSD(
            spatial_adapter=spatial_adapter,
            use_coherence_gate=use_coherence_gate,
            use_magnitude_fallback=use_magnitude_fallback,
        )

    def forward(self, features1, features2, target, teacher_pack):
        del target
        structure = self.structure_builder(teacher_pack)
        evidence = self.evidence_builder(structure)
        encoder = self.encoder(features1, features2, evidence)
        directional = evidence["structural_code"].float()
        magnitude = evidence["change_magnitude_code"].float()
        return {
            "total": encoder["total"],
            "encoder": encoder["total"],
            "cr_direction": encoder["direction"],
            "cr_magnitude": encoder["magnitude"],
            "cr_stable": encoder["stable"],
            "code_boundary_mean": directional[:, 0:1].mean(),
            "code_local_mean": directional[:, 1:2].mean(),
            "code_geometry_mean": directional[:, 2:3].mean(),
            "code_stable_mean": directional[:, 3:4].mean(),
            "magnitude_mean": magnitude.mean(),
            "trust_change_mean": evidence["trust_change"].mean(),
            "trust_stable_mean": evidence["trust_stable"].mean(),
            "coverage_change_mean": evidence["coverage_change"].mean(),
            "coverage_stable_mean": evidence["coverage_stable"].mean(),
            "structural_uncertainty_mean": evidence["uncertainty"].mean(),
            **_coherence_diagnostics(evidence),
            **{key: value for key, value in encoder.items() if key.startswith("stage")},
        }
