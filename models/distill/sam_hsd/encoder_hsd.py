"""Stage-specific directional Encoder-HSD."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    signed_structural_code_loss,
    structural_code_loss,
    weighted_mean,
)


class EncoderHSD(nn.Module):
    """Four feature probes plus one direction-shared 16->1 prediction head."""

    def __init__(self, channels=(32, 64, 128, 256), probe_channels=16,
                 stage_weights=(0.35, 0.30, 0.20, 0.15), softplus_beta=4.0):
        super().__init__()
        self.probes = nn.ModuleList(
            nn.Conv2d(channel, probe_channels, 1, bias=False) for channel in channels
        )
        self.direction_head = nn.Conv2d(probe_channels, 1, 1, bias=True)
        self.stage_weights = tuple(float(weight) for weight in stage_weights)
        self.softplus_beta = float(softplus_beta)

    @staticmethod
    def _resize(value, size, mode="bilinear"):
        value = value.float()
        if mode == "nearest":
            return F.interpolate(value, size=size, mode=mode)
        return F.interpolate(value, size=size, mode=mode, align_corners=False)

    def forward(self, features1, features2, evidence):
        if not (len(features1) == len(features2) == len(self.probes) == 4):
            raise ValueError("Encoder-HSD requires four matching feature stages")
        if len(evidence["stage_targets"]) != 4:
            raise ValueError("PI-DTRS must provide four hierarchical targets")

        total = features1[0].new_zeros(())
        details = {}
        for index, (feature1, feature2, probe, stage_weight, teacher) in enumerate(
            zip(
                features1, features2, self.probes, self.stage_weights,
                evidence["stage_targets"],
            ),
            start=1,
        ):
            z1 = probe(feature1)
            z2 = probe(feature2)
            if evidence.get("encoder_unsigned", False):
                directional = torch.abs(z2 - z1)
                prediction_plus = prediction_minus = torch.sigmoid(
                    self.direction_head(directional)
                )
            else:
                beta = self.softplus_beta
                d_plus = F.softplus(beta * (z2 - z1)) / beta
                d_minus = F.softplus(beta * (z1 - z2)) / beta
                prediction_plus = torch.sigmoid(self.direction_head(d_plus))
                prediction_minus = torch.sigmoid(self.direction_head(d_minus))
            prediction_stable = (
                F.cosine_similarity(z1, z2, dim=1, eps=1e-6).unsqueeze(1) + 1.0
            ) * 0.5

            size = feature1.shape[-2:]
            targets = {
                name: self._resize(teacher[name], size)
                for name in ("plus", "minus", "stable")
            }
            small = self._resize(evidence["small_object"], size, mode="nearest")
            small_balance = 1.0 + 0.5 * small
            confidences = {
                "plus": self._resize(evidence["plus_conf"], size),
                "minus": self._resize(evidence["minus_conf"], size),
                "stable": self._resize(evidence["stable_conf"], size),
            }

            stage_loss = feature1.new_zeros(())
            for name, prediction in (
                ("plus", prediction_plus),
                ("minus", prediction_minus),
                ("stable", prediction_stable),
            ):
                weight = (confidences[name] * small_balance).detach()
                loss_map = F.smooth_l1_loss(
                    prediction.float(), targets[name].detach(), reduction="none",
                )
                component_loss = weighted_mean(loss_map, weight)
                stage_loss = stage_loss + component_loss
                details[f"stage{index}_{name}"] = component_loss
            total = total + stage_weight * stage_loss
            details[f"stage{index}"] = stage_loss
        return {"total": total, **details}


class SpatialResidualProbe(nn.Module):
    """Detachable local structure adapter used by EIR-HSD encoder stages."""

    def __init__(self, in_channels, hidden=16, spatial=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.spatial = bool(spatial)
        if self.spatial:
            self.depthwise = nn.Conv2d(
                hidden, hidden, 3, padding=1, groups=hidden, bias=False,
            )
            self.pointwise = nn.Conv2d(hidden, hidden, 1, bias=False)

    def forward(self, feature):
        projected = self.proj(feature)
        if not self.spatial:
            return projected
        residual = self.pointwise(F.gelu(self.depthwise(projected)))
        return projected + residual


class ExchangeInvariantEncoderHSD(nn.Module):
    """Predict one shared four-channel structural code at four encoder stages."""

    def __init__(self, channels=(32, 64, 128, 256), hidden=16,
                 spatial_adapter=True,
                 stage_weights=(0.35, 0.30, 0.20, 0.15),
                 directional_restore=False):
        super().__init__()
        self.probes = nn.ModuleList(
            SpatialResidualProbe(channel, hidden, spatial=spatial_adapter)
            for channel in channels
        )
        self.structure_head = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 4, 1, bias=True),
        )
        self.stage_weights = tuple(float(weight) for weight in stage_weights)
        self.directional_restore = bool(directional_restore)

    def _pair_representation(self, left, right):
        residual = right - left
        if not self.directional_restore:
            residual = torch.abs(residual)
        common = left * right
        return torch.cat((residual, common), dim=1)

    def forward(self, features1, features2, evidence):
        if not (len(features1) == len(features2) == len(self.probes) == 4):
            raise ValueError("EIR Encoder-HSD requires four matching feature stages")
        target = evidence["structural_code"]
        channel_routes = evidence["stage_channel_weights"]
        if len(channel_routes) != 4:
            raise ValueError("EIR-HSD requires four stage channel routes")

        total = features1[0].new_zeros(())
        details = {}
        for index, (feature1, feature2, probe, stage_weight, route) in enumerate(
            zip(
                features1, features2, self.probes,
                self.stage_weights, channel_routes,
            ),
            start=1,
        ):
            z1, z2 = probe(feature1), probe(feature2)
            prediction = torch.sigmoid(
                self.structure_head(self._pair_representation(z1, z2))
            )
            stage_loss, components = structural_code_loss(
                prediction, target,
                evidence["trust_change"], evidence["trust_stable"],
                channel_weights=route,
            )
            total = total + stage_weight * stage_loss
            details[f"stage{index}"] = stage_loss
            for name, value in components.items():
                details[f"stage{index}_{name}"] = value
        return {"total": total, **details}


class Z2PairProjector(nn.Module):
    """Project an ordered pair onto the even/odd irreducible Z2 subspaces."""

    def __init__(self, channels=16):
        super().__init__()
        self.ordered_encoder = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1, bias=True),
            nn.GELU(),
        )

    @staticmethod
    def _ordered_pair(left, right):
        return torch.cat((left, right, right - left, left * right), dim=1)

    def forward(self, left, right):
        forward = self.ordered_encoder(self._ordered_pair(left, right))
        reverse = self.ordered_encoder(self._ordered_pair(right, left))
        even = 0.5 * (forward + reverse)
        odd = 0.5 * (forward - reverse)
        return even, odd


class Z2StructuralEncoderHSD(nn.Module):
    """Training-only Z2 even/odd structural residual distillation.

    ``pair_mode='group'`` is the proposed method.  ``invariant`` reproduces
    the R2 abs/product representation, while ``mixed`` reproduces the R8-style
    signed channels in one sigmoid head.  These modes are mutually exclusive
    so each Run3 recipe changes exactly one representational assumption.
    """

    PAIR_MODES = {"group", "invariant", "mixed"}

    def __init__(self, channels=(32, 64, 128, 256), hidden=16,
                 spatial_adapter=False,
                 stage_weights=(0.35, 0.30, 0.20, 0.15),
                 pair_mode="group", use_even=True, use_odd=True,
                 odd_weight=1.0):
        super().__init__()
        if pair_mode not in self.PAIR_MODES:
            raise ValueError(f"Unsupported Z2 pair mode: {pair_mode}")
        if odd_weight < 0:
            raise ValueError("odd_weight must be non-negative")
        if pair_mode == "group" and not (use_even or use_odd):
            raise ValueError("The group projector requires at least one active head")
        if pair_mode != "group" and (not use_even or use_odd):
            raise ValueError(
                "Invariant/mixed controls require use_even=True and use_odd=False"
            )
        self.probes = nn.ModuleList(
            SpatialResidualProbe(channel, hidden, spatial=spatial_adapter)
            for channel in channels
        )
        self.stage_weights = tuple(float(weight) for weight in stage_weights)
        self.pair_mode = pair_mode
        self.use_even = bool(use_even)
        self.use_odd = bool(use_odd)
        self.odd_weight = float(odd_weight)
        if pair_mode == "group":
            self.pair_projector = Z2PairProjector(hidden)
            if self.use_even:
                self.even_head = nn.Conv2d(hidden, 4, 1, bias=True)
            if self.use_odd:
                # A bias would violate f(-h)=-f(h) after tanh, so the odd
                # representation head must remain strictly bias-free.
                self.odd_head = nn.Conv2d(hidden, 3, 1, bias=False)
        else:
            self.control_head = nn.Sequential(
                nn.Conv2d(2 * hidden, hidden, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden, 4, 1, bias=True),
            )

    @staticmethod
    def _control_pair(left, right, signed):
        residual = right - left
        if not signed:
            residual = torch.abs(residual)
        return torch.cat((residual, left * right), dim=1)

    @staticmethod
    def _latent_diagnostics(even, odd):
        even_vector = even.float().flatten(2).mean(dim=2)
        odd_vector = odd.float().flatten(2).mean(dim=2)
        cosine = F.cosine_similarity(
            even_vector, odd_vector, dim=1, eps=1e-6,
        ).abs().mean()
        return even.abs().mean(), odd.abs().mean(), cosine

    def forward(self, features1, features2, evidence):
        if not (len(features1) == len(features2) == len(self.probes) == 4):
            raise ValueError("Z2-SRD requires four matching feature stages")
        channel_routes = evidence["stage_channel_weights"]
        if len(channel_routes) != 4:
            raise ValueError("Z2-SRD requires four stage channel routes")

        zero = features1[0].new_zeros(())
        total = zero
        even_total = zero
        odd_total = zero
        even_feature_total = zero
        odd_feature_total = zero
        cross_total = zero
        details = {}
        for index, (feature1, feature2, probe, stage_weight, route) in enumerate(
            zip(
                features1, features2, self.probes,
                self.stage_weights, channel_routes,
            ),
            start=1,
        ):
            z1, z2 = probe(feature1), probe(feature2)
            stage_even = zero
            stage_odd = zero
            if self.pair_mode == "group":
                even_feature, odd_feature = self.pair_projector(z1, z2)
                even_abs, odd_abs, cross = self._latent_diagnostics(
                    even_feature, odd_feature,
                )
                even_feature_total = even_feature_total + stage_weight * even_abs
                odd_feature_total = odd_feature_total + stage_weight * odd_abs
                cross_total = cross_total + stage_weight * cross
                if self.use_even:
                    even_prediction = torch.sigmoid(self.even_head(even_feature))
                    stage_even, components = structural_code_loss(
                        even_prediction, evidence["structural_code"],
                        evidence["trust_change"], evidence["trust_stable"],
                        channel_weights=route,
                    )
                    for name, value in components.items():
                        details[f"stage{index}_even_{name}"] = value
                if self.use_odd:
                    odd_prediction = torch.tanh(self.odd_head(odd_feature))
                    stage_odd, components = signed_structural_code_loss(
                        odd_prediction, evidence["signed_structural_code"],
                        evidence["trust_change"], channel_weights=route[:3],
                    )
                    for name, value in components.items():
                        details[f"stage{index}_odd_{name}"] = value
                stage_loss = stage_even + self.odd_weight * stage_odd
            else:
                pair = self._control_pair(
                    z1, z2, signed=self.pair_mode == "mixed",
                )
                prediction = torch.sigmoid(self.control_head(pair))
                stage_even, components = structural_code_loss(
                    prediction, evidence["structural_code"],
                    evidence["trust_change"], evidence["trust_stable"],
                    channel_weights=route,
                )
                stage_loss = stage_even
                for name, value in components.items():
                    details[f"stage{index}_control_{name}"] = value

            total = total + stage_weight * stage_loss
            even_total = even_total + stage_weight * stage_even
            odd_total = odd_total + stage_weight * stage_odd
            details[f"stage{index}"] = stage_loss
            details[f"stage{index}_even"] = stage_even
            details[f"stage{index}_odd"] = stage_odd
        return {
            "total": total,
            "even": even_total,
            "odd": odd_total,
            "even_feature": even_feature_total,
            "odd_feature": odd_feature_total,
            "even_odd_cosine": cross_total,
            **details,
        }
