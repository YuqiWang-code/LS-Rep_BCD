"""Decoder feature/prediction hierarchy distillation and SCGR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import (
    SCALE_WEIGHTS,
    affinity_loss,
    gt_conditioned_relation_loss,
    multiscale_scgr_loss,
    relation_loss,
    residual_correction_loss,
    signed_boundary_loss,
    structural_code_loss,
)


class DecoderHSD(nn.Module):
    def __init__(self, use_scgr=True, boundary_band=7,
                 scgr_weight=0.35, boundary_weight=0.25,
                 relation_weight=0.25, affinity_weight=0.15):
        super().__init__()
        self.use_scgr = bool(use_scgr)
        self.boundary_band = int(boundary_band)
        self.weights = {
            "scgr": float(scgr_weight),
            "boundary": float(boundary_weight),
            "relation": float(relation_weight),
            "affinity": float(affinity_weight),
        }

    def forward(self, decoder_features, predictions, target, evidence):
        if len(decoder_features) != 4 or len(predictions) != 4:
            raise ValueError("Decoder-HSD requires four features and predictions")
        zero = target.new_zeros(())
        scgr, route_stats = multiscale_scgr_loss(predictions, target, evidence) \
            if self.use_scgr else (zero, {
                "reliable_positive_ratio": zero,
                "reliable_negative_ratio": zero,
                "hard_negative_ratio": zero,
                "hard_positive_ratio": zero,
            })
        boundary = signed_boundary_loss(
            predictions, target, evidence, self.boundary_band,
        )
        relation = relation_loss(
            predictions, decoder_features, target, evidence,
        )
        affinity = affinity_loss(predictions, target, evidence)
        total = (
            self.weights["scgr"] * scgr
            + self.weights["boundary"] * boundary["total"]
            + self.weights["relation"] * relation["total"]
            + self.weights["affinity"] * affinity
        )
        return {
            "total": total,
            "scgr": scgr,
            "boundary": boundary["total"],
            "boundary_positive": boundary["positive"],
            "boundary_negative": boundary["negative"],
            "relation": relation["total"],
            "relation_changed": relation["changed"],
            "stable_suppression": relation["stable_suppression"],
            "prediction_contrast": relation["contrast"],
            "feature_relation": relation["feature"],
            "affinity": affinity,
            **route_stats,
        }


class DecoderStructureHead(nn.Module):
    """Shared training-only structure head for the four 64-channel stages."""

    def __init__(self, in_channels=64, hidden=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden, bias=False,
        )
        self.out = nn.Conv2d(hidden, 4, 1, bias=True)

    def forward(self, feature):
        projected = self.proj(feature)
        structural_feature = projected + F.gelu(self.depthwise(projected))
        code = torch.sigmoid(self.out(structural_feature))
        return code, structural_feature


class ResidualDecoderHSD(nn.Module):
    """EIR-HSD decoder representation learning plus selective correction."""

    def __init__(self, use_correction=True, relation_weight=0.5,
                 correction_weight=0.5, fp_weight=1.3, fn_weight=1.0):
        super().__init__()
        self.structure_head = DecoderStructureHead()
        self.use_correction = bool(use_correction)
        self.relation_weight = float(relation_weight)
        self.correction_weight = float(correction_weight)
        self.fp_weight = float(fp_weight)
        self.fn_weight = float(fn_weight)

    def forward(self, decoder_features, predictions, target, evidence):
        if len(decoder_features) != 4 or len(predictions) != 4:
            raise ValueError("EIR Decoder-HSD requires four features and predictions")
        structural_total = target.new_zeros(())
        structural_features = []
        details = {}
        for index, (scale_weight, feature, route) in enumerate(
            zip(
                SCALE_WEIGHTS, decoder_features,
                evidence["stage_channel_weights"],
            ),
            start=1,
        ):
            prediction, structural_feature = self.structure_head(feature)
            structural_features.append(structural_feature)
            current, components = structural_code_loss(
                prediction, evidence["structural_code"],
                evidence["trust_change"], evidence["trust_stable"],
                channel_weights=route,
            )
            structural_total = structural_total + scale_weight * current
            details[f"stage{index}_structure"] = current
            for name, value in components.items():
                details[f"stage{index}_{name}"] = value

        relation = gt_conditioned_relation_loss(
            structural_features, target, evidence,
        )
        if self.use_correction:
            correction = residual_correction_loss(
                predictions, target, evidence,
                false_negative_weight=self.fn_weight,
                false_positive_weight=self.fp_weight,
            )
        else:
            zero = target.new_zeros(())
            correction = {
                "total": zero, "false_negative": zero,
                "false_positive": zero, "fn_ratio": zero, "fp_ratio": zero,
            }
        total = (
            structural_total
            + self.relation_weight * relation["total"]
            + self.correction_weight * correction["total"]
        )
        return {
            "total": total,
            "structure": structural_total,
            "relation": relation["total"],
            "relation_same": relation["same"],
            "relation_contrast": relation["contrast"],
            "correction": correction["total"],
            "correction_fn": correction["false_negative"],
            "correction_fp": correction["false_positive"],
            "fn_correction_ratio": correction["fn_ratio"],
            "fp_correction_ratio": correction["fp_ratio"],
            **details,
        }
