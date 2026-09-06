"""Top-level asymmetric hierarchical SAM structure distillation adapter."""

from __future__ import annotations

import torch.nn as nn

from .decoder_hsd import DecoderHSD, ResidualDecoderHSD
from .encoder_hsd import EncoderHSD, ExchangeInvariantEncoderHSD
from .relational_structure import RelationalStructureBuilder
from .temporal_evidence import (
    DirectionalTemporalEvidence,
    ExchangeInvariantStructuralEvidence,
)


class SAMHSDAdapter(nn.Module):
    """PI-DTRS + optional Encoder-HSD and Decoder-HSD, training only."""

    def __init__(self, mode="full", evidence_mode="directional",
                 use_scgr=True, robust_filter=True, boundary_band=7):
        super().__init__()
        if mode not in {"encoder", "decoder", "full"}:
            raise ValueError(f"Unsupported SAM-HSD mode: {mode}")
        self.mode = mode
        self.structure_builder = RelationalStructureBuilder()
        self.evidence_builder = DirectionalTemporalEvidence(
            mode=evidence_mode, robust_filter=robust_filter,
        )
        if mode in {"encoder", "full"}:
            self.encoder = EncoderHSD()
        if mode in {"decoder", "full"}:
            self.decoder = DecoderHSD(use_scgr=use_scgr, boundary_band=boundary_band)

    def forward(self, features1, features2, decoder_features,
                predictions, target, teacher_pack):
        structure = self.structure_builder(teacher_pack)
        evidence = self.evidence_builder(structure)
        zero = target.new_zeros(())
        encoder = self.encoder(features1, features2, evidence) \
            if hasattr(self, "encoder") else {"total": zero}
        decoder = self.decoder(decoder_features, predictions, target, evidence) \
            if hasattr(self, "decoder") else {
                "total": zero, "scgr": zero, "boundary": zero,
                "relation": zero, "affinity": zero,
                "boundary_positive": zero, "boundary_negative": zero,
                "relation_changed": zero, "stable_suppression": zero,
                "prediction_contrast": zero, "feature_relation": zero,
                "reliable_positive_ratio": zero, "reliable_negative_ratio": zero,
                "hard_negative_ratio": zero, "hard_positive_ratio": zero,
            }
        if hasattr(self, "encoder") and hasattr(self, "decoder"):
            total = 0.5 * encoder["total"] + 0.5 * decoder["total"]
        elif hasattr(self, "encoder"):
            total = encoder["total"]
        else:
            total = decoder["total"]
        return {
            "total": total,
            "encoder": encoder["total"],
            "decoder": decoder["total"],
            "scgr": decoder.get("scgr", zero),
            "boundary": decoder.get("boundary", zero),
            "boundary_positive": decoder.get("boundary_positive", zero),
            "boundary_negative": decoder.get("boundary_negative", zero),
            "relation": decoder.get("relation", zero),
            "relation_changed": decoder.get("relation_changed", zero),
            "stable_suppression": decoder.get("stable_suppression", zero),
            "prediction_contrast": decoder.get("prediction_contrast", zero),
            "feature_relation": decoder.get("feature_relation", zero),
            "affinity": decoder.get("affinity", zero),
            "e_plus_mean": evidence["plus"].mean(),
            "e_minus_mean": evidence["minus"].mean(),
            "e_stable_mean": evidence["stable"].mean(),
            "e_uncertain_mean": evidence["uncertainty"].mean(),
            "plus_conf_mean": evidence["plus_conf"].mean(),
            "minus_conf_mean": evidence["minus_conf"].mean(),
            "stable_conf_mean": evidence["stable_conf"].mean(),
            "change_conf_mean": evidence["change_conf"].mean(),
            **{key: decoder.get(key, zero) for key in (
                "reliable_positive_ratio", "reliable_negative_ratio",
                "hard_negative_ratio", "hard_positive_ratio",
            )},
        }


class EIRHSDAdapter(nn.Module):
    """Exchange-Invariant Residual Hierarchical Structure Distillation."""

    def __init__(self, mode="full", spatial_adapter=True,
                 use_correction=True, fixed_fusion=False,
                 directional_restore=False, boundary_tolerance=2,
                 trust_gamma=1.0):
        super().__init__()
        if mode not in {"encoder", "decoder", "full"}:
            raise ValueError(f"Unsupported EIR-HSD mode: {mode}")
        self.mode = mode
        self.fixed_fusion = bool(fixed_fusion)
        self.structure_builder = RelationalStructureBuilder()
        self.evidence_builder = ExchangeInvariantStructuralEvidence(
            boundary_tolerance=boundary_tolerance,
            trust_gamma=trust_gamma,
            directional_restore=directional_restore,
        )
        if mode in {"encoder", "full"}:
            self.encoder = ExchangeInvariantEncoderHSD(
                spatial_adapter=spatial_adapter,
                directional_restore=directional_restore,
            )
        if mode in {"decoder", "full"}:
            self.decoder = ResidualDecoderHSD(
                use_correction=use_correction,
            )

    def forward(self, features1, features2, decoder_features,
                predictions, target, teacher_pack):
        structure = self.structure_builder(teacher_pack)
        evidence = self.evidence_builder(structure)
        zero = target.new_zeros(())
        encoder = self.encoder(features1, features2, evidence) \
            if hasattr(self, "encoder") else {"total": zero}
        decoder = self.decoder(
            decoder_features, predictions, target, evidence,
        ) if hasattr(self, "decoder") else {
            "total": zero, "structure": zero, "relation": zero,
            "relation_same": zero, "relation_contrast": zero,
            "correction": zero, "correction_fn": zero,
            "correction_fp": zero, "fn_correction_ratio": zero,
            "fp_correction_ratio": zero,
        }

        if hasattr(self, "encoder") and hasattr(self, "decoder"):
            if self.fixed_fusion:
                total = 0.5 * encoder["total"] + 0.5 * decoder["total"]
            else:
                total = encoder["total"] + decoder["total"]
        elif hasattr(self, "encoder"):
            total = encoder["total"]
        else:
            total = decoder["total"]

        code = evidence["structural_code"].float()
        return {
            "total": total,
            "encoder": encoder["total"],
            "decoder": decoder["total"],
            "decoder_structure": decoder.get("structure", zero),
            "relation": decoder.get("relation", zero),
            "relation_same": decoder.get("relation_same", zero),
            "relation_contrast": decoder.get("relation_contrast", zero),
            "correction": decoder.get("correction", zero),
            "correction_fn": decoder.get("correction_fn", zero),
            "correction_fp": decoder.get("correction_fp", zero),
            "fn_correction_ratio": decoder.get("fn_correction_ratio", zero),
            "fp_correction_ratio": decoder.get("fp_correction_ratio", zero),
            "code_boundary_mean": code[:, 0:1].mean(),
            "code_local_mean": code[:, 1:2].mean(),
            "code_geometry_mean": code[:, 2:3].mean(),
            "code_stable_mean": code[:, 3:4].mean(),
            "trust_change_mean": evidence["trust_change"].mean(),
            "trust_stable_mean": evidence["trust_stable"].mean(),
            "coverage_change_mean": evidence["coverage_change"].mean(),
            "coverage_stable_mean": evidence["coverage_stable"].mean(),
            "structural_uncertainty_mean": evidence["uncertainty"].mean(),
        }
