"""Unchanged student with a removable, loss-only direction-C auxiliary."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.lwganet import LWGANet_L0_1242_e32_k11_GELU
from .decoder.a2net_decoder import (
    Decoder,
    NeighborFeatureAggregation,
    TemporalFusionModule,
)


class A2Net_LWGANet_L0(nn.Module):
    """
    A2Net-LWGANet-L0 change-detection student.

    Deploy graph is ALWAYS:

        backbone
            -> NeighborFeatureAggregation (SWA)
            -> TemporalFusionModule (TFM)
            -> Decoder

    All Direction-C / dynamic-teacher components are TRAINING-ONLY,
    loss-only auxiliaries.

    They are never injected into the deploy feature path.

    Supported training mechanisms
    -----------------------------
    auxiliary_mode="none":
        Clean baseline.

    auxiliary_mode="direction_c", mechanism="dynamic_teacher":
        RDT-CD reciprocal dynamic teachers implemented in
        models/distill/dynamic_teacher.py.

    Deployment contract
    -------------------
    switch_to_deploy() physically removes training_auxiliary and leaves
    exactly the unchanged student graph.
    """

    def __init__(
        self,
        pretrained=True,
        pretrained_path=None,
        auxiliary_mode="none",
        routing_cfg=None,
    ):
        super().__init__()

        if auxiliary_mode not in {
            "none",
            "direction_c",
        }:
            raise ValueError(
                "Only none / direction_c are supported; "
                "old auxiliary modes were removed"
            )

        # ------------------------------------------------------------------
        # Unchanged deployable student
        # ------------------------------------------------------------------
        self.backbone = LWGANet_L0_1242_e32_k11_GELU(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )

        self.mid_d = 64

        self.swa = NeighborFeatureAggregation(
            [32, 32, 64, 128, 256],
            self.mid_d,
        )

        self.tfm = TemporalFusionModule(
            self.mid_d,
            self.mid_d,
        )

        self.decoder = Decoder(
            self.mid_d,
        )

        # ------------------------------------------------------------------
        # Training-only auxiliary
        # ------------------------------------------------------------------
        self.auxiliary_mode = auxiliary_mode

        # Explicitly record the selected training mechanism.
        #
        # This is metadata/control state only. It does not enter the main
        # prediction graph.
        self.training_mechanism = "none"

        # Whether forward(..., compute_auxiliary=True) requires an actual
        # SAMStruct + OVCDistill teacher_pack.
        #
        # dynamic_teacher + cache_conditioning=True:  True
        # dynamic_teacher + cache_conditioning=False: False
        #
        # The False case is the capacity-matched no-cache ablation (D2).
        self.training_auxiliary_requires_cache = False

        if auxiliary_mode == "direction_c":
            cfg = dict(routing_cfg or {})

            mechanism = cfg.pop(
                "mechanism",
                "legacy",
            )

            self.training_mechanism = str(
                mechanism
            )

            # --------------------------------------------------------------
            # RDT-CD:
            # Reciprocal Dynamic Teachers.
            #
            # Cache values are priors rather than immutable final teacher
            # answers. Fast teacher experts are optimized online and EMA
            # target teachers generate the proposals seen by the student.
            # --------------------------------------------------------------
            if mechanism == "dynamic_teacher":
                from .distill.dynamic_teacher import (
                    DynamicTeacherDirectionC as DirectionC,
                )

                # Full RDT-CD requires both SAMStruct and OVCDistill.
                #
                # For the no-cache capacity control (D2):
                #
                #     cache_conditioning=False
                #
                # dynamic_teacher.py deliberately supports teacher_pack=None.
                self.training_auxiliary_requires_cache = bool(
                    cfg.get(
                        "cache_conditioning",
                        True,
                    )
                )

            else:
                raise ValueError(
                    "Unknown training mechanism: "
                    + str(mechanism)
                )

            # --------------------------------------------------------------
            # IMPORTANT REPRODUCIBILITY CONTRACT
            # --------------------------------------------------------------
            #
            # Auxiliary parameter initialization must not consume the global
            # torch CPU RNG stream.
            #
            # This keeps clean baseline and auxiliary experiments identical
            # under the same seed with respect to subsequent random
            # input/data-augmentation behavior.
            #
            # The auxiliary is initialized on CPU during model construction;
            # therefore fork_rng(devices=[]) is sufficient here.
            # --------------------------------------------------------------
            with torch.random.fork_rng(
                devices=[]
            ):
                self.training_auxiliary = DirectionC(
                    self.mid_d,
                    **cfg,
                )

    # ----------------------------------------------------------------------
    # Training-auxiliary state
    # ----------------------------------------------------------------------

    @property
    def use_training_auxiliary(self):
        """
        True only while a removable training auxiliary physically exists.
        """
        return (
            self.auxiliary_mode != "none"
            and hasattr(
                self,
                "training_auxiliary",
            )
        )

    # ----------------------------------------------------------------------
    # Unchanged deployable main path
    # ----------------------------------------------------------------------

    def extract_pair_features(
        self,
        x1,
        x2,
    ):
        """
        Siamese backbone extraction.

        This function is part of the deployable student and contains no
        teacher information.
        """
        return (
            tuple(self.backbone(x1)),
            tuple(self.backbone(x2)),
        )

    def _forward_main_path(
        self,
        features1,
        features2,
        output_size,
        return_decoder_features=False,
    ):
        """
        Execute the unchanged deploy graph.

        Teacher/cache tensors never enter this function.
        """
        aggregated1 = self.swa(
            *features1
        )

        aggregated2 = self.swa(
            *features2
        )

        change = self.tfm(
            *aggregated1,
            *aggregated2,
        )

        decoder_features_and_logits = self.decoder(
            *change
        )

        # Decoder contract:
        #
        #   first four outputs:
        #       p2 / p3 / p4 / p5 decoder features
        #
        #   remaining outputs:
        #       four-scale prediction logits
        decoder_features = tuple(
            decoder_features_and_logits[:4]
        )

        raw_logits = decoder_features_and_logits[
            4:
        ]

        predictions = tuple(
            torch.sigmoid(
                F.interpolate(
                    logits,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            for logits in raw_logits
        )

        if return_decoder_features:
            return (
                predictions,
                decoder_features,
            )

        return predictions

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        x1,
        x2,
        target=None,
        teacher_pack=None,
        compute_auxiliary=True,
        force_action=None,
    ):
        """
        Forward pass.

        Main-output invariance contract
        -------------------------------
        The main predictions are computed BEFORE the training auxiliary and
        are never modified by auxiliary outputs.

        Consequently:

            compute_auxiliary=False

        and:

            compute_auxiliary=True

        must produce bit-identical main predictions when the main modules are
        in the same deterministic state.

        Evaluation/deployment
        ---------------------
        If self.training == False, only the prediction tuple is returned.

        Training
        --------
        Returns:

            predictions, auxiliary

        where auxiliary["direction_c"] contains the selected training
        mechanism's loss/diagnostic dictionary.
        """

        # ------------------------------------------------------------------
        # 1. Student encoder
        # ------------------------------------------------------------------
        features1, features2 = (
            self.extract_pair_features(
                x1,
                x2,
            )
        )

        # Auxiliary is computed only during an actual training forward.
        need_aux = (
            self.training
            and compute_auxiliary
            and self.use_training_auxiliary
        )

        # ------------------------------------------------------------------
        # 2. Unchanged main prediction path
        #
        # Decoder features are exposed only when the auxiliary needs them.
        # Their exposure does not change the main computation itself.
        # ------------------------------------------------------------------
        output = self._forward_main_path(
            features1,
            features2,
            x1.shape[-2:],
            return_decoder_features=need_aux,
        )

        if need_aux:
            predictions, decoder_features = (
                output
            )
        else:
            predictions = output
            decoder_features = None

        # Inference/evaluation graph ends here.
        if not self.training:
            return predictions

        # ------------------------------------------------------------------
        # 3. Training-only loss auxiliary
        # ------------------------------------------------------------------
        auxiliary = {}

        if need_aux:
            if target is None:
                raise ValueError(
                    "Direction C requires training GT "
                    "when compute_auxiliary=True"
                )

            if (
                self.training_auxiliary_requires_cache
                and teacher_pack is None
            ):
                raise ValueError(
                    f"{self.training_mechanism} requires "
                    "both synchronously replayed teacher caches"
                )

            # --------------------------------------------------------------
            # IMPORTANT:
            #
            # Only decoder_features[0] is OBSERVED by the training auxiliary.
            #
            # Nothing from training_auxiliary is fed back into:
            #   backbone
            #   SWA
            #   TFM
            #   Decoder
            #   main predictions
            #
            # The auxiliary can influence student parameters ONLY through its
            # loss during backward().
            # --------------------------------------------------------------
            auxiliary["direction_c"] = (
                self.training_auxiliary(
                    decoder_features[0],
                    predictions[0],
                    target,
                    teacher_pack,
                    force_action=force_action,
                )
            )

        return (
            predictions,
            auxiliary,
        )

    # ----------------------------------------------------------------------
    # Deployment
    # ----------------------------------------------------------------------

    def switch_to_deploy(self):
        """
        Idempotently remove every training-only auxiliary component.

        After this operation the model contains only:

            backbone
                -> SWA
                -> TFM
                -> Decoder

        No SAMStruct/OVCDistill cache, router, dynamic teacher, fast teacher,
        EMA target teacher, or other training-only module remains registered.

        Calling this function more than once is safe.
        """

        if hasattr(
            self,
            "training_auxiliary",
        ):
            del self.training_auxiliary

        self.auxiliary_mode = "none"
        self.training_mechanism = "none"
        self.training_auxiliary_requires_cache = False

        return self