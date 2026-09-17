"""A2Net-LWGANet-L0 with a removable Run3 BT-SAM-RDT training auxiliary."""

from __future__ import annotations

from typing import Any, Dict, Optional

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
    A2Net-LWGANet-L0 binary change-detection student.

    Deploy graph
    ------------
    The deployable student is always exactly:

        backbone
            -> NeighborFeatureAggregation (SWA)
            -> TemporalFusionModule (TFM)
            -> Decoder

    Run3
    ----
    auxiliary_mode="bt_sam_rdt" enables the TRAINING-ONLY BT-SAM-RDT
    auxiliary implemented in ``models/distill/dynamic_teacher.py``.

    The auxiliary may observe:
        - one decoder feature map;
        - one main student prediction;
        - binary GT;
        - synchronously replayed SAMStruct + OVCDistill teacher caches.

    It must never inject teacher/cache tensors into the deploy feature path.
    Its only influence on the student is through an auxiliary loss during
    backward propagation.

    Deployment contract
    -------------------
    ``switch_to_deploy()`` physically removes the complete training auxiliary.
    After removal, the registered model graph is only the unchanged student.
    """

    _SUPPORTED_AUXILIARY_MODES = {
        "none",
        "bt_sam_rdt",
    }

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        auxiliary_mode: str = "none",
        auxiliary_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        auxiliary_mode = str(auxiliary_mode)

        if auxiliary_mode not in self._SUPPORTED_AUXILIARY_MODES:
            raise ValueError(
                "auxiliary_mode must be one of "
                f"{sorted(self._SUPPORTED_AUXILIARY_MODES)}, "
                f"got {auxiliary_mode!r}"
            )

        # ------------------------------------------------------------------
        # 1. Unchanged deployable student
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
        # 2. Training-only Run3 auxiliary
        # ------------------------------------------------------------------
        self.auxiliary_mode = auxiliary_mode
        self.training_mechanism = "none"
        self.training_auxiliary_requires_cache = False

        if auxiliary_mode == "bt_sam_rdt":
            # Lazy import:
            # the clean/deploy student does not depend on the Run3 auxiliary
            # unless the auxiliary is explicitly enabled.
            from .distill.dynamic_teacher import BTSAMRDT

            self.training_mechanism = "bt_sam_rdt"

            # Run3 uses synchronously replayed SAMStruct + OVCDistill caches.
            #
            # Teacher ablations such as SAM-only are handled inside BTSAMRDT;
            # the outer model/cache-loading contract remains unchanged so that
            # all experiments use the same data pipeline.
            self.training_auxiliary_requires_cache = True

            cfg = dict(auxiliary_cfg or {})

            # --------------------------------------------------------------
            # Reproducibility contract
            # --------------------------------------------------------------
            # Initializing training-only parameters must not advance the
            # global CPU RNG stream.
            #
            # This keeps B0 and Run3 identical under the same random seed with
            # respect to subsequent stochastic data augmentation / sampling.
            #
            # The auxiliary is created on CPU during model construction, so
            # preserving the CPU RNG stream is sufficient here.
            # --------------------------------------------------------------
            with torch.random.fork_rng(
                devices=[],
            ):
                self.training_auxiliary = BTSAMRDT(
                    channels=self.mid_d,
                    **cfg,
                )

    # ----------------------------------------------------------------------
    # Training auxiliary state
    # ----------------------------------------------------------------------

    @property
    def use_training_auxiliary(self) -> bool:
        """
        True only while the removable Run3 training auxiliary exists.
        """
        return (
            self.auxiliary_mode != "none"
            and hasattr(
                self,
                "training_auxiliary",
            )
        )

    # ----------------------------------------------------------------------
    # Unchanged deployable student
    # ----------------------------------------------------------------------

    def extract_pair_features(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ):
        """
        Siamese backbone feature extraction.

        No teacher/cache information is accepted by this function.
        """
        return (
            tuple(
                self.backbone(x1)
            ),
            tuple(
                self.backbone(x2)
            ),
        )

    def _forward_main_path(
        self,
        features1,
        features2,
        output_size,
        return_decoder_features: bool = False,
    ):
        """
        Execute the unchanged deployable A2Net main path.

        Teacher/cache tensors must never enter this function.
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

        # Decoder output contract:
        #
        # first four outputs:
        #     p2 / p3 / p4 / p5 decoder feature maps
        #
        # remaining outputs:
        #     four-scale prediction logits
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
        x1: torch.Tensor,
        x2: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_pack: Optional[Dict[str, Any]] = None,
        compute_auxiliary: bool = True,
    ):
        """
        Forward pass.

        Main-output invariance
        ----------------------
        Main predictions are computed before the Run3 training auxiliary and
        are never modified by auxiliary outputs.

        Therefore, when all deployable modules are in the same deterministic
        state:

            compute_auxiliary=False

        and:

            compute_auxiliary=True

        must produce identical main predictions.

        Evaluation / deployment
        -----------------------
        In ``eval()`` mode only the main prediction tuple is returned.

        Training
        --------
        Returns:

            predictions, auxiliary

        For Run3:

            auxiliary["bt_sam_rdt"]

        contains the student KD loss, fast-teacher fitting loss and diagnostic
        quantities produced by the training-only BT-SAM-RDT auxiliary.
        """

        # ------------------------------------------------------------------
        # 1. Student Siamese encoder
        # ------------------------------------------------------------------
        features1, features2 = self.extract_pair_features(
            x1,
            x2,
        )

        # Auxiliary computation is allowed only during training.
        need_aux = (
            self.training
            and bool(
                compute_auxiliary
            )
            and self.use_training_auxiliary
        )

        # ------------------------------------------------------------------
        # 2. Unchanged A2Net main prediction path
        #
        # Decoder features are exposed only when the Run3 auxiliary needs
        # them. Exposing them does not alter the main-path computation.
        # ------------------------------------------------------------------
        output = self._forward_main_path(
            features1,
            features2,
            x1.shape[-2:],
            return_decoder_features=need_aux,
        )

        if need_aux:
            (
                predictions,
                decoder_features,
            ) = output

        else:
            predictions = output
            decoder_features = None

        # Evaluation / deployment graph ends here.
        if not self.training:
            return predictions

        # ------------------------------------------------------------------
        # 3. Training-only Run3 auxiliary
        # ------------------------------------------------------------------
        auxiliary: Dict[str, Any] = {}

        if need_aux:
            if target is None:
                raise ValueError(
                    "BT-SAM-RDT requires binary training GT when "
                    "compute_auxiliary=True"
                )

            if (
                self.training_auxiliary_requires_cache
                and teacher_pack is None
            ):
                raise ValueError(
                    "BT-SAM-RDT requires a synchronously replayed "
                    "SAMStruct + OVCDistill teacher_pack"
                )

            # --------------------------------------------------------------
            # Critical deployment / gradient-isolation contract
            # --------------------------------------------------------------
            #
            # Only decoder_features[0] and predictions[0] are OBSERVED by the
            # training auxiliary.
            #
            # Nothing produced by BTSAMRDT is fed back into:
            #
            #     backbone
            #     SWA
            #     TFM
            #     Decoder
            #     main predictions
            #
            # Teacher information therefore cannot alter the forward main
            # prediction graph.
            #
            # The auxiliary may influence deployable student parameters only
            # through its student-side loss during backward().
            #
            # Student/teacher detach boundaries themselves are implemented
            # inside models/distill/dynamic_teacher.py.
            # --------------------------------------------------------------
            auxiliary[
                "bt_sam_rdt"
            ] = self.training_auxiliary(
                feature=decoder_features[0],
                prediction=predictions[0],
                target=target,
                teacher_pack=teacher_pack,
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
        Physically remove every Run3 training-only component.

        After this operation the registered model contains only:

            backbone
                -> SWA
                -> TFM
                -> Decoder

        No component belonging to:

            SAMStruct cache processing
            OVCDistill cache processing
            bi-temporal SAM instance matching
            SAM/OV prior fusion
            Fast Teacher
            EMA Target Teacher
            GT task-space audit
            auxiliary diagnostics

        remains registered in the model.

        Calling this function repeatedly is safe.
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