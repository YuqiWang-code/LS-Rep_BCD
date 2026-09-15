"""A2Net-LWGANet-L0 with a removable RDT-CD + SCGR training auxiliary.

The deployable student graph is unchanged.  RDT-CD and SCGR are used only
while training and can influence the student only through loss/backward.
No teacher/cache tensor is injected into the deploy feature path.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

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
    """A2Net-LWGANet-L0 binary change-detection student.

    Deploy graph
    ------------
    The deploy graph is always::

        backbone
            -> NeighborFeatureAggregation (SWA)
            -> TemporalFusionModule (TFM)
            -> Decoder

    Training-only auxiliary
    -----------------------
    ``auxiliary_mode="direction_c"`` enables the current RDT-CD mechanism.
    The auxiliary observes:

    - ``decoder_features[0]``: the feature immediately before the final-scale
      classifier path used by SCGR for analytical gradient concordance;
    - ``predictions[0]``: the current student probability;
    - GT: only for teacher fitting / SCGR auditing / diagnostics;
    - synchronously replayed SAMStruct + OVCDistill cache when
      ``cache_conditioning=True``.

    The auxiliary output is never fed back into backbone/SWA/TFM/Decoder.
    Therefore the main prediction graph is identical with auxiliary ON/OFF.

    Deployment contract
    -------------------
    ``switch_to_deploy()`` physically removes the entire training auxiliary.
    After conversion, only the unchanged student remains registered.
    """

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        auxiliary_mode: str = "none",
        routing_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        if auxiliary_mode not in {"none", "direction_c"}:
            raise ValueError(
                "Only auxiliary_mode='none' or 'direction_c' is supported"
            )

        # ------------------------------------------------------------------
        # 1. Unchanged deployable student.
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
        self.decoder = Decoder(self.mid_d)

        # ------------------------------------------------------------------
        # 2. Training-only auxiliary metadata/state.
        # ------------------------------------------------------------------
        self.auxiliary_mode = str(auxiliary_mode)
        self.training_mechanism = "none"
        self.training_auxiliary_requires_cache = False

        if self.auxiliary_mode == "direction_c":
            cfg = dict(routing_cfg or {})

            # Current codebase intentionally supports only the final mechanism:
            # RDT-CD + SCGR.  Old router/teacher mechanisms are not silently
            # accepted because that would make experiment provenance ambiguous.
            mechanism = str(
                cfg.pop("mechanism", "dynamic_teacher")
            )
            if mechanism != "dynamic_teacher":
                raise ValueError(
                    "Only mechanism='dynamic_teacher' is supported by the "
                    "current RDT-CD + SCGR implementation; got "
                    f"{mechanism!r}"
                )

            self.training_mechanism = mechanism
            self.training_auxiliary_requires_cache = bool(
                cfg.get("cache_conditioning", True)
            )

            from .distill.dynamic_teacher import (
                DynamicTeacherDirectionC as DirectionC,
            )

            # --------------------------------------------------------------
            # Reproducibility contract.
            # --------------------------------------------------------------
            # Constructing the training-only auxiliary must not consume the
            # global CPU RNG stream.  Otherwise B0 and D1/D2 could see
            # different subsequent data-augmentation/random-input streams even
            # under the same seed simply because the auxiliary was enabled.
            #
            # The model is constructed on CPU before .to(device), therefore
            # fork_rng(devices=[]) is sufficient here.
            # --------------------------------------------------------------
            with torch.random.fork_rng(devices=[]):
                self.training_auxiliary = DirectionC(
                    self.mid_d,
                    **cfg,
                )

    # ----------------------------------------------------------------------
    # Training-auxiliary state.
    # ----------------------------------------------------------------------
    @property
    def use_training_auxiliary(self) -> bool:
        """Whether a removable training auxiliary currently exists."""
        return (
            self.auxiliary_mode != "none"
            and hasattr(self, "training_auxiliary")
        )

    # ----------------------------------------------------------------------
    # Unchanged deployable student path.
    # ----------------------------------------------------------------------
    def extract_pair_features(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Run the weight-shared Siamese backbone for both timestamps."""
        return (
            tuple(self.backbone(x1)),
            tuple(self.backbone(x2)),
        )

    def _forward_main_path(
        self,
        features1: Sequence[torch.Tensor],
        features2: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
        return_decoder_features: bool = False,
    ):
        """Execute the unchanged deploy graph.

        Teacher/cache tensors never enter this function.

        ``Decoder`` is expected to return eight tensors:

        - first four: p2/p3/p4/p5 decoder features;
        - remaining four: multi-scale prediction logits.
        """
        aggregated1 = self.swa(*features1)
        aggregated2 = self.swa(*features2)

        change = self.tfm(
            *aggregated1,
            *aggregated2,
        )

        decoder_outputs = self.decoder(*change)
        if len(decoder_outputs) < 8:
            raise RuntimeError(
                "Decoder contract violation: expected at least 8 outputs "
                "(4 decoder features + 4 prediction logits), got "
                f"{len(decoder_outputs)}"
            )

        decoder_features = tuple(decoder_outputs[:4])
        raw_logits = tuple(decoder_outputs[4:])

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
            return predictions, decoder_features
        return predictions

    # ----------------------------------------------------------------------
    # Forward.
    # ----------------------------------------------------------------------
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_pack: Optional[Dict] = None,
        compute_auxiliary: bool = True,
        force_action: Optional[int] = None,
    ):
        """Run student prediction and, during training only, RDT-CD + SCGR.

        Main-output invariance
        ----------------------
        Main predictions are produced before the auxiliary is invoked and are
        never modified by its outputs.  Under the same deterministic model
        state, ``compute_auxiliary=False`` and ``True`` therefore must produce
        identical main predictions.

        Evaluation / deployment
        -----------------------
        When ``self.training is False`` only the prediction tuple is returned.

        Training
        --------
        Returns ``(predictions, auxiliary)``.  If enabled,
        ``auxiliary['direction_c']`` is the RDT-CD + SCGR loss/diagnostic dict.
        """
        if x1.ndim != 4 or x2.ndim != 4:
            raise ValueError("x1 and x2 must both be [B,C,H,W]")
        if x1.shape != x2.shape:
            raise ValueError(
                "x1 and x2 must have identical shapes; got "
                f"{tuple(x1.shape)} and {tuple(x2.shape)}"
            )

        # ------------------------------------------------------------------
        # A. Student encoder.
        # ------------------------------------------------------------------
        features1, features2 = self.extract_pair_features(x1, x2)

        need_aux = (
            self.training
            and bool(compute_auxiliary)
            and self.use_training_auxiliary
        )

        # ------------------------------------------------------------------
        # B. Unchanged deployable prediction path.
        # Decoder features are merely exposed when the loss-only auxiliary
        # needs them; this does not alter the main computation.
        # ------------------------------------------------------------------
        main_output = self._forward_main_path(
            features1,
            features2,
            tuple(x1.shape[-2:]),
            return_decoder_features=need_aux,
        )

        if need_aux:
            predictions, decoder_features = main_output
        else:
            predictions = main_output
            decoder_features = None

        # Evaluation/deployment ends exactly at the main student graph.
        if not self.training:
            return predictions

        # ------------------------------------------------------------------
        # C. Training-only, loss-only auxiliary.
        # ------------------------------------------------------------------
        auxiliary: Dict[str, Dict[str, torch.Tensor]] = {}

        if need_aux:
            if target is None:
                raise ValueError(
                    "RDT-CD + SCGR requires training GT when "
                    "compute_auxiliary=True"
                )

            if target.ndim != 4 or target.shape[1] != 1:
                raise ValueError("target must be [B,1,H,W]")
            if target.shape[0] != x1.shape[0]:
                raise ValueError(
                    "target batch size must match x1/x2 batch size"
                )
            if tuple(target.shape[-2:]) != tuple(x1.shape[-2:]):
                raise ValueError(
                    "target spatial size must match the input spatial size"
                )

            if (
                self.training_auxiliary_requires_cache
                and teacher_pack is None
            ):
                raise ValueError(
                    "RDT-CD cache-conditioned mode requires synchronously "
                    "replayed SAMStruct + OVCDistill teacher caches"
                )

            if decoder_features is None or len(decoder_features) == 0:
                raise RuntimeError(
                    "SCGR requires decoder_features[0], but decoder features "
                    "were not exposed by the main path"
                )

            # --------------------------------------------------------------
            # Critical SCGR observation point.
            # --------------------------------------------------------------
            # decoder_features[0] remains part of the ordinary student graph.
            # The auxiliary observes it in two different ways downstream:
            #
            # 1) fast/EMA dynamic teachers detach it before teacher fitting or
            #    proposal generation;
            # 2) SCGR uses its detached value to analytically estimate the
            #    final-classifier gradient direction in each sparse region.
            #
            # No auxiliary tensor is injected back into any student feature.
            # The only Student effect is the differentiable KD loss returned
            # as output['total'] and applied by train.py during backward().
            # --------------------------------------------------------------
            auxiliary["direction_c"] = self.training_auxiliary(
                feature=decoder_features[0],
                prediction=predictions[0],
                target=target,
                teacher_pack=teacher_pack,
                force_action=force_action,
            )

        return predictions, auxiliary

    # ----------------------------------------------------------------------
    # Deployment.
    # ----------------------------------------------------------------------
    def switch_to_deploy(self):
        """Idempotently delete every training-only auxiliary component.

        After conversion the registered graph contains only::

            backbone -> SWA -> TFM -> Decoder

        No dynamic teacher, EMA target teacher, SAM/OV cache interface, SCGR
        router, or other training-only module remains in the model state.
        """
        if hasattr(self, "training_auxiliary"):
            del self.training_auxiliary

        self.auxiliary_mode = "none"
        self.training_mechanism = "none"
        self.training_auxiliary_requires_cache = False

        return self
