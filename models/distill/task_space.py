"""Run3 BT-SAM-RDT task-space priors and GT safety auditing.

SAMStruct is converted into a genuinely bi-temporal structural-change prior:
T1/T2 instances are matched by IoU, while directional coverage localizes
expansion/shrinkage. OVCDistill supplies semantic soft-change evidence.
The two sources are fused into one teacher prior. GT is used only afterwards
for a pixel-wise positive-Brier-gain safety audit.

Nothing in this file belongs to the deploy graph.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .losses import resize


def _require_single_channel_map(
    x: torch.Tensor,
    name: str,
) -> None:
    """Require a batched [B,1,H,W] tensor."""
    if (
        not torch.is_tensor(x)
        or x.ndim != 4
        or x.shape[1] != 1
    ):
        raise ValueError(
            f"{name} must be a [B,1,H,W] tensor"
        )


def _sample_scatter_sum(
    values: torch.Tensor,
    owners: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Sum per-instance values back to their owning batch samples."""
    out = torch.zeros(
        batch_size,
        device=values.device,
        dtype=torch.float32,
    )

    if values.numel() > 0:
        out.scatter_add_(
            0,
            owners.long(),
            values.float(),
        )

    return out


# ============================================================================
# Bi-temporal SAM structural prior
# ============================================================================


@torch.no_grad()
def build_bitemporal_structural_prior(
    sam_pack: Dict[str, Dict[str, torch.Tensor]],
    output_size: Tuple[int, int] | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Build a student-independent, GT-independent bi-temporal SAM prior.

    Expected cache
    --------------
    sam_pack["t1"]["instance_id"] : [B,1,H,W]
    sam_pack["t1"]["quality"]     : [B,1,H,W]
    sam_pack["t1"]["boundary"]    : [B,1,H,W]

    sam_pack["t2"]["instance_id"] : [B,1,H,W]
    sam_pack["t2"]["quality"]     : [B,1,H,W]
    sam_pack["t2"]["boundary"]    : [B,1,H,W]

    Core mechanism
    --------------
    1. Select T1<->T2 counterparts using instance IoU.
    2. Use directional coverage:
           C12 = intersection / area(T1)
           C21 = intersection / area(T2)
       to model persistence, expansion and shrinkage.
    3. Project correspondence back into the image plane.
    4. Use SAM quality and boundary uncertainty to form reliability.

    Crucially:
        no student prediction enters this function;
        no student feature enters this function;
        no GT enters this function.
    """
    for temporal in (
        "t1",
        "t2",
    ):
        if temporal not in sam_pack:
            raise KeyError(
                f"SAM pack missing {temporal!r}"
            )

        for key in (
            "instance_id",
            "quality",
            "boundary",
        ):
            if key not in sam_pack[temporal]:
                raise KeyError(
                    f"SAM pack[{temporal!r}] missing {key!r}"
                )

            _require_single_channel_map(
                sam_pack[temporal][key],
                f"sam[{temporal}][{key}]",
            )

    ids1 = (
        sam_pack["t1"]["instance_id"]
        .detach()
        .long()
    )

    ids2 = (
        sam_pack["t2"]["instance_id"]
        .detach()
        .long()
    )

    q1 = (
        sam_pack["t1"]["quality"]
        .detach()
        .float()
    )

    q2 = (
        sam_pack["t2"]["quality"]
        .detach()
        .float()
    )

    b1 = (
        sam_pack["t1"]["boundary"]
        .detach()
        .float()
    )

    b2 = (
        sam_pack["t2"]["boundary"]
        .detach()
        .float()
    )

    if not (
        ids1.shape
        == ids2.shape
        == q1.shape
        == q2.shape
        == b1.shape
        == b2.shape
    ):
        raise ValueError(
            "All T1/T2 SAM fields must have identical [B,1,H,W] shapes"
        )

    batch_size = ids1.shape[0]

    if batch_size < 1:
        raise ValueError(
            "SAM batch must be non-empty"
        )

    # ----------------------------------------------------------------------
    # Make opaque instance IDs unique across the batch.
    #
    # ID=1 in sample A and ID=1 in sample B are unrelated objects.
    # ----------------------------------------------------------------------
    stride = (
        torch.maximum(
            ids1.max(),
            ids2.max(),
        )
        + 1
    )

    sample_index = torch.arange(
        batch_size,
        device=ids1.device,
        dtype=torch.long,
    ).view(
        batch_size,
        1,
        1,
        1,
    )

    composite1 = (
        ids1
        + sample_index * stride
    ).reshape(-1)

    composite2 = (
        ids2
        + sample_index * stride
    ).reshape(-1)

    # Compact IDs so memory depends on the number of observed instances,
    # not on the magnitude of the original SAM instance IDs.
    unique1, index1 = torch.unique(
        composite1,
        sorted=True,
        return_inverse=True,
    )

    unique2, index2 = torch.unique(
        composite2,
        sorted=True,
        return_inverse=True,
    )

    owner1 = torch.div(
        unique1,
        stride,
        rounding_mode="floor",
    )

    owner2 = torch.div(
        unique2,
        stride,
        rounding_mode="floor",
    )

    local_id1 = (
        unique1
        - owner1 * stride
    )

    local_id2 = (
        unique2
        - owner2 * stride
    )

    n1 = unique1.numel()
    n2 = unique2.numel()

    area1 = torch.bincount(
        index1,
        minlength=n1,
    ).float()

    area2 = torch.bincount(
        index2,
        minlength=n2,
    ).float()

    # ----------------------------------------------------------------------
    # Per-instance correspondence state.
    # ----------------------------------------------------------------------
    best_iou1 = torch.zeros(
        n1,
        device=ids1.device,
        dtype=torch.float32,
    )

    best_iou2 = torch.zeros(
        n2,
        device=ids1.device,
        dtype=torch.float32,
    )

    best_cov12 = torch.zeros_like(
        best_iou1
    )

    best_cov21 = torch.zeros_like(
        best_iou2
    )

    # For T1 instance i:
    #     best_j[i] = best matching T2 compact instance.
    #
    # For T2 instance j:
    #     best_i[j] = best matching T1 compact instance.
    #
    # -1 means no valid cross-temporal counterpart.
    best_j = torch.full(
        (n1,),
        -1,
        device=ids1.device,
        dtype=torch.long,
    )

    best_i = torch.full(
        (n2,),
        -1,
        device=ids1.device,
        dtype=torch.long,
    )

    flat_ids1 = ids1.reshape(-1)
    flat_ids2 = ids2.reshape(-1)

    valid1 = (
        flat_ids1 > 0
    )

    valid2 = (
        flat_ids2 > 0
    )

    overlap_valid = (
        valid1
        & valid2
    )

    # ----------------------------------------------------------------------
    # Sparse overlap table.
    #
    # pair_code identifies one compact (T1,T2) instance pair.
    # No dense N1 x N2 overlap matrix is allocated.
    # ----------------------------------------------------------------------
    pair_code = (
        index1[
            overlap_valid
        ]
        * n2
        + index2[
            overlap_valid
        ]
    )

    pair_code, intersection = torch.unique(
        pair_code,
        sorted=True,
        return_counts=True,
    )

    pair_i = torch.div(
        pair_code,
        n2,
        rounding_mode="floor",
    )

    pair_j = pair_code.remainder(
        n2
    )

    intersection = (
        intersection.float()
    )

    # Defensive checks:
    # - background cannot form a pair;
    # - pairs must belong to the same sample.
    pair_is_valid = (
        (
            local_id1[
                pair_i
            ]
            > 0
        )
        & (
            local_id2[
                pair_j
            ]
            > 0
        )
        & (
            owner1[
                pair_i
            ]
            == owner2[
                pair_j
            ]
        )
    )

    pair_i = pair_i[
        pair_is_valid
    ]

    pair_j = pair_j[
        pair_is_valid
    ]

    intersection = intersection[
        pair_is_valid
    ]

    if pair_i.numel() > 0:
        union = (
            area1[
                pair_i
            ]
            + area2[
                pair_j
            ]
            - intersection
        ).clamp_min(
            1.0
        )

        pair_iou = (
            intersection
            / union
        ).clamp(
            0.0,
            1.0,
        )

        # Directional coverage.
        pair_cov12 = (
            intersection
            / area1[
                pair_i
            ].clamp_min(
                1.0
            )
        ).clamp(
            0.0,
            1.0,
        )

        pair_cov21 = (
            intersection
            / area2[
                pair_j
            ].clamp_min(
                1.0
            )
        ).clamp(
            0.0,
            1.0,
        )

        # ------------------------------------------------------------------
        # Select counterparts by maximum IoU.
        # ------------------------------------------------------------------
        best_iou1.scatter_reduce_(
            0,
            pair_i,
            pair_iou,
            reduce="amax",
            include_self=True,
        )

        best_iou2.scatter_reduce_(
            0,
            pair_j,
            pair_iou,
            reduce="amax",
            include_self=True,
        )

        tolerance = (
            8
            * torch.finfo(
                pair_iou.dtype
            ).eps
        )

        # Deterministic tie-break:
        # choose the smallest compact counterpart ID.
        best_for_i = (
            pair_iou
            >= (
                best_iou1[
                    pair_i
                ]
                - tolerance
            )
        )

        candidate_j = torch.where(
            best_for_i,
            pair_j,
            torch.full_like(
                pair_j,
                n2,
            ),
        )

        best_j_tmp = torch.full(
            (n1,),
            n2,
            device=ids1.device,
            dtype=torch.long,
        )

        best_j_tmp.scatter_reduce_(
            0,
            pair_i,
            candidate_j,
            reduce="amin",
            include_self=True,
        )

        best_j = torch.where(
            best_j_tmp < n2,
            best_j_tmp,
            torch.full_like(
                best_j_tmp,
                -1,
            ),
        )

        best_for_j = (
            pair_iou
            >= (
                best_iou2[
                    pair_j
                ]
                - tolerance
            )
        )

        candidate_i = torch.where(
            best_for_j,
            pair_i,
            torch.full_like(
                pair_i,
                n1,
            ),
        )

        best_i_tmp = torch.full(
            (n2,),
            n1,
            device=ids1.device,
            dtype=torch.long,
        )

        best_i_tmp.scatter_reduce_(
            0,
            pair_j,
            candidate_i,
            reduce="amin",
            include_self=True,
        )

        best_i = torch.where(
            best_i_tmp < n1,
            best_i_tmp,
            torch.full_like(
                best_i_tmp,
                -1,
            ),
        )

        # ------------------------------------------------------------------
        # Keep directional coverage belonging specifically to the
        # IoU-selected counterpart.
        # ------------------------------------------------------------------
        selected_for_i = (
            pair_j
            == best_j[
                pair_i
            ]
        )

        best_cov12.scatter_reduce_(
            0,
            pair_i,
            torch.where(
                selected_for_i,
                pair_cov12,
                torch.zeros_like(
                    pair_cov12
                ),
            ),
            reduce="amax",
            include_self=True,
        )

        selected_for_j = (
            pair_i
            == best_i[
                pair_j
            ]
        )

        best_cov21.scatter_reduce_(
            0,
            pair_j,
            torch.where(
                selected_for_j,
                pair_cov21,
                torch.zeros_like(
                    pair_cov21
                ),
            ),
            reduce="amax",
            include_self=True,
        )

    # ----------------------------------------------------------------------
    # Project correspondence back to pixels.
    # ----------------------------------------------------------------------
    pixel_cov12 = (
        best_cov12[
            index1
        ]
    )

    pixel_cov21 = (
        best_cov21[
            index2
        ]
    )

    consistent12 = (
        valid1
        & valid2
        & (
            best_j[
                index1
            ]
            == index2
        )
    )

    consistent21 = (
        valid1
        & valid2
        & (
            best_i[
                index2
            ]
            == index1
        )
    )

    # ----------------------------------------------------------------------
    # Structural change evidence.
    #
    # T1 side:
    #   no object                         -> no T1 evidence
    #   no local best counterpart         -> full change evidence
    #   correct counterpart at this pixel -> 1 - C12
    #
    # T2 side is symmetric.
    #
    # Directional coverage is important:
    #
    # expansion:
    #   C12 ~= 1
    #   C21 < 1
    #
    # shrinkage:
    #   C12 < 1
    #   C21 ~= 1
    # ----------------------------------------------------------------------
    side1_change = torch.where(
        valid1,
        torch.where(
            consistent12,
            1.0
            - pixel_cov12,
            torch.ones_like(
                pixel_cov12
            ),
        ),
        torch.zeros_like(
            pixel_cov12
        ),
    )

    side2_change = torch.where(
        valid2,
        torch.where(
            consistent21,
            1.0
            - pixel_cov21,
            torch.ones_like(
                pixel_cov21
            ),
        ),
        torch.zeros_like(
            pixel_cov21
        ),
    )

    active_sides = (
        valid1.float()
        + valid2.float()
    )

    structural_prior = torch.where(
        active_sides > 0,
        (
            side1_change
            + side2_change
        )
        / active_sides.clamp_min(
            1.0
        ),
        torch.zeros_like(
            active_sides
        ),
    ).clamp(
        0.0,
        1.0,
    )

    # ----------------------------------------------------------------------
    # Structural reliability.
    #
    # High reliability:
    #   clearly persistent coverage
    #   OR
    #   clearly novel coverage.
    #
    # Lowest reliability:
    #   ambiguous ~0.5 correspondence.
    # ----------------------------------------------------------------------
    certainty1 = torch.maximum(
        pixel_cov12,
        1.0
        - pixel_cov12,
    )

    certainty2 = torch.maximum(
        pixel_cov21,
        1.0
        - pixel_cov21,
    )

    flat_q1 = q1.reshape(-1)
    flat_q2 = q2.reshape(-1)

    flat_b1 = b1.reshape(-1)
    flat_b2 = b2.reshape(-1)

    # Boundary evidence is attenuated rather than removed completely.
    reliability1 = (
        flat_q1
        * (
            1.0
            - 0.5
            * flat_b1
        )
        * certainty1
        * valid1.float()
    )

    reliability2 = (
        flat_q2
        * (
            1.0
            - 0.5
            * flat_b2
        )
        * certainty2
        * valid2.float()
    )

    structural_reliability = torch.where(
        active_sides > 0,
        (
            reliability1
            + reliability2
        )
        / active_sides.clamp_min(
            1.0
        ),
        torch.zeros_like(
            active_sides
        ),
    ).clamp(
        0.0,
        1.0,
    )

    structural_prior = (
        structural_prior
        .reshape_as(
            q1
        )
        .float()
    )

    structural_reliability = (
        structural_reliability
        .reshape_as(
            q1
        )
        .float()
    )

    # ----------------------------------------------------------------------
    # Per-sample diagnostics.
    # ----------------------------------------------------------------------
    active_instance1 = (
        local_id1 > 0
    )

    active_instance2 = (
        local_id2 > 0
    )

    count1 = _sample_scatter_sum(
        torch.ones_like(
            best_iou1[
                active_instance1
            ]
        ),
        owner1[
            active_instance1
        ],
        batch_size,
    )

    count2 = _sample_scatter_sum(
        torch.ones_like(
            best_iou2[
                active_instance2
            ]
        ),
        owner2[
            active_instance2
        ],
        batch_size,
    )

    matched1 = _sample_scatter_sum(
        (
            best_iou1[
                active_instance1
            ]
            > 0
        ).float(),
        owner1[
            active_instance1
        ],
        batch_size,
    )

    matched2 = _sample_scatter_sum(
        (
            best_iou2[
                active_instance2
            ]
            > 0
        ).float(),
        owner2[
            active_instance2
        ],
        batch_size,
    )

    total_count = (
        count1
        + count2
    )

    pair_match_ratio = torch.where(
        total_count > 0,
        (
            matched1
            + matched2
        )
        / total_count.clamp_min(
            1.0
        ),
        torch.zeros_like(
            total_count
        ),
    )

    iou_sum1 = _sample_scatter_sum(
        best_iou1[
            active_instance1
        ],
        owner1[
            active_instance1
        ],
        batch_size,
    )

    iou_sum2 = _sample_scatter_sum(
        best_iou2[
            active_instance2
        ],
        owner2[
            active_instance2
        ],
        batch_size,
    )

    pair_match_iou = torch.where(
        total_count > 0,
        (
            iou_sum1
            + iou_sum2
        )
        / total_count.clamp_min(
            1.0
        ),
        torch.zeros_like(
            total_count
        ),
    )

    cov12_sum = _sample_scatter_sum(
        best_cov12[
            active_instance1
        ],
        owner1[
            active_instance1
        ],
        batch_size,
    )

    cov21_sum = _sample_scatter_sum(
        best_cov21[
            active_instance2
        ],
        owner2[
            active_instance2
        ],
        batch_size,
    )

    pair_match_cov12 = torch.where(
        count1 > 0,
        cov12_sum
        / count1.clamp_min(
            1.0
        ),
        torch.zeros_like(
            count1
        ),
    )

    pair_match_cov21 = torch.where(
        count2 > 0,
        cov21_sum
        / count2.clamp_min(
            1.0
        ),
        torch.zeros_like(
            count2
        ),
    )

    # Match discrete instances BEFORE interpolation.
    if output_size is not None:
        output_size = tuple(
            int(v)
            for v in output_size
        )

        if (
            structural_prior.shape[-2:]
            != output_size
        ):
            structural_prior = resize(
                structural_prior,
                output_size,
            ).clamp(
                0.0,
                1.0,
            )

            structural_reliability = resize(
                structural_reliability,
                output_size,
            ).clamp(
                0.0,
                1.0,
            )

    return {
        "prior": (
            structural_prior
        ),
        "reliability": (
            structural_reliability
        ),
        "pair_match_ratio": (
            pair_match_ratio
        ),
        "pair_match_iou": (
            pair_match_iou
        ),
        "pair_match_cov12": (
            pair_match_cov12
        ),
        "pair_match_cov21": (
            pair_match_cov21
        ),
    }


# ============================================================================
# OV semantic prior
# ============================================================================


@torch.no_grad()
def build_ov_prior(
    ov_pack: Dict[str, Dict[str, torch.Tensor]],
    output_size: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    """
    Build confidence-weighted OVCDistill semantic soft-change prior.

    OV relation channels are deliberately not used by Run3 because the
    structural innovation is provided by BT-SAM correspondence, while OV is
    used only as semantic change evidence.
    """
    output_size = tuple(
        int(v)
        for v in output_size
    )

    values = []
    confidences = []

    for level in (
        "l1",
        "l2",
    ):
        try:
            soft_change = (
                ov_pack[
                    "soft_change"
                ][level]
            )

            confidence = (
                ov_pack[
                    "confidence"
                ][level]
            )

        except KeyError as exc:
            raise KeyError(
                "OV pack must contain soft_change/confidence for l1 and l2"
            ) from exc

        _require_single_channel_map(
            soft_change,
            f"ov[soft_change][{level}]",
        )

        _require_single_channel_map(
            confidence,
            f"ov[confidence][{level}]",
        )

        if (
            soft_change.shape[0]
            != confidence.shape[0]
        ):
            raise ValueError(
                f"OV {level} soft_change/confidence batch sizes must match"
            )

        values.append(
            resize(
                soft_change.detach(),
                output_size,
            ).clamp(
                0.0,
                1.0,
            )
        )

        confidences.append(
            resize(
                confidence.detach(),
                output_size,
            ).clamp(
                0.0,
                1.0,
            )
        )

    confidence_mass = (
        confidences[0]
        + confidences[1]
    )

    semantic_prior = torch.where(
        confidence_mass > 0,
        (
            values[0]
            * confidences[0]
            + values[1]
            * confidences[1]
        )
        / confidence_mass.clamp_min(
            1e-8
        ),
        torch.zeros_like(
            confidence_mass
        ),
    ).clamp(
        0.0,
        1.0,
    )

    semantic_reliability = (
        0.5
        * confidence_mass
    ).clamp(
        0.0,
        1.0,
    )

    return {
        "prior": (
            semantic_prior
        ),
        "reliability": (
            semantic_reliability
        ),
    }


# ============================================================================
# SAM + OV complementary fusion
# ============================================================================


@torch.no_grad()
def fuse_foundation_priors(
    sam_prior: torch.Tensor,
    sam_reliability: torch.Tensor,
    ov_prior: torch.Tensor,
    ov_reliability: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Fuse structural and semantic evidence into ONE foundation prior.

    Prior:
        q_f =
            (r_s * q_s + r_o * q_o)
            /
            (r_s + r_o)

    Reliability:
        average available-source confidence
            x
        SAM/OV agreement

    SAM and OV therefore complement each other rather than compete through a
    teacher-selection router.
    """
    for tensor, name in (
        (
            sam_prior,
            "sam_prior",
        ),
        (
            sam_reliability,
            "sam_reliability",
        ),
        (
            ov_prior,
            "ov_prior",
        ),
        (
            ov_reliability,
            "ov_reliability",
        ),
    ):
        _require_single_channel_map(
            tensor,
            name,
        )

    if not (
        sam_prior.shape
        == sam_reliability.shape
        == ov_prior.shape
        == ov_reliability.shape
    ):
        raise ValueError(
            "SAM/OV priors and reliability maps must have identical shapes"
        )

    sam_prior = (
        sam_prior
        .detach()
        .float()
        .clamp(
            0.0,
            1.0,
        )
    )

    ov_prior = (
        ov_prior
        .detach()
        .float()
        .clamp(
            0.0,
            1.0,
        )
    )

    sam_reliability = (
        sam_reliability
        .detach()
        .float()
        .clamp(
            0.0,
            1.0,
        )
    )

    ov_reliability = (
        ov_reliability
        .detach()
        .float()
        .clamp(
            0.0,
            1.0,
        )
    )

    mass = (
        sam_reliability
        + ov_reliability
    )

    fused_prior = torch.where(
        mass > 0,
        (
            sam_prior
            * sam_reliability
            + ov_prior
            * ov_reliability
        )
        / mass.clamp_min(
            1e-8
        ),
        torch.zeros_like(
            mass
        ),
    ).clamp(
        0.0,
        1.0,
    )

    sam_available = (
        sam_reliability > 0
    )

    ov_available = (
        ov_reliability > 0
    )

    source_count = (
        sam_available.float()
        + ov_available.float()
    )

    mean_confidence = torch.where(
        source_count > 0,
        mass
        / source_count.clamp_min(
            1.0
        ),
        torch.zeros_like(
            mass
        ),
    )

    raw_conflict = (
        sam_prior
        - ov_prior
    ).abs()

    both_available = (
        sam_available
        & ov_available
    )

    agreement = torch.where(
        both_available,
        1.0
        - raw_conflict,
        torch.ones_like(
            raw_conflict
        ),
    )

    fused_reliability = (
        mean_confidence
        * agreement
    ).clamp(
        0.0,
        1.0,
    )

    conflict = torch.where(
        both_available,
        raw_conflict,
        torch.zeros_like(
            raw_conflict
        ),
    )

    return {
        "prior": (
            fused_prior
        ),
        "reliability": (
            fused_reliability
        ),
        "conflict": (
            conflict
        ),
        "both_available": (
            both_available
        ),
    }


@torch.no_grad()
def build_foundation_prior(
    teacher_pack: Dict[str, Dict],
    output_size: Tuple[int, int],
    use_ov: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build Run3 foundation priors without Student or GT access.

    Main Run3:
        use_ov=True

        BT-SAM structural prior
                 +
        OV semantic prior
                 ↓
        fused foundation prior

    SAM-only ablation:
        use_ov=False
    """
    if teacher_pack is None:
        raise ValueError(
            "teacher_pack is required for BT-SAM-RDT"
        )

    if "sam" not in teacher_pack:
        raise KeyError(
            "teacher_pack missing 'sam'"
        )

    output_size = tuple(
        int(v)
        for v in output_size
    )

    sam = build_bitemporal_structural_prior(
        teacher_pack[
            "sam"
        ],
        output_size=output_size,
    )

    result = {
        "sam_prior": (
            sam[
                "prior"
            ]
        ),
        "sam_reliability": (
            sam[
                "reliability"
            ]
        ),
        "pair_match_ratio": (
            sam[
                "pair_match_ratio"
            ]
        ),
        "pair_match_iou": (
            sam[
                "pair_match_iou"
            ]
        ),
        "pair_match_cov12": (
            sam[
                "pair_match_cov12"
            ]
        ),
        "pair_match_cov21": (
            sam[
                "pair_match_cov21"
            ]
        ),
    }

    if use_ov:
        if "ov" not in teacher_pack:
            raise KeyError(
                "teacher_pack missing 'ov'"
            )

        ov = build_ov_prior(
            teacher_pack[
                "ov"
            ],
            output_size=output_size,
        )

        fused = fuse_foundation_priors(
            sam_prior=sam[
                "prior"
            ],
            sam_reliability=sam[
                "reliability"
            ],
            ov_prior=ov[
                "prior"
            ],
            ov_reliability=ov[
                "reliability"
            ],
        )

        result.update(
            {
                "ov_prior": (
                    ov[
                        "prior"
                    ]
                ),
                "ov_reliability": (
                    ov[
                        "reliability"
                    ]
                ),
                "fused_prior": (
                    fused[
                        "prior"
                    ]
                ),
                "fused_reliability": (
                    fused[
                        "reliability"
                    ]
                ),
                "sam_ov_conflict": (
                    fused[
                        "conflict"
                    ]
                ),
                "both_available": (
                    fused[
                        "both_available"
                    ]
                ),
            }
        )

    else:
        zeros = torch.zeros_like(
            sam[
                "prior"
            ]
        )

        result.update(
            {
                "ov_prior": (
                    zeros
                ),
                "ov_reliability": (
                    zeros
                ),
                "fused_prior": (
                    sam[
                        "prior"
                    ]
                ),
                "fused_reliability": (
                    sam[
                        "reliability"
                    ]
                ),
                "sam_ov_conflict": (
                    zeros
                ),
                "both_available": (
                    torch.zeros_like(
                        zeros,
                        dtype=torch.bool,
                    )
                ),
            }
        )

    return result


# ============================================================================
# Single fused-teacher GT audit
# ============================================================================


@torch.no_grad()
def task_space_audit(
    prediction: torch.Tensor,
    target: torch.Tensor,
    proposal: torch.Tensor,
    reliability: torch.Tensor,
    policy: str = "advantage",
) -> Dict[str, torch.Tensor]:
    """
    Run3 GT safety audit using exact pixel-wise Brier improvement.

    Main policy
    -----------
        E_s = (p - y)^2
        E_t = (q - y)^2

        gain = E_s - E_t

        admit teacher iff:
            reliability > 0
            and
            gain > numerical tolerance

    GT never constructs the teacher prior/proposal.

    Run3 deliberately contains no:
        SAM-vs-OV competition
        regional router
        gradient-concordance gate
        learned action selector
        utility-margin tuning
    """
    for tensor, name in (
        (
            prediction,
            "prediction",
        ),
        (
            target,
            "target",
        ),
        (
            proposal,
            "proposal",
        ),
        (
            reliability,
            "reliability",
        ),
    ):
        _require_single_channel_map(
            tensor,
            name,
        )

    if not (
        prediction.shape
        == target.shape
        == proposal.shape
        == reliability.shape
    ):
        raise ValueError(
            "prediction/target/proposal/reliability must have identical shapes"
        )

    p = (
        prediction
        .detach()
        .float()
    )

    y = (
        target
        .detach()
        .float()
    )

    q = (
        proposal
        .detach()
        .float()
        .clamp(
            0.0,
            1.0,
        )
    )

    r = (
        reliability
        .detach()
        .float()
        .clamp(
            0.0,
            1.0,
        )
    )

    student_error = (
        p
        - y
    ).square()

    teacher_error = (
        q
        - y
    ).square()

    gain = (
        student_error
        - teacher_error
    )

    relative_gain = (
        gain
        / student_error.clamp_min(
            1e-8
        )
    ).clamp(
        -1.0,
        1.0,
    )

    available = (
        r > 0
    )

    tolerance = (
        8
        * torch.finfo(
            p.dtype
        ).eps
    )

    if policy == "advantage":
        eligible = (
            available
            & (
                gain
                > tolerance
            )
        )

    elif policy == "quality":
        # Negative/control ablation:
        # remove the GT positive-gain safety check.
        eligible = available

    else:
        raise ValueError(
            "policy must be 'advantage' or 'quality'"
        )

    # GT only decides admit/reject.
    # Distillation strength comes from fused teacher reliability.
    effective = (
        r
        * eligible.float()
    )

    return {
        "student_error_map": (
            student_error
        ),
        "teacher_error_map": (
            teacher_error
        ),
        "gain_map": (
            gain
        ),
        "relative_gain_map": (
            relative_gain
        ),
        "available_map": (
            available
        ),
        "eligible_map": (
            eligible
        ),
        "accepted_map": (
            eligible
        ),
        "effective_map": (
            effective
        ),
    }


# ============================================================================
# Bernoulli task-space distillation
# ============================================================================


def bernoulli_kl(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    KL(Ber(target) || Ber(prediction)).

    Same gradient with respect to Student prediction as soft BCE, while
    removing irreducible target entropy from the logged distillation loss.
    """
    p = (
        prediction
        .float()
        .clamp(
            1e-6,
            1.0 - 1e-6,
        )
    )

    t = (
        target
        .detach()
        .float()
        .clamp(
            0.0,
            1.0,
        )
    )

    return (
        F.binary_cross_entropy(
            p.expand_as(t),
            t,
            reduction="none",
        )
        + torch.special.xlogy(
            t,
            t,
        )
        + torch.special.xlogy(
            1.0 - t,
            1.0 - t,
        )
    )