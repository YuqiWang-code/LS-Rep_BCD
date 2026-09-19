"""Run3 BT-SAM-RDT task-space priors and GT safety auditing.

Run3 converts SAMStruct into a genuinely bi-temporal structural-change prior:

    T1/T2 SAM instances
        -> sparse cross-temporal instance matching
        -> correspondence reliability
        -> pixel-localized structural difference

OVCDistill provides complementary semantic soft-change evidence.

The structural and semantic priors are fused into one foundation prior, after
which GT is used only for a pixel-wise positive-Brier-gain safety audit.

Nothing in this file belongs to the deploy graph.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ============================================================================
# Generic helpers
# ============================================================================


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


def _resize_map(
    x: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """FP32 bilinear resize for continuous probability/reliability maps."""
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
    Build a student-independent and GT-independent BT-SAM structural prior.

    Expected cache
    --------------
    sam_pack["t1"]["instance_id"] : [B,1,H,W]
    sam_pack["t1"]["quality"]     : [B,1,H,W]
    sam_pack["t1"]["boundary"]    : [B,1,H,W]

    sam_pack["t2"]["instance_id"] : [B,1,H,W]
    sam_pack["t2"]["quality"]     : [B,1,H,W]
    sam_pack["t2"]["boundary"]    : [B,1,H,W]

    Instance matching
    -----------------
    For T1 instance i and T2 instance j:

        I_ij   = |M_i^1 ∩ M_j^2|

        IoU_ij =
            I_ij
            /
            (|M_i^1| + |M_j^2| - I_ij)

        C12_ij =
            I_ij / |M_i^1|

        C21_ij =
            I_ij / |M_j^2|

    The best counterpart is selected by maximum IoU.

    Pixel-localized structural prior
    --------------------------------
    IoU / directional coverage are NOT directly used as pixel-level change
    probabilities.

    Instead, after instance correspondence is established:

        matched overlap
            -> structural change = 0

        T1-only region
            -> disappearance / shrinkage = 1

        T2-only region
            -> appearance / expansion = 1

        overlap of non-corresponding instances
            -> structural disagreement = 1

    Example: expansion
    ------------------
        T1 object:
            XXXX
            XXXX

        T2 object:
          XXXXXXXX
          XXXXXXXX

    The shared old core remains 0, while the newly occupied T2 region is 1.

    This is the key Run3 correction over the previous implementation, where
    directional coverage was directly converted into a region-wide change
    score and therefore incorrectly assigned non-zero change probability to
    stable overlap cores.

    Reliability
    -----------
    SAM quality and boundary confidence are combined with instance
    correspondence confidence.

    For matched instances:

        association =
            sqrt(
                best_IoU
                *
                max(C12, C21)
            )

    This formulation gives:
        - high confidence for exact matches;
        - moderate confidence for clear expansion/shrinkage;
        - low confidence for tiny accidental overlap.

    For completely unmatched instances, structural novelty itself is treated
    as valid evidence and reliability is determined by SAM quality/boundary.

    Crucially:
        - no Student prediction enters this function;
        - no Student feature enters this function;
        - no GT enters this function.
    """

    # ======================================================================
    # 1. Validate input
    # ======================================================================

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
            if key not in sam_pack[
                temporal
            ]:
                raise KeyError(
                    f"SAM pack[{temporal!r}] missing {key!r}"
                )

            _require_single_channel_map(
                sam_pack[
                    temporal
                ][
                    key
                ],
                f"sam[{temporal}][{key}]",
            )

    ids1 = (
        sam_pack[
            "t1"
        ][
            "instance_id"
        ]
        .detach()
        .long()
    )

    ids2 = (
        sam_pack[
            "t2"
        ][
            "instance_id"
        ]
        .detach()
        .long()
    )

    q1 = (
        sam_pack[
            "t1"
        ][
            "quality"
        ]
        .detach()
        .float()
    )

    q2 = (
        sam_pack[
            "t2"
        ][
            "quality"
        ]
        .detach()
        .float()
    )

    b1 = (
        sam_pack[
            "t1"
        ][
            "boundary"
        ]
        .detach()
        .float()
    )

    b2 = (
        sam_pack[
            "t2"
        ][
            "boundary"
        ]
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

    batch_size = int(
        ids1.shape[
            0
        ]
    )

    if batch_size < 1:
        raise ValueError(
            "SAM batch must be non-empty"
        )

    # ======================================================================
    # 2. Make opaque instance IDs unique across samples
    #
    # SAM instance IDs have meaning only inside one sample.
    # ======================================================================

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
        + sample_index
        * stride
    ).reshape(
        -1
    )

    composite2 = (
        ids2
        + sample_index
        * stride
    ).reshape(
        -1
    )

    # Compact observed IDs so memory scales with the actual number of
    # instances instead of the numerical magnitude of SAM IDs.
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
        - owner1
        * stride
    )

    local_id2 = (
        unique2
        - owner2
        * stride
    )

    n1 = int(
        unique1.numel()
    )

    n2 = int(
        unique2.numel()
    )

    area1 = torch.bincount(
        index1,
        minlength=n1,
    ).float()

    area2 = torch.bincount(
        index2,
        minlength=n2,
    ).float()

    # ======================================================================
    # 3. Per-instance matching state
    # ======================================================================

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

    # Coverage in the instance's own direction.
    best_cov12 = torch.zeros_like(
        best_iou1
    )

    best_cov21 = torch.zeros_like(
        best_iou2
    )

    # Reverse-direction coverage belonging to the selected counterpart.
    #
    # T1 instance i:
    #     selected pair has C12 and C21.
    #
    # T2 instance j:
    #     selected pair has C21 and C12.
    best_reverse_cov21_for_i = (
        torch.zeros_like(
            best_iou1
        )
    )

    best_reverse_cov12_for_j = (
        torch.zeros_like(
            best_iou2
        )
    )

    # best_j[i] = compact T2 counterpart selected for T1 instance i.
    # best_i[j] = compact T1 counterpart selected for T2 instance j.
    #
    # -1 means no overlapping counterpart.
    best_j = torch.full(
        (
            n1,
        ),
        -1,
        device=ids1.device,
        dtype=torch.long,
    )

    best_i = torch.full(
        (
            n2,
        ),
        -1,
        device=ids1.device,
        dtype=torch.long,
    )

    flat_ids1 = (
        ids1.reshape(
            -1
        )
    )

    flat_ids2 = (
        ids2.reshape(
            -1
        )
    )

    valid1 = (
        flat_ids1
        > 0
    )

    valid2 = (
        flat_ids2
        > 0
    )

    overlap_valid = (
        valid1
        & valid2
    )

    # ======================================================================
    # 4. Sparse overlap table
    #
    # No dense N1 x N2 matrix is constructed.
    # ======================================================================

    pair_code = (
        index1[
            overlap_valid
        ]
        * n2
        + index2[
            overlap_valid
        ]
    )

    pair_code, intersection = (
        torch.unique(
            pair_code,
            sorted=True,
            return_counts=True,
        )
    )

    pair_i = torch.div(
        pair_code,
        n2,
        rounding_mode="floor",
    )

    pair_j = (
        pair_code.remainder(
            n2
        )
    )

    intersection = (
        intersection.float()
    )

    # Defensive filtering:
    # background may never be treated as an object correspondence.
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

    # ======================================================================
    # 5. Match instances by maximum IoU
    # ======================================================================

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
        # Best IoU per T1 instance
        # ------------------------------------------------------------------

        best_iou1.scatter_reduce_(
            0,
            pair_i,
            pair_iou,
            reduce="amax",
            include_self=True,
        )

        # ------------------------------------------------------------------
        # Best IoU per T2 instance
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Deterministic T1 -> T2 counterpart
        # ------------------------------------------------------------------

        is_best_for_i = (
            pair_iou
            >= (
                best_iou1[
                    pair_i
                ]
                - tolerance
            )
        )

        candidate_j = torch.where(
            is_best_for_i,
            pair_j,
            torch.full_like(
                pair_j,
                n2,
            ),
        )

        best_j_tmp = torch.full(
            (
                n1,
            ),
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
            best_j_tmp
            < n2,
            best_j_tmp,
            torch.full_like(
                best_j_tmp,
                -1,
            ),
        )

        # ------------------------------------------------------------------
        # Deterministic T2 -> T1 counterpart
        # ------------------------------------------------------------------

        is_best_for_j = (
            pair_iou
            >= (
                best_iou2[
                    pair_j
                ]
                - tolerance
            )
        )

        candidate_i = torch.where(
            is_best_for_j,
            pair_i,
            torch.full_like(
                pair_i,
                n1,
            ),
        )

        best_i_tmp = torch.full(
            (
                n2,
            ),
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
            best_i_tmp
            < n1,
            best_i_tmp,
            torch.full_like(
                best_i_tmp,
                -1,
            ),
        )

        # ------------------------------------------------------------------
        # Directional coverages belonging to selected T1 -> T2 pair
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

        best_reverse_cov21_for_i.scatter_reduce_(
            0,
            pair_i,
            torch.where(
                selected_for_i,
                pair_cov21,
                torch.zeros_like(
                    pair_cov21
                ),
            ),
            reduce="amax",
            include_self=True,
        )

        # ------------------------------------------------------------------
        # Directional coverages belonging to selected T2 -> T1 pair
        # ------------------------------------------------------------------

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

        best_reverse_cov12_for_j.scatter_reduce_(
            0,
            pair_j,
            torch.where(
                selected_for_j,
                pair_cov12,
                torch.zeros_like(
                    pair_cov12
                ),
            ),
            reduce="amax",
            include_self=True,
        )

    # ======================================================================
    # 6. Instance-level correspondence reliability
    # ======================================================================

    has_match1 = (
        best_j
        >= 0
    )

    has_match2 = (
        best_i
        >= 0
    )

    # Coverage max captures clear containment during expansion/shrinkage.
    coverage_strength1 = torch.maximum(
        best_cov12,
        best_reverse_cov21_for_i,
    )

    coverage_strength2 = torch.maximum(
        best_cov21,
        best_reverse_cov12_for_j,
    )

    association1 = torch.sqrt(
        (
            best_iou1
            * coverage_strength1
        ).clamp(
            0.0,
            1.0,
        )
    )

    association2 = torch.sqrt(
        (
            best_iou2
            * coverage_strength2
        ).clamp(
            0.0,
            1.0,
        )
    )

    # If an object has no overlapping counterpart at all, the unmatched
    # object itself is direct structural novelty evidence.
    #
    # SAM quality/boundary will still control the final reliability.
    association1 = torch.where(
        has_match1,
        association1,
        torch.ones_like(
            association1
        ),
    )

    association2 = torch.where(
        has_match2,
        association2,
        torch.ones_like(
            association2
        ),
    )

    # Background compact IDs must never carry reliability.
    association1 = (
        association1
        * (
            local_id1
            > 0
        ).float()
    )

    association2 = (
        association2
        * (
            local_id2
            > 0
        ).float()
    )

    # ======================================================================
    # 7. Pixel-localized structural change prior
    # ======================================================================

    # At a pixel where both temporal images contain an instance:
    #
    # treat that overlap as structurally stable when either directional
    # matcher identifies the other instance as its best counterpart.
    #
    # Using directional OR rather than strict mutual-best makes the prior
    # more robust to SAM over-/under-segmentation and split/merge behavior.
    correspondence_consistent = (
        valid1
        & valid2
        & (
            (
                best_j[
                    index1
                ]
                == index2
            )
            | (
                best_i[
                    index2
                ]
                == index1
            )
        )
    )

    any_object = (
        valid1
        | valid2
    )

    # Core Run3 definition:
    #
    # no object in either time:
    #     0
    #
    # corresponding overlap:
    #     0
    #
    # one-sided object or non-corresponding overlap:
    #     1
    structural_prior_flat = (
        any_object
        & (
            ~correspondence_consistent
        )
    ).float()

    structural_prior = (
        structural_prior_flat
        .reshape_as(
            q1
        )
        .float()
    )

    # ======================================================================
    # 8. Pixel structural reliability
    # ======================================================================

    flat_q1 = (
        q1.reshape(
            -1
        )
        .clamp(
            0.0,
            1.0,
        )
    )

    flat_q2 = (
        q2.reshape(
            -1
        )
        .clamp(
            0.0,
            1.0,
        )
    )

    flat_b1 = (
        b1.reshape(
            -1
        )
        .clamp(
            0.0,
            1.0,
        )
    )

    flat_b2 = (
        b2.reshape(
            -1
        )
        .clamp(
            0.0,
            1.0,
        )
    )

    # SAM boundary pixels are less reliable but are not completely discarded.
    local_quality1 = (
        flat_q1
        * (
            1.0
            - 0.5
            * flat_b1
        )
    )

    local_quality2 = (
        flat_q2
        * (
            1.0
            - 0.5
            * flat_b2
        )
    )

    pixel_association1 = (
        association1[
            index1
        ]
    )

    pixel_association2 = (
        association2[
            index2
        ]
    )

    reliability1 = (
        local_quality1
        * pixel_association1
        * valid1.float()
    )

    reliability2 = (
        local_quality2
        * pixel_association2
        * valid2.float()
    )

    active_sides = (
        valid1.float()
        + valid2.float()
    )

    structural_reliability_flat = torch.where(
        active_sides
        > 0,
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

    structural_reliability = (
        structural_reliability_flat
        .reshape_as(
            q1
        )
        .float()
    )

    # ======================================================================
    # 9. Per-sample instance diagnostics
    # ======================================================================

    active_instance1 = (
        local_id1
        > 0
    )

    active_instance2 = (
        local_id2
        > 0
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
        total_count
        > 0,
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
        total_count
        > 0,
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
        count1
        > 0,
        cov12_sum
        / count1.clamp_min(
            1.0
        ),
        torch.zeros_like(
            count1
        ),
    )

    pair_match_cov21 = torch.where(
        count2
        > 0,
        cov21_sum
        / count2.clamp_min(
            1.0
        ),
        torch.zeros_like(
            count2
        ),
    )

    # Additional diagnostic:
    # mean instance-level correspondence confidence.
    association_sum1 = _sample_scatter_sum(
        association1[
            active_instance1
        ],
        owner1[
            active_instance1
        ],
        batch_size,
    )

    association_sum2 = _sample_scatter_sum(
        association2[
            active_instance2
        ],
        owner2[
            active_instance2
        ],
        batch_size,
    )

    pair_match_confidence = torch.where(
        total_count
        > 0,
        (
            association_sum1
            + association_sum2
        )
        / total_count.clamp_min(
            1.0
        ),
        torch.zeros_like(
            total_count
        ),
    )

    # ======================================================================
    # 10. Continuous-map resize
    #
    # Discrete instance matching is always performed BEFORE interpolation.
    # ======================================================================

    if output_size is not None:
        output_size = tuple(
            int(v)
            for v in output_size
        )

        if (
            structural_prior.shape[
                -2:
            ]
            != output_size
        ):
            structural_prior = (
                _resize_map(
                    structural_prior,
                    output_size,
                )
                .clamp(
                    0.0,
                    1.0,
                )
            )

            structural_reliability = (
                _resize_map(
                    structural_reliability,
                    output_size,
                )
                .clamp(
                    0.0,
                    1.0,
                )
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

        "pair_match_confidence": (
            pair_match_confidence
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

    Run3 deliberately does not interpret OV relation channels because their
    semantic meaning is not part of the current validated mechanism.

    OV contributes only:
        soft_change
        confidence
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
                ][
                    level
                ]
            )

            confidence = (
                ov_pack[
                    "confidence"
                ][
                    level
                ]
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
            soft_change.shape[
                0
            ]
            != confidence.shape[
                0
            ]
        ):
            raise ValueError(
                f"OV {level} soft_change/confidence batch sizes must match"
            )

        values.append(
            _resize_map(
                soft_change.detach(),
                output_size,
            ).clamp(
                0.0,
                1.0,
            )
        )

        confidences.append(
            _resize_map(
                confidence.detach(),
                output_size,
            ).clamp(
                0.0,
                1.0,
            )
        )

    confidence_mass = (
        confidences[
            0
        ]
        + confidences[
            1
        ]
    )

    semantic_prior = torch.where(
        confidence_mass
        > 0,
        (
            values[
                0
            ]
            * confidences[
                0
            ]
            + values[
                1
            ]
            * confidences[
                1
            ]
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

    Prior
    -----
        q_f =
            (
                r_s * q_s
                +
                r_o * q_o
            )
            /
            (
                r_s + r_o
            )

    Reliability
    -----------
        mean available-source confidence
            x
        SAM/OV agreement

    SAM and OV therefore complement one another upstream instead of competing
    as independent residual teachers.
    """

    for (
        tensor,
        name,
    ) in (
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

    sam_reliability = (
        sam_reliability
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
        mass
        > 0,
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
        sam_reliability
        > 0
    )

    ov_available = (
        ov_reliability
        > 0
    )

    source_count = (
        sam_available.float()
        + ov_available.float()
    )

    mean_confidence = torch.where(
        source_count
        > 0,
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
        (
            1.0
            - raw_conflict
        ),
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


# ============================================================================
# Complete foundation prior
# ============================================================================


@torch.no_grad()
def build_foundation_prior(
    teacher_pack: Dict[str, Dict],
    output_size: Tuple[int, int],
    use_ov: bool = True,
    use_sam: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Construct Run3 foundation priors without Student or GT access.

    Full Run3
    ---------
        BT-SAM structural prior
                 +
        OV semantic prior
                 ↓
        fused foundation prior

    R3A ablation
    -------------
        use_ov=False

        BT-SAM structural prior
                 ↓
        foundation prior
    """
    if teacher_pack is None:
        raise ValueError(
            "teacher_pack is required for BT-SAM-RDT"
        )

    if not use_sam:
        # --------------------------------------------------------------
        # R3O ablation: OV semantic prior only (no BT-SAM structural prior).
        # The online Fast/EMA teacher and GT audit remain unchanged.
        # --------------------------------------------------------------
        if not use_ov:
            raise ValueError(
                "use_sam=False requires use_ov=True"
            )

        if "ov" not in teacher_pack:
            raise KeyError(
                "teacher_pack missing 'ov'"
            )

        ov = build_ov_prior(
            teacher_pack["ov"],
            output_size=tuple(int(v) for v in output_size),
        )

        ov_prior = ov["prior"]
        ov_reliability = ov["reliability"]
        zeros = torch.zeros_like(ov_prior)

        return {
            "sam_prior": zeros,
            "sam_reliability": zeros,
            "pair_match_ratio": zeros,
            "pair_match_iou": zeros,
            "pair_match_cov12": zeros,
            "pair_match_cov21": zeros,
            "pair_match_confidence": zeros,
            "ov_prior": ov_prior,
            "ov_reliability": ov_reliability,
            "fused_prior": ov_prior,
            "fused_reliability": ov_reliability,
            "sam_ov_conflict": zeros,
            "both_available": torch.zeros_like(zeros, dtype=torch.bool),
        }

    if (
        "sam"
        not in teacher_pack
    ):
        raise KeyError(
            "teacher_pack missing 'sam'"
        )

    output_size = tuple(
        int(v)
        for v in output_size
    )

    sam = (
        build_bitemporal_structural_prior(
            teacher_pack[
                "sam"
            ],
            output_size=output_size,
        )
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

        "pair_match_confidence": (
            sam[
                "pair_match_confidence"
            ]
        ),
    }

    # ======================================================================
    # Full Run3: SAM + OV
    # ======================================================================

    if use_ov:
        if (
            "ov"
            not in teacher_pack
        ):
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
            sam_prior=(
                sam[
                    "prior"
                ]
            ),
            sam_reliability=(
                sam[
                    "reliability"
                ]
            ),
            ov_prior=(
                ov[
                    "prior"
                ]
            ),
            ov_reliability=(
                ov[
                    "reliability"
                ]
            ),
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

    # ======================================================================
    # R3A: BT-SAM only
    # ======================================================================

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
# Single fused-teacher GT safety audit
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
        E_s =
            (p - y)^2

        E_t =
            (q - y)^2

        gain =
            E_s - E_t

        teacher is admitted iff:

            reliability > 0
            AND
            gain > numerical tolerance

    GT never constructs the foundation prior or dynamic teacher proposal.

    Run3 deliberately contains no:
        - SAM-vs-OV teacher competition;
        - regional router;
        - gradient-concordance gate;
        - learned action selector;
        - utility-margin tuning.
    """

    for (
        tensor,
        name,
    ) in (
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
            "prediction/target/proposal/reliability "
            "must have identical shapes"
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
        r
        > 0
    )

    tolerance = (
        8
        * torch.finfo(
            p.dtype
        ).eps
    )

    if (
        policy
        == "advantage"
    ):
        eligible = (
            available
            & (
                gain
                > tolerance
            )
        )

    elif (
        policy
        == "quality"
    ):
        # Negative/control ablation:
        # teacher quality/support only, without GT positive-gain gating.
        eligible = available

    else:
        raise ValueError(
            "policy must be 'advantage' or 'quality'"
        )

    # GT decides only admit/reject.
    #
    # Distillation strength remains controlled by the teacher's independent
    # reliability map.
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

    This has the same gradient with respect to Student probability as soft BCE
    while removing the target's irreducible Bernoulli entropy from the logged
    KD value.
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
            p.expand_as(
                t
            ),
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