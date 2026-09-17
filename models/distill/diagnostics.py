"""Run3 BT-SAM-RDT detached student-difficulty diagnostics.

This module contains only Student/GT diagnostic utilities.

It does NOT:
    - construct SAM teacher priors;
    - perform T1/T2 SAM instance matching;
    - fuse SAM and OV;
    - route teachers;
    - create Student gradients.

Bi-temporal SAM diagnostics such as pair-match IoU / directional coverage are
computed directly in task_space.py, where the structural correspondence is
actually constructed.

All connected-component analysis here uses 4-connectivity.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


# ============================================================================
# Public diagnostic schema
# ============================================================================


H_NAMES = (
    "boundary_miss",
    "fragmentation",
    "small_change_miss",
    "bg_false_alarm",
    "uncertainty",
    "change_ratio",
)


# ============================================================================
# Generic masked statistic
# ============================================================================


def masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a per-image masked mean.

    Empty support returns exactly zero rather than inventing a pseudo-count.

    Parameters
    ----------
    x:
        Tensor with batch dimension first.

    mask:
        Tensor broadcastable to x.

    Returns
    -------
    torch.Tensor
        Per-image values with shape [B].
    """
    if not torch.is_tensor(x):
        raise TypeError(
            "x must be a torch.Tensor"
        )

    if not torch.is_tensor(mask):
        raise TypeError(
            "mask must be a torch.Tensor"
        )

    if x.ndim < 2:
        raise ValueError(
            "x must contain a batch dimension and at least one data dimension"
        )

    if mask.shape[0] != x.shape[0]:
        raise ValueError(
            "x and mask must have the same batch size"
        )

    try:
        expanded_mask = (
            mask.expand_as(x)
        )
    except RuntimeError as exc:
        raise ValueError(
            "mask must be broadcastable to x"
        ) from exc

    dims = tuple(
        range(
            1,
            x.ndim,
        )
    )

    numerator = (
        x
        * expanded_mask
    ).sum(
        dims
    )

    denominator = (
        expanded_mask
        .sum(
            dims
        )
        .clamp_min(
            1.0
        )
    )

    return (
        numerator
        / denominator
    )


# ============================================================================
# Student difficulty diagnosis
# ============================================================================


@torch.no_grad()
def build_cd_difficulty(
    prediction: torch.Tensor,
    target: torch.Tensor,
    boundary_radius: int = 2,
    small_area: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    Build detached binary change-detection difficulty diagnostics.

    The returned six-dimensional difficulty vector is:

        h_i = [
            boundary_miss,
            fragmentation,
            small_change_miss,
            bg_false_alarm,
            uncertainty,
            change_ratio,
        ]

    These statistics describe the CURRENT Student failure state. They are used
    only for diagnostics / detached weighting in Run3.

    No gradient from this function reaches the Student.

    Parameters
    ----------
    prediction:
        Student probability map [B,1,H,W].

    target:
        Binary GT [B,1,H,W].

    boundary_radius:
        Radius used to construct the inner GT boundary band.

    small_area:
        Connected GT components with area <= small_area are treated as
        small-change regions.

    Returns
    -------
    dict
        h:
            [B,6] difficulty vector.

        valid:
            [B,4] availability flags for:
                boundary
                interior
                small
                background

        masks:
            Detached spatial masks.

        importance:
            Detached [B,1,H,W] weighting map used by the Run3 auxiliary.
    """
    boundary_radius = int(
        boundary_radius
    )

    small_area = int(
        small_area
    )

    if boundary_radius < 1:
        raise ValueError(
            "boundary_radius must be positive"
        )

    if small_area < 1:
        raise ValueError(
            "small_area must be positive"
        )

    if (
        not torch.is_tensor(prediction)
        or not torch.is_tensor(target)
    ):
        raise TypeError(
            "prediction and target must be torch tensors"
        )

    if (
        prediction.ndim != 4
        or prediction.shape[1] != 1
    ):
        raise ValueError(
            "prediction must be [B,1,H,W]"
        )

    if (
        target.ndim != 4
        or target.shape[1] != 1
    ):
        raise ValueError(
            "target must be [B,1,H,W]"
        )

    if (
        prediction.shape
        != target.shape
    ):
        raise ValueError(
            "prediction and target must have identical shapes"
        )

    if (
        prediction.shape[0]
        < 1
    ):
        raise ValueError(
            "batch must be non-empty"
        )

    p = (
        prediction
        .detach()
        .float()
        .clamp(
            1e-6,
            1.0 - 1e-6,
        )
    )

    y = (
        target
        .detach()
        .float()
    )

    if not torch.all(
        (
            y == 0
        )
        | (
            y == 1
        )
    ):
        raise ValueError(
            "target must contain only binary 0/1 values"
        )

    # ======================================================================
    # 1. GT boundary / interior
    # ======================================================================

    kernel_size = (
        2
        * boundary_radius
        + 1
    )

    # Erode the positive GT mask through complementary max pooling.
    interior = (
        1.0
        - F.max_pool2d(
            1.0 - y,
            kernel_size=kernel_size,
            stride=1,
            padding=boundary_radius,
        )
    ).clamp(
        0.0,
        1.0,
    )

    inner_boundary = (
        y
        - interior
    ).clamp(
        0.0,
        1.0,
    )

    # ======================================================================
    # 2. Hard Student error maps
    # ======================================================================

    hard_prediction = (
        p > 0.5
    ).float()

    false_negative = (
        y
        * (
            1.0
            - hard_prediction
        )
    )

    false_positive = (
        (
            1.0
            - y
        )
        * hard_prediction
    )

    # ======================================================================
    # 3. Connected-component Student diagnostics
    #
    # This remains detached CPU analysis.
    # ======================================================================

    batch_size = (
        y.shape[0]
    )

    height = (
        y.shape[2]
    )

    width = (
        y.shape[3]
    )

    small_np = np.zeros(
        (
            batch_size,
            1,
            height,
            width,
        ),
        dtype=np.float32,
    )

    fragmented_np = np.zeros_like(
        small_np
    )

    fragmentation_scores = []

    gt_np = (
        y[
            :,
            0,
        ]
        .cpu()
        .numpy()
        .astype(
            bool
        )
    )

    prediction_np = (
        hard_prediction[
            :,
            0,
        ]
        .cpu()
        .numpy()
        .astype(
            bool
        )
    )

    # SciPy's default 2-D structure uses 4-connectivity.
    for sample_index, (
        gt_mask,
        predicted_mask,
    ) in enumerate(
        zip(
            gt_np,
            prediction_np,
        )
    ):
        # --------------------------------------------------------------
        # GT connected components
        # --------------------------------------------------------------
        gt_labels, gt_component_count = (
            ndimage.label(
                gt_mask
            )
        )

        if (
            gt_component_count
            > 0
        ):
            gt_areas = np.bincount(
                gt_labels.ravel(),
                minlength=(
                    gt_component_count
                    + 1
                ),
            )

            is_small_component = (
                (
                    gt_areas
                    <= small_area
                )
                & (
                    gt_areas
                    > 0
                )
            )

            # Background label 0 is never a change object.
            is_small_component[
                0
            ] = False

            small_np[
                sample_index,
                0,
            ] = (
                is_small_component[
                    gt_labels
                ]
                .astype(
                    np.float32
                )
            )

        # --------------------------------------------------------------
        # Fragmentation
        #
        # Examine connected predicted pieces that lie inside GT objects.
        # --------------------------------------------------------------
        predicted_inside_gt = (
            predicted_mask
            & gt_mask
        )

        piece_labels, piece_count = (
            ndimage.label(
                predicted_inside_gt
            )
        )

        if (
            gt_component_count > 0
            and piece_count > 0
        ):
            piece_ids = np.arange(
                1,
                piece_count + 1,
            )

            # Each predicted piece lies inside GT. Determine which GT object
            # owns that piece.
            parent_gt = (
                ndimage.maximum(
                    gt_labels,
                    labels=piece_labels,
                    index=piece_ids,
                )
                .astype(
                    np.int64
                )
            )

            piece_count_per_gt = (
                np.bincount(
                    parent_gt,
                    minlength=(
                        gt_component_count
                        + 1
                    ),
                )
            )

        else:
            piece_count_per_gt = (
                np.zeros(
                    gt_component_count
                    + 1,
                    dtype=np.int64,
                )
            )

        if (
            gt_component_count
            > 0
        ):
            fragmented_components = (
                piece_count_per_gt
                > 1
            )

            fragmented_np[
                sample_index,
                0,
            ] = (
                (
                    fragmented_components[
                        gt_labels
                    ]
                )
                & gt_mask
            ).astype(
                np.float32
            )

            extra_pieces = (
                np.maximum(
                    piece_count_per_gt[
                        1:
                    ]
                    - 1,
                    0,
                )
                .sum()
            )

            total_piece_mass = (
                np.maximum(
                    piece_count_per_gt[
                        1:
                    ],
                    1,
                )
                .sum()
            )

            fragmentation_score = float(
                extra_pieces
                / max(
                    total_piece_mass,
                    1,
                )
            )

        else:
            fragmentation_score = 0.0

        fragmentation_scores.append(
            fragmentation_score
        )

    small_change = (
        torch.from_numpy(
            small_np
        )
        .to(
            device=p.device,
            dtype=p.dtype,
        )
    )

    fragmented = (
        torch.from_numpy(
            fragmented_np
        )
        .to(
            device=p.device,
            dtype=p.dtype,
        )
    )

    fragmentation_score = (
        torch.tensor(
            fragmentation_scores,
            device=p.device,
            dtype=p.dtype,
        )
    )

    # ======================================================================
    # 4. Student uncertainty
    # ======================================================================

    entropy = -(
        p
        * p.log()
        + (
            1.0
            - p
        )
        * (
            1.0
            - p
        ).log()
    ) / float(
        np.log(
            2.0
        )
    )

    entropy = (
        entropy
        .clamp(
            0.0,
            1.0,
        )
    )

    # ======================================================================
    # 5. Six-dimensional Student difficulty vector
    # ======================================================================

    boundary_miss = masked_mean(
        false_negative,
        inner_boundary,
    )

    small_change_miss = masked_mean(
        false_negative,
        small_change,
    )

    background_false_alarm = masked_mean(
        false_positive,
        1.0 - y,
    )

    uncertainty = (
        entropy.mean(
            (
                1,
                2,
                3,
            )
        )
    )

    change_ratio = (
        y.mean(
            (
                1,
                2,
                3,
            )
        )
    )

    h = torch.stack(
        (
            boundary_miss,
            fragmentation_score,
            small_change_miss,
            background_false_alarm,
            uncertainty,
            change_ratio,
        ),
        dim=1,
    )

    # ======================================================================
    # 6. Spatial masks
    # ======================================================================

    masks = {
        "boundary": (
            inner_boundary
        ),
        "interior": (
            interior
        ),
        "small": (
            small_change
        ),
        "background": (
            1.0 - y
        ),
        "fragmented": (
            fragmented
        ),
        "false_negative": (
            false_negative
        ),
        "false_positive": (
            false_positive
        ),
        "uncertainty": (
            entropy
        ),
    }

    # ======================================================================
    # 7. Detached auxiliary importance map
    #
    # This is not itself a teacher.
    #
    # It only emphasizes current Student failure regions when computing:
    #     - Student KD;
    #     - Fast Teacher fitting;
    #     - diagnostic summaries.
    #
    # No gradient enters through this map.
    # ======================================================================

    importance = (
        1.0
        + false_negative
        * inner_boundary
        + fragmented
        * interior
        + false_negative
        * small_change
        + false_positive
        + entropy
    )

    importance = (
        importance
        .detach()
        .float()
    )

    # ======================================================================
    # 8. Diagnostic validity
    #
    # Some images contain no small object or no positive change at all.
    # Such missing regions must not be interpreted as measured zero error.
    # ======================================================================

    valid = torch.stack(
        (
            masks[
                "boundary"
            ].sum(
                (
                    1,
                    2,
                    3,
                )
            )
            > 0,

            masks[
                "interior"
            ].sum(
                (
                    1,
                    2,
                    3,
                )
            )
            > 0,

            masks[
                "small"
            ].sum(
                (
                    1,
                    2,
                    3,
                )
            )
            > 0,

            masks[
                "background"
            ].sum(
                (
                    1,
                    2,
                    3,
                )
            )
            > 0,
        ),
        dim=1,
    )

    return {
        "h": (
            h.detach()
        ),
        "masks": (
            masks
        ),
        "importance": (
            importance
        ),
        "valid": (
            valid.detach()
        ),
    }