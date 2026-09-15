"""SCGR task-space routing for RDT-CD.

SCGR = Sparse-Change Gradient-Concordant Routing.

This file is TRAINING-ONLY and parameter-free. It provides:
  1) SAMStruct / OVCDistill cache priors in binary-change probability space;
  2) sparse-change-aware regional teacher routing;
  3) classifier-feature gradient-concordance auditing;
  4) remaining-error-mass-conserving KD weights.

Teacher order is fixed:
  0 -> SAMStruct structural teacher
  1 -> OVCDistill semantic teacher

A teacher is admitted only when it is available, gives positive GT-audited
Brier utility, and has positive analytical classifier-gradient concordance
with the GT direction. Otherwise the region is rejected.

Nothing in this file belongs to the deploy graph.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .losses import resize


NUM_TEACHERS = 2
SAM_INDEX = 0
OV_INDEX = 1
REJECT_INDEX = 2
DEFAULT_REGION_SIZE = 16
EPS = 1e-8


def _check_4d(name: str, x: torch.Tensor, channels: int) -> None:
    if x.ndim != 4 or x.shape[1] != channels:
        raise ValueError(
            f"{name} must be [B,{channels},H,W], got {tuple(x.shape)}"
        )
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} contains non-finite values")


def _check_binary_target(target: torch.Tensor) -> None:
    _check_4d("target", target, 1)
    if not bool(torch.all((target == 0) | (target == 1))):
        raise ValueError("target must be binary 0/1")


def _region_sum(x: torch.Tensor, region_size: int) -> torch.Tensor:
    """Non-overlapping regional sum; incomplete borders are zero-padded."""
    if region_size <= 0:
        raise ValueError("region_size must be positive")
    h, w = x.shape[-2:]
    pad_h = (-h) % region_size
    pad_w = (-w) % region_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return F.avg_pool2d(
        x,
        kernel_size=region_size,
        stride=region_size,
    ) * float(region_size * region_size)


def _expand_regions(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x.float(), size=size, mode="nearest")


@torch.no_grad()
def instance_transport(
    probability: torch.Tensor,
    ids: torch.Tensor,
    quality: torch.Tensor,
    boundary: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SAM instance transport without treating SAM IDs as change labels.

    Each supported pixel receives a quality-weighted leave-one-out mean of the
    detached student probability from the same SAM instance. Boundary pixels
    are downweighted as sources. Unsupported/singleton pixels return identity
    with confidence zero.
    """
    if ids.shape != probability.shape:
        raise ValueError("SAM instance_id must match prediction shape")
    if quality.shape != probability.shape or boundary.shape != probability.shape:
        raise ValueError("SAM quality/boundary must match prediction shape")

    proposal_list = []
    confidence_list = []

    for p, labels, q, edge in zip(
        probability.float(),
        ids,
        quality.float(),
        boundary.float(),
    ):
        p = p.flatten()
        labels = labels.flatten()
        q = q.flatten().clamp(0.0, 1.0)
        edge = edge.flatten().clamp(0.0, 1.0)

        _, compact = torch.unique(labels, sorted=True, return_inverse=True)
        source = q * (1.0 - edge) * (labels > 0).to(q.dtype)

        mass = torch.zeros_like(p).scatter_add_(0, compact, source)
        value_sum = torch.zeros_like(p).scatter_add_(0, compact, source * p)

        remaining = mass[compact] - source
        valid = (labels > 0) & (remaining > 1e-6)
        transported = (
            (value_sum[compact] - source * p) / remaining.clamp_min(1e-6)
        ).clamp(0.0, 1.0)

        proposal_list.append(torch.where(valid, transported, p))
        confidence_list.append(q * valid.to(q.dtype))

    shape = probability.shape
    return (
        torch.stack(proposal_list).reshape(shape),
        torch.stack(confidence_list).reshape(shape),
    )


@torch.no_grad()
def build_task_proposals(
    prediction: torch.Tensor,
    pack: Dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build detached [SAM, OV] cache priors; GT is never used here."""
    _check_4d("prediction", prediction, 1)
    if not isinstance(pack, dict) or "sam" not in pack or "ov" not in pack:
        raise ValueError("pack must contain synchronously replayed 'sam' and 'ov'")

    p = prediction.detach().float().clamp(0.0, 1.0)

    sam_values = []
    sam_conf = []
    for temporal in ("t1", "t2"):
        cache = pack["sam"][temporal]
        value, conf = instance_transport(
            p,
            cache["instance_id"].detach(),
            cache["quality"].detach().float(),
            cache["boundary"].detach().float(),
        )
        sam_values.append(value)
        sam_conf.append(conf)

    sam_mass = sam_conf[0] + sam_conf[1]
    sam = torch.where(
        sam_mass > 0,
        (
            sam_values[0] * sam_conf[0]
            + sam_values[1] * sam_conf[1]
        ) / sam_mass.clamp_min(EPS),
        p,
    )
    q_sam = (0.5 * sam_mass).clamp(0.0, 1.0)

    ov_values = []
    ov_conf = []
    for level in ("l1", "l2"):
        ov_values.append(
            resize(
                pack["ov"]["soft_change"][level].detach(),
                p.shape[-2:],
            ).clamp(0.0, 1.0)
        )
        ov_conf.append(
            resize(
                pack["ov"]["confidence"][level].detach(),
                p.shape[-2:],
            ).clamp(0.0, 1.0)
        )

    ov_mass = ov_conf[0] + ov_conf[1]
    ov = torch.where(
        ov_mass > 0,
        (
            ov_values[0] * ov_conf[0]
            + ov_values[1] * ov_conf[1]
        ) / ov_mass.clamp_min(EPS),
        p,
    )
    q_ov = (0.5 * ov_mass).clamp(0.0, 1.0)

    return torch.cat((sam, ov), 1), torch.cat((q_sam, q_ov), 1)


def bernoulli_kl(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """KL(Ber(target) || Ber(prediction)); derivative equals soft BCE."""
    p = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    t = target.detach().float().clamp(0.0, 1.0)
    return (
        F.binary_cross_entropy(p.expand_as(t), t, reduction="none")
        + torch.special.xlogy(t, t)
        + torch.special.xlogy(1.0 - t, 1.0 - t)
    )


@torch.no_grad()
def sparse_class_balance(target: torch.Tensor) -> torch.Tensor:
    """Give change/background equal spatial budget when both are present."""
    _check_binary_target(target)
    y = target.float()
    n = float(y.shape[-2] * y.shape[-1])

    n_change = y.sum((2, 3), keepdim=True)
    n_bg = (1.0 - y).sum((2, 3), keepdim=True)
    has_change = n_change > 0
    has_bg = n_bg > 0
    both = has_change & has_bg

    change_budget = torch.where(
        both,
        torch.full_like(n_change, 0.5),
        has_change.to(y.dtype),
    )
    bg_budget = torch.where(
        both,
        torch.full_like(n_bg, 0.5),
        has_bg.to(y.dtype),
    )

    change_scale = torch.where(
        has_change,
        change_budget * n / n_change.clamp_min(1.0),
        torch.zeros_like(n_change),
    )
    bg_scale = torch.where(
        has_bg,
        bg_budget * n / n_bg.clamp_min(1.0),
        torch.zeros_like(n_bg),
    )
    return y * change_scale + (1.0 - y) * bg_scale


@torch.no_grad()
def build_error_mass(
    prediction: torch.Tensor,
    target: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Remaining balanced student error mass used by SCGR."""
    _check_4d("prediction", prediction, 1)
    _check_binary_target(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")

    p = prediction.detach().float().clamp(0.0, 1.0)
    y = target.detach().float()
    if importance is None:
        imp = torch.ones_like(p)
    else:
        if importance.shape != p.shape:
            raise ValueError("importance must match prediction shape")
        imp = importance.detach().float().clamp_min(0.0)

    class_balance = sparse_class_balance(y)
    absolute_error = (p - y).abs()
    error_mass = absolute_error * imp * class_balance
    return {
        "absolute_error": absolute_error,
        "importance": imp,
        "class_balance": class_balance,
        "error_mass": error_mass,
    }


@torch.no_grad()
def classifier_gradient_concordance(
    feature: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    proposals: torch.Tensor,
    gradient_weight: torch.Tensor,
    region_grid: Tuple[int, int],
) -> torch.Tensor:
    """Low-cost analytical proxy of final 1x1-classifier gradient cosine.

    `feature` must be decoder p2, i.e. the input of A2Net `decoder.cls`.
    For BCE/KL in logit space, local coefficients are approximately p-y and
    p-q. They are reduced to p2 resolution and multiplied by p2 features;
    regional channel vectors then approximate classifier-weight gradients.
    """
    f = feature.detach().float()
    p = prediction.detach().float()
    y = target.detach().float()
    q = proposals.detach().float()
    w = gradient_weight.detach().float().clamp_min(0.0)
    feature_size = f.shape[-2:]

    p_f = F.interpolate(p, size=feature_size, mode="area")
    y_f = F.interpolate(y, size=feature_size, mode="area")
    w_f = F.interpolate(w, size=feature_size, mode="area")

    gt_vector = F.adaptive_avg_pool2d(
        f * ((p_f - y_f) * w_f),
        output_size=region_grid,
    )
    gt_norm = gt_vector.square().sum(1).sqrt()

    cosines = []
    for k in range(NUM_TEACHERS):
        q_f = F.interpolate(
            q[:, k : k + 1],
            size=feature_size,
            mode="area",
        )
        kd_vector = F.adaptive_avg_pool2d(
            f * ((p_f - q_f) * w_f),
            output_size=region_grid,
        )
        kd_norm = kd_vector.square().sum(1).sqrt()
        dot = (gt_vector * kd_vector).sum(1)
        denom = gt_norm * kd_norm
        cosine = torch.where(
            denom > 1e-12,
            dot / denom.clamp_min(1e-12),
            torch.zeros_like(dot),
        ).clamp(-1.0, 1.0)
        cosines.append(cosine.unsqueeze(1))

    return torch.cat(cosines, 1)


def _active_teacher_vector(
    teacher: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if teacher == "both":
        values = (1.0, 1.0)
    elif teacher == "sam":
        values = (1.0, 0.0)
    elif teacher == "ov":
        values = (0.0, 1.0)
    else:
        raise ValueError("teacher must be both/sam/ov")
    return torch.tensor(values, device=device, dtype=dtype)


@torch.no_grad()
def task_space_route(
    prediction: torch.Tensor,
    target: torch.Tensor,
    proposals: torch.Tensor,
    quality: torch.Tensor,
    *,
    feature: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
    policy: str = "scgr",
    teacher: str = "both",
    force_action: Optional[int] = None,
    region_size: int = DEFAULT_REGION_SIZE,
) -> Dict[str, torch.Tensor]:
    """Sparse-Change Gradient-Concordant Routing.

    The function intentionally requires `feature`; there is no fallback to the
    old pixel-only Brier router.
    """
    if policy != "scgr":
        raise ValueError("task_space_route only supports policy='scgr'")
    if region_size <= 0:
        raise ValueError("region_size must be positive")

    _check_4d("prediction", prediction, 1)
    _check_binary_target(target)
    _check_4d("proposals", proposals, NUM_TEACHERS)
    _check_4d("quality", quality, NUM_TEACHERS)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    expected = (
        prediction.shape[0],
        NUM_TEACHERS,
        prediction.shape[-2],
        prediction.shape[-1],
    )
    if tuple(proposals.shape) != expected or tuple(quality.shape) != expected:
        raise ValueError(f"proposals/quality must both have shape {expected}")
    if feature.ndim != 4 or feature.shape[0] != prediction.shape[0]:
        raise ValueError("feature must be [B,C,Hf,Wf] with matching batch size")
    if not bool(torch.isfinite(feature).all()):
        raise ValueError("feature contains non-finite values")

    p = prediction.detach().float().clamp(0.0, 1.0)
    y = target.detach().float()
    q = proposals.detach().float().clamp(0.0, 1.0)
    reliability = quality.detach().float().clamp(0.0, 1.0)

    if importance is None:
        imp = torch.ones_like(p)
    else:
        if importance.shape != p.shape:
            raise ValueError("importance must match prediction shape")
        imp = importance.detach().float().clamp_min(0.0)

    mass_info = build_error_mass(p, y, imp)
    error_mass = mass_info["error_mass"]
    class_balance = mass_info["class_balance"]

    # 1) GT-audited regional output utility.
    base_error = (p - y).square()
    proposal_error = (q - y).square()
    gain_map = base_error - proposal_error

    region_mass = _region_sum(error_mass, region_size)
    gain_numerator = _region_sum(gain_map * error_mass, region_size)
    region_utility = gain_numerator / region_mass.clamp_min(EPS)

    base_numerator = _region_sum(base_error * error_mass, region_size)
    region_relative_utility = (
        gain_numerator / base_numerator.clamp_min(EPS)
    ).clamp(-1.0, 1.0)

    region_quality = (
        _region_sum(reliability * error_mass, region_size)
        / region_mass.clamp_min(EPS)
    ).clamp(0.0, 1.0)

    region_grid = region_mass.shape[-2:]

    # 2) Final-classifier gradient concordance.
    # Do not multiply by absolute error here: p-y already carries residual size.
    gradient_weight = imp * class_balance
    region_concordance = classifier_gradient_concordance(
        feature=feature,
        prediction=p,
        target=y,
        proposals=q,
        gradient_weight=gradient_weight,
        region_grid=region_grid,
    )

    # 3) Candidate availability / ablation restrictions.
    active = _active_teacher_vector(
        teacher,
        p.device,
        p.dtype,
    ).view(1, NUM_TEACHERS, 1, 1)

    if force_action is not None:
        forced_action = int(force_action)
        if forced_action not in (SAM_INDEX, OV_INDEX, REJECT_INDEX):
            raise ValueError("force_action must be 0, 1, or 2")
        if forced_action == REJECT_INDEX:
            active = torch.zeros_like(active)
        else:
            forced = torch.zeros_like(active)
            forced[:, forced_action] = 1.0
            active = active * forced

    region_available = (
        (region_quality > 0.0)
        & (region_mass > EPS)
        & (active > 0.0)
    )

    tolerance = 8.0 * torch.finfo(p.dtype).eps
    region_positive_utility = region_utility > tolerance
    region_positive_concordance = region_concordance > 0.0
    region_eligible = (
        region_available
        & region_positive_utility
        & region_positive_concordance
    )

    # 4) Teacher routing score.
    region_score = (
        region_relative_utility.clamp_min(0.0)
        * region_concordance.clamp_min(0.0)
        * region_quality
        * region_eligible.to(p.dtype)
    )
    score_sum = region_score.sum(1, keepdim=True)
    region_accepted = score_sum > 0.0
    region_mixture = region_score / score_sum.clamp_min(1e-12)

    region_action = torch.where(
        region_accepted[:, 0],
        region_score.argmax(1),
        torch.full(
            region_score.shape[:1] + region_score.shape[2:],
            REJECT_INDEX,
            device=p.device,
            dtype=torch.long,
        ),
    )

    # 5) Expand region decisions and conserve remaining error mass.
    spatial_size = p.shape[-2:]
    mixture_map = _expand_regions(region_mixture, spatial_size)
    local_support = (reliability > 0.0).to(p.dtype) * active
    mixture_map = mixture_map * local_support

    accepted_map = mixture_map.sum(1, keepdim=True) > 0.0
    kd_weight_map = mixture_map * error_mass

    action = _expand_regions(
        region_action.unsqueeze(1).float(),
        spatial_size,
    )[:, 0].long()
    action = torch.where(
        accepted_map[:, 0],
        action,
        torch.full_like(action, REJECT_INDEX),
    )

    relative_map = _expand_regions(region_relative_utility, spatial_size)
    utility_map = _expand_regions(region_utility, spatial_size)
    concordance_map = _expand_regions(region_concordance, spatial_size)
    eligible_map = _expand_regions(
        region_eligible.float(),
        spatial_size,
    ).bool()
    available_map = (reliability > 0.0) & (active > 0.0)

    gradient_conflict_region = (
        region_available
        & region_positive_utility
        & (~region_positive_concordance)
    )
    gradient_conflict_map = _expand_regions(
        gradient_conflict_region.float(),
        spatial_size,
    ).bool()

    return {
        # Common RDT/task-space maps.
        "gain_map": gain_map,
        "relative_map": relative_map,
        "eligible_map": eligible_map,
        "available_map": available_map,
        "mixture_map": mixture_map,
        "effective_map": mixture_map,
        "action": action,
        "accepted_map": accepted_map,
        # SCGR pixel maps.
        "error_mass_map": error_mass,
        "class_balance_map": class_balance,
        "kd_weight_map": kd_weight_map,
        "utility_map": utility_map,
        "concordance_map": concordance_map,
        "gradient_conflict_map": gradient_conflict_map,
        # SCGR region evidence.
        "region_mass": region_mass,
        "region_utility": region_utility,
        "region_relative_utility": region_relative_utility,
        "region_quality": region_quality,
        "region_concordance": region_concordance,
        "region_available": region_available,
        "region_positive_utility": region_positive_utility,
        "region_positive_concordance": region_positive_concordance,
        "region_eligible": region_eligible,
        "region_score": region_score,
        "region_mixture": region_mixture,
        "region_accepted": region_accepted,
        "region_action": region_action,
    }


def routed_bernoulli_kd(
    prediction: torch.Tensor,
    proposals: torch.Tensor,
    route: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Error-mass-conserving teacher -> student KD.

    The denominator is total remaining balanced student error mass, not accepted
    mass and not HxW image area. Therefore sparse useful change regions are not
    drowned by background area, while a tiny accepted island is still unable to
    become full-image-strength supervision.
    """
    _check_4d("prediction", prediction, 1)
    _check_4d("proposals", proposals, NUM_TEACHERS)
    if "kd_weight_map" not in route or "error_mass_map" not in route:
        raise KeyError("SCGR route lacks kd_weight_map/error_mass_map")

    kd_map = bernoulli_kl(prediction, proposals)
    kd_weight = route["kd_weight_map"].detach().float()
    error_mass = route["error_mass_map"].detach().float()

    if kd_weight.shape != kd_map.shape:
        raise ValueError("kd_weight_map must match teacher KD map shape")
    if error_mass.shape != prediction.shape:
        raise ValueError("error_mass_map must match prediction shape")

    denominator = error_mass.sum((2, 3)).clamp_min(EPS)
    per_teacher = (kd_map * kd_weight).sum((2, 3)) / denominator
    total = per_teacher.sum(1).mean()

    accepted_error_mass = kd_weight.sum(1).sum((1, 2))
    total_error_mass = error_mass.sum((1, 2, 3)).clamp_min(EPS)

    return {
        "total": total,
        "loss_per_teacher": per_teacher,
        "kd_map": kd_map,
        "accepted_error_mass": accepted_error_mass,
        "total_error_mass": total_error_mass,
        "effective_per_error_mass_per_image": (
            accepted_error_mass / total_error_mass
        ),
    }
