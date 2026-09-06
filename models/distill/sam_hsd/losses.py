"""Parameter-free Decoder-HSD objectives and confidence-aware SCGR."""

from __future__ import annotations

import torch
import torch.nn.functional as F


SCALE_WEIGHTS = (0.40, 0.30, 0.20, 0.10)
SHIFTS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def weighted_mean(loss, weight, eps=1e-6):
    return (loss * weight).sum() / weight.sum().clamp_min(eps)


def soft_boundary(probability, kernel=3):
    pad = kernel // 2
    dilated = F.max_pool2d(probability, kernel, stride=1, padding=pad)
    eroded = -F.max_pool2d(-probability, kernel, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


def _valid_roll_mask(reference, dy, dx):
    valid = torch.ones_like(reference, dtype=torch.bool)
    if dy < 0:
        valid[..., dy:, :] = False
    elif dy > 0:
        valid[..., :dy, :] = False
    if dx < 0:
        valid[..., :, dx:] = False
    elif dx > 0:
        valid[..., :, :dx] = False
    return valid


def _resize(value, size, mode="bilinear"):
    value = value.float()
    if mode == "nearest":
        return F.interpolate(value, size=size, mode=mode)
    return F.interpolate(value, size=size, mode=mode, align_corners=False)


def scgr_loss(prediction, target, evidence, hard_negative_weight=1.8,
              hard_positive_weight=1.3, confidence_threshold=0.5):
    """Route only high-confidence SAM/GT agreements and conflicts."""
    gt = target > 0.5
    change_score = (evidence["delta"] * evidence["change_conf"]).detach()
    stable_score = (evidence["stable"] * evidence["stable_conf"]).detach()
    confident_change = change_score > confidence_threshold
    confident_stable = stable_score > confidence_threshold

    hard_negative = (~gt) & confident_change
    hard_positive = gt & confident_stable
    reliable_positive = gt & confident_change & (~hard_positive)
    reliable_negative = (~gt) & confident_stable & (~hard_negative)

    weight = torch.zeros_like(prediction, dtype=torch.float32)
    weight = torch.where(reliable_positive | reliable_negative, 1.0, weight)
    weight = torch.where(hard_positive, hard_positive_weight, weight)
    weight = torch.where(hard_negative, hard_negative_weight, weight)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        loss_map = F.binary_cross_entropy(
            prediction.float().clamp(1e-5, 1 - 1e-5),
            target.float(), reduction="none",
        )
    total_pixels = float(target.numel())
    stats = {
        "reliable_positive_ratio": reliable_positive.float().sum() / total_pixels,
        "reliable_negative_ratio": reliable_negative.float().sum() / total_pixels,
        "hard_negative_ratio": hard_negative.float().sum() / total_pixels,
        "hard_positive_ratio": hard_positive.float().sum() / total_pixels,
    }
    return weighted_mean(loss_map, weight), stats


def multiscale_scgr_loss(predictions, target, evidence):
    total = target.new_zeros(())
    stats = None
    for scale_weight, prediction in zip(SCALE_WEIGHTS, predictions):
        loss, current = scgr_loss(prediction, target, evidence)
        total = total + scale_weight * loss
        stats = current if stats is None else stats
    return total, stats


def signed_boundary_loss(predictions, target, evidence, boundary_band=7):
    """Positive structural boundary alignment plus false-edge suppression."""
    gt = target > 0.5
    gt_boundary = soft_boundary(target.float())
    gt_band = F.max_pool2d(
        gt_boundary, boundary_band, stride=1, padding=boundary_band // 2,
    )
    boundary_target = (
        0.85 * evidence["boundary_delta"]
        + 0.15 * evidence["boundary_union"]
    ).clamp(0, 1)
    positive_weight = (boundary_target * gt_band).detach()
    stable_interior = (
        evidence["stable"] * evidence["stable_conf"]
        * (1.0 - evidence["boundary_union"])
    )
    negative_weight = (
        stable_interior * (~gt).float() * (1.0 - gt_band)
    ).detach()

    positive = target.new_zeros(())
    negative = target.new_zeros(())
    for scale_weight, prediction in zip(SCALE_WEIGHTS, predictions):
        predicted_boundary = soft_boundary(prediction.float())
        positive_map = F.smooth_l1_loss(
            predicted_boundary, torch.ones_like(predicted_boundary), reduction="none",
        )
        negative_map = F.smooth_l1_loss(
            predicted_boundary, torch.zeros_like(predicted_boundary), reduction="none",
        )
        positive = positive + scale_weight * weighted_mean(
            positive_map, positive_weight,
        )
        negative = negative + scale_weight * weighted_mean(
            negative_map, negative_weight,
        )
    return {
        "total": 0.5 * positive + 0.5 * negative,
        "positive": positive,
        "negative": negative,
    }


def _composite_ids(instance, stride):
    offsets = torch.arange(
        instance.shape[0], device=instance.device, dtype=torch.long,
    ).view(-1, 1, 1, 1) * stride
    return instance.long() + offsets, stride * instance.shape[0]


def _instance_balanced_pair_loss(prediction, target, evidence, mode):
    """Average pairs inside each instance, then average instances equally."""
    if mode not in {"changed", "affinity", "stable"}:
        raise ValueError(f"Unknown instance pair mode: {mode}")
    change = target > 0.5
    delta_score = (evidence["delta"] * evidence["change_conf"]).detach()
    stable_score = (evidence["stable"] * evidence["stable_conf"]).detach()
    instance_loss_sum = prediction.new_zeros(())
    instance_count = prediction.new_zeros(())

    for time_key in ("t1", "t2"):
        instance = evidence["structure"][time_key]["instance_id"]
        quality = evidence["structure"][time_key]["quality"].float()
        composite, buckets = _composite_ids(instance, evidence["instance_stride"])
        loss_by_instance = prediction.new_zeros(buckets)
        weight_by_instance = prediction.new_zeros(buckets)
        for dy, dx in SHIFTS:
            shifted_instance = torch.roll(instance, (dy, dx), (-2, -1))
            shifted_quality = torch.roll(quality, (dy, dx), (-2, -1))
            pair_quality = torch.minimum(quality, shifted_quality)
            valid = (
                (instance == shifted_instance) & (instance > 0)
                & (pair_quality > 0.7) & _valid_roll_mask(instance, dy, dx)
            )
            if mode == "changed":
                shifted_change = torch.roll(change, (dy, dx), (-2, -1))
                valid = valid & change & shifted_change
            elif mode == "affinity":
                shifted_score = torch.roll(delta_score, (dy, dx), (-2, -1))
                valid = valid & (delta_score > 0.5) & (shifted_score > 0.5)
            else:
                shifted_change = torch.roll(change, (dy, dx), (-2, -1))
                shifted_score = torch.roll(stable_score, (dy, dx), (-2, -1))
                valid = (
                    valid & (~change) & (~shifted_change)
                    & (stable_score > 0.5) & (shifted_score > 0.5)
                )
            difference = torch.abs(
                prediction - torch.roll(prediction, (dy, dx), (-2, -1))
            )
            weight = valid.float() * pair_quality
            ids = composite.reshape(-1)
            loss_by_instance.scatter_add_(
                0, ids, (difference * weight).reshape(-1),
            )
            weight_by_instance.scatter_add_(0, ids, weight.reshape(-1))
        present = weight_by_instance > 0
        # Bucket zero belongs to background in the first sample and never has
        # valid weight, so no special-case subtraction is required.
        if present.any():
            means = loss_by_instance[present] / weight_by_instance[present]
            instance_loss_sum = instance_loss_sum + means.sum()
            instance_count = instance_count + present.sum()
    return instance_loss_sum / instance_count.clamp_min(1.0)


def _instance_balanced_stable_suppression(prediction, target, evidence):
    stable_score = (evidence["stable"] * evidence["stable_conf"]).detach()
    unchanged = target <= 0.5
    total, count = prediction.new_zeros(()), prediction.new_zeros(())
    for time_key in ("t1", "t2"):
        instance = evidence["structure"][time_key]["instance_id"]
        quality = evidence["structure"][time_key]["quality"].float()
        composite, buckets = _composite_ids(instance, evidence["instance_stride"])
        valid = (
            (instance > 0) & (quality > 0.7) & unchanged
            & (stable_score > 0.5)
        )
        weight = valid.float() * quality
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            value = F.binary_cross_entropy(
                prediction.float().clamp(1e-5, 1 - 1e-5),
                torch.zeros_like(prediction).float(), reduction="none",
            )
        sums = prediction.new_zeros(buckets)
        weights = prediction.new_zeros(buckets)
        ids = composite.reshape(-1)
        sums.scatter_add_(0, ids, (value * weight).reshape(-1))
        weights.scatter_add_(0, ids, weight.reshape(-1))
        present = weights > 0
        if present.any():
            total = total + (sums[present] / weights[present]).sum()
            count = count + present.sum()
    return total / count.clamp_min(1.0)


def _prediction_contrast(prediction, target, evidence, margin=0.5):
    gt = target > 0.5
    boundary_score = (
        evidence["boundary_delta"] * evidence["change_conf"]
    ).detach()
    total, weight_total = prediction.new_zeros(()), prediction.new_zeros(())
    for dy, dx in SHIFTS:
        shifted_gt = torch.roll(gt, (dy, dx), (-2, -1))
        shifted_score = torch.roll(boundary_score, (dy, dx), (-2, -1))
        pair_score = torch.maximum(boundary_score, shifted_score)
        valid = (
            (gt != shifted_gt) & (pair_score > 0.3)
            & _valid_roll_mask(gt, dy, dx)
        )
        difference = torch.abs(
            prediction - torch.roll(prediction, (dy, dx), (-2, -1))
        )
        weight = valid.float() * pair_score
        total = total + (F.relu(margin - difference) * weight).sum()
        weight_total = weight_total + weight.sum()
    return total / weight_total.clamp_min(1e-6)


def _feature_stable_relation(feature, target, evidence):
    size = feature.shape[-2:]
    stable_score = _resize(
        evidence["stable"] * evidence["stable_conf"], size,
    ).detach()
    unchanged = _resize(target, size, mode="nearest") <= 0.5
    total, count = feature.new_zeros(()), feature.new_zeros(())
    normalized = F.normalize(feature.float(), dim=1, eps=1e-6)
    for time_key in ("t1", "t2"):
        source = evidence["structure"][time_key]
        instance = _resize(source["instance_id"], size, mode="nearest").round().to(torch.int32)
        quality = _resize(source["quality"], size)
        composite, buckets = _composite_ids(instance, evidence["instance_stride"])
        sums = feature.new_zeros(buckets)
        weights = feature.new_zeros(buckets)
        for dy, dx in SHIFTS:
            shifted_instance = torch.roll(instance, (dy, dx), (-2, -1))
            shifted_quality = torch.roll(quality, (dy, dx), (-2, -1))
            shifted_stable = torch.roll(stable_score, (dy, dx), (-2, -1))
            shifted_unchanged = torch.roll(unchanged, (dy, dx), (-2, -1))
            pair_quality = torch.minimum(quality, shifted_quality)
            valid = (
                (instance == shifted_instance) & (instance > 0)
                & (pair_quality > 0.7) & unchanged & shifted_unchanged
                & (stable_score > 0.5) & (shifted_stable > 0.5)
                & _valid_roll_mask(instance, dy, dx)
            )
            cosine = (
                normalized * torch.roll(normalized, (dy, dx), (-2, -1))
            ).sum(dim=1, keepdim=True)
            weight = valid.float() * pair_quality
            ids = composite.reshape(-1)
            sums.scatter_add_(0, ids, ((1.0 - cosine) * weight).reshape(-1))
            weights.scatter_add_(0, ids, weight.reshape(-1))
        present = weights > 0
        if present.any():
            total = total + (sums[present] / weights[present]).sum()
            count = count + present.sum()
    return total / count.clamp_min(1.0)


def _feature_break_contrast(feature, target, evidence, margin_out=0.2):
    size = feature.shape[-2:]
    gt = _resize(target, size, mode="nearest") > 0.5
    break_score = _resize(
        evidence["boundary_delta"] * evidence["change_conf"], size,
    ).detach()
    normalized = F.normalize(feature.float(), dim=1, eps=1e-6)
    total, weight_total = feature.new_zeros(()), feature.new_zeros(())
    for dy, dx in SHIFTS:
        shifted_gt = torch.roll(gt, (dy, dx), (-2, -1))
        shifted_score = torch.roll(break_score, (dy, dx), (-2, -1))
        pair_score = torch.maximum(break_score, shifted_score)
        valid = (
            (gt != shifted_gt) & (pair_score > 0.3)
            & _valid_roll_mask(gt, dy, dx)
        )
        cosine = (
            normalized * torch.roll(normalized, (dy, dx), (-2, -1))
        ).sum(dim=1, keepdim=True)
        weight = valid.float() * pair_score
        total = total + (F.relu(cosine - margin_out) * weight).sum()
        weight_total = weight_total + weight.sum()
    return total / weight_total.clamp_min(1e-6)


def feature_relation_loss(decoder_features, target, evidence):
    total = target.new_zeros(())
    for scale_weight, feature in zip(SCALE_WEIGHTS, decoder_features):
        stable = _feature_stable_relation(feature, target, evidence)
        contrast = _feature_break_contrast(feature, target, evidence)
        total = total + scale_weight * (0.65 * stable + 0.35 * contrast)
    return total


def relation_loss(predictions, decoder_features, target, evidence):
    changed = target.new_zeros(())
    stable = target.new_zeros(())
    contrast = target.new_zeros(())
    for scale_weight, prediction in zip(SCALE_WEIGHTS, predictions):
        changed = changed + scale_weight * _instance_balanced_pair_loss(
            prediction, target, evidence, mode="changed",
        )
        stable = stable + scale_weight * _instance_balanced_stable_suppression(
            prediction, target, evidence,
        )
        contrast = contrast + scale_weight * _prediction_contrast(
            prediction, target, evidence,
        )
    feature = feature_relation_loss(decoder_features, target, evidence)
    total = 0.30 * changed + 0.25 * stable + 0.15 * contrast + 0.30 * feature
    return {
        "total": total,
        "changed": changed,
        "stable_suppression": stable,
        "contrast": contrast,
        "feature": feature,
    }


def affinity_loss(predictions, target, evidence):
    return sum(
        weight * _instance_balanced_pair_loss(
            prediction, target, evidence, mode="affinity",
        )
        for weight, prediction in zip(SCALE_WEIGHTS, predictions)
    )


def structural_code_loss(prediction, teacher_code, trust_change, trust_stable,
                         channel_weights=(0.25, 0.25, 0.25, 0.25)):
    """Trust-weighted regression of Boundary/Local/Geometry/Stable codes."""
    if prediction.ndim != 4 or prediction.shape[1] != 4:
        raise ValueError(f"Structural prediction must be [B,4,H,W], got {prediction.shape}")
    if len(channel_weights) != 4:
        raise ValueError("Structural channel_weights must contain four values")
    size = prediction.shape[-2:]
    teacher = _resize(teacher_code, size).detach()
    change_weight = _resize(trust_change, size).detach()
    stable_weight = _resize(trust_stable, size).detach()
    route_total = sum(float(value) for value in channel_weights)
    if route_total <= 0:
        raise ValueError("Structural channel weights must have positive sum")
    route = prediction.new_tensor(
        [float(value) / route_total for value in channel_weights],
        dtype=torch.float32,
    )

    loss_map = F.smooth_l1_loss(
        prediction.float(), teacher.float(), reduction="none",
    )
    names = ("boundary", "local", "geometry", "stable")
    components = {}
    total = prediction.new_zeros(())
    for index, name in enumerate(names):
        weight = stable_weight if name == "stable" else change_weight
        component = weighted_mean(loss_map[:, index:index + 1], weight)
        components[name] = component
        total = total + route[index] * component
    return total, components


def _gt_conditioned_relation_at_scale(feature, target, evidence,
                                       same_weight=0.65, contrast_weight=0.35,
                                       contrast_margin=0.2):
    """Relate features only where SAM, GT and continuous trust agree."""
    size = feature.shape[-2:]
    gt = _resize(target, size, mode="nearest") > 0.5
    trust_change = _resize(evidence["trust_change"], size).detach()
    trust_stable = _resize(evidence["trust_stable"], size).detach()
    boundary_support = _resize(
        evidence["boundary_residual"] * evidence["trust_change"], size,
    ).detach()
    normalized = F.normalize(feature.float(), dim=1, eps=1e-6)

    same_total = feature.new_zeros(())
    same_denominator = feature.new_zeros(())
    for time_key in ("t1", "t2"):
        source = evidence["structure"][time_key]
        instance = _resize(
            source["instance_id"], size, mode="nearest",
        ).round().to(torch.int32)
        quality = _resize(source["quality"], size).detach()
        for dy, dx in SHIFTS:
            shifted_instance = torch.roll(instance, (dy, dx), (-2, -1))
            shifted_gt = torch.roll(gt, (dy, dx), (-2, -1))
            shifted_quality = torch.roll(quality, (dy, dx), (-2, -1))
            class_trust = torch.where(gt, trust_change, trust_stable)
            shifted_trust = torch.roll(class_trust, (dy, dx), (-2, -1))
            pair_weight = (
                torch.minimum(quality, shifted_quality)
                * torch.minimum(class_trust, shifted_trust)
            )
            valid = (
                (instance == shifted_instance) & (instance > 0)
                & (gt == shifted_gt) & (pair_weight > 0)
                & _valid_roll_mask(instance, dy, dx)
            )
            cosine = (
                normalized * torch.roll(normalized, (dy, dx), (-2, -1))
            ).sum(dim=1, keepdim=True)
            weight = pair_weight * valid.float()
            same_total = same_total + ((1.0 - cosine) * weight).sum()
            same_denominator = same_denominator + weight.sum()
    same_loss = same_total / same_denominator.clamp_min(1e-6)

    contrast_total = feature.new_zeros(())
    contrast_denominator = feature.new_zeros(())
    for dy, dx in SHIFTS:
        shifted_gt = torch.roll(gt, (dy, dx), (-2, -1))
        shifted_boundary = torch.roll(boundary_support, (dy, dx), (-2, -1))
        pair_support = torch.maximum(boundary_support, shifted_boundary)
        valid = (
            (gt != shifted_gt) & (pair_support > 0)
            & _valid_roll_mask(gt, dy, dx)
        )
        cosine = (
            normalized * torch.roll(normalized, (dy, dx), (-2, -1))
        ).sum(dim=1, keepdim=True)
        weight = pair_support * valid.float()
        contrast_total = contrast_total + (
            F.relu(cosine - contrast_margin) * weight
        ).sum()
        contrast_denominator = contrast_denominator + weight.sum()
    contrast_loss = contrast_total / contrast_denominator.clamp_min(1e-6)
    return {
        "total": same_weight * same_loss + contrast_weight * contrast_loss,
        "same": same_loss,
        "contrast": contrast_loss,
    }


def gt_conditioned_relation_loss(structural_features, target, evidence):
    total = target.new_zeros(())
    same = target.new_zeros(())
    contrast = target.new_zeros(())
    for scale_weight, feature in zip(SCALE_WEIGHTS, structural_features):
        current = _gt_conditioned_relation_at_scale(feature, target, evidence)
        total = total + scale_weight * current["total"]
        same = same + scale_weight * current["same"]
        contrast = contrast + scale_weight * current["contrast"]
    return {"total": total, "same": same, "contrast": contrast}


def residual_correction_loss(predictions, target, evidence,
                             false_negative_weight=1.0,
                             false_positive_weight=1.3):
    """Correct only current student errors supported by SAM structure and GT."""
    gt = target > 0.5
    change_support = (
        evidence["change_strength"] * evidence["trust_change"]
    ).detach()
    stable_support = (
        evidence["stable_consensus"] * evidence["trust_stable"]
    ).detach()
    fn_total = target.new_zeros(())
    fp_total = target.new_zeros(())
    finest_fn = finest_fp = None
    for scale_index, (scale_weight, prediction) in enumerate(
        zip(SCALE_WEIGHTS, predictions)
    ):
        detached = prediction.detach()
        false_negative = gt & (detached < 0.5) & (change_support > 0)
        false_positive = (~gt) & (detached > 0.5) & (stable_support > 0)
        if scale_index == 0:
            finest_fn, finest_fp = false_negative, false_positive
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            loss_map = F.binary_cross_entropy(
                prediction.float().clamp(1e-5, 1 - 1e-5),
                target.float(), reduction="none",
            )
        fn_total = fn_total + scale_weight * weighted_mean(
            loss_map, change_support * false_negative.float(),
        )
        fp_total = fp_total + scale_weight * weighted_mean(
            loss_map, stable_support * false_positive.float(),
        )
    pixel_count = float(target.numel())
    return {
        "total": false_negative_weight * fn_total + false_positive_weight * fp_total,
        "false_negative": fn_total,
        "false_positive": fp_total,
        "fn_ratio": finest_fn.float().sum() / pixel_count,
        "fp_ratio": finest_fp.float().sum() / pixel_count,
    }
