"""DART-R-TS: GT-audited task-space proposals, pixel routing and rejection.

SAM supplies partitions, not change labels. It transports detached student
probabilities within each temporal instance (leave-one-out). OV supplies soft
change probabilities. GT is used ONLY to audit proposal utility, never to make
the proposals. No auxiliary classifier, MLP, gradient probe or persistent cache.
This is a new hypothesis, not a demonstrated accuracy improvement.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .diagnostics import build_cd_difficulty, masked_mean
from .losses import resize


@torch.no_grad()
def instance_transport(probability, ids, quality, boundary):
    """Quality-weighted leave-one-out mean, independently for each image/ID.

    IDs are opaque labels; unique compaction prevents huge-ID allocations.
    A singleton, uncovered pixel or zero source support returns identity with
    confidence zero. Boundary pixels are downweighted as *sources*, not targets.
    """
    if ids.shape != probability.shape:
        raise ValueError('SAM IDs must match the augmented student raster')
    proposals, confidences = [], []
    for p, labels, q, edge in zip(probability, ids, quality, boundary):
        p, labels, q, edge = [v.flatten() for v in (p, labels, q, edge)]
        _, index = torch.unique(labels, sorted=True, return_inverse=True)
        source = q * (1-edge) * (labels > 0)
        # N bins is an upper bound independent of ID values, avoiding item().
        mass = torch.zeros_like(p).scatter_add_(0, index, source)
        sums = torch.zeros_like(p).scatter_add_(0, index, source*p)
        remaining = mass[index]-source
        valid = (labels > 0) & (remaining > 1e-6)
        value = ((sums[index]-source*p)/remaining.clamp_min(1e-6)).clamp(0, 1)
        proposals.append(torch.where(valid, value, p))
        confidences.append(q*valid)
    shape = probability.shape
    return torch.stack(proposals).reshape(shape), torch.stack(confidences).reshape(shape)


@torch.no_grad()
def build_task_proposals(prediction, pack):
    """No GT access. Return [B,2,H,W] candidates/confidence in change space."""
    p = prediction.detach().float()
    sam_values, sam_quality = [], []
    for temporal in ('t1', 't2'):
        cache = pack['sam'][temporal]
        value, quality = instance_transport(
            p, cache['instance_id'].detach(), cache['quality'].detach().float(),
            cache['boundary'].detach().float())
        sam_values.append(value)
        sam_quality.append(quality)
    mass = sam_quality[0]+sam_quality[1]
    sam = torch.where(mass > 0,
                      (sam_values[0]*sam_quality[0]+sam_values[1]*sam_quality[1]) /
                      mass.clamp_min(1e-8), p)
    q_sam = mass*.5
    values, qualities = [], []
    for level in ('l1', 'l2'):
        values.append(resize(pack['ov']['soft_change'][level].detach(), p.shape[-2:]))
        qualities.append(resize(pack['ov']['confidence'][level].detach(), p.shape[-2:]))
    mass = qualities[0]+qualities[1]
    ov = torch.where(mass > 0, (values[0]*qualities[0]+values[1]*qualities[1]) /
                     mass.clamp_min(1e-8), p)
    # Opaque OV relation channels deliberately unused: their generation semantics
    # are not established consistently for historical and rebuilt caches.
    return torch.cat((sam, ov), 1), torch.cat((q_sam, mass*.5), 1)


@torch.no_grad()
def task_space_route(prediction, target, proposals, quality, policy='advantage',
                     teacher='both', force_action=None):
    """Audit exact Brier improvement; no feature-gradient cosine or learned gate.

    Relative improvement sets teacher proportions. Absolute confidence attenuates
    the mixture ONCE. Rejected pixels remain in the spatial loss denominator.
    force_action is a debug override of teacher choice, NOT of the GT safety gate.
    """
    p, y = prediction.detach().float(), target.detach().float()
    base_error = (p-y).square()
    gain = base_error-(proposals-y).square()
    relative = (gain/base_error.clamp_min(1e-8)).clamp(-1, 1)
    available = quality > 0
    if teacher not in ('both', 'sam', 'ov'):
        raise ValueError('teacher must be both/sam/ov')
    if teacher != 'both':
        available[:, 1 if teacher == 'sam' else 0] = False
    if force_action is not None:
        if int(force_action) not in (0, 1, 2):
            raise ValueError('force_action must be 0, 1 or 2')
        for k in (0, 1):
            if int(force_action) != k:
                available[:, k] = False
    tolerance = 8*torch.finfo(p.dtype).eps  # numerical equality, not utility margin
    if policy == 'advantage':
        eligible = available & (gain > tolerance)
        score = quality*relative.clamp_min(0)*eligible
    elif policy == 'quality':
        # Negative/ablation control: removes GT utility auditing.
        eligible = available
        score = quality*eligible
    else:
        raise ValueError('policy must be advantage/quality')
    total = score.sum(1, keepdim=True)
    admitted = total > 0
    mixture = score/total.clamp_min(1e-12)
    strength = (mixture*quality).sum(1, keepdim=True)
    effective = mixture*strength
    action = torch.where(admitted[:, 0], score.argmax(1),
                         torch.full_like(score[:, 0], 2, dtype=torch.long))
    return {'gain_map': gain, 'relative_map': relative,
            'eligible_map': eligible, 'available_map': available,
            'mixture_map': mixture, 'effective_map': effective,
            'action': action, 'accepted_map': admitted}


def bernoulli_kl(prediction, target):
    """KL(Ber(target)||Ber(prediction)); identical derivative to soft BCE.

    Subtracting target entropy avoids logging irreducible soft-target entropy as
    distillation error. FP32 supports the existing probability-output network.
    """
    p = prediction.float().clamp(1e-6, 1-1e-6)
    t = target.detach().float()
    return F.binary_cross_entropy(p.expand_as(t), t, reduction='none') + \
        torch.special.xlogy(t, t) + torch.special.xlogy(1-t, 1-t)


class TaskSpaceDirectionC(nn.Module):
    """Removable, parameter-free training mechanism; all maps are local values."""
    def __init__(self, channels=64, boundary_radius=2, small_area=64,
                 policy='advantage', teacher='both', difficulty=True):
        super().__init__()
        self.boundary_radius, self.small_area = int(boundary_radius), int(small_area)
        self.policy, self.teacher, self.difficulty = policy, teacher, bool(difficulty)

    def forward(self, feature, prediction, target, teacher_pack, force_action=None):
        # feature is accepted for the existing call contract; no auxiliary head.
        diagnosis = build_cd_difficulty(prediction, target, self.boundary_radius, self.small_area)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            proposals, quality = build_task_proposals(prediction, teacher_pack)
            route = task_space_route(prediction, target, proposals, quality,
                                     self.policy, self.teacher, force_action)
            importance = diagnosis['importance'] if self.difficulty else torch.ones_like(prediction)
            losses = bernoulli_kl(prediction, proposals)
            effective = route['effective_map']
            # Full-image denominator: never turn a tiny admitted region into a
            # full-image-strength loss, and never dilute through sample rejection.
            per_teacher = (losses*effective*importance).sum((2, 3)) / \
                importance.sum((2, 3)).clamp_min(1.)
            kd = per_teacher.sum(1).mean()
        def summary(x):
            return (x.detach().float()*importance).sum((2, 3)) / importance.sum((2, 3)).clamp_min(1.)
        q = summary(quality)
        support = route['accepted_map'].float()
        base_error = (prediction.detach().float()-target).square()
        changed_support = masked_mean(support, target).mean()
        bg_support = masked_mean(support, 1-target).mean()
        return {'total': kd, 'loss_per_teacher': per_teacher,
                'h': diagnosis['h'], 'h_valid': diagnosis['valid'], 'q': q,
                'effective_weights': summary(effective),
                'weights': torch.cat((summary(route['mixture_map']), summary(1-support)), 1),
                'action': route['action'], 'proposals': proposals,
                'pixel_reject_ratio': (1-support).mean(),
                'image_reject_ratio': (~route['accepted_map'].flatten(1).any(1)).float().mean(),
                'accepted_change_ratio': changed_support, 'accepted_bg_ratio': bg_support,
                'proposal_gain': summary(route['gain_map']),
                'relative_gain': summary(route['relative_map']),
                'eligible_ratio': route['eligible_map'].float().mean((2, 3)),
                'available_ratio': route['available_map'].float().mean((2, 3)),
                'student_brier': base_error.mean(),
                'proposal_brier': summary((proposals-target).square()),
                'effective_mass': effective.sum(1).mean(),
                'sam_transport': per_teacher[:, 0].mean(),
                'ov_task': per_teacher[:, 1].mean()}
