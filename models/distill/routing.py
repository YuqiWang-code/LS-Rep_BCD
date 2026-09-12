"""Direction C v1: detached difficulty + GT-gradient suitability + MLP + Reject.

The router is supervised by an explicit first-order feature-gradient proxy,
NOT by minimizing its own weighted KD losses. This is a design implementation,
not a claim of held-out improvement or a reproduction of an existing paper.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .diagnostics import build_cd_difficulty
from .losses import TeacherHeads, resize


def estimate_teacher_fit(losses, feature, prediction, target, importance):
    """Detached per-image cosine between teacher and GT gradients at decoder p2.

Positive alignment means a small descent step in teacher loss is locally aligned
with descent in the GT proxy. It is not a measured full-optimizer-step gain.
"""
    if not torch.is_grad_enabled() or not feature.requires_grad:
        return losses.new_zeros(losses.shape)
    p = prediction.float().clamp(1e-6,1-1e-6)
    y = target.detach().float()
    gt_bce = F.binary_cross_entropy(p,y,reduction='none').mean((1,2,3))
    dice = 1-(2*(p*y).sum((1,2,3))+1e-5)/(p.sum((1,2,3))+y.sum((1,2,3))+1e-5)
    g_gt = torch.autograd.grad((gt_bce+dice).sum(),feature,retain_graph=True)[0].detach()
    weight = resize(importance.detach(),feature.shape[-2:])
    fits = []
    for k in range(losses.shape[1]):
        g_teacher = torch.autograd.grad(losses[:,k].sum(),feature,retain_graph=True)[0].detach()
        dot = (g_gt*g_teacher*weight).sum((1,2,3))
        denom = ((g_gt.square()*weight).sum((1,2,3)) *
                 (g_teacher.square()*weight).sum((1,2,3))).sqrt()
        fits.append(torch.where(denom>1e-20,dot/denom.clamp_min(1e-20),torch.zeros_like(dot)))
    return torch.stack(fits,dim=1).clamp(-1,1).detach()


def reduce_teacher_losses(losses, weights, force_action=None):
    """Reject=K zeros EVERY teacher contribution; no teacher-only renormalization."""
    k = losses.shape[1]
    action = weights.detach().argmax(1)
    if force_action is not None:
        if not 0 <= int(force_action) <= k:
            raise ValueError('force_action must be a teacher index or K (Reject)')
        action = torch.full_like(action,int(force_action))
    effective = weights[:,:k].detach() * (action != k).unsqueeze(1)
    if force_action is not None and force_action < k:
        # Test/debug only: isolate the gradient of a selected teacher.
        effective = F.one_hot(action,k).to(losses.dtype)
    return (effective*losses).sum(1).mean(), effective, action


class DirectionC(nn.Module):
    def __init__(self, channels=64, hidden=16, boundary_radius=2, small_area=64,
                 utility_margin=.02):
        super().__init__()
        self.heads = TeacherHeads(channels)
        self.router = nn.Sequential(nn.Linear(8,hidden),nn.ReLU(),nn.Linear(hidden,3))
        self.boundary_radius = int(boundary_radius)
        self.small_area = int(small_area)
        self.utility_margin = float(utility_margin)

    def forward(self, feature, prediction, target, teacher_pack, force_action=None):
        diagnosis = build_cd_difficulty(prediction,target,self.boundary_radius,self.small_area)
        losses, quality, components = self.heads(feature,teacher_pack,diagnosis['importance'])
        fit = estimate_teacher_fit(losses,feature,prediction,target,diagnosis['importance'])
        reliability = (quality*fit.clamp_min(0)).detach()
        router_input = torch.cat((diagnosis['h'],reliability),dim=1).detach()
        logits = self.router(router_input)
        weights = logits.softmax(1)
        # Explicit Reject score=0. On ties/all unsuitable, choose Reject.
        utility = quality*fit
        best_score, best_teacher = utility.max(1)
        labels = torch.where(best_score>self.utility_margin,best_teacher,
                             torch.full_like(best_teacher,2)).detach()
        router_loss = F.cross_entropy(logits,labels)
        kd, effective, action = reduce_teacher_losses(losses,weights,force_action)
        return {
            'total':kd, 'router_loss':router_loss, 'loss_per_teacher':losses,
            'h':diagnosis['h'], 'h_valid':diagnosis['valid'], 'q':quality,
            'd':fit, 'r':reliability, 'weights':weights, 'effective_weights':effective,
            'action':action, 'target_action':labels, **components,
        }
