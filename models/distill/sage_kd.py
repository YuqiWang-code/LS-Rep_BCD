"""SAGE-CD knowledge distillation with logit standardization and gradient budget.

Fixes the FA-SCRD failure mode where ``kd ~ 12-15`` overwhelmed ``main ~ 3.6``:
1. Distill the full-res *task target* (semantic change logit), not raw
   Foundation feature relations.
2. Standardize logits per-sample/per-spatial before the soft target so the loss
   scale is decoupled from the teacher/student magnitude gap.
3. A gradient budget bounds each KD loss so it never exceeds ``rho`` of the main
   loss gradient on the shared student change feature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-5


def standardize_logit(z: torch.Tensor) -> torch.Tensor:
    """Per-sample spatial z-score over HW."""
    mu = z.mean(dim=(2, 3), keepdim=True)
    sigma = z.std(dim=(2, 3), keepdim=True)
    return (z - mu) / (sigma + _EPS)


def semantic_kd_loss(
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    temperature: float = 1.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bernoulli soft-target distillation between standardized semantic logits.

    Args:
        student_logit: [B,1,H,W] student auxiliary logit.
        teacher_logit: [B,1,H,W] teacher semantic change logit (cache).
        temperature:    soft-target temperature.
        weight:         optional [B,1,H,W] region weight (SAGE router; None = uniform).

    Returns:
        scalar KD loss.
    """
    z_s = standardize_logit(student_logit.float())
    z_t = standardize_logit(teacher_logit.float())
    q_s = torch.sigmoid(z_s / temperature)
    q_t = torch.sigmoid(z_t / temperature)

    bce = F.binary_cross_entropy(q_s, q_t, reduction="none")  # [B,1,H,W]
    if weight is not None:
        denom = weight.sum() + _EPS
        return (bce * weight).sum() / denom
    return bce.mean()


def grad_norm(loss: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """L2 norm of d(loss)/d(tensor), zero if the loss does not depend on it."""
    grad = torch.autograd.grad(loss, tensor, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        return torch.zeros((), device=loss.device, dtype=loss.dtype)
    return grad.norm(2)


def gradient_budget_lambda(
    main_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    shared_feature: torch.Tensor,
    rho: float = 0.25,
    lambda_max: float = 1.0,
) -> torch.Tensor:
    """Dynamic KD weight so that ||grad(kd)|| <= rho * ||grad(main)||.

    Returns a detached scalar lambda (clipped to [0, lambda_max]).
    """
    g_main = grad_norm(main_loss, shared_feature)
    g_kd = grad_norm(kd_loss, shared_feature)
    raw = rho * (g_main + _EPS) / (g_kd + _EPS)
    return raw.clamp(0.0, lambda_max).detach()


__all__ = [
    "standardize_logit",
    "semantic_kd_loss",
    "grad_norm",
    "gradient_budget_lambda",
]
