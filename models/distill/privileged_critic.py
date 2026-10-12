#!/usr/bin/env python3
"""CTP-Critic: Counterfactual Teacher-Privileged Critic (third-round main mechanism).

Report §5. The mechanism keeps the nine-teacher pool and the selection Agent
untouched; it changes only the **training-time knowledge channel**. The teacher's
multi-channel symmetric evidence is injected into a *train-only privileged
critic* attached to the student's stride-4 change features, and the student's
main prediction path is never touched:

    Z        = psi(C2)                        student change features, stride 4
    U        = P([Resize(E), Resize(M)])      teacher evidence, train-only adapter
    l_pri    = h(Z + g(Z, U))                 privileged view      (teacher present)
    l_nul    = h(Z + g(Z, 0))                 teacher-free view    (counterfactual)
    L_task   = BCE(l_nul, Y_down)
    L_pri    = BCE(l_pri, Y_down)
    L_trans  = tau^2 * KL( sg sigmoid(l_pri/tau) || sigmoid(l_nul/tau) )
    L        = L_main + lambda_a (L_task + L_pri) + lambda_t L_trans

Gradients from every auxiliary term flow into the student through ``Z``; the
teacher evidence is a constant (it comes from the offline cache), and ``P`` is
train-only. At deploy time nothing of this exists, so the student stays at
3,280,274 params and the same FLOPs.

Counterfactual modes (the whole point -- they make the teacher claim falsifiable):

``real``     paired U from the same sample.
``zero``     U replaced by zeros *at the point of use*; the adapter P still runs on
             the real cache so the two arms differ in exactly one thing.
``shuffle``  U taken from a different sample in the same batch: identical magnitude
             distribution, pairing destroyed. This is what separates "teacher
             knowledge" from "an extra regularizer".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("real", "zero", "shuffle")


@dataclass
class CTPOutput:
    loss_task: torch.Tensor
    loss_pri: torch.Tensor
    loss_transfer: torch.Tensor
    logits_pri: torch.Tensor
    logits_nul: torch.Tensor
    u_norm: float
    n_valid: int


class PrivilegedCritic(nn.Module):
    """Train-only critic. Never referenced by the deployable student."""

    def __init__(self, stu_ch: int = 64, teach_ch: int = 768, d: int = 32,
                 z_dim: int = 64, tau: float = 1.0, mode: str = "real"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.d = d
        self.z_dim = z_dim
        self.tau = float(tau)

        self.psi = nn.Conv2d(stu_ch, z_dim, 1)
        self.P = nn.Sequential(
            nn.Conv2d(2 * teach_ch, d, 1, bias=False),
            nn.GroupNorm(1, d),
            nn.GELU(),
        )
        # gated residual, deliberately a single 1x1 conv (report: no attention/expert stacking)
        self.g = nn.Conv2d(z_dim + d, z_dim, 1)
        self.h = nn.Conv2d(z_dim, 1, 1)

    # ------------------------------------------------------------------ pieces
    def encode_teacher(self, evidence: torch.Tensor, context: torch.Tensor,
                       size: tuple[int, int]) -> torch.Tensor:
        """P([Resize(E), Resize(M)]) -> [B, d, H, W] on the student's grid."""
        e = F.interpolate(evidence.float(), size=size, mode="bilinear", align_corners=False)
        m = F.interpolate(context.float(), size=size, mode="bilinear", align_corners=False)
        return self.P(torch.cat([e, m], dim=1))

    def _apply_mode(self, u: torch.Tensor) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros_like(u)
        if self.mode == "shuffle":
            if u.shape[0] < 2:
                return u
            # deterministic roll, no RNG: magnitude distribution preserved, pairing broken
            return torch.roll(u, shifts=1, dims=0)
        return u

    # ----------------------------------------------------------------- forward
    def forward(self, change_c2: torch.Tensor, evidence: torch.Tensor,
                context: torch.Tensor, y_down: torch.Tensor) -> CTPOutput:
        z = self.psi(change_c2)                       # [B, z_dim, H, W]
        u_raw = self.encode_teacher(evidence, context, z.shape[-2:])
        u = self._apply_mode(u_raw)

        gated = self.g(torch.cat([z, u], dim=1))
        logits_pri = self.h(z + gated)
        gated_nul = self.g(torch.cat([z, torch.zeros_like(u)], dim=1))
        logits_nul = self.h(z + gated_nul)

        y = y_down
        if y.shape[-2:] != logits_nul.shape[-2:]:
            y = F.interpolate(y.float(), size=logits_nul.shape[-2:], mode="nearest")
        y = y.float()
        if y.dim() == 3:
            y = y[:, None]

        loss_task = F.binary_cross_entropy_with_logits(logits_nul, y)
        loss_pri = F.binary_cross_entropy_with_logits(logits_pri, y)
        loss_transfer = bernoulli_kl_transfer(logits_pri, logits_nul, self.tau)
        return CTPOutput(loss_task=loss_task, loss_pri=loss_pri,
                         loss_transfer=loss_transfer, logits_pri=logits_pri,
                         logits_nul=logits_nul, u_norm=float(u.detach().abs().mean()),
                         n_valid=int(y.numel()))


def bernoulli_kl_transfer(logits_pri: torch.Tensor, logits_nul: torch.Tensor,
                          tau: float = 1.0) -> torch.Tensor:
    """``tau^2 * KL( sg sigmoid(l_pri/tau) || sigmoid(l_nul/tau) )``.

    The target is DETACHED: the privileged view is a moving reference, it must not
    be pulled toward the teacher-free view by this term (that would let the
    teacher-conditioned branch collapse instead of transferring).
    """
    p = torch.sigmoid(logits_pri / tau).detach().clamp(1e-6, 1 - 1e-6)
    q = torch.sigmoid(logits_nul / tau).clamp(1e-6, 1 - 1e-6)
    kl = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
    return kl.mean() * (tau ** 2)


def ctp_aux_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["CTPOutput", "MODES", "PrivilegedCritic", "bernoulli_kl_transfer", "ctp_aux_params"]
