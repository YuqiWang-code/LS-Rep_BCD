"""Symmetric change-relation knowledge distillation (FA-SCRD).

Both teacher and student change features are first L2-normalized along the
channel axis, then the spatial affinity within each local patch is distilled
via KL. No channel alignment is required, so a 768-ch ViT teacher and a
64-ch student can be distilled directly.

The relation is temporal-symmetric: swapping T1/T2 swaps C_T (and C_S) but
leaves |Norm(F_T1) - Norm(F_T2)| unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_relation_kd(
    student_change: torch.Tensor,
    teacher_change: torch.Tensor,
    weight: torch.Tensor,
    tau: float = 0.07,
    patch_size: int = 4,
) -> torch.Tensor:
    """Distill local spatial change-relation from teacher to student.

    Args:
        student_change: [B, Cs, H, W] student change feature (e.g. TFM output).
        teacher_change: [B, Ct, H, W] teacher change feature (|Norm(F_T1)-Norm(F_T2)|).
        weight:         [B, 1, H, W] (or [B, H, W]) failure-aware soft weight.
        tau:            affinity temperature.
        patch_size:     local patch side length (must divide H and W).

    Returns:
        scalar relation-KD loss.
    """
    student_change = F.normalize(student_change.float(), p=2, dim=1)
    teacher_change = F.normalize(teacher_change.float(), p=2, dim=1)

    B, _, H, W = student_change.shape
    P = patch_size
    if H % P != 0 or W % P != 0:
        raise ValueError(f"patch_size {P} must divide feature size {H}x{W}")

    nph, npw = H // P, W // P
    pp = P * P

    # [B, C, nph, P, npw, P] -> [B, nph*npw, C, P*P]
    def to_patches(x):
        x = x.unfold(2, P, P).unfold(3, P, P)  # [B,C,nph,npw,P,P]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        return x.reshape(B, nph * npw, -1, pp)

    s_p = to_patches(student_change)  # [B, Np, Cs, pp]
    t_p = to_patches(teacher_change)  # [B, Np, Ct, pp]

    # spatial affinity within each patch: A[b,n,p,q] = <x[:,p], x[:,q]> / tau
    a_t = torch.einsum("bncp,bncq->bnpq", t_p, t_p) / tau  # [B, Np, pp, pp]
    a_s = torch.einsum("bncp,bncq->bnpq", s_p, s_p) / tau

    log_a_t = F.log_softmax(a_t, dim=-1)
    log_a_s = F.log_softmax(a_s, dim=-1)
    a_t = torch.exp(log_a_t)

    # KL(teacher || student), summed over both affinity rows
    kl = (a_t * (log_a_t - log_a_s)).sum(dim=(-1, -2))  # [B, Np]

    if weight.dim() == 4:
        weight = weight.squeeze(1)  # [B,H,W]
    w_p = weight.unfold(1, P, P).unfold(2, P, P).reshape(B, nph * npw, pp).mean(dim=-1)

    return (kl * w_p).mean()
