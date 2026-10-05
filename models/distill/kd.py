"""Training-only KD head and loss for CATA-CD dense-change distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeEvidenceHead(nn.Module):
    """Project one change scale (64ch) to 128ch evidence for dense-change KD.

    ``scale_index`` selects which change scale to use: 0=c2(stride4), 1=c3(stride8),
    2=c4(stride16), 3=c5(stride32). Default 2 matches most teacher stride-16 caches.
    """

    def __init__(self, in_d: int = 64, out_d: int = 128, scale_index: int = 2):
        super().__init__()
        self.scale_index = scale_index
        self.proj = nn.Conv2d(in_d, out_d, 1)

    def forward(self, change):
        return self.proj(change[self.scale_index])


def _align(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if src.shape[-2:] != ref.shape[-2:]:
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return src


def dense_change_kd(student_evidence, teacher_change, confidence, cap: float = 2.0):
    """Per-pixel cosine KD between student evidence and cached teacher change evidence.

    ``teacher_change``/``confidence`` are detached and spatially aligned to the
    student evidence; confidence weights the per-pixel loss. Returns a scalar loss.
    """
    t = _align(teacher_change.detach(), student_evidence)
    c = _align(confidence.detach(), student_evidence).squeeze(1)

    s = F.normalize(student_evidence, dim=1)
    t = F.normalize(t, dim=1)
    sim = (s * t).sum(dim=1)                 # [B,H,W] cosine in [-1,1]
    loss = (1.0 - sim).clamp(min=0.0)
    w = c.clamp(0.0, 1.0)
    loss = (loss * w).sum() / (w.sum() + 1e-6)
    return loss.clamp(max=cap)


def gradient_budget_lambda(main_loss, kd_loss, shared, rho: float = 0.25, lam_max: float = 1.0):
    """Scale KD so its gradient norm on ``shared`` is at most ``rho`` x main's.

    Returns a detached scalar lambda in [0, lam_max].
    """
    g_main = torch.autograd.grad(main_loss, shared, retain_graph=True, create_graph=False)[0]
    g_kd = torch.autograd.grad(kd_loss, shared, retain_graph=True, create_graph=False)[0]
    norm_main = g_main.norm() + 1e-6
    norm_kd = g_kd.norm() + 1e-6
    lam = (rho * norm_main / norm_kd).clamp(max=lam_max)
    return lam.detach()


__all__ = ["ChangeEvidenceHead", "dense_change_kd", "gradient_budget_lambda"]
