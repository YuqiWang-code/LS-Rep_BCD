#!/usr/bin/env python3
"""GT-Conditioned Reliable Task Distillation (J1, review §4.3 / §7.1).

The J1 mechanism question is narrow and falsifiable:

    Is a teacher's *soft change evidence* worth anything **on top of** the same
    auxiliary head trained only on GT, once both see the same pixels?

To answer it we split the auxiliary supervision into three arms that differ in
exactly one thing each:

    gt      1-channel aux head on c2 (stride 4), target = hard GT, all pixels
    gate    same head / same optimizer / same data stream, but supervision is
            restricted to the teacher-consistent region ``G``; target is still GT
    taskkd  identical region ``G`` and identical balancing, but the target is the
            frozen teacher response ``q_T`` instead of the binary GT

``gate`` vs ``taskkd`` is the clean contrast (same pixels, same capacity, only the
target differs). ``gt`` vs ``gate`` additionally changes the effective sample
count, which is why :func:`aux_task_loss` always returns region statistics that
the trainer must log.

Definitions (all in **256x256 GT space**, then area-pooled onto the aux grid):

    z_T(p)   = max_c |norm_c F_A(p) - norm_c F_B(p)|      (cached "confidence")
    q_T(p)   = sigmoid((z_T(p) - m_T) / (s_T + eps))      monotone squash, NOT a
                                                          calibrated probability
    G+(p)    = y(p)      AND q_T(p) >= t_plus
    G-(p)    = (1-y(p))  AND q_T(p) <= t_minus
    G        = (G+ u G-) minus a 2 px band around the GT boundary

``m_T, s_T, t_plus, t_minus`` are estimated from the **train split only** and
frozen (see :mod:`models.tools.audit_taskkd`); nothing here reads val or test.

If a batch yields an empty ``G`` the auxiliary loss is skipped (returns ``None``),
never silently replaced by a fabricated zero target.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_TASKS = ("none", "gt", "gate", "taskkd")
# The plan fixes the boundary uncertainty band at r = 2 px in the 256x256 GT frame.
DEFAULT_BOUNDARY_R = 2


@dataclass
class TaskReliableConfig:
    """Frozen, train-derived calibration for one (teacher, dataset) pair."""

    teacher_package: str
    dataset: str
    z_median: float          # m_T
    z_q90: float             # used to build s_T
    t_plus: float            # q_T threshold for confidently-changed teacher pixels
    t_minus: float           # q_T threshold for confidently-unchanged teacher pixels
    boundary_r: int = DEFAULT_BOUNDARY_R
    native_hw: tuple[int, int] = (16, 16)
    n_samples: int = 0
    ece: float | None = None
    note: str = "q_T is a monotone squash of the change response, not a calibrated probability"

    @property
    def s_T(self) -> float:
        return max(self.z_q90 - self.z_median, 1e-6)

    @classmethod
    def from_json(cls, path: str | Path) -> "TaskReliableConfig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        hw = d.get("native_hw", [16, 16])
        return cls(
            teacher_package=d["teacher_package"], dataset=d["dataset"],
            z_median=float(d["z_median"]), z_q90=float(d["z_q90"]),
            t_plus=float(d["t_plus"]), t_minus=float(d["t_minus"]),
            boundary_r=int(d.get("boundary_r", DEFAULT_BOUNDARY_R)),
            native_hw=(int(hw[0]), int(hw[1])), n_samples=int(d.get("n_samples", 0)),
            ece=(None if d.get("ece") is None else float(d["ece"])),
        )

    def to_json(self, path: str | Path) -> None:
        d = asdict(self)
        d["native_hw"] = list(self.native_hw)
        d["s_T"] = self.s_T
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")


class TaskReliableAuxHead(nn.Module):
    """1-channel training-only head on c2 (stride 4). Exactly 65 parameters."""

    def __init__(self, in_d: int = 64, scale_index: int = 0):
        super().__init__()
        self.scale_index = scale_index
        self.proj = nn.Conv2d(in_d, 1, kernel_size=1)

    def forward(self, change) -> torch.Tensor:
        return self.proj(change[self.scale_index])


# --------------------------------------------------------------------- helpers
def compute_q(z: torch.Tensor, cfg: TaskReliableConfig) -> torch.Tensor:
    """Monotone squash of the cached teacher change response into (0, 1)."""
    return torch.sigmoid((z - cfg.z_median) / (cfg.s_T + 1e-6))


def _to_256(x: torch.Tensor, size: int = 256, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == (size, size):
        return x
    return F.interpolate(x, size=(size, size), mode=mode,
                         align_corners=False if mode == "bilinear" else None)


def boundary_band(y256: torch.Tensor, r: int) -> torch.Tensor:
    """Soft band of width ``r`` px (256 frame) around the GT change boundary.

    ``y256``: [B,1,H,W] in {0,1}. Returns [B,1,H,W] float in {0,1}.
    Implemented with max-pool dilations so it stays cheap and exact on the grid.
    """
    if r <= 0:
        return torch.zeros_like(y256)
    k = 2 * r + 1
    dil = F.max_pool2d(y256, kernel_size=k, stride=1, padding=r)
    ero = -F.max_pool2d(-y256, kernel_size=k, stride=1, padding=r)
    return (dil - ero).clamp(0.0, 1.0)


def build_regions_256(y256: torch.Tensor, q256: torch.Tensor,
                      cfg: TaskReliableConfig) -> dict[str, torch.Tensor]:
    """Teacher-consistent, boundary-excluded masks, all at 256x256."""
    band = boundary_band(y256, cfg.boundary_r)
    keep = 1.0 - band
    g_pos = y256 * (q256 >= cfg.t_plus).float() * keep
    g_neg = (1.0 - y256) * (q256 <= cfg.t_minus).float() * keep
    return {"g_pos": g_pos, "g_neg": g_neg, "band": band}


def _area_pool(weights256: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Area-average a 256-space weight map onto the aux grid -> soft weights in [0,1]."""
    return F.adaptive_avg_pool2d(weights256, size)


def _balanced_bce(logits: torch.Tensor, target: torch.Tensor,
                  w_pos: torch.Tensor, w_neg: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Positive/negative halves normalised separately (change ratio is 4%-21%)."""
    pos_den = w_pos.sum()
    neg_den = w_neg.sum()
    if float(pos_den) <= 0.0 and float(neg_den) <= 0.0:
        return None, {"n_pos": 0.0, "n_neg": 0.0, "skipped": True}

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    parts, ws = [], []
    if float(pos_den) > 0.0:
        parts.append((bce * w_pos).sum() / (pos_den + 1e-6))
        ws.append(1.0)
    if float(neg_den) > 0.0:
        parts.append((bce * w_neg).sum() / (neg_den + 1e-6))
        ws.append(1.0)
    loss = torch.stack(parts).sum() / float(len(parts))
    return loss, {"n_pos": float(pos_den), "n_neg": float(neg_den), "skipped": False}


# ----------------------------------------------------------------- public loss
def aux_task_loss(task: str, logits: torch.Tensor, target256: torch.Tensor,
                  cfg: TaskReliableConfig | None = None,
                  confidence: torch.Tensor | None = None) -> tuple[torch.Tensor | None, dict]:
    """Auxiliary loss for one batch.

    ``logits``      [B,1,h,w] raw aux-head output on the c2 grid
    ``target256``   [B,1,256,256] hard GT in {0,1}
    ``confidence``  [B,1,Ht,Wt] cached teacher change response (gate/taskkd only)

    Returns ``(loss_or_None, stats)``. ``None`` means "this batch has no usable
    region, skip the auxiliary term" -- the caller must still run the main loss.
    """
    if task not in AUX_TASKS:
        raise ValueError(f"unknown aux task {task!r}; expected one of {AUX_TASKS}")
    size = logits.shape[-2:]
    y256 = _to_256(target256.float(), 256, "nearest")

    if task == "gt":
        # All pixels, no teacher filter, no boundary exclusion: the GT-only control.
        tgt = _area_pool(y256, size)
        loss, stats = _balanced_bce(logits, tgt, tgt, 1.0 - tgt)
        stats["region"] = "all_pixels"
        stats["target"] = "gt"
        return loss, stats

    if cfg is None or confidence is None:
        raise ValueError(f"aux task {task!r} requires both cfg and confidence")

    q256 = _to_256(compute_q(confidence.float(), cfg), 256, "bilinear")
    reg = build_regions_256(y256, q256, cfg)
    w_pos = _area_pool(reg["g_pos"], size)
    w_neg = _area_pool(reg["g_neg"], size)

    if task == "gate":
        tgt256 = y256
    elif task == "taskkd":
        tgt256 = q256               # frozen teacher soft response
    else:  # pragma: no cover - guarded above
        raise ValueError(task)

    tgt = _area_pool(tgt256, size)
    loss, stats = _balanced_bce(logits, tgt, w_pos, w_neg)
    stats["region"] = "teacher_consistent_minus_boundary"
    stats["target"] = "gt" if task == "gate" else "teacher_q"
    stats["band_frac"] = float(reg["band"].mean())
    stats["g_pos_frac"] = float(reg["g_pos"].mean())
    stats["g_neg_frac"] = float(reg["g_neg"].mean())
    return loss, stats


__all__ = [
    "AUX_TASKS",
    "DEFAULT_BOUNDARY_R",
    "TaskReliableConfig",
    "TaskReliableAuxHead",
    "aux_task_loss",
    "boundary_band",
    "build_regions_256",
    "compute_q",
]
