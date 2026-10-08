"""Train-only teacher capability probes (used by the selection Agent, not for test).

These are cheap, offline statistics computed on the TRAIN split from cached
teacher evidence + labels. They do not decide whether a teacher works; the full
40K student run does. They only provide the Agent an interpretable input state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_4d(x: torch.Tensor) -> torch.Tensor:
    """Normalize [H,W] / [1,H,W] / [N,1,H,W] to 4D."""
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x


@torch.no_grad()
def separability_scores(local_change: torch.Tensor, label: torch.Tensor):
    """Change vs unchanged separability of cached teacher evidence.

    ``local_change``: [N, C, H, W] (or [C,H,W]); ``label``: [H,W] / [N,1,H,W] binary.
    """
    if local_change.dim() == 3:
        local_change = local_change.unsqueeze(0)
    label = _to_4d(label)
    label = F.interpolate(label.float(), size=local_change.shape[-2:], mode="nearest").squeeze(1)

    mag = local_change.abs().mean(dim=1)  # [N,H,W]
    pos = label.bool()                     # [N,H,W]
    neg = ~pos
    change_mean = mag[pos].mean().item() if pos.any() else 0.0
    unchanged_mean = mag[neg].mean().item() if neg.any() else 0.0
    ratio = (change_mean + 1e-6) / (unchanged_mean + 1e-6)
    return {
        "change_mean": change_mean,
        "unchanged_mean": unchanged_mean,
        "ratio": ratio,
    }


@torch.no_grad()
def confidence_calibration(confidence: torch.Tensor, label: torch.Tensor):
    """Mean confidence on changed vs unchanged pixels."""
    if confidence.dim() == 3:
        confidence = confidence.unsqueeze(0)
    label = _to_4d(label)
    label = F.interpolate(label.float(), size=confidence.shape[-2:], mode="nearest").squeeze(1)
    conf = confidence.mean(dim=1)  # [N,H,W]
    pos = label.bool()
    neg = ~pos
    return {
        "conf_change": conf[pos].mean().item() if pos.any() else 0.0,
        "conf_unchanged": conf[neg].mean().item() if neg.any() else 0.0,
    }


__all__ = ["separability_scores", "confidence_calibration", "extended_probes", "balanced_auroc"]


# =====================================================================================
# Extended probes P1-P5 (review §7.4). All are TRAIN-only, Cache-only, CPU-only.
# z = the teacher change RESPONSE (the per-pixel `confidence` map = max over channels
# of the normalised A/B evidence). It is NOT a calibrated probability; the old name
# "confidence" is kept only for cache-schema compatibility.
# =====================================================================================

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def balanced_auroc(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator, max_per_class: int = 4096):
    """AUROC between two score sets with per-class subsampling (tie-aware)."""
    if pos.size == 0 or neg.size == 0:
        return None
    if pos.size > max_per_class:
        pos = rng.choice(pos, max_per_class, replace=False)
    if neg.size > max_per_class:
        neg = rng.choice(neg, max_per_class, replace=False)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average ranks for ties
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(uniq.size)
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    n_pos = pos.size
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * neg.size))


def _boundary_band(mask: np.ndarray, r: int = 2):
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    m = mask.astype(np.uint8)
    dil = cv2.dilate(m, k)
    ero = cv2.erode(m, k)
    inner_edge = ((dil - ero) > 0) & mask          # changed pixels on the boundary
    outer_band = (dil > 0) & (~mask)               # unchanged pixels in the narrow outer band
    return inner_edge, outer_band


@torch.no_grad()
def extended_probes(confidence: torch.Tensor, label: torch.Tensor, seed: int = 2333) -> dict:
    """P1-P5 probes for one sample. Returns None for any probe that is undefined."""
    rng = np.random.default_rng(seed)
    z = confidence
    if z.dim() == 3:
        z = z.squeeze(0)
    z = z.float().numpy()
    lab = label
    if lab.dim() == 3:
        lab = lab.squeeze(0)
    m = (lab.float().numpy() > 0.5)
    if z.shape != m.shape:
        zt = torch.as_tensor(z).unsqueeze(0).unsqueeze(0)
        z = F.interpolate(zt, size=m.shape, mode="bilinear", align_corners=False)[0, 0].numpy()

    pos, neg = z[m], z[~m]
    out: dict = {}

    # P1: balanced pixel AUROC
    out["balanced_auroc"] = balanced_auroc(pos, neg, rng)

    # P2: hard-negative separation  Q50(z|M=1) - Q90(z|M=0), scaled by MAD(z)
    if pos.size and neg.size:
        mad = float(np.median(np.abs(z - np.median(z))) ) + 1e-6
        out["hard_negative_separation"] = float(
            (np.percentile(pos, 50) - np.percentile(neg, 90)) / mad)
    else:
        out["hard_negative_separation"] = None

    # P3: boundary contrast
    inner, outer = _boundary_band(m, r=2)
    if inner.any() and outer.any():
        out["boundary_contrast"] = float(z[inner].mean() - z[outer].mean())
    else:
        out["boundary_contrast"] = None

    # P4: size-conditioned AUROC gap (small vs large GT components, a0 = 0.5% of image)
    n, labmap, stats, _c = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    a0 = 0.005 * m.size
    if n > 1:
        small_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < a0]
        large_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= a0]
        def _auc_for(ids):
            if not ids or neg.size == 0:
                return None
            sel = np.isin(labmap, ids)
            return balanced_auroc(z[sel], neg, rng)
        a_small, a_large = _auc_for(small_ids), _auc_for(large_ids)
        out["size_conditioned_auroc_gap"] = (
            None if (a_small is None or a_large is None) else float(a_small - a_large))
    else:
        out["size_conditioned_auroc_gap"] = None

    # P5: unchanged leakage ratio
    out["unchanged_leakage_ratio"] = (
        None if (not pos.size or not neg.size) else float(neg.mean() / (pos.mean() + 1e-6)))

    # extra: zero-change flag so aggregates can separate empty maps
    out["_empty_pair"] = float(not m.any())
    return out
