"""Dataset Signature: hand-crafted, label-based change geometry + bi-temporal appearance.

Computed on the TRAIN split (labels are available; this is a supervised task).
No large model is used. The returned vector is the Agent's ``s_D`` state.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# Fixed feature order (must stay stable across runs).
GEOM_NAMES = [
    "change_ratio", "n_components_per_img",
    "area_p25", "area_p50", "area_p75", "area_p90",
    "small_fraction", "large_fraction",
    "boundary_area_ratio", "compactness",
]
APPEAR_NAMES = [
    "rgb_mean_shift", "rgb_std_shift", "hist_js",
    "edge_disagreement", "grad_diff", "lowfreq_diff", "highfreq_diff",
]
REG_NAMES = ["unchanged_photo_diff"]
SIGNATURE_NAMES = GEOM_NAMES + APPEAR_NAMES + REG_NAMES


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Failed to read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"Failed to read {path}")
    return img


def _hist_js(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    ha, _ = np.histogram(a.ravel(), bins=bins, range=(0, 255))
    hb, _ = np.histogram(b.ravel(), bins=bins, range=(0, 255))
    pa = ha / (ha.sum() + 1e-9)
    pb = hb / (hb.sum() + 1e-9)
    m = 0.5 * (pa + pb)

    def kl(p, q):
        p = np.where(p > 0, p, 1e-12)
        q = np.where(q > 0, q, 1e-12)
        return (p * np.log(p / q)).sum()

    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def _sample_features(data_root: Path, sample_id: str) -> np.ndarray:
    pre = _read_rgb(data_root / "A" / sample_id)
    post = _read_rgb(data_root / "B" / sample_id)
    label = _read_gray(data_root / "label" / sample_id)
    if pre.shape[:2] != post.shape[:2] or pre.shape[:2] != label.shape[:2]:
        raise ValueError(f"shape mismatch for {sample_id}")

    lab = (label >= 128).astype(np.uint8)
    H, W = lab.shape
    area = H * W

    change_ratio = lab.mean()

    n, labmap, stats, _ = cv2.connectedComponentsWithStats(lab, connectivity=8)
    comp_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)  # exclude background
    n_components = max(len(comp_areas), 1)
    quantiles = np.percentile(comp_areas, [25, 50, 75, 90]) if len(comp_areas) else [0, 0, 0, 0]
    small_fraction = float((comp_areas < 0.005 * area).sum() / max(len(comp_areas), 1))
    large_fraction = float((comp_areas > 0.05 * area).sum() / max(len(comp_areas), 1))

    contours, _ = cv2.findContours(lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    changed_area = lab.sum()
    boundary_area_ratio = float(perimeter / (changed_area + 1e-6))
    compactness = float((perimeter ** 2) / (4 * np.pi * changed_area + 1e-6)) if changed_area else 0.0

    pre_f = pre.astype(np.float32)
    post_f = post.astype(np.float32)
    rgb_mean_shift = float(np.abs(post_f - pre_f).mean())
    rgb_std_shift = float(np.abs(post_f.std(axis=(0, 1)) - pre_f.std(axis=(0, 1))).mean())
    hist_js = float(np.mean([_hist_js(pre_f[..., c], post_f[..., c]) for c in range(3)]))

    pre_g = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY)
    post_g = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY)
    pre_e = cv2.Canny(pre_g, 50, 150)
    post_e = cv2.Canny(post_g, 50, 150)
    edge_disagreement = float(np.abs(pre_e.astype(np.float32) - post_e.astype(np.float32)).mean())
    grad_diff = float(np.abs(cv2.Sobel(post_g, cv2.CV_32F, 1, 1) - cv2.Sobel(pre_g, cv2.CV_32F, 1, 1)).mean())

    def _freq_energy(img, low=True):
        f = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
        H, W = img.shape
        cy, cx = H // 2, W // 2
        yy, xx = np.ogrid[:H, :W]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = (r < 16) if low else (r >= 16)
        return float(np.abs(f[mask]).mean())

    lowfreq_diff = float(np.abs(_freq_energy(post_g, True) - _freq_energy(pre_g, True)))
    highfreq_diff = float(np.abs(_freq_energy(post_g, False) - _freq_energy(pre_g, False)))

    unchanged = (lab == 0)
    unchanged_photo_diff = float(
        np.abs(post_f - pre_f)[unchanged].mean() if unchanged.any() else 0.0
    )

    return np.array([
        change_ratio, n_components, *quantiles,
        small_fraction, large_fraction,
        boundary_area_ratio, compactness,
        rgb_mean_shift, rgb_std_shift, hist_js,
        edge_disagreement, grad_diff, lowfreq_diff, highfreq_diff,
        unchanged_photo_diff,
    ], dtype=np.float32)


def compute_dataset_signature(data_root, dataset_name, split: str = "train", max_samples: int | None = None):
    """Aggregate per-sample features into one dataset-level signature vector."""
    data_root = Path(data_root)
    ids = [
        line.strip()
        for line in (data_root / "list" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_samples is not None:
        ids = ids[:max_samples]

    feats = [_sample_features(data_root, sid) for sid in ids]
    mat = np.stack(feats, axis=0)  # [N, D]
    signature = mat.mean(axis=0)
    return signature.astype(np.float32), {name: float(v) for name, v in zip(SIGNATURE_NAMES, signature)}


__all__ = ["compute_dataset_signature", "SIGNATURE_NAMES"]
