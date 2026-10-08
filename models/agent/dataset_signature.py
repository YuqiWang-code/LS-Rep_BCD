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


# =====================================================================================
# schema_version = 2  (review 2026-10-08)
#   - fixes: n_components counts 0 for empty label maps; gradient uses the Sobel
#     MAGNITUDE sqrt(gx^2+gy^2) instead of the mixed derivative; object-area
#     quantiles are aggregated GLOBALLY over all components (not per-image means);
#   - adds : empty_pair_ratio, boundary_density, change_area_cv, global area
#     quantiles, illumination_unchanged, hard_pseudo_change_fraction;
#   - path : resolves A/B/label with the same CDD test_/png<->jpg fallback as
#     models/datasets/cd_dataset.py::CDDataset._resolve_path.
# v1 is kept untouched for reproducibility; v2 is written to a separate file.
# =====================================================================================

SIGNATURE_V2_VERSION = 2
SIGNATURE_V2_NAMES = [
    # change geometry (8)
    "change_ratio", "empty_pair_ratio", "n_components_per_img",
    "small_fraction", "large_fraction", "boundary_density", "compactness", "change_area_cv",
    # global object-area quantiles (3)
    "area_p25_g", "area_p50_g", "area_p90_g",
    # bi-temporal appearance (7)
    "rgb_mean_shift", "rgb_std_shift", "hist_js", "edge_disagreement",
    "grad_magnitude_diff", "lowfreq_diff", "highfreq_diff",
    # registration / pseudo-change (2)
    "illumination_unchanged", "hard_pseudo_change_fraction",
]


def _resolve(data_root, folder: str, entry: str, dataset_name: str) -> Path:
    """Same fallback logic as ``CDDataset._resolve_path`` (CDD test_ prefix, png<->jpg)."""
    relative = Path(entry)
    candidates = [Path(data_root) / folder / relative]
    if str(dataset_name).upper() == "CDD" and not relative.name.startswith("test_"):
        candidates.append(Path(data_root) / folder / relative.with_name(f"test_{relative.name}"))
    for base in tuple(candidates):
        suffix = base.suffix.lower()
        if suffix == ".png":
            candidates.append(base.with_suffix(".jpg"))
        elif suffix in {".jpg", ".jpeg"}:
            candidates.append(base.with_suffix(".png"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve {folder}/{entry} under {data_root}")


def _grad_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _sample_features_v2(data_root, sample_id, dataset_name):
    pre = _read_rgb(_resolve(data_root, "A", sample_id, dataset_name))
    post = _read_rgb(_resolve(data_root, "B", sample_id, dataset_name))
    label = _read_gray(_resolve(data_root, "label", sample_id, dataset_name))
    if pre.shape[:2] != post.shape[:2] or pre.shape[:2] != label.shape[:2]:
        raise ValueError(f"shape mismatch for {sample_id}")

    lab = (label >= 128).astype(np.uint8)
    H, W = lab.shape
    area = H * W
    change_ratio = float(lab.mean())

    n, _labmap, stats, _c = cv2.connectedComponentsWithStats(lab, connectivity=8)
    comp_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    n_components = int(len(comp_areas))  # 0 for an empty (unchanged) label map
    small_fraction = float((comp_areas < 0.005 * area).sum() / max(len(comp_areas), 1))
    large_fraction = float((comp_areas > 0.05 * area).sum() / max(len(comp_areas), 1))

    contours, _ = cv2.findContours(lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    changed = int(lab.sum())
    boundary_density = float(perimeter / (changed + 1e-6))
    compactness = float((perimeter ** 2) / (4 * np.pi * changed + 1e-6)) if changed else 0.0

    pre_f, post_f = pre.astype(np.float32), post.astype(np.float32)
    absdiff = np.abs(post_f - pre_f)
    rgb_mean_shift = float(absdiff.mean())
    rgb_std_shift = float(np.abs(post_f.std(axis=(0, 1)) - pre_f.std(axis=(0, 1))).mean())
    hist_js = float(np.mean([_hist_js(pre_f[..., c], post_f[..., c]) for c in range(3)]))

    pre_g = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY)
    post_g = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY)
    pre_e = cv2.Canny(pre_g, 50, 150)
    post_e = cv2.Canny(post_g, 50, 150)
    edge_disagreement = float(np.abs(pre_e.astype(np.float32) - post_e.astype(np.float32)).mean())
    grad_magnitude_diff = float(np.abs(_grad_magnitude(post_g) - _grad_magnitude(pre_g)).mean())

    def _freq(img, low=True):
        f = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
        h, w = img.shape
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
        mask = (r < 16) if low else (r >= 16)
        return float(np.abs(f[mask]).mean())

    lowfreq_diff = float(np.abs(_freq(post_g, True) - _freq(pre_g, True)))
    highfreq_diff = float(np.abs(_freq(post_g, False) - _freq(pre_g, False)))

    unchanged = lab == 0
    illumination_unchanged = float(absdiff[unchanged].mean() if unchanged.any() else 0.0)
    changed_mask = lab == 1

    vec = np.array([
        change_ratio, float(changed == 0), n_components,
        small_fraction, large_fraction, boundary_density, compactness, 0.0,  # cv filled in aggregate
        *[0.0, 0.0, 0.0],                                                    # global area filled in aggregate
        rgb_mean_shift, rgb_std_shift, hist_js,
        edge_disagreement, grad_magnitude_diff, lowfreq_diff, highfreq_diff,
        illumination_unchanged, 0.0,                                         # hard-pseudo filled in aggregate
    ], dtype=np.float32)
    return vec, comp_areas / float(area), absdiff, changed_mask, unchanged


def _global_object_area_quantiles(all_areas: list[np.ndarray]) -> list[float]:
    if not all_areas:
        return [0.0, 0.0, 0.0]
    flat = np.concatenate(all_areas)
    if flat.size == 0:
        return [0.0, 0.0, 0.0]
    q = np.percentile(flat, [25, 50, 90])
    return [float(v) for v in q]


def compute_dataset_signature_v2(data_root, dataset_name, split: str = "train",
                                 max_samples: int | None = None, pilot: int = 200):
    """schema_version=2 dataset signature (global object-area quantiles + fixed geometry)."""
    data_root = Path(data_root)
    dataset_name = str(dataset_name).upper()
    ids = [
        line.strip()
        for line in (data_root / "list" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_samples is not None:
        ids = ids[:max_samples]

    # pilot pass: dataset-level threshold = q90 of |A-B| on changed pixels (TRAIN only)
    pilot_ids = ids[: min(pilot, len(ids))]
    changed_diffs = []
    for sid in pilot_ids:
        pre = _read_rgb(_resolve(data_root, "A", sid, dataset_name)).astype(np.float32)
        post = _read_rgb(_resolve(data_root, "B", sid, dataset_name)).astype(np.float32)
        lab = (_read_gray(_resolve(data_root, "label", sid, dataset_name)) >= 128)
        if lab.any():
            changed_diffs.append(np.abs(post - pre)[lab].mean())
    thr = float(np.percentile(changed_diffs, 90)) if changed_diffs else 0.0

    vecs, area_list, cv_list, ratios = [], [], [], []
    hard_fracs = []
    for sid in ids:
        vec, areas_norm, absdiff, changed_mask, unchanged = _sample_features_v2(data_root, sid, dataset_name)
        vecs.append(vec)
        area_list.append(areas_norm)
        ratio = float(vec[0])
        ratios.append(ratio)
        # hard pseudo-change: fraction of unchanged pixels whose |A-B| exceeds the
        # changed-pixel q90 threshold (difficulty of separating pseudo change)
        if unchanged.any() and thr > 0:
            hard_fracs.append(float((absdiff.max(axis=2)[unchanged] > thr).mean()))
        else:
            hard_fracs.append(0.0)

        # per-image boundary/area CV input
        n_comp = float(vec[2])
        cv_list.append(n_comp)

    mat = np.stack(vecs, axis=0)
    sig = mat.mean(axis=0)
    areas_q = _global_object_area_quantiles(area_list)
    sig[8:11] = areas_q
    sig[7] = float(np.std(ratios) / (np.mean(ratios) + 1e-6))       # change_area_cv
    sig[-1] = float(np.mean(hard_fracs))                             # hard_pseudo_change_fraction

    details = {name: float(v) for name, v in zip(SIGNATURE_V2_NAMES, sig)}
    details["_pilot_threshold_absdiff"] = thr
    details["_n_samples"] = len(ids)
    return sig.astype(np.float32), details


__all__ = [
    "compute_dataset_signature",
    "compute_dataset_signature_v2",
    "SIGNATURE_NAMES",
    "SIGNATURE_V2_NAMES",
    "SIGNATURE_V2_VERSION",
]
