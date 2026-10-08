#!/usr/bin/env python3
"""Freeze the train-only calibration for J1's GT-conditioned reliable distillation.

Review 2026-10-08 §4.3 / §7.1 step A-B. This tool reads ONLY the **train** split
plus the train teacher cache and produces the frozen thresholds used by
``models.distill.task_reliable_kd``:

    m_T      = median of the raw change response z_T over all train pixels
    s_T      = Q0.90(z_T) - Q0.50(z_T)
    q_T(p)   = sigmoid((z_T(p) - m_T) / s_T)        (monotone squash, NOT a probability)
    t_plus   = median of q_T over GT-changed pixels
    t_minus  = Q0.90 of q_T over GT-unchanged pixels

Everything is derived from train only; val and test are never opened. The
quantile-calibration quality is judged on a **train-internal** meta split
(80/20), which is the only honest way to decide whether q_T is usable without
looking at the evaluation sets.

Diagnostics emitted alongside the calibration:
  * quadrant counts (GT+-/response-high-low): how much of the teacher response is
    "changed" that GT calls unchanged, i.e. the pseudo-change leakage H3 predicts;
  * balanced AUROC of q_T against GT, and the same stratified by object size and
    by boundary-vs-interior (H4 predicts boundary degradation);
  * ECE of q_T and of a refit affine calibration on the held-out meta split.

Usage:
    python -m models.tools.audit_taskkd \
        --cache_root /share_datasets/CD_teacher_cache/CATA_CD_v2/dinov3_lvd \
        --data_root /share_datasets/CD/SYSU-CD-256 \
        --dataset_name SYSU --teacher_package dinov3_lvd \
        --split train --max_samples 512 \
        --out outputs/CATA-CD/Run2/calibration/SYSU_dinov3_lvd.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets.cd_dataset import get_loader  # noqa: E402
from models.distill.cache_v2 import TeacherCacheReaderV2, apply_cache_state  # noqa: E402
from models.distill.task_reliable_kd import (  # noqa: E402
    DEFAULT_BOUNDARY_R,
    TaskReliableConfig,
    boundary_band,
)

SIZE_BINS = ((0, 256), (256, 1024), (1024, 10 ** 9))   # px at 256x256
SIZE_LABELS = ("small_lt256", "medium_256_1024", "large_ge1024")
N_ECE_BINS = 15
META_FRACTION = 0.2
RNG_SEED = 2333


def balanced_auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Threshold-free balanced AUROC (ties-aware via mid-ranks), NaN if degenerate."""
    pos, neg = score[label == 1], score[label == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def ece(prob: np.ndarray, label: np.ndarray, n_bins: int = N_ECE_BINS) -> tuple[float, list]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, out = label.size, []
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
        if not m.any():
            continue
        conf, acc, w = float(prob[m].mean()), float(label[m].mean()), float(m.mean())
        e += w * abs(conf - acc)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "conf": conf, "acc": acc})
    return float(e), out


def fit_affine_logit(q: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """2-parameter Platt-style scaling in logit space, fitted by 1-D Newton steps."""
    z = np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    a, b = 1.0, 0.0
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        ga = float(((p - y) * z).mean())
        gb = float((p - y).mean())
        haa = float((p * (1 - p) * z * z).mean()) + 1e-9
        hab = float((p * (1 - p) * z).mean())
        hbb = float((p * (1 - p)).mean()) + 1e-9
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a - 0.5 * da, b - 0.5 * db
    return float(a), float(b)


def object_size_stats(y256: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel object-size bin id and boundary flag for a binary 256x256 mask."""
    yb = (y256 > 0.5).astype(np.uint8)
    n, lab = cv2.connectedComponents(yb, connectivity=8)
    size_id = np.full(yb.shape, -1, dtype=np.int32)
    for c in range(1, n):
        area = int((lab == c).sum())
        for bi, (lo, hi) in enumerate(SIZE_BINS):
            if lo <= area < hi:
                size_id[lab == c] = bi
                break
    band = boundary_band(torch.from_numpy(yb).float()[None, None], DEFAULT_BOUNDARY_R)[0, 0]
    return size_id, band.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dataset_name", required=True, choices=("SYSU", "WHU", "CDD", "LEVIR"))
    ap.add_argument("--teacher_package", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_samples", type=int, default=512)
    ap.add_argument("--trainsize", type=int, default=256)
    ap.add_argument("--boundary_r", type=int, default=DEFAULT_BOUNDARY_R)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = get_loader(args.data_root, str(Path(args.data_root) / "list" / f"{args.split}.txt"),
                      batchsize=1, trainsize=args.trainsize, num_workers=0,
                      return_state=True, seed=RNG_SEED).dataset
    base.set_epoch(0)
    reader = TeacherCacheReaderV2(args.cache_root, args.dataset_name)

    n = min(args.max_samples, len(base))
    step = max(1, len(base) // n)
    idxs = list(range(0, len(base), step))[:n]

    z_native: list[np.ndarray] = []
    q_changed: list[np.ndarray] = []
    q_unchanged: list[np.ndarray] = []
    per_sample: list[dict] = []
    native_hw = None

    for k, idx in enumerate(idxs):
        image, label, sample_id, state = base[idx]
        cache = apply_cache_state(reader.load(args.split, sample_id), state)
        conf = cache["confidence"].float()
        if conf.dim() == 3:
            conf = conf[0]
        z = conf.numpy().astype(np.float64)
        if native_hw is None:
            native_hw = (int(z.shape[0]), int(z.shape[1]))
        z_native.append(z.reshape(-1))

        y = label.detach().float().numpy()
        y = y[0] if y.ndim == 3 else y
        if y.shape != (args.trainsize, args.trainsize):
            y = cv2.resize(y, (args.trainsize, args.trainsize), interpolation=cv2.INTER_NEAREST)
        yb = (y > 0.5).astype(np.float64)

        # Match training exactly: q is computed on the NATIVE grid, then upsampled
        # to 256 and compared against the 256 label.
        per_sample.append({"idx": idx, "sample_id": sample_id, "z": z, "y": yb})
        if k % 64 == 0:
            print(f"  [taskkd] loaded {k + 1}/{len(idxs)}", flush=True)

    z_all = np.concatenate(z_native)
    z_median = float(np.median(z_all))
    z_q90 = float(np.quantile(z_all, 0.90))

    # second pass: q at 256 using the frozen m_T,s_T
    s_T = max(z_q90 - z_median, 1e-6)
    size_score: dict[str, list] = {lab: [[], []] for lab in SIZE_LABELS}   # [score, label]
    edge_score, inner_score = [[], []], [[], []]
    quad = {"gt1_q_hi": 0, "gt1_q_lo": 0, "gt0_q_hi": 0, "gt0_q_lo": 0}
    all_q, all_y = [], []

    for s in per_sample:
        z = torch.from_numpy(s["z"])[None, None].float()
        q_native = torch.sigmoid((z - z_median) / s_T)
        q256 = torch.nn.functional.interpolate(q_native, size=(args.trainsize, args.trainsize),
                                               mode="bilinear", align_corners=False)[0, 0].numpy()
        yb = s["y"]
        qc, qu = q256[yb > 0.5], q256[yb < 0.5]
        if qc.size:
            q_changed.append(qc if qc.size < 20000 else np.random.default_rng(0).choice(qc, 20000, replace=False))
        if qu.size:
            q_unchanged.append(qu if qu.size < 20000 else np.random.default_rng(0).choice(qu, 20000, replace=False))
        ss = np.random.default_rng(s["idx"]).choice(q256.size, min(4000, q256.size), replace=False)
        all_q.append(q256.reshape(-1)[ss])
        all_y.append(yb.reshape(-1)[ss])

        size_id, band = object_size_stats(yb)
        # Size-stratified AUROC needs positives from the bin AND the unchanged
        # pixels as negatives. Scoring only the bin's own pixels would be all-one-
        # class and degenerate to NaN.
        neg_mask = yb < 0.5
        if neg_mask.any():
            neg_scores = q256[neg_mask]
            neg_labels = np.zeros(neg_scores.size, dtype=np.float64)
            rng_neg = np.random.default_rng(s["idx"] + 7)
            if neg_scores.size > 40000:
                keep = rng_neg.choice(neg_scores.size, 40000, replace=False)
                neg_scores, neg_labels = neg_scores[keep], neg_labels[keep]
            for bi, slab in enumerate(SIZE_LABELS):
                m = size_id == bi
                if m.any():
                    size_score[slab][0].append(np.concatenate([q256[m], neg_scores]))
                    size_score[slab][1].append(
                        np.concatenate([np.ones(int(m.sum())), neg_labels]))
        mb = band > 0
        if mb.any():
            edge_score[0].append(q256[mb]); edge_score[1].append(yb[mb])
        mi = ~mb
        if mi.any():
            inner_score[0].append(q256[mi]); inner_score[1].append(yb[mi])

    qc_all = np.concatenate(q_changed) if q_changed else np.array([])
    qu_all = np.concatenate(q_unchanged) if q_unchanged else np.array([])
    t_plus = float(np.median(qc_all)) if qc_all.size else 0.5
    t_minus = float(np.quantile(qu_all, 0.90)) if qu_all.size else 0.5

    # quadrant counts / leakage at the frozen thresholds, over all visited pixels
    for s in per_sample:
        z = torch.from_numpy(s["z"])[None, None].float()
        q256 = torch.nn.functional.interpolate(
            torch.sigmoid((z - z_median) / s_T), size=(args.trainsize, args.trainsize),
            mode="bilinear", align_corners=False)[0, 0].numpy()
        yb = s["y"] > 0.5
        hi, lo = q256 >= t_plus, q256 <= t_minus
        quad["gt1_q_hi"] += int((yb & hi).sum())
        quad["gt1_q_lo"] += int((yb & ~hi).sum())
        quad["gt0_q_lo"] += int((~yb & lo).sum())
        quad["gt0_q_hi"] += int((~yb & ~lo).sum())
    tot1 = quad["gt1_q_hi"] + quad["gt1_q_lo"]
    tot0 = quad["gt0_q_lo"] + quad["gt0_q_hi"]

    Q = np.concatenate(all_q); Y = np.concatenate(all_y)
    auroc = balanced_auroc(Q, Y)

    def _cat(pairs):
        if not pairs[0]:
            return None, None
        return np.concatenate(pairs[0]), np.concatenate(pairs[1])

    size_auroc = {}
    for lab in SIZE_LABELS:
        sc, lb = _cat(size_score[lab])
        size_auroc[lab] = (None if sc is None or sc.size < 10
                           else balanced_auroc(sc, lb))
    esc, elb = _cat(edge_score)
    isc, ilb = _cat(inner_score)
    edge_auroc = None if esc is None else balanced_auroc(esc, elb)
    inner_auroc = None if isc is None else balanced_auroc(isc, ilb)

    # ECE on a train-internal meta split (only place we evaluate calibration quality)
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(Q.size)
    n_meta = int(Q.size * META_FRACTION)
    meta, fit = perm[:n_meta], perm[n_meta:]
    ece_raw = ece(Q[meta], Y[meta])[0]
    aa, bb = fit_affine_logit(Q[fit], Y[fit])
    qq = np.clip(Q[meta], 1e-6, 1 - 1e-6)
    zz = np.log(qq / (1 - qq))
    ece_affine = ece(1.0 / (1.0 + np.exp(-(aa * zz + bb))), Y[meta])[0]

    cfg = TaskReliableConfig(
        teacher_package=args.teacher_package, dataset=args.dataset_name,
        z_median=z_median, z_q90=z_q90, t_plus=t_plus, t_minus=t_minus,
        boundary_r=args.boundary_r, native_hw=native_hw or (16, 16),
        n_samples=len(idxs), ece=float(ece_raw),
        note=("q_T is a monotone squash of the cached change response, NOT a calibrated "
              "probability; thresholds are train-only quantiles"),
    )
    cfg.to_json(args.out)

    diag = {
        "_meta": {"split": args.split, "n_samples": len(idxs), "data_root": args.data_root,
                  "cache_root": args.cache_root, "trainsize": args.trainsize,
                  "boundary_r": args.boundary_r, "rng_seed": RNG_SEED,
                  "note": "train split only; val/test never opened"},
        "calibration": {"z_median": z_median, "z_q90": z_q90, "s_T": s_T,
                        "t_plus": t_plus, "t_minus": t_minus,
                        "native_hw": list(native_hw or (16, 16))},
        "quadrants": {**quad,
                      "changed_recall_at_t_plus": quad["gt1_q_hi"] / max(tot1, 1),
                      "unchanged_leakage_at_t_plus": quad["gt0_q_hi"] / max(tot0, 1),
                      "unchanged_confident_frac": quad["gt0_q_lo"] / max(tot0, 1),
                      "n_changed": tot1, "n_unchanged": tot0},
        "auroc": {"balanced_all": auroc, "by_size": size_auroc,
                  "boundary": edge_auroc, "interior": inner_auroc,
                  "boundary_minus_interior": (
                      None if edge_auroc is None or inner_auroc is None
                      else edge_auroc - inner_auroc)},
        "threshold_sanity": {
            "t_plus": t_plus, "t_minus": t_minus,
            # t_plus <= t_minus means the change response barely separates changed
            # from unchanged: the "confidently changed" bar is no higher than the
            # "confidently unchanged" bar. Reported, not hidden.
            "t_plus_le_t_minus": bool(t_plus <= t_minus),
            "median_q_changed": t_plus, "q90_q_unchanged": t_minus,
        },
        "calibration_quality": {"ece_raw": float(ece_raw), "ece_affine_refit": float(ece_affine),
                                "affine_a": aa, "affine_b": bb,
                                "meta_fraction": META_FRACTION},
    }
    diag_path = Path(args.out).with_suffix(".diagnostics.json")
    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(f"\n[taskkd] calibration -> {args.out}")
    print(f"[taskkd] native_hw={native_hw} m_T={z_median:.4f} s_T={s_T:.4f} "
          f"t+={t_plus:.4f} t-={t_minus:.4f}")
    print(f"[taskkd] balanced AUROC(q_T, GT) = {auroc:.4f}")
    print(f"[taskkd] changed recall @t+ = {quad['gt1_q_hi']/max(tot1,1):.4f}   "
          f"UNCHANGED LEAKAGE @t+ = {quad['gt0_q_hi']/max(tot0,1):.4f}")
    print(f"[taskkd] AUROC by size: " + " ".join(
        f"{k}={v:.3f}" if v is not None else f"{k}=n/a" for k, v in size_auroc.items()))
    print(f"[taskkd] boundary={edge_auroc} interior={inner_auroc}")
    print(f"[taskkd] ECE raw={ece_raw:.4f} vs affine-refit={ece_affine:.4f}")
    print(f"[taskkd] diagnostics -> {diag_path}")


if __name__ == "__main__":
    main()
