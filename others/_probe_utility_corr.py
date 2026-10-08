#!/usr/bin/env python3
"""Do the train-only probes predict teacher KD utility at all? (negative-result evidence)

Review 2026-10-08 §3.3 / §8.1 (P1) corrections applied to the original draft:

  * the original used ``argsort(argsort(x))`` as a rank transform, which is NOT
    ties-aware (equal values get arbitrary distinct ranks). We now use
    ``scipy.stats.spearmanr``, which applies the standard mid-rank correction.
  * per-domain rank correlation is the primary quantity; the across-domain number
    is reported as a **descriptive macro-average of the four domain values**, not
    as a pooled correlation over 36 supposedly independent points.
  * **no p-values are reported.** Four domains x nine teachers is not 36
    independent samples (teachers are shared across domains, and within a domain
    the nine teachers are not a random sample), so a p-value here would be
    meaningless. Instead a **shuffled-pairing null** shows the range of |r| that
    random teacher->utility assignment produces at this sample size.
  * effective n per domain is printed so the reader can see how little data
    supports each number.

Usage:
    python others/_probe_utility_corr.py
    python others/_probe_utility_corr.py --probes_dir /share_datasets/CD_teacher_cache/CATA_CD_v2/probes_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "CATA-CD"
DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]
TEACHERS = ["sam2", "dinov2", "dinov3_lvd", "dinov3_sat", "remoteclip", "mars",
            "anysat", "universat", "radio"]
KEYS = ["balanced_auroc", "hard_negative_separation", "boundary_contrast",
        "size_conditioned_auroc_gap", "unchanged_leakage_ratio"]
N_PERM = 4000


def load(p) -> dict:
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def spearman_tiesafe(xs, ys) -> float:
    """Ties-aware Spearman rho; NaN when degenerate (constant input)."""
    if len(xs) < 3:
        return float("nan")
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes_dir", default=str(OUT / "probes_v2"))
    ap.add_argument("--registry", default=str(OUT / "registry_v3" / "agent_train_registry.json"))
    ap.add_argument("--out_json", default=str(OUT / "probe_utility_corr.json"))
    ap.add_argument("--seed", type=int, default=2333)
    args = ap.parse_args()

    probe_dir, reg = Path(args.probes_dir), load(args.registry)
    rng = np.random.default_rng(args.seed)

    # Fail loudly on a missing/empty registry: silently producing n=0 for every
    # domain would look like a legitimate "no correlation" result.
    if not reg:
        raise SystemExit(f"[corr] FATAL: registry is empty or missing: {args.registry}")
    if not probe_dir.is_dir():
        raise SystemExit(f"[corr] FATAL: probes dir missing: {probe_dir}")
    n_teachers = len([k for k in reg if not k.startswith("_")])
    print(f"[corr] registry={args.registry} ({n_teachers} teachers); probes={probe_dir}")
    if n_teachers < 3:
        raise SystemExit(f"[corr] FATAL: only {n_teachers} teachers in the registry")

    # ---- P1 balanced AUROC per (dataset, teacher) ----
    print("=== P1 balanced AUROC per (dataset, teacher) ===")
    per_ds_auroc = {}
    for ds in DATASETS:
        vals = {}
        for t in TEACHERS:
            au = load(probe_dir / f"{ds}_{t}.json").get("balanced_auroc")
            if au is not None:
                vals[t] = au
        per_ds_auroc[ds] = vals
        print("  %-6s " % ds + "  ".join(f"{t}={v:.3f}" for t, v in vals.items()))

    # ---- per-domain ties-aware Spearman + macro-average + shuffled null ----
    print("\n=== Spearman(probe, val_delta_f1_pp) - ties-aware, per domain ===")
    print("  (no p-values: 4 domains x 9 teachers are not independent samples;")
    print(f"   a shuffled-pairing null ({N_PERM} draws) shows the noise range instead)\n")
    results = {}
    for k in KEYS:
        per_ds = {}
        for ds in DATASETS:
            xs, ys = [], []
            for t in TEACHERS:
                v = load(probe_dir / f"{ds}_{t}.json").get(k)
                y = reg.get(t, {}).get(ds, {}).get("val_delta_f1_pp")
                if v is not None and y is not None and np.isfinite(v) and np.isfinite(y):
                    xs.append(float(v))
                    ys.append(float(y))
            r = spearman_tiesafe(xs, ys)
            per_ds[ds] = {"r": r, "n": len(xs)}
            if len(xs) >= 3 and np.isfinite(r):
                arr_y = np.asarray(ys, float)
                draws = [spearman_tiesafe(xs, rng.permutation(arr_y)) for _ in range(N_PERM)]
                draws = np.asarray([d for d in draws if np.isfinite(d)])
                lo, hi = np.percentile(draws, [2.5, 97.5]) if draws.size else (np.nan, np.nan)
                per_ds[ds]["null_95"] = [float(lo), float(hi)]
        macro = float(np.nanmean([per_ds[d]["r"] for d in DATASETS]))
        los = [per_ds[d].get("null_95", [np.nan, np.nan])[0] for d in DATASETS]
        his = [per_ds[d].get("null_95", [np.nan, np.nan])[1] for d in DATASETS]
        all_lo, all_hi = float(np.nanmean(los)), float(np.nanmean(his))
        results[k] = {"per_domain": per_ds, "macro_avg_r": macro,
                      "null_macro_95": [all_lo, all_hi]}
        cells = "  ".join(
            (f"{d}={per_ds[d]['r']:+.2f}(n={per_ds[d]['n']})"
             if np.isfinite(per_ds[d]["r"]) else f"{d}=n/a(n={per_ds[d]['n']})")
            for d in DATASETS)
        flag = ""
        if np.isfinite(macro) and np.isfinite(all_lo):
            flag = ("  [inside null 95%]" if all_lo <= macro <= all_hi
                    else "  [outside null 95%]")
        print(f"  {k:30s} {cells}   macro={macro:+.2f}{flag}")
        print(f"  {'':30s} shuffled-null macro 95% = [{all_lo:+.2f}, {all_hi:+.2f}]")

    # ---- teacher mean AUROC vs mean utility ----
    print("\n=== teacher mean AUROC vs its mean val utility ===")
    rows = []
    for t in TEACHERS:
        aus = [per_ds_auroc[d].get(t) for d in DATASETS]
        ys = [reg.get(t, {}).get(d, {}).get("val_delta_f1_pp") for d in DATASETS]
        aus = [v for v in aus if v is not None]
        ys = [v for v in ys if v is not None]
        if aus and ys:
            rows.append({"teacher": t, "mean_auroc": float(np.mean(aus)),
                         "mean_val_delta_pp": float(np.mean(ys))})
    for r in sorted(rows, key=lambda r: -r["mean_auroc"]):
        print(f"  {r['teacher']:12s} meanAuroc={r['mean_auroc']:.3f}  "
              f"meanValDelta={r['mean_val_delta_pp']:+.3f}pp")
    if len(rows) >= 3:
        au = [r["mean_auroc"] for r in rows]
        r_au = spearman_tiesafe(au, [r["mean_val_delta_pp"] for r in rows])
        arr = np.asarray([r["mean_val_delta_pp"] for r in rows], float)
        draws = np.asarray([d for d in
                            (spearman_tiesafe(au, rng.permutation(arr)) for _ in range(N_PERM))
                            if np.isfinite(d)])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        inside = lo <= r_au <= hi
        print(f"\n  across-teacher Spearman(meanAUROC, meanValDelta) = {r_au:+.3f}   "
              f"shuffled-null 95% = [{lo:+.2f}, {hi:+.2f}]  (n={len(rows)} teachers)"
              f"  {'[inside null]' if inside else '[outside null]'}")
        results["_teacher_mean"] = {"rho": r_au, "null_95": [float(lo), float(hi)],
                                    "n_teachers": len(rows), "inside_null": bool(inside)}

    Path(args.out_json).write_text(json.dumps({
        "_meta": {
            "method": "scipy.stats.spearmanr (ties-aware, mid-rank)",
            "aggregation": "per-domain rho + descriptive macro-average; NO p-values",
            "why_no_p": "4 domains x 9 teachers are not independent samples",
            "null": f"{N_PERM} shuffled pairings per domain, seed {args.seed}",
            "probes_dir": str(args.probes_dir),
            "registry": str(args.registry),
        },
        "per_dataset_balanced_auroc": per_ds_auroc,
        "probe_vs_utility": results,
        "teacher_mean": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n[corr] wrote {args.out_json}")


if __name__ == "__main__":
    main()
