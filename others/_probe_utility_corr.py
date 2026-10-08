#!/usr/bin/env python3
"""Do the train-only probes predict teacher KD utility at all? (negative-result evidence)"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "CATA-CD"
DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]
TEACHERS = ["sam2", "dinov2", "dinov3_lvd", "dinov3_sat", "remoteclip", "mars",
            "anysat", "universat", "radio"]
KEYS = ["balanced_auroc", "hard_negative_separation", "boundary_contrast",
        "size_conditioned_auroc_gap", "unchanged_leakage_ratio"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).is_file() else {}


reg = load(OUT / "registry" / "agent_train_registry.json")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


print("=== P1 balanced AUROC per (dataset, teacher) ===")
rows = []
for ds in DATASETS:
    line = [f"{ds}:"]
    for t in TEACHERS:
        p = load(OUT / "probes_v2" / f"{ds}_{t}.json")
        au = p.get("balanced_auroc")
        y = reg.get(t, {}).get(ds, {}).get("val_delta_f1_pp")
        if au is not None:
            line.append(f"{t}={au:.3f}")
        rows.append({"ds": ds, "t": t, "auroc": au, "y": y})
    print("  " + "  ".join(line))

print("\n=== Spearman(probe, val_delta_f1_pp) ===")
for k in KEYS:
    per_ds = {}
    for ds in DATASETS:
        xs, ys = [], []
        for t in TEACHERS:
            v = load(OUT / "probes_v2" / f"{ds}_{t}.json").get(k)
            y = reg.get(t, {}).get(ds, {}).get("val_delta_f1_pp")
            if v is not None and y is not None:
                xs.append(v)
                ys.append(y)
        per_ds[ds] = spearman(xs, ys)
    # pooled (within-dataset z-scored to remove dataset offset)
    px, py = [], []
    for ds in DATASETS:
        xs, ys = [], []
        for t in TEACHERS:
            v = load(OUT / "probes_v2" / f"{ds}_{t}.json").get(k)
            y = reg.get(t, {}).get(ds, {}).get("val_delta_f1_pp")
            if v is not None and y is not None:
                xs.append(v)
                ys.append(y)
        if len(xs) >= 3:
            px.extend((np.array(xs) - np.mean(xs)).tolist())
            py.extend((np.array(ys) - np.mean(ys)).tolist())
    pooled = spearman(px, py)
    print(f"  {k:32s} " + "  ".join(f"{ds}={per_ds[ds]:+.2f}" for ds in DATASETS) + f"   pooled={pooled:+.2f}")

print("\n=== 教师 AUROC 均值 vs 其 val 效用均值 ===")
for t in TEACHERS:
    aus = [load(OUT / "probes_v2" / f"{ds}_{t}.json").get("balanced_auroc") for ds in DATASETS]
    ys = [reg.get(t, {}).get(ds, {}).get("val_delta_f1_pp") for ds in DATASETS]
    aus = [v for v in aus if v is not None]
    ys = [v for v in ys if v is not None]
    print(f"  {t:12s} meanAuroc={np.mean(aus):.3f}  meanValDelta={np.mean(ys):+.3f}pp")
