#!/usr/bin/env python3
"""Verify every quantitative claim the 2026-10-08 review makes about our own tables.

The review (docs/temporary/CATA-CD_v2_负结果严格复盘_..._20261008.md §2.2/§6.2) states
that two of our previously reported summaries were WRONG:

  * "36 组中 30 组非正"        -> review says 26 are <=0 / 33 are not strong-positive
  * "所有 val 效用在 ±0.15pp 内" -> review says positive val <= +0.146pp but
                                   negative val reaches about -1.076pp

It also re-derives the noise floor sigma and the engineering thresholds tau_D.
This script recomputes all of it from the frozen artifacts so the README wording
is corrected on evidence, not on assertion.

Run:
    python others/_verify_review_claims.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "CATA-CD"
DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]
TEACHERS = ["sam2", "dinov2", "dinov3_lvd", "dinov3_sat", "remoteclip", "mars",
            "anysat", "universat", "radio"]
STRONG_PP = 0.20


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


audit = load(OUT / "audit_v3" / "test_block_audit.json")
research = load(OUT / "registry" / "research_report.json")
agent = load(OUT / "registry" / "agent_train_registry.json")

recs = {(r["experiment_id"], r["dataset"]): r for r in audit.get("records", [])}
EXP = {"sam2": "TV-SAM", "dinov2": "TV-D2", "dinov3_lvd": "TV-D3N", "dinov3_sat": "TV-D3S",
       "remoteclip": "TV-RCLIP", "mars": "TV-MARS", "anysat": "TV-ANYSAT",
       "universat": "TV-UNISAT", "radio": "TV-RADIO"}

print("=" * 78)
print("A. TEST deltas recomputed from the frozen logs (not from the derived JSON)")
print("=" * 78)

# Independent recomputation: audit file -> deltas, then compare with research_report.
audit_test_delta = {}
for t, exp in EXP.items():
    for ds in DATASETS:
        tv, c1 = recs.get((exp, ds)), recs.get(("C1", ds))
        if tv and c1 and tv.get("test_pp") and c1.get("test_pp"):
            audit_test_delta[(t, ds)] = {
                "df1": tv["test_pp"]["F1"] - c1["test_pp"]["F1"],
                "diou": tv["test_pp"]["IoU"] - c1["test_pp"]["IoU"],
            }

reg_test_delta = {
    (t, ds): {"df1": research[t][ds].get("test_delta_f1_pp"),
              "diou": research[t][ds].get("test_delta_iou_pp")}
    for t in TEACHERS for ds in DATASETS if research.get(t, {}).get(ds)
}

mismatch = []
for k, v in audit_test_delta.items():
    rv = reg_test_delta.get(k, {})
    if rv.get("df1") is None:
        mismatch.append((k, "missing in registry"))
    elif abs(v["df1"] - rv["df1"]) > 1e-6:
        mismatch.append((k, f"df1 audit={v['df1']:.6f} registry={rv['df1']:.6f}"))
print(f"independent recompute vs research_report.json: {len(audit_test_delta)} pairs, "
      f"{len(mismatch)} mismatches")
for m in mismatch:
    print("   MISMATCH", m)

n = len(audit_test_delta)
pos = [k for k, v in audit_test_delta.items() if v["df1"] > 0]
nonpos = [k for k, v in audit_test_delta.items() if v["df1"] <= 0]
strong = [k for k, v in audit_test_delta.items()
          if v["df1"] >= STRONG_PP and v["diou"] > 0]
print(f"\n  n                            = {n}")
print(f"  df1 >  0                     = {len(pos)}")
print(f"  df1 <= 0                     = {len(nonpos)}")
print(f"  strong positive (dF1>={STRONG_PP} & dIoU>0) = {len(strong)}")
print(f"  not strong positive          = {n - len(strong)}")
print("  review claims: 10 positive / 26 non-positive / 3 strong / 33 not-strong")
ok = (len(pos) == 10 and len(nonpos) == 26 and len(strong) == 3 and n - len(strong) == 33)
print(f"  --> review claim {'CONFIRMED' if ok else 'DISCREPANT'}")
print("  positive pairs:", ", ".join(f"{t}/{ds}({audit_test_delta[(t, ds)]['df1']:+.3f})"
                                     for t, ds in sorted(pos)))

print()
print("=" * 78)
print("B. VAL deltas: is the '+/-0.15pp' claim correct?")
print("=" * 78)

val_deltas = []
for t in TEACHERS:
    for ds in DATASETS:
        v = agent.get(t, {}).get(ds, {}).get("val_delta_f1_pp")
        if v is not None:
            val_deltas.append((t, ds, v))

posv = [(t, ds, v) for t, ds, v in val_deltas if v > 0]
negv = [(t, ds, v) for t, ds, v in val_deltas if v < 0]
print(f"  n = {len(val_deltas)}   positive = {len(posv)}   negative = {len(negv)}")
print(f"  max positive val delta = {max(v for _, _, v in val_deltas):+.4f} pp")
print(f"  min negative val delta = {min(v for _, _, v in val_deltas):+.4f} pp")
print(f"  max |val delta|        = {max(abs(v) for _, _, v in val_deltas):.4f} pp")
print("  --> claim 'all val within +/-0.15pp' is",
      "CONFIRMED" if all(abs(v) <= 0.15 for _, _, v in val_deltas) else "REFUTED")
print("  worst 5 val deltas:")
for t, ds, v in sorted(val_deltas, key=lambda x: x[2])[:5]:
    print(f"     {t:12s} {ds:6s} {v:+.4f} pp")

print("\n  per-dataset oracle (max) val delta = the Agent's best possible label:")
oracle = {}
for ds in DATASETS:
    vals = [(t, agent[t][ds]["val_delta_f1_pp"]) for t in TEACHERS
            if agent.get(t, {}).get(ds, {}).get("val_delta_f1_pp") is not None]
    if vals:
        bt, bv = max(vals, key=lambda x: x[1])
        oracle[ds] = bv
        print(f"     {ds:6s} max={bv:+.4f} pp  ({bt})   delta gate = +{STRONG_PP}pp  "
              f"-> {'PASS' if bv > STRONG_PP else 'FAIL'}")

print()
print("=" * 78)
print("C. Noise floor sigma from the 4 C1-baseline repeats (C1 + R1/R2/R3)")
print("=" * 78)

runs = ["C1", "R1", "R2", "R3"]
NOISE_EXP = {"C1": None, "R1": "R1", "R2": "R2", "R3": "R3"}
print("  test F1 (pp) per repeat; C1 lives in teacher_adaptation, R* in noise/")
test_vals, val_vals = {}, {}
for ds in DATASETS:
    tv, vv = [], []
    for run in runs:
        if run == "C1":
            r = recs.get(("C1", ds))
        else:
            r = recs.get((run, ds))
        if r and r.get("test_pp"):
            tv.append(r["test_pp"]["F1"])
        if r and r.get("best_val_f1_pp") is not None:
            vv.append(r["best_val_f1_pp"])
    test_vals[ds] = tv
    val_vals[ds] = vv

print(f"\n  {'ds':6s} {'n':>2s} {'test sigma':>11s} {'test range':>11s} "
      f"{'val sigma':>10s} {'tau_D=max(0.2,2sqrt2 s)':>24s}")
tau = {}
for ds in DATASETS:
    ts = statistics.stdev(test_vals[ds]) if len(test_vals[ds]) > 1 else float("nan")
    vs = statistics.stdev(val_vals[ds]) if len(val_vals[ds]) > 1 else float("nan")
    rg = max(test_vals[ds]) - min(test_vals[ds])
    tau[ds] = max(STRONG_PP, 2 * (2 ** 0.5) * ts)
    print(f"  {ds:6s} {len(test_vals[ds]):2d} {ts:11.4f} {rg:11.4f} {vs:10.4f} {tau[ds]:24.4f}")
print("\n  review claims sigma_test = SYSU 0.289 / WHU 0.537 / CDD 0.013 / LEVIR 0.195")
print("  review claims sigma_val  = SYSU 0.411 / WHU 0.311 / CDD 0.013 / LEVIR 0.061")

print("\n  largest single observed teacher test gain vs tau_D (can 'perfect selection' pass?):")
for ds in DATASETS:
    best = max((audit_test_delta[(t, ds)]["df1"] for t in TEACHERS
                if (t, ds) in audit_test_delta), default=float("nan"))
    print(f"     {ds:6s} best teacher dF1={best:+.4f}  tau_D={tau[ds]:.4f}  "
          f"-> {'PASS' if best > tau[ds] else 'does NOT clear noise gate'}")

print()
print("=" * 78)
print("D. Hard-target gaps for C1 and for the best teacher per dataset")
print("=" * 78)
TARGET = {"SYSU": 85.0, "LEVIR": 92.5, "WHU": 95.0, "CDD": 98.0}
for ds in DATASETS:
    c1 = recs[("C1", ds)]["test_pp"]["F1"]
    best = max((audit_test_delta[(t, ds)]["df1"] for t in TEACHERS
                if (t, ds) in audit_test_delta), default=0.0)
    print(f"  {ds:6s} C1={c1:7.4f} target={TARGET[ds]:5.1f} gap={TARGET[ds]-c1:+.4f}pp   "
          f"C1+best_teacher={c1 + best:7.4f}  remaining gap={TARGET[ds]-c1-best:+.4f}pp")

sys.exit(0)
