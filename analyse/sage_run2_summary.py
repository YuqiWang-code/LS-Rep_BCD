#!/usr/bin/env python3
"""SAGE-CD Run2 result summary: formal test blocks + KD/router diagnostics.

Reads outputs/SAGE-CD/Run2/**/train_log.txt (last complete TEST RESULTS block)
and prints:
  1. formal test metric table (percentage points)
  2. per-arm F1 deltas (adjacent pairs and vs clean anchor R0)
  3. gate verdicts for the Run2 success/failure criteria
  4. last-10-epoch mean KD / lambda / expert-utility diagnostics
  5. historical clean-anchor spread (same protocol, batch 64 / 40k / seed 2333)
     from docs/experiment_metrics.xlsx

Usage: python -B analyse/sage_run2_summary.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
RUN2 = ROOT / "outputs" / "SAGE-CD" / "Run2"
XLSX = ROOT / "docs" / "experiment_metrics.xlsx"
ARMS = ("R0", "R1", "R2", "R3", "R4")
DATASETS = ("SYSU", "WHU")
METRICS = ("Recall", "Precision", "OA", "F1", "IoU", "Kappa")
DIAG = ("kd", "lam_d", "lam_s", "u_d", "u_s")

BLOCK_RE = re.compile(
    r"^=== TEST RESULTS ===[ \t]*\r?$\n(?P<body>.*?)^=== END TEST RESULTS ===[ \t]*\r?$",
    re.MULTILINE | re.DOTALL,
)
KV_RE = re.compile(r"^([^:\r\n]+):[ \t]*(.*?)\r?$", re.MULTILINE)
EPOCH_RE = re.compile(r"^Epoch \[(\d+)/\d+\](?P<body>.*)$", re.MULTILINE)
FIELD_RE = re.compile(r"(\w+)=([-\d.eE+]+)")


def last_test(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = list(BLOCK_RE.finditer(text))
    if not blocks:
        raise SystemExit(f"no complete TEST RESULTS block: {path}")
    return {m.group(1).strip(): m.group(2).strip()
            for m in KV_RE.finditer(blocks[-1].group("body"))}


def last_epochs(path: Path, n: int = 10) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = [FIELD_RE.findall(m.group("body")) for m in EPOCH_RE.finditer(text)]
    tail = rows[-n:]
    out = {}
    for key in DIAG:
        vals = [float(v) for r in tail for k, v in r if k == key]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def main() -> None:
    tests: dict[tuple[str, str], dict[str, str]] = {}
    diags: dict[tuple[str, str], dict[str, float]] = {}
    for arm in ARMS:
        for ds in DATASETS:
            log = RUN2 / arm / ds / "train_log.txt"
            if not log.is_file():
                raise SystemExit(f"missing log: {log}")
            tests[(arm, ds)] = last_test(log)
            diags[(arm, ds)] = last_epochs(log)

    print("=" * 78)
    print("1. SAGE-CD Run2 formal test metrics (last complete TEST RESULTS block, %)")
    print("=" * 78)
    head = f"{'Arm':4} {'Dataset':8}" + "".join(f"{m:>10}" for m in METRICS)
    print(head)
    for arm in ARMS:
        for ds in DATASETS:
            t = tests[(arm, ds)]
            print(f"{arm:4} {ds:8}" + "".join(f"{float(t[m]) * 100:10.2f}" for m in METRICS))

    print()
    print("=" * 78)
    print("2. F1 deltas (percentage points)")
    print("=" * 78)
    f1 = {k: float(v["F1"]) * 100 for k, v in tests.items()}
    pairs = [("R1", "R0", "joint temporal BN"),
             ("R2", "R1", "SCF + identity gate"),
             ("R3", "R2", "dual teacher (uniform)"),
             ("R4", "R3", "failure-type routing"),
             ("R4", "R2", "MAIN GATE: full method vs student reform"),
             ("R4", "R0", "full method vs clean anchor"),
             ("R2", "R0", "student reform vs clean anchor")]
    print(f"{'Comparison':38} {'Variable':30} {'SYSU':>8} {'WHU':>8}")
    for hi, lo, label in pairs:
        row = f"{hi}-{lo:<34} {label:30}"
        for ds in DATASETS:
            row += f"{f1[(hi, ds)] - f1[(lo, ds)]:8.2f}"
        print(row)

    print()
    print("=" * 78)
    print("3. Run2 gate verdicts (frozen criteria, docs/README §4.6)")
    print("=" * 78)
    checks = [
        ("MAIN: R4-R2 >= +0.30 on SYSU and WHU", f1[("R4", "SYSU")] - f1[("R2", "SYSU")],
         f1[("R4", "WHU")] - f1[("R2", "WHU")], 0.30),
        ("FAIL-1: R2 >= R1 (SCF works)", f1[("R2", "SYSU")] - f1[("R1", "SYSU")],
         f1[("R2", "WHU")] - f1[("R1", "WHU")], 0.0),
        ("FAIL-2: R4 > R2 (teacher net gain)", f1[("R4", "SYSU")] - f1[("R2", "SYSU")],
         f1[("R4", "WHU")] - f1[("R2", "WHU")], 0.0),
        ("FAIL-3: routing > uniform (R4-R3)", f1[("R4", "SYSU")] - f1[("R3", "SYSU")],
         f1[("R4", "WHU")] - f1[("R3", "WHU")], 0.0),
    ]
    for label, sysu, whu, thr in checks:
        ok = sysu >= thr and whu >= thr
        print(f"{label:42} SYSU {sysu:+6.2f}  WHU {whu:+6.2f}  -> {'PASS' if ok else 'FAIL'}")

    print()
    print("=" * 78)
    print("4. Last-10-epoch training diagnostics (mean)")
    print("=" * 78)
    print(f"{'Arm':4} {'Dataset':8}" + "".join(f"{k:>9}" for k in DIAG) + f"{'u_d+u_s':>10}")
    for arm in ARMS:
        for ds in DATASETS:
            d = diags[(arm, ds)]
            print(f"{arm:4} {ds:8}" + "".join(f"{d[k]:9.3f}" for k in DIAG)
                  + f"{d['u_d'] + d['u_s']:10.3f}")

    print()
    print("=" * 78)
    print("5. Clean-anchor spread, same protocol (batch 64 / 40000 steps / seed 2333)")
    print("=" * 78)
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["Experiment Results"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c) for c in rows[0]]
    idx = {name: header.index(name) for name in
           ("Run", "Experiment ID", "Dataset", "Batch Size", "Max Steps", "Seed", "F1")}
    clean_anchors = {
        ("DART-R", "B0"), ("RDT-CD-Run1", "B0"), ("RDT-CD-Run2", "B0"),
        ("RDT-CD-Run3", "B0"), ("SCTC-Run1", "S0"), ("FA-SCRD-Run1", "C0"),
        ("SAGE-CD-Run1", "G0"), ("SAGE-CD-Run2", "R0"),
    }
    for ds in ("SYSU-CD-256", "WHU-CD-256"):
        vals = []
        for r in rows[1:]:
            key = (r[idx["Run"]], r[idx["Experiment ID"]])
            if (key in clean_anchors and r[idx["Dataset"]] == ds
                    and r[idx["Batch Size"]] == 64 and r[idx["Max Steps"]] == 40000
                    and r[idx["Seed"]] == 2333):
                vals.append((f"{r[idx['Run']]}/{r[idx['Experiment ID']]}", float(r[idx["F1"]])))
        if not vals:
            continue
        numbers = [v for _, v in vals]
        mean = sum(numbers) / len(numbers)
        sd = (sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1)) ** 0.5
        print(f"\n{ds}: n={len(vals)}  min={min(numbers):.2f}  max={max(numbers):.2f}  "
              f"spread={max(numbers) - min(numbers):.2f}  mean={mean:.2f}  sd={sd:.2f}")
        for name, value in vals:
            print(f"    {name:28} {value:8.2f}")


if __name__ == "__main__":
    sys.exit(main())
