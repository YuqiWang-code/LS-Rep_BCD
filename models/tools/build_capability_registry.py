#!/usr/bin/env python3
"""Build the Teacher Capability Registry (Stage 3) from completed train_log.txt files.

Outputs two strictly separated JSON files:
  research_report.json       -- formal TEST deltas (paper evidence only; never Agent input)
  agent_train_registry.json  -- VAL deltas only (Agent training input; no test leakage)

Usage:
    python -m models.tools.build_capability_registry \
        --logs_root outputs/CATA-CD/teacher_adaptation --registry_dir outputs/CATA-CD/registry
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# teacher_package -> experiment_id
TEACHERS = {
    "sam2": "TV-SAM",
    "dinov2": "TV-D2",
    "dinov3_lvd": "TV-D3N",
    "dinov3_sat": "TV-D3S",
    "remoteclip": "TV-RCLIP",
    "mars": "TV-MARS",
    "anysat": "TV-ANYSAT",
    "universat": "TV-UNISAT",
    "radio": "TV-RADIO",
}
DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]
EPOCH_RE = re.compile(r"Epoch \[\d+/\d+\].*?F1=([\d.]+) IoU=([\d.]+)")
TEST_RE = re.compile(r"=== TEST RESULTS ===\n(.*?)=== END TEST RESULTS ===", re.S)


def best_val(text: str):
    f1s, ious = [], []
    for m in EPOCH_RE.finditer(text):
        f1s.append(float(m.group(1)))
        ious.append(float(m.group(2)))
    if not f1s:
        return None
    i = max(range(len(f1s)), key=lambda k: f1s[k])
    return {"f1": f1s[i], "iou": ious[i]}


def test_metrics(text: str):
    m = TEST_RE.search(text)
    if not m:
        return None
    vals = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k in ("F1", "IoU", "Recall", "Precision", "OA", "Kappa"):
                try:
                    vals[k] = float(v.strip())
                except ValueError:
                    pass
    return vals or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", required=True)
    ap.add_argument("--registry_dir", required=True)
    args = ap.parse_args()

    logs_root = Path(args.logs_root)
    reg_dir = Path(args.registry_dir)
    reg_dir.mkdir(parents=True, exist_ok=True)

    def read(exp: str, ds: str) -> str:
        p = logs_root / exp / ds / "train_log.txt"
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""

    # anchors
    c1_val = {ds: best_val(read("C1", ds)) for ds in DATASETS}
    c1_test = {ds: test_metrics(read("C1", ds)) for ds in DATASETS}
    c0_test = {ds: test_metrics(read("C0", ds)) for ds in DATASETS}

    research = {"_anchors": {"C0_test": c0_test, "C1_test": c1_test, "C1_val": c1_val}}
    agent = {}

    for teacher, exp in TEACHERS.items():
        research[teacher] = {}
        agent[teacher] = {}
        for ds in DATASETS:
            txt = read(exp, ds)
            tv = best_val(txt)
            tt = test_metrics(txt)
            cv = c1_val[ds]
            ct = c1_test[ds]
            entry_r = {"experiment_id": exp, "has_test": tt is not None}
            entry_a = {"experiment_id": exp, "has_val": tv is not None}
            if tv and cv:
                entry_a["val_f1"] = tv["f1"]
                entry_a["val_iou"] = tv["iou"]
                entry_a["val_delta_f1"] = tv["f1"] - cv["f1"]
                entry_a["val_delta_iou"] = tv["iou"] - cv["iou"]
            if tt and ct:
                entry_r["test_f1"] = tt["F1"]
                entry_r["test_iou"] = tt["IoU"]
                entry_r["test_delta_f1"] = tt["F1"] - ct["F1"]
                entry_r["test_delta_iou"] = tt["IoU"] - ct["IoU"]
            research[teacher][ds] = entry_r
            agent[teacher][ds] = entry_a

    (reg_dir / "research_report.json").write_text(json.dumps(research, indent=2), encoding="utf-8")
    (reg_dir / "agent_train_registry.json").write_text(json.dumps(agent, indent=2), encoding="utf-8")

    print(f"[registry] wrote {reg_dir/'research_report.json'} and {reg_dir/'agent_train_registry.json'}")
    print("\n== VAL utility (best val F1 relative to C1), Agent input ==")
    for ds in DATASETS:
        row = [f"{ds}: C1={c1_val[ds]['f1']*100:.2f}" if c1_val[ds] else f"{ds}: C1=?"]
        for teacher in TEACHERS:
            e = agent[teacher][ds]
            if "val_delta_f1" in e:
                row.append(f"{teacher}={e['val_delta_f1']*100:+.2f}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
