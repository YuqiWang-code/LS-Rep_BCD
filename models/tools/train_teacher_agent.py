#!/usr/bin/env python3
"""Train the offline contextual bandit Agent with Leave-One-Dataset-Out (Stage 4).

Builds (dataset, teacher) pair features from dataset signatures + teacher probes,
fits the tiny MLP on validation utilities, and runs LODO selection. Test metrics
are never used.

Usage:
    python -m models.tools.train_teacher_agent \
        --signatures_dir signatures --probes_dir probes \
        --registry_dir registry --output_dir agent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.agent.offline_bandit import train_policy  # noqa: E402

DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CATA-CD teacher-selection Agent (LODO)")
    p.add_argument("--signatures_dir", required=True)
    p.add_argument("--probes_dir", required=True)
    p.add_argument("--registry_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    args = parse_args()
    sig_dir = Path(args.signatures_dir)
    probe_dir = Path(args.probes_dir)
    reg_dir = Path(args.registry_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agent-train registry: {teacher_id: {dataset: {val_delta_f1, ...}}}
    registry = load_json(reg_dir / "agent_train_registry.json")

    rows = []  # (dataset, teacher_id, pair_feature, val_utility)
    teacher_ids = sorted(registry.keys())
    for teacher in teacher_ids:
        entries = registry[teacher]
        for ds in DATASETS:
            e = entries.get(ds)
            if not e:
                continue
            sig = load_json(sig_dir / f"{ds}.json").get("vector", [])
            probe = load_json(probe_dir / f"{ds}_{teacher}.json")
            p_sep = probe.get("separability", {})
            p_cal = probe.get("calibration", {})
            probe_vec = [p_sep.get("change_mean", 0.0), p_sep.get("unchanged_mean", 0.0),
                         p_sep.get("ratio", 0.0), p_cal.get("conf_change", 0.0), p_cal.get("conf_unchanged", 0.0)]
            pair = list(sig) + probe_vec
            utility = float(e.get("val_delta_f1", 0.0)) + 0.5 * float(e.get("val_delta_iou", 0.0))
            rows.append((ds, teacher, pair, utility))

    if not rows:
        raise SystemExit("No agent-train registry entries found")

    # LODO: leave one dataset out, train, then predict selection on held-out dataset.
    report = {}
    for holdout in DATASETS:
        tr = [r for r in rows if r[0] != holdout]
        te = [r for r in rows if r[0] == holdout]
        if not tr or not te:
            report[holdout] = "skipped (no data)"
            continue
        X = np.stack([r[2] for r in tr]).astype(np.float32)
        y = np.stack([r[3] for r in tr]).astype(np.float32)
        policy, loss = train_policy(X, y, epochs=2000, lr=1e-3, weight_decay=1e-4)
        import torch
        Xt = torch.tensor(np.stack([r[2] for r in te]), dtype=torch.float32)
        scores = policy(Xt).detach().numpy()
        best = int(scores.argmax())
        chosen = te[best][1] if scores[best] > 0 else "None"
        report[holdout] = {"chosen": chosen, "score": float(scores[best]), "train_loss": loss}

    out_dir.joinpath("lodo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[agent] LODO report:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
