#!/usr/bin/env python3
"""Train the offline contextual bandit Agent with Leave-One-Dataset-Out (Stage 4).

State x(D,T) = [dataset signature s_D, teacher train-only probe c(D,T)].
Reward = validation utility relative to C1 (never test). Action = one Teacher
Package or None (abstention when all predicted utilities <= 0).

Compared against baselines A-H (change-ratio heuristic), A-F (best global fixed
teacher) and A-REG (oracle-val, upper bound).

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
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.agent.offline_bandit import TeacherAgentPolicy  # noqa: E402

DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CATA-CD teacher-selection Agent (LODO)")
    p.add_argument("--signatures_dir", required=True)
    p.add_argument("--probes_dir", required=True)
    p.add_argument("--registry_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=800)
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def build_rows(sig_dir, probe_dir, registry):
    rows = []
    for teacher in sorted(registry.keys()):
        for ds in DATASETS:
            e = registry[teacher].get(ds) or {}
            if "val_delta_f1" not in e:
                continue
            sig = load_json(sig_dir / f"{ds}.json").get("vector", [])
            probe = load_json(probe_dir / f"{ds}_{teacher}.json")
            sep = probe.get("separability", {})
            cal = probe.get("calibration", {})
            pv = [sep.get("change_mean", 0.0), sep.get("unchanged_mean", 0.0), sep.get("ratio", 0.0),
                  cal.get("conf_change", 0.0), cal.get("conf_unchanged", 0.0)]
            utility = float(e["val_delta_f1"]) + 0.5 * float(e.get("val_delta_iou", 0.0))
            rows.append({"ds": ds, "teacher": teacher, "x": list(sig) + pv, "y": utility})
    return rows


def fit_mlp(X, y, epochs, seed=0):
    torch.manual_seed(seed)
    policy = TeacherAgentPolicy(X.shape[1])
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3, weight_decay=1e-2)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(policy(xt), yt)
        loss.backward()
        opt.step()
    return policy


def fit_ridge(X, y, lam=1.0):
    Xa = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xa.T @ y)
    return lambda Z: np.concatenate([Z, np.ones((len(Z), 1), dtype=np.float32)], axis=1) @ w


def main() -> None:
    args = parse_args()
    sig_dir, probe_dir = Path(args.signatures_dir), Path(args.probes_dir)
    reg_dir, out_dir = Path(args.registry_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_json(reg_dir / "agent_train_registry.json")
    rows = build_rows(sig_dir, probe_dir, registry)
    if not rows:
        raise SystemExit("no agent-train registry rows")

    X_all = np.stack([r["x"] for r in rows]).astype(np.float32)
    y_all = np.stack([r["y"] for r in rows]).astype(np.float32)

    report = {}
    for holdout in DATASETS:
        tr = [i for i, r in enumerate(rows) if r["ds"] != holdout]
        te = [i for i, r in enumerate(rows) if r["ds"] == holdout]
        if not tr or not te:
            report[holdout] = "skipped"
            continue

        Xtr, ytr = X_all[tr], y_all[tr]
        Xte, yte = X_all[te], y_all[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        ym, ys = ytr.mean(), ytr.std() + 1e-9
        Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
        ytr_n = (ytr - ym) / ys

        # --- A-LODO: MLP (scaled targets) ---
        policy = fit_mlp(Xtr_n, ytr_n, args.epochs)
        with torch.no_grad():
            score_mlp = policy(torch.tensor(Xte_n, dtype=torch.float32)).numpy() * ys + ym
        # --- A-LODO-linear: ridge ---
        ridge = fit_ridge(Xtr_n, ytr_n, lam=1.0)
        score_ridge = ridge(Xte_n) * ys + ym

        teachers_te = [rows[i]["teacher"] for i in te]
        util_te = [float(rows[i]["y"]) for i in te]

        def pick(scores):
            k = int(np.argmax(scores))
            return (teachers_te[k], float(scores[k])) if scores[k] > 0 else (None, 0.0)

        mlp_pick = pick(score_mlp)
        ridge_pick = pick(score_ridge)
        oracle_k = int(np.argmax(util_te))
        oracle = (teachers_te[oracle_k], util_te[oracle_k]) if util_te[oracle_k] > 0 else (None, 0.0)

        # A-F: best fixed teacher by mean val utility on the training datasets
        train_teachers = sorted({rows[i]["teacher"] for i in tr})
        mean_util = {t: float(np.mean([rows[i]["y"] for i in tr if rows[i]["teacher"] == t])) for t in train_teachers}
        fixed_t = max(mean_util, key=mean_util.get)
        fixed_util = float(np.mean([rows[i]["y"] for i in te if rows[i]["teacher"] == fixed_t]))

        # A-H: change-ratio heuristic -> highest change_ratio dataset picks the teacher with the
        # highest change_mean probe; simplest implementable rule.
        sig = load_json(sig_dir / f"{holdout}.json").get("details", {})
        change_ratio = sig.get("change_ratio", 0.0)
        h_idx = int(np.argmax([rows[i]["x"][len(rows[i]["x"]) - 5] for i in te]))  # probe change_mean
        heuristic = (teachers_te[h_idx], util_te[h_idx]) if change_ratio > 0 else (None, 0.0)

        def util_of(pick_tuple):
            t = pick_tuple[0]
            if t is None:
                return 0.0
            vals = [rows[i]["y"] for i in te if rows[i]["teacher"] == t]
            return float(vals[0]) if vals else 0.0

        report[holdout] = {
            "change_ratio": change_ratio,
            "oracle_val": {"teacher": oracle[0], "val_utility": oracle[1]},
            "A_LODO_mlp": {"teacher": mlp_pick[0], "pred": mlp_pick[1], "val_utility": util_of(mlp_pick)},
            "A_LODO_ridge": {"teacher": ridge_pick[0], "pred": ridge_pick[1], "val_utility": util_of(ridge_pick)},
            "A_F_fixed": {"teacher": fixed_t, "val_utility": fixed_util},
            "A_H_heuristic": {"teacher": heuristic[0], "val_utility": util_of(heuristic)},
        }

    (out_dir / "lodo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== LODO report (validation utility, percentage points) ===")
    for ds in DATASETS:
        r = report.get(ds)
        if not isinstance(r, dict):
            print(f"{ds}: {r}")
            continue
        print(f"\n{ds} (change_ratio={r['change_ratio']:.4f})")
        for key in ("oracle_val", "A_LODO_mlp", "A_LODO_ridge", "A_F_fixed", "A_H_heuristic"):
            v = r[key]
            print(f"  {key:14s} teacher={str(v['teacher']):10s} val_utility={v['val_utility']*100:+.3f}pp")
        oracle_t = r["oracle_val"]["teacher"]
        for key in ("A_LODO_mlp", "A_LODO_ridge", "A_F_fixed", "A_H_heuristic"):
            match = "MATCH" if r[key]["teacher"] == oracle_t else "miss"
            gap = (r["oracle_val"]["val_utility"] - r[key]["val_utility"]) * 100
            print(f"  -> {key:14s} {match}, gap_vs_oracle={gap:+.3f}pp")


if __name__ == "__main__":
    main()
