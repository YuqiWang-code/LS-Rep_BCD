#!/usr/bin/env python3
"""Train / evaluate the U-SafeAgent (Agent-only, offline, val-only) — review §9.1.

Arms (all reuse the same 44 teacher runs; no new student training):
  AG-00  legacy MLP (23-dim signature+probe, max>0 gate)        [historical control]
  AG-01  shared weighted ridge, delta=0                          [only change the fitter]
  AG-02  ridge + delta=+0.20pp gate                              [only change the None gate]
  AG-03  AG-02 + group-jackknife margin (mu - kappa*u)           [main proposal]
  AG-04  AG-03 + extended probes (P1-P5)                          [only if available]
  AG-05  AG-03 + teacher metadata                                 [single-variable control]
  AG-CF  best fixed teacher by mean TRAIN val, else None          [fair fixed baseline]
  AG-CH  train-only probe heuristic, else None                    [fair heuristic baseline]

Diagnostics per fold: selected action, held-out val reward, regret, harm / false-action /
coverage / positive-recall. Test values are NEVER read (hard guard on the registry).

Usage:
    python -m models.tools.train_safe_teacher_agent \
        --registry_dir outputs/CATA-CD/registry \
        --signatures_dir outputs/CATA-CD/signatures \
        --probes_dir outputs/CATA-CD/probes \
        --logs_root outputs/CATA-CD \
        --out_dir outputs/CATA-CD/agent_safe --seed 2333 --delta_pp 0.20 --kappa 1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.agent.offline_bandit import TeacherAgentPolicy  # noqa: E402
from models.agent.safe_selector import (  # noqa: E402
    DEFAULT_DELTA_PP,
    DEFAULT_KAPPA,
    RidgeSelector,
    SafeSelector,
)
from models.agent.teacher_metadata import META_FEATURE_NAMES, meta_features  # noqa: E402

DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]
HARM_PP = -0.20

# Compact, physically motivated feature subset (review §4.1 advises <=10-12 inputs).
# NOTE: the legacy ``separability_ratio`` has an unbounded dynamic range (division by a
# near-zero unchanged response; observed 0.29 - 870), so the P1/P2 AUROC-style probes
# are preferred here instead.
COMPACT_SIG = ["change_ratio", "empty_pair_ratio", "area_p50_g", "small_fraction",
               "boundary_density", "hard_pseudo_change_fraction", "illumination_unchanged"]
COMPACT_PROBE = ["probe_balanced_auroc", "probe_hard_negative_separation",
                 "probe_unchanged_leakage_ratio"]
COMPACT_META = ["meta_log2_stride", "meta_pyramid", "meta_dom_satellite"]

EPOCH_RE = re.compile(r"Epoch \[\d+/\d+\].*?F1=([\d.]+) IoU=([\d.]+)")
TEST_RE = re.compile(r"=== TEST RESULTS ===[ \t]*\r?\n(.*?)\r?\n?=== END TEST RESULTS ===", re.S)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="U-SafeAgent training / LODO evaluation")
    p.add_argument("--registry_dir", required=True)
    p.add_argument("--signatures_dir", required=True)
    p.add_argument("--probes_dir", required=True)
    p.add_argument("--logs_root", default=None, help="for the val run-level noise proxy")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=2333)
    p.add_argument("--delta_pp", type=float, default=DEFAULT_DELTA_PP)
    p.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--feature_set", default="sig_probe_ext_meta",
                   choices=("legacy", "sig_only", "sig_probe", "sig_probe_ext",
                            "sig_probe_ext_meta", "full", "ladder"))
    return p.parse_args()


# --------------------------------------------------------------------------- data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def val_noise_sigma(logs_root: Path | None) -> dict:
    """Per-dataset VAL run-level sigma from C1 + R1/R2/R3 best-val F1 (test never read)."""
    out = {}
    if logs_root is None:
        return out
    for ds in DATASETS:
        vals = []
        for rel in [f"teacher_adaptation/C1/{ds}/train_log.txt",
                    f"noise/R1/{ds}/train_log.txt",
                    f"noise/R2/{ds}/train_log.txt",
                    f"noise/R3/{ds}/train_log.txt"]:
            p = logs_root / rel
            if not p.is_file():
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
            f1s = [float(m.group(1)) for m in EPOCH_RE.finditer(txt)]
            if f1s:
                vals.append(max(f1s) * 100)
        if len(vals) >= 2:
            out[ds] = float(np.std(vals, ddof=1))
    return out


def probe_vector(probe: dict) -> dict:
    sep = probe.get("separability", {})
    cal = probe.get("calibration", {})
    change = sep.get("change_mean", None)
    unchange = sep.get("unchanged_mean", None)
    ratio = sep.get("ratio", None)
    v = {
        "probe_change_mean": change,
        "probe_unchanged_mean": unchange,
        "separability_ratio": ratio,
        "probe_conf_change": cal.get("conf_change", None),
        "probe_conf_unchanged": cal.get("conf_unchanged", None),
        "hard_negative_gap": (None if (change is None or unchange is None) else change - unchange),
    }
    # extended probes (P1-P5) live in the same JSON when present
    for k in ("balanced_auroc", "hard_negative_separation", "boundary_contrast",
              "size_conditioned_auroc_gap", "unchanged_leakage_ratio"):
        if k in probe:
            v[f"probe_{k}"] = probe[k]
    return v


def signature_vector(sig_dir: Path, ds: str) -> dict:
    data = load_json(sig_dir / f"{ds}.json")
    names = data.get("names", [])
    vec = data.get("vector", [])
    if len(names) != len(vec):
        raise ValueError(f"{ds}: signature names/vector length mismatch")
    return {n: float(v) for n, v in zip(names, vec)}


def build_rows(registry: dict, sig_dir: Path, probe_dir: Path, feature_set: str):
    rows, missing = [], []
    for teacher in sorted(k for k in registry if not k.startswith("_")):
        for ds in DATASETS:
            e = registry[teacher].get(ds) or {}
            if "val_delta_f1_pp" not in e:
                missing.append((ds, teacher, "no val_delta_f1_pp"))
                continue
            f = {}
            f.update(signature_vector(sig_dir, ds))
            f.update(probe_vector(load_json(probe_dir / f"{ds}_{teacher}.json")))
            f.update(meta_features(teacher))
            rows.append({"ds": ds, "teacher": teacher, "f": f, "y": float(e["val_delta_f1_pp"])})
    return rows, missing


def feature_names(rows, feature_set: str) -> list[str]:
    """Feature ladder for single-variable ablations (review §9.1).

    legacy           : v1 signature + legacy probes + no metadata  (AG-00 historical)
    sig_only         : compact v2 signature only
    sig_probe        : + legacy separability probes
    sig_probe_ext    : + P1-P5 extended probes      (AG-04 variable)
    sig_probe_ext_meta: + teacher metadata           (AG-05 variable)
    full             : every available feature
    """
    all_names = sorted(rows[0]["f"].keys())
    if feature_set == "full":
        return all_names
    if feature_set == "legacy":
        return [n for n in all_names if not n.startswith("meta_")]

    available = set(all_names)
    want = [n for n in COMPACT_SIG if n in available]
    if feature_set == "sig_only":
        return want
    want += [n for n in ("probe_change_mean", "hard_negative_gap") if n in available]
    if feature_set == "sig_probe":
        return want
    want += [n for n in COMPACT_PROBE if n in available]
    if feature_set == "sig_probe_ext":
        return want
    want += [n for n in COMPACT_META if n in available]
    if feature_set == "sig_probe_ext_meta":
        return want
    raise ValueError(f"unknown feature_set {feature_set}")


def matrix(rows, names):
    X, y, groups, tids = [], [], [], []
    for r in rows:
        v = []
        for n in names:
            val = r["f"].get(n, np.nan)
            if val is None:
                raise ValueError(f"missing feature {n} for {r['ds']}/{r['teacher']} "
                                 f"(missing probe must be explicit, not 0)")
            v.append(float(val))
        X.append(v)
        y.append(r["y"])
        groups.append(r["ds"])
        tids.append(r["teacher"])
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), groups, tids


# --------------------------------------------------------------------------- arms
def fit_legacy_mlp(X, y, groups, epochs=800, seed=2333):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ym, ys = y.mean(), y.std() + 1e-9
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32)
    policy = TeacherAgentPolicy(X.shape[1])
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(policy(Xs), yt)
        loss.backward()
        opt.step()

    def predict(Z):
        with torch.no_grad():
            return policy(torch.tensor((Z - mu) / sd, dtype=torch.float32)).numpy() * ys + ym
    return predict


def decide_simple(pred_pp, teacher_ids, delta_pp):
    k = int(np.argmax(pred_pp))
    if pred_pp[k] > delta_pp:
        return teacher_ids[k], float(pred_pp[k]), "accepted"
    return None, 0.0, "no_supported_positive_utility"


def run_lodo(rows, names, arm, args, u_noise: dict):
    X, y, groups, tids = matrix(rows, names)
    report = {}
    for hold in DATASETS:
        tr = [i for i, g in enumerate(groups) if g != hold]
        te = [i for i, g in enumerate(groups) if g == hold]
        if not tr or not te:
            continue
        Xtr, ytr, gtr = X[tr], y[tr], [groups[i] for i in tr]
        Xte, tids_te = X[te], [tids[i] for i in te]

        if arm == "AG-00":
            predict = fit_legacy_mlp(Xtr, ytr, gtr)
            teacher, pred, reason = decide_simple(predict(Xte), tids_te, 0.0)
            detail = {"pred_pp": [float(v) for v in predict(Xte)]}
        elif arm == "AG-CF":
            mean_tr = {t: float(np.mean([y[i] for i in tr if tids[i] == t])) for t in set(tids_te)}
            best = max(mean_tr, key=mean_tr.get)
            teacher, pred, reason = (best, mean_tr[best], "fixed_train_mean") \
                if mean_tr[best] > args.delta_pp else (None, 0.0, "fixed_below_delta")
            detail = {"train_mean_pp": mean_tr}
        elif arm == "AG-CH":
            idx = int(np.argmax([rows[i]["f"].get("probe_change_mean", 0.0) for i in te]))
            chosen = tids_te[idx]
            util = float(np.mean([y[i] for i in te if tids[i] == chosen]))
            teacher, pred, reason = (chosen, util, "probe_heuristic") if util > args.delta_pp \
                else (None, 0.0, "heuristic_below_delta")
            detail = {}
        else:
            kappa = {"AG-01": 0.0, "AG-02": 0.0, "AG-03": args.kappa, "AG-04": args.kappa,
                     "AG-05": args.kappa}[arm]
            delta = 0.0 if arm == "AG-01" else args.delta_pp
            sel = (RidgeSelector(lam=args.lam) if arm == "AG-01"
                   else SafeSelector(lam=args.lam, delta_pp=delta, kappa=kappa))
            sel.fit(Xtr, ytr, gtr, names)
            d = sel.decide(Xte, tids_te, u_noise_pp=u_noise.get(hold, 0.0))
            teacher, pred, reason = d.teacher, d.margin_pp, d.reason
            detail = {"per_teacher": d.per_teacher}

        chosen_util = 0.0
        if teacher is not None:
            vals = [y[i] for i in te if tids[i] == teacher]
            chosen_util = float(vals[0]) if vals else 0.0
        report[hold] = {
            "arm": arm, "teacher": teacher, "score_pp": pred, "reason": reason,
            "heldout_val_pp": chosen_util, "detail": detail,
        }
    return report


def summarize(report):
    n = len(report)
    if n == 0:
        return {}
    harms = sum(1 for r in report.values() if r["teacher"] is not None and r["heldout_val_pp"] < HARM_PP)
    false_actions = sum(1 for r in report.values() if r["teacher"] is not None and r["heldout_val_pp"] <= 0)
    coverage = sum(1 for r in report.values() if r["teacher"] is not None)
    regrets = []
    for ds, r in report.items():
        oracle = r.get("_oracle_pp")
        if oracle is None:
            continue
        regrets.append(max(0.0, oracle) - max(0.0, r["heldout_val_pp"]))
    pos_domains = [ds for ds, r in report.items() if (r.get("_oracle_pp") or 0) > 0]
    hit = sum(1 for ds in pos_domains
              if report[ds]["teacher"] is not None and report[ds]["heldout_val_pp"] > 0)
    return {
        "n_folds": n,
        "harm_rate": harms / n,
        "false_action_rate": false_actions / n,
        "coverage": coverage / n,
        "mean_regret_pp": float(np.mean(regrets)) if regrets else None,
        "median_regret_pp": float(np.median(regrets)) if regrets else None,
        "any_positive_recall": (hit / len(pos_domains)) if pos_domains else None,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    reg_dir = Path(args.registry_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_path = reg_dir / "agent_train_registry.json"
    # HARD GUARD: never load the research (test) report here.
    registry = load_json(registry_path)
    for k in registry:
        if isinstance(k, str) and k.startswith("test_"):
            raise PermissionError("agent_train_registry contains a test field")
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest() if registry_path.is_file() else "n/a"

    rows, missing = build_rows(registry, Path(args.signatures_dir), Path(args.probes_dir), args.feature_set)
    if missing:
        print(f"[warn] {len(missing)} rows skipped: {missing[:4]}")
    u_noise = val_noise_sigma(Path(args.logs_root)) if args.logs_root else {}

    # oracle per fold = best held-out VAL reward (evaluation only, never used for fitting)
    oracle = {}
    for ds in DATASETS:
        vals = [r["y"] for r in rows if r["ds"] == ds]
        if vals:
            oracle[ds] = max(vals)

    ARMS = ("AG-00", "AG-01", "AG-02", "AG-03", "AG-05", "AG-CF", "AG-CH")
    rungs = (["legacy", "sig_only", "sig_probe", "sig_probe_ext", "sig_probe_ext_meta"]
             if args.feature_set == "ladder" else [args.feature_set])

    ladder = {}
    names = None
    for rung in rungs:
        names = feature_names(rows, rung)
        results, summaries = {}, {}
        for arm in ARMS:
            rep = run_lodo(rows, names, arm, args, u_noise)
            for ds, r in rep.items():
                r["_oracle_pp"] = oracle.get(ds)
            results[arm] = rep
            summaries[arm] = summarize(rep)
        ladder[rung] = {"n_features": len(names), "summaries": summaries,
                        "actions": {ds: results["AG-03"][ds]["teacher"] for ds in DATASETS}}
        print(f"\n--- feature rung '{rung}' ({len(names)} features) ---")
        for ds in DATASETS:
            line = [f"{ds}(oracle={oracle.get(ds, float('nan')):+.2f})"]
            for arm in ARMS:
                t = results[arm][ds]["teacher"]
                line.append(f"{arm}={t if t else 'None'}({results[arm][ds]['heldout_val_pp']:+.2f})")
            print("  " + "  ".join(line))
        print("  summary: " + "  ".join(
            f"{a}:h{summaries[a]['harm_rate']:.2f}/f{summaries[a]['false_action_rate']:.2f}/c{summaries[a]['coverage']:.2f}"
            for a in ARMS))

    # frozen actions of the PRIMARY rung (AG-03)
    primary_rung = rungs[-1]
    primary = ladder[primary_rung]
    actions = {ds: {"teacher": primary["actions"][ds],
                    "reason": primary["summaries"] and "safe_set_or_none"} for ds in DATASETS}
    (out_dir / "selected_actions.json").write_text(json.dumps({
        "arm": "AG-03", "feature_set": primary_rung, "delta_pp": args.delta_pp,
        "kappa": args.kappa, "seed": args.seed, "registry_sha256": registry_sha[:16],
        "feature_names": names, "actions": actions,
    }, indent=2), encoding="utf-8")
    (out_dir / "feature_schema.json").write_text(json.dumps({
        "feature_names": names, "n_features": len(names), "feature_set": args.feature_set,
        "signature_dims": len(signature_vector(Path(args.signatures_dir), DATASETS[0])),
        "probe_keys": sorted(k for k in rows[0]["f"] if k.startswith(("probe_", "separability_", "hard_negative"))),
        "meta_keys": META_FEATURE_NAMES,
    }, indent=2), encoding="utf-8")
    (out_dir / "lodo_report.json").write_text(json.dumps(
        {"ladder": ladder, "oracle_val_pp": oracle,
         "val_sigma_pp": u_noise, "registry_sha256": registry_sha[:16]}, indent=2), encoding="utf-8")

    print(f"\n[agent] frozen AG-03 actions ({primary_rung}) -> {out_dir/'selected_actions.json'}")
    print(f"[agent] ladder report -> {out_dir/'lodo_report.json'}")


if __name__ == "__main__":
    main()
