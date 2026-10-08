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

P0 fixes from the 2026-10-08 review (§2.3 P0-A, P0-E):

  * **P0-A (AG-CH held-out val leak, fixed).** AG-CH used to read the held-out
    dataset's own ``y`` to decide whether to accept its teacher
    (``util = mean(y[i] for i in te ...)``), i.e. it looked at the answer before
    abstaining. Its zero false-action rate was therefore not a fair LODO baseline.
    Now BOTH the selection and the accept/abstain gate are derived from training
    domains only, and the held-out ``y`` is used exclusively for post-hoc scoring.
    ``--verify_no_leak`` proves it by permuting the held-out labels and asserting
    the actions are bit-identical.
  * **P0-E (noise floor source, split).** ``--noise_mode transductive`` reproduces
    the old protocol (the target dataset's own repeated-C1 sigma is used as the
    prior; legitimate only when the target dataset already has repeated training
    runs). ``--noise_mode inductive`` (default) estimates the floor only from the
    three TRAINING domains, which is the honest setting for "a dataset we have
    never trained or validated on".

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
    p.add_argument("--noise_mode", default="inductive", choices=("inductive", "transductive"),
                   help="inductive: noise floor from the TRAINING domains only (P0-E); "
                        "transductive: legacy protocol that reads the target dataset's own "
                        "repeated-C1 val sigma")
    p.add_argument("--verify_no_leak", action="store_true",
                   help="permute each fold's held-out labels and assert the actions are "
                        "unchanged (P0-A leak audit); writes leak_audit.json")
    return p.parse_args()


# --------------------------------------------------------------------------- data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def val_noise_sigma(logs_root: Path | None, datasets: list[str] | None = None) -> dict:
    """Per-dataset VAL run-level sigma from C1 + R1/R2/R3 best-val F1 (test never read).

    P0-E: callers must pass the dataset subset they are allowed to observe.
    ``inductive`` mode passes the TRAINING domains only, so the held-out dataset's
    own repeated-C1 sigma never enters the decision.
    """
    out = {}
    if logs_root is None:
        return out
    for ds in (datasets or DATASETS):
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


def _probe_gain_on(rows, idxs) -> float:
    """Mean val gain of the ``probe_change_mean``-argmax teacher over a set of rows.

    Pure feature-side choice: it reads ``probe_change_mean`` (a train-only probe
    feature) and nothing else. Used to build AG-CH's gate from TRAINING domains.
    """
    if not idxs:
        return 0.0
    j = int(np.argmax([rows[i]["f"].get("probe_change_mean", 0.0) for i in idxs]))
    return float(rows[idxs[j]]["y"])


def run_lodo(rows, names, arm, args, logs_root: Path | None,
             noise_mode: str = "inductive", permute_holdout: bool = False,
             rng: np.random.Generator | None = None):
    X, y, groups, tids = matrix(rows, names)
    report = {}
    for hold in DATASETS:
        tr = [i for i, g in enumerate(groups) if g != hold]
        te = [i for i, g in enumerate(groups) if g == hold]
        if not tr or not te:
            continue
        Xtr, ytr, gtr = X[tr], y[tr], [groups[i] for i in tr]
        Xte, tids_te = X[te], [tids[i] for i in te]

        # ---- P0-A leak audit: the held-out LABELS must not reach any decision ----
        # ``y_all`` is what post-hoc scoring reads. Permuting ``y[te]`` destroys the
        # teacher->reward mapping inside the held-out domain while leaving the
        # training labels untouched, so any action change is a genuine leak.
        y_all = y.copy()
        if permute_holdout:
            rng = rng or np.random.default_rng(args.seed)
            y_all[te] = y[te][rng.permutation(len(te))]

        # ---- P0-E: where does the noise floor come from? ----
        if noise_mode == "transductive":
            u_pp = val_noise_sigma(logs_root, [hold]).get(hold, 0.0)
            u_src = f"transductive:{hold}"
        else:
            train_domains = sorted(set(gtr))
            sigma_tr = val_noise_sigma(logs_root, train_domains)
            # Conservative: the largest training-domain floor, never the held-out one.
            u_pp = max(sigma_tr.values()) if sigma_tr else 0.0
            u_src = "inductive:" + ",".join(train_domains)

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
            # ---- P0-A FIX ----
            # Selection: best train-only probe response (no labels read).
            chosen = tids_te[int(np.argmax(
                [rows[i]["f"].get("probe_change_mean", 0.0) for i in te]))]
            # Gate: calibrated ONLY on the training domains -- for each training
            # domain, what did the same probe rule actually earn there?
            per_domain = {g: _probe_gain_on(rows, [i for i in tr if groups[i] == g])
                          for g in sorted(set(gtr))}
            gate = float(np.mean(list(per_domain.values()))) if per_domain else 0.0
            teacher, pred, reason = (
                (chosen, gate, "probe_heuristic_train_gated") if gate > args.delta_pp
                else (None, 0.0, "heuristic_below_delta_train"))
            detail = {"train_probe_argmax_gain_pp": per_domain, "gate_pp": gate}
        else:
            kappa = {"AG-01": 0.0, "AG-02": 0.0, "AG-03": args.kappa, "AG-04": args.kappa,
                     "AG-05": args.kappa}[arm]
            delta = 0.0 if arm == "AG-01" else args.delta_pp
            sel = (RidgeSelector(lam=args.lam) if arm == "AG-01"
                   else SafeSelector(lam=args.lam, delta_pp=delta, kappa=kappa))
            sel.fit(Xtr, ytr, gtr, names)
            d = sel.decide(Xte, tids_te, u_noise_pp=u_pp)
            teacher, pred, reason = d.teacher, d.margin_pp, d.reason
            detail = {"per_teacher": d.per_teacher}

        chosen_util = 0.0
        if teacher is not None:
            vals = [y_all[i] for i in te if tids[i] == teacher]
            chosen_util = float(vals[0]) if vals else 0.0
        report[hold] = {
            "arm": arm, "teacher": teacher, "score_pp": pred, "reason": reason,
            "heldout_val_pp": chosen_util, "u_noise_pp": u_pp, "u_noise_source": u_src,
            "detail": detail,
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
    logs_root = Path(args.logs_root) if args.logs_root else None
    # Reporting copy only. The decision path recomputes the floor per fold under
    # ``args.noise_mode`` and, in inductive mode, never opens the held-out log (P0-E).
    u_noise = val_noise_sigma(logs_root) if logs_root else {}

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
            rep = run_lodo(rows, names, arm, args, logs_root, args.noise_mode)
            for ds, r in rep.items():
                r["_oracle_pp"] = oracle.get(ds)
            results[arm] = rep
            summaries[arm] = summarize(rep)
        ladder[rung] = {
            "n_features": len(names),
            "summaries": summaries,
            "actions": {ds: results["AG-03"][ds]["teacher"] for ds in DATASETS},
            # Audit trail: which noise floor the decision SAW, and why it abstained.
            "noise_floor": {ds: {"u_noise_pp": results["AG-03"][ds]["u_noise_pp"],
                                 "source": results["AG-03"][ds]["u_noise_source"]}
                            for ds in DATASETS},
            "decisions": {ds: {"teacher": results["AG-03"][ds]["teacher"],
                               "reason": results["AG-03"][ds]["reason"],
                               "score_pp": results["AG-03"][ds]["score_pp"],
                               "heldout_val_pp": results["AG-03"][ds]["heldout_val_pp"]}
                          for ds in DATASETS},
        }
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

    # ---- P0-A leak audit: permute each fold's held-out labels; actions must not move ----
    leak_audit = {"ran": bool(args.verify_no_leak), "noise_mode": args.noise_mode}
    if args.verify_no_leak:
        rng = np.random.default_rng(args.seed + 1)
        per_rung, all_ok = {}, True
        for rung in rungs:
            nms = feature_names(rows, rung)
            rung_res = {}
            for arm in ARMS:
                base = run_lodo(rows, nms, arm, args, logs_root, args.noise_mode)
                perm = run_lodo(rows, nms, arm, args, logs_root, args.noise_mode,
                                permute_holdout=True, rng=np.random.default_rng(args.seed + 1))
                changed = {ds: {"base": base[ds]["teacher"], "permuted": perm[ds]["teacher"]}
                           for ds in base if base[ds]["teacher"] != perm[ds]["teacher"]}
                rung_res[arm] = {"ok": not changed, "changed": changed}
                all_ok = all_ok and not changed
            per_rung[rung] = rung_res
        leak_audit.update({"all_arms_invariant": all_ok, "per_rung": per_rung})
        (out_dir / "leak_audit.json").write_text(
            json.dumps(leak_audit, indent=2), encoding="utf-8")
        verdict = "PASS" if all_ok else "FAIL"
        print(f"\n[leak-audit] held-out label permutation -> actions invariant: {verdict}")
        for rung, arms in per_rung.items():
            for arm, r in arms.items():
                if not r["ok"]:
                    print(f"  LEAK {rung}/{arm}: {r['changed']}")

    # frozen actions of the PRIMARY rung (AG-03)
    primary_rung = rungs[-1]
    primary = ladder[primary_rung]
    actions = {ds: {"teacher": primary["actions"][ds],
                    "reason": primary["decisions"][ds]["reason"],
                    "score_pp": primary["decisions"][ds]["score_pp"],
                    "u_noise_pp": primary["noise_floor"][ds]["u_noise_pp"],
                    "u_noise_source": primary["noise_floor"][ds]["source"]}
               for ds in DATASETS}
    (out_dir / "selected_actions.json").write_text(json.dumps({
        "arm": "AG-03", "feature_set": primary_rung, "delta_pp": args.delta_pp,
        "kappa": args.kappa, "seed": args.seed, "noise_mode": args.noise_mode,
        "registry_sha256": registry_sha[:16],
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
         "val_sigma_pp": u_noise, "noise_mode": args.noise_mode,
         "leak_audit": leak_audit,
         "registry_sha256": registry_sha[:16]}, indent=2), encoding="utf-8")

    print(f"\n[agent] frozen AG-03 actions ({primary_rung}) -> {out_dir/'selected_actions.json'}")
    print(f"[agent] ladder report -> {out_dir/'lodo_report.json'}")


if __name__ == "__main__":
    main()
