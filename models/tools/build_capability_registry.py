#!/usr/bin/env python3
"""Build the Teacher Capability Registry (Stage 3) from completed train_log.txt files.

P0 correctness (review 2026-10-08):
  - parse the LAST complete ``=== TEST RESULTS === ... === END TEST RESULTS ===``
    block (project discipline), not the first;
  - all deltas are emitted in **pp** (percentage points);
  - every record carries the source log path + sha256 for traceability;
  - research (test) and agent_train (val only) registries are strictly separated,
    and the agent_train file must never contain a ``test_*`` field.

Outputs:
  research_report.json       -- formal TEST deltas (paper evidence only)
  agent_train_registry.json  -- VAL deltas only (Agent input; no test leakage)

Usage:
    python -m models.tools.build_capability_registry \
        --logs_root outputs/CATA-CD/teacher_adaptation \
        --registry_dir outputs/CATA-CD/registry
"""

from __future__ import annotations

import argparse
import hashlib
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

# A complete block only; finditer + last keeps the project "final block" rule.
TEST_RE = re.compile(r"=== TEST RESULTS ===[ \t]*\r?\n(.*?)\r?\n?=== END TEST RESULTS ===", re.S)
# Full validation row: all six reported metrics are required, so a half-written
# line can never be mistaken for a validation point.
EPOCH_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\][^\n]*?F1=([\d.]+) IoU=([\d.]+) Recall=([\d.]+) "
    r"Precision=([\d.]+) OA=([\d.]+) Kappa=([\d.]+)"
)
BEST_VAL_METRICS = ("f1", "iou", "recall", "precision", "oa", "kappa")
REQUIRED_TEST = ("F1", "IoU", "Recall", "Precision", "OA", "Kappa")

AGENT_FORBIDDEN_PREFIX = "test_"


def last_test_block(text: str) -> dict | None:
    """Return the LAST complete TEST block as a dict, or None."""
    blocks = list(TEST_RE.finditer(text))
    if not blocks:
        return None
    vals: dict[str, float] = {}
    for line in blocks[-1].group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        if k in REQUIRED_TEST:
            try:
                vals[k] = float(v.strip())
            except ValueError:
                pass
    return vals if len(vals) == len(REQUIRED_TEST) else None


def _best_val_row(text: str) -> tuple[dict | None, int | None]:
    """Best (max F1) validation row -> (metrics_dict, epoch).

    The returned metrics dict carries ONLY the six reported metrics, so callers may
    safely map it through ``to_pp`` (which multiplies every value by 100). The epoch
    index is returned separately for exactly that reason.
    """
    best, best_epoch = None, None
    for m in EPOCH_RE.finditer(text):
        f1 = float(m.group(3))
        if best is None or f1 > best["f1"]:
            best = dict(zip(BEST_VAL_METRICS, (float(m.group(i)) for i in range(3, 9))))
            best_epoch = int(m.group(1))
    return best, best_epoch


def best_val(text: str) -> dict | None:
    """Best (max F1) validation row of a run - checkpoint-selected validation utility."""
    return _best_val_row(text)[0]


def best_val_epoch(text: str) -> int | None:
    return _best_val_row(text)[1]


def read(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "", ""
    data = path.read_bytes()
    return data.decode("utf-8", errors="replace"), hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", required=True)
    ap.add_argument("--registry_dir", required=True)
    ap.add_argument("--allow_overwrite", action="store_true",
                    help="required to write into a directory that already holds a registry "
                         "(v3 discipline: never silently overwrite the v2 anchors)")
    ap.add_argument("--audit_json", default=None,
                    help="optional outputs/CATA-CD/audit_v3/test_block_audit.json; when given, "
                         "the C0/C1 anchors are cross-checked against it")
    args = ap.parse_args()

    logs_root = Path(args.logs_root)
    reg_dir = Path(args.registry_dir)
    if (reg_dir / "research_report.json").is_file() and not args.allow_overwrite:
        raise SystemExit(
            f"[registry] refusing to overwrite the existing registry in {reg_dir}.\n"
            f"           Choose a fresh --registry_dir (e.g. .../registry_v3) or pass "
            f"--allow_overwrite explicitly.")
    reg_dir.mkdir(parents=True, exist_ok=True)

    def load(exp: str, ds: str):
        return read(logs_root / exp / ds / "train_log.txt")

    c1_val, c1_test, c0_test, c0_val = {}, {}, {}, {}
    for ds in DATASETS:
        t, _h = load("C1", ds)
        c1_val[ds] = best_val(t)
        c1_test[ds] = last_test_block(t)
        t0, _h0 = load("C0", ds)
        c0_test[ds] = last_test_block(t0)
        c0_val[ds] = best_val(t0)

    def to_pp(d):
        return {k: (None if v is None else {m: v[m] * 100 for m in v}) for k, v in d.items()}

    research = {
        "_meta": {
            "unit": "pp",
            "version": "v3",
            "test_block_rule": "last complete TEST RESULTS block",
            "val_rule": "best (max F1) epoch row = checkpoint-selected validation utility",
            # P0-B (review 2026-10-08): v2 emitted C0_test_pp as a raw 0-1 ratio while
            # C1_* were already scaled, so the C0 anchor disagreed with its own unit
            # label by 100x. Every anchor below now goes through the same to_pp().
            "p0_b_c0_unit_fix": "all _anchors.*_pp fields are pp (x100 of the log ratio)",
        },
        "_anchors": {
            "C0_test_pp": to_pp(c0_test),
            "C0_val_pp": to_pp(c0_val),
            "C1_test_pp": to_pp(c1_test),
            "C1_val_pp": to_pp(c1_val),
        },
    }
    agent: dict = {"_meta": {"unit": "pp", "reward_field": "val_delta_f1_pp",
                             "note": "no test_* field by construction"}}

    for teacher, exp in TEACHERS.items():
        research[teacher] = {}
        agent[teacher] = {}
        for ds in DATASETS:
            txt, sha = load(exp, ds)
            src = (logs_root / exp / ds / "train_log.txt").as_posix()
            tv, tt = best_val(txt), last_test_block(txt)
            cv, ct = c1_val[ds], c1_test[ds]

            # ---- agent-train (VAL only; never any test field) ----
            a = {"experiment_id": exp, "source_log": src, "log_sha256": sha,
                 "best_val_epoch": best_val_epoch(txt)}
            if tv is not None and cv is not None:
                a["val_f1_pp"] = tv["f1"] * 100
                a["val_iou_pp"] = tv["iou"] * 100
                a["val_delta_f1_pp"] = (tv["f1"] - cv["f1"]) * 100
                a["val_delta_iou_pp"] = (tv["iou"] - cv["iou"]) * 100
            agent[teacher][ds] = a

            # ---- research (test; paper evidence only) ----
            r = {"experiment_id": exp, "source_log": src, "log_sha256": sha}
            if tt is not None and ct is not None:
                r["test_f1_pp"] = tt["F1"] * 100
                r["test_iou_pp"] = tt["IoU"] * 100
                r["test_delta_f1_pp"] = (tt["F1"] - ct["F1"]) * 100
                r["test_delta_iou_pp"] = (tt["IoU"] - ct["IoU"]) * 100
            research[teacher][ds] = r

    # Hard guard: the agent-train registry must never carry a test_* key.
    def assert_no_test(node, path="root"):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.startswith(AGENT_FORBIDDEN_PREFIX):
                    raise AssertionError(f"agent_train_registry leak: {path}.{k}")
                assert_no_test(v, f"{path}.{k}")

    assert_no_test(agent)

    # ---- optional cross-check against the independent TEST-block audit (step 1) ----
    if args.audit_json:
        audit_path = Path(args.audit_json)
        if not audit_path.is_file():
            raise SystemExit(f"[registry] --audit_json not found: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        aud = {(r["experiment_id"], r["dataset"]): r for r in audit.get("records", [])}
        bad = []
        for exp, anchors in (("C0", c0_test), ("C1", c1_test)):
            for ds in DATASETS:
                r = aud.get((exp, ds))
                if not r or not r.get("test"):
                    bad.append(f"{exp}/{ds}: missing from audit")
                    continue
                if abs(r["test"]["F1"] - anchors[ds]["F1"]) > 1e-9:
                    bad.append(f"{exp}/{ds}: registry={anchors[ds]['F1']} audit={r['test']['F1']}")
        if bad:
            raise SystemExit("[registry] audit cross-check FAILED:\n  " + "\n  ".join(bad))
        print(f"[registry] audit cross-check OK vs {audit_path.name} "
              f"({len(aud)} audited logs)")

    (reg_dir / "research_report.json").write_text(json.dumps(research, indent=2), encoding="utf-8")
    (reg_dir / "agent_train_registry.json").write_text(json.dumps(agent, indent=2), encoding="utf-8")

    print(f"[registry] wrote research_report.json + agent_train_registry.json (pp) -> {reg_dir}")
    print("\n== dF1_val(pp), Agent reward ==")
    for ds in DATASETS:
        base = c1_val[ds]["f1"] * 100 if c1_val[ds] else float("nan")
        parts = [f"{ds}(C1={base:.2f})"]
        for teacher in TEACHERS:
            e = agent[teacher][ds]
            if "val_delta_f1_pp" in e:
                parts.append(f"{teacher}={e['val_delta_f1_pp']:+.2f}")
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()
