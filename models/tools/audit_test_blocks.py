#!/usr/bin/env python3
"""Audit every CATA-CD ``train_log.txt``: last complete TEST block + provenance SHA.

Step 1 of the 2026-10-08 review plan
(``docs/temporary/CATA-CD_v2_负结果严格复盘_下一轮可证伪迭代方案_GitHub最新_20261008.md``
§11.1 item 1 / §11.2):

    for all 44+12+4 ``train_log.txt`` produce the LAST COMPLETE TEST block summary
    together with the source SHA, without overwriting any existing file.

Strictness this tool enforces (each maps to a review finding):

* **P0-C** -- the block actually used is the *last complete* one. A block counts as
  complete only if it carries every required metric **and** a ``Total time``. An
  unterminated trailing block is reported as ``trailing_incomplete_block`` and is
  never used as the result.
* **§8.3 item 7** -- the block must also carry params / FLOPs / swap error /
  deploy error / declared steps, so we can tell a real 40K terminal block from a
  truncated or resumed one.
* **§11.2** -- ``Total time`` is parsed so GPU-hour budgets can be computed from
  evidence instead of guessed.

Read-only guarantee: this tool never writes, renames or truncates a log; it only
reads and emits new audit artifacts into a fresh directory.
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

TEST_BEGIN_RE = re.compile(r"^=== TEST RESULTS ===[ \t]*$", re.M)
TEST_END_RE = re.compile(r"^=== END TEST RESULTS ===[ \t]*$", re.M)
HEADER_END_RE = re.compile(r"^-{10,}\s*$", re.M)

# Full epoch row: the six reported metrics are all required so that a partially
# written line can never be mistaken for a validation point.
EPOCH_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\][^\n]*?F1=([\d.]+) IoU=([\d.]+) Recall=([\d.]+) "
    r"Precision=([\d.]+) OA=([\d.]+) Kappa=([\d.]+)"
)
STEP_RE = re.compile(r"global_step:\s*(\d+)")
HMS_RE = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")

REQUIRED_TEST = ("F1", "IoU", "Recall", "Precision", "OA", "Kappa")
# Fields the plan (§8.3 item 7) requires inside a usable terminal TEST block.
REQUIRED_BLOCK_FIELDS = REQUIRED_TEST + (
    "Total time",
    "Train Params",
    "Infer Params",
    "FLOPs",
    "Max Steps",
    "Batch Size",
)


def parse_hms_to_seconds(value: str) -> float | None:
    m = HMS_RE.match(value.strip())
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def parse_header(text: str) -> dict[str, str]:
    """Config block: key/value lines before the first dashed separator."""
    m = HEADER_END_RE.search(text)
    head = text[: m.start()] if m else text[:8000]
    out: dict[str, str] = {}
    for line in head.splitlines():
        if line.startswith("=") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def find_test_blocks(text: str) -> list[tuple[int, int | None]]:
    """All TEST blocks as ``(body_start, body_end_or_None)``; None = unterminated."""
    begins = [m.end() for m in TEST_BEGIN_RE.finditer(text)]
    ends = [m.start() for m in TEST_END_RE.finditer(text)]
    blocks: list[tuple[int, int | None]] = []
    for b in begins:
        e = next((x for x in ends if x > b), None)
        blocks.append((b, e))
    return blocks


def parse_block_fields(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def is_complete(fields: dict[str, str]) -> tuple[bool, list[str]]:
    missing = [k for k in REQUIRED_BLOCK_FIELDS if k not in fields]
    for k in REQUIRED_TEST:
        if k in fields:
            try:
                float(fields[k])
            except ValueError:
                missing.append(f"{k}(non-numeric)")
    if "Total time" in fields and parse_hms_to_seconds(fields["Total time"]) is None:
        missing.append("Total time(unparseable)")
    return (not missing), missing


def best_val(text: str) -> dict | None:
    best = None
    for m in EPOCH_RE.finditer(text):
        f1 = float(m.group(3))
        if best is None or f1 > best["f1"]:
            best = {
                "epoch": int(m.group(1)),
                "max_epochs": int(m.group(2)),
                "f1": f1,
                "iou": float(m.group(4)),
                "recall": float(m.group(5)),
                "precision": float(m.group(6)),
                "oa": float(m.group(7)),
                "kappa": float(m.group(8)),
            }
    return best


def last_global_step(text: str) -> int | None:
    steps = [int(m.group(1)) for m in STEP_RE.finditer(text)]
    return steps[-1] if steps else None


def audit_one(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    header = parse_header(text)
    blocks = find_test_blocks(text)

    trailing_incomplete = bool(blocks) and blocks[-1][1] is None
    terminated = [b for b in blocks if b[1] is not None]

    test_fields, missing = None, []
    if terminated:
        start, end = terminated[-1]
        cand = parse_block_fields(text[start : end if end is not None else len(text)])
        complete, missing = is_complete(cand)
        test_fields = cand if complete else None
        if not complete:
            missing = missing
    elif trailing_incomplete:
        missing = ["no terminated TEST block at all"]

    n_terminated = len(terminated)
    total_time_s = (
        parse_hms_to_seconds(test_fields["Total time"]) if test_fields else None
    )
    bv = best_val(text)
    step = last_global_step(text)
    declared = header.get("max_steps")

    rec: dict = {
        "experiment_id": header.get("experiment_id"),
        "dataset": header.get("dataset_name"),
        "source_log": path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix(),
        "absolute_log": path.as_posix(),
        "log_sha256": hashlib.sha256(raw).hexdigest(),
        "log_bytes": len(raw),
        "implementation_version": header.get("implementation_version"),
        "seed": header.get("seed"),
        "batch_size": header.get("batch_size"),
        "max_steps_declared": int(declared) if declared and declared.isdigit() else declared,
        "num_workers": header.get("num_workers"),
        "gpu_id": header.get("gpu_id"),
        "use_teacher": header.get("use_teacher"),
        "teacher_package": header.get("teacher_package"),
        "dca_mode": header.get("dca_mode"),
        "pretrained_path": header.get("pretrained_path"),
        "data_fingerprint": header.get("data_fingerprint"),
        "n_test_blocks_total": len(blocks),
        "n_test_blocks_terminated": n_terminated,
        "trailing_incomplete_block": trailing_incomplete,
        "usable_last_block": test_fields is not None,
        "unusable_reason": None if test_fields is not None else "; ".join(missing),
        "last_global_step": step,
    }

    if test_fields:
        rec["test"] = {k: float(test_fields[k]) for k in REQUIRED_TEST}
        rec["test_pp"] = {k: float(test_fields[k]) * 100 for k in REQUIRED_TEST}
        rec["train_params"] = test_fields.get("Train Params")
        rec["infer_params"] = test_fields.get("Infer Params")
        rec["flops"] = test_fields.get("FLOPs")
        rec["temporal_swap_max_error"] = test_fields.get("Temporal swap max error")
        rec["deploy_max_error"] = test_fields.get("Deploy max error")
        rec["block_max_steps"] = test_fields.get("Max Steps")
        rec["block_batch_size"] = test_fields.get("Batch Size")
        rec["total_time_s"] = total_time_s
        rec["total_time_h"] = total_time_s / 3600 if total_time_s else None
        rec["total_time_raw"] = test_fields.get("Total time")
        rec["block_seed"] = test_fields.get("Seed")
    else:
        rec["test_pp"] = None

    if bv:
        rec["best_val_f1_pp"] = bv["f1"] * 100
        rec["best_val_iou_pp"] = bv["iou"] * 100
        rec["best_val_epoch"] = bv["epoch"]
        rec["best_val_max_epochs"] = bv["max_epochs"]
    else:
        rec["best_val_f1_pp"] = None

    # Consistency checks (report, never silently "fix").
    flags = []
    if rec["max_steps_declared"] != 40000:
        flags.append(f"max_steps_declared={rec['max_steps_declared']}!=40000")
    if rec["seed"] != "2333":
        flags.append(f"seed={rec['seed']}!=2333")
    if rec["batch_size"] != "64":
        flags.append(f"batch_size={rec['batch_size']}!=64")
    if test_fields and rec.get("block_max_steps") not in (None, str(rec["max_steps_declared"])):
        flags.append(
            f"block_max_steps={rec['block_max_steps']}!=declared={rec['max_steps_declared']}"
        )
    if test_fields and step is not None and rec["max_steps_declared"] == 40000 and step != 40000:
        flags.append(f"last_global_step={step}!=40000")
    if test_fields and rec.get("block_seed") not in (None, rec["seed"]):
        flags.append(f"block_seed={rec['block_seed']}!=header_seed={rec['seed']}")
    if test_fields and rec.get("temporal_swap_max_error") not in (None, "0.00000000e+00"):
        flags.append(f"temporal_swap_max_error={rec['temporal_swap_max_error']}")
    if test_fields and rec.get("deploy_max_error") not in (None, "0.00000000e+00"):
        flags.append(f"deploy_max_error={rec['deploy_max_error']}")
    rec["flags"] = flags

    return rec


def collect(root: Path) -> list[dict]:
    return [audit_one(p, root) for p in sorted(root.rglob("train_log.txt"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        default=[
            "outputs/CATA-CD/teacher_adaptation",
            "outputs/CATA-CD/noise",
            "outputs/CATA-CD/Run1",
        ],
        help="directories scanned recursively for train_log.txt",
    )
    ap.add_argument("--out_dir", default="outputs/CATA-CD/audit_v3")
    ap.add_argument("--json_name", default="test_block_audit.json")
    ap.add_argument("--markdown_name", default="test_block_audit.md")
    args = ap.parse_args()

    records: list[dict] = []
    for r in args.roots:
        root = PROJECT_ROOT / r
        if not root.is_dir():
            print(f"[audit] WARN missing root: {root}", file=sys.stderr)
            continue
        recs = collect(root)
        print(f"[audit] {r}: {len(recs)} train_log.txt")
        records.extend(recs)

    usable = [x for x in records if x["usable_last_block"]]
    unusable = [x for x in records if not x["usable_last_block"]]
    flagged = [x for x in records if x["flags"]]

    gpu_h_total = sum(x["total_time_h"] for x in usable if x.get("total_time_h"))
    by_exp: dict[str, float] = {}
    for x in usable:
        if x.get("total_time_h"):
            key = f"{x['experiment_id']}/{x['dataset']}"
            by_exp[key] = x["total_time_h"]

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "_meta": {
            "purpose": "step 1 of the 2026-10-08 review plan: freeze lab evidence",
            "test_block_rule": "last COMPLETE TEST RESULTS block (requires metrics + Total time + params/FLOPs/steps)",
            "read_only": True,
            "n_logs": len(records),
            "n_usable": len(usable),
            "n_unusable": len(unusable),
            "n_flagged": len(flagged),
            "total_gpu_hours_parsed": gpu_h_total,
        },
        "unusable": [
            {
                "source_log": x["source_log"],
                "experiment_id": x["experiment_id"],
                "dataset": x["dataset"],
                "reason": x["unusable_reason"],
                "trailing_incomplete_block": x["trailing_incomplete_block"],
                "n_test_blocks_total": x["n_test_blocks_total"],
                "log_sha256": x["log_sha256"],
            }
            for x in unusable
        ],
        "flagged": [
            {"source_log": x["source_log"], "flags": x["flags"]} for x in flagged
        ],
        "gpu_hours_by_run": by_exp,
        "records": records,
    }
    (out_dir / args.json_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# CATA-CD train_log TEST-block audit (v3, step 1)",
        "",
        f"- logs scanned: **{len(records)}** (usable **{len(usable)}**, unusable **{len(unusable)}**, flagged **{len(flagged)}**)",
        f"- parsed GPU hours total: **{gpu_h_total:.2f} h**",
        "- block rule: last COMPLETE block (metrics + Total time + params/FLOPs/steps)",
        "",
    ]
    if unusable:
        lines += ["## Unusable logs", "", "| log | exp | ds | reason |", "|---|---|---|---|"]
        lines += [
            f"| `{x['source_log']}` | {x['experiment_id']} | {x['dataset']} | {x['unusable_reason']} |"
            for x in unusable
        ]
        lines.append("")
    if flagged:
        lines += ["## Flagged logs (consistency)", "", "| log | flags |", "|---|---|"]
        lines += [
            f"| `{x['source_log']}` | {'; '.join(x['flags'])} |" for x in flagged
        ]
        lines.append("")

    lines += [
        "## TEST block results (pp)",
        "",
        "| experiment | dataset | F1 | IoU | Recall | Precision | OA | Kappa | infer params | FLOPs | time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for x in sorted(usable, key=lambda r: (str(r["experiment_id"]), str(r["dataset"]))):
        t = x["test_pp"]
        lines.append(
            f"| {x['experiment_id']} | {x['dataset']} | {t['F1']:.4f} | {t['IoU']:.4f} | "
            f"{t['Recall']:.4f} | {t['Precision']:.4f} | {t['OA']:.4f} | {t['Kappa']:.4f} | "
            f"{x.get('infer_params')} | {x.get('flops')} | {x.get('total_time_raw')} |"
        )
    (out_dir / args.markdown_name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[audit] usable={len(usable)} unusable={len(unusable)} flagged={len(flagged)}")
    print(f"[audit] GPU hours parsed: {gpu_h_total:.2f}")
    print(f"[audit] wrote -> {out_dir}")
    for x in unusable:
        print(f"  UNUSABLE {x['source_log']}: {x['unusable_reason']}")
    for x in flagged:
        print(f"  FLAG {x['source_log']}: {'; '.join(x['flags'])}")


if __name__ == "__main__":
    main()
