#!/usr/bin/env python3
"""Collect completed SAM-HSD test blocks into docs/experiment_metrics.xlsx.

Scans saved_models/SAM-HSD recursively plus outputs/SAM-HSD/Run3 recursively.
Only train_log.txt files and their last complete formal TEST RESULTS blocks are
accepted; launcher tee logs are excluded. Metrics are percentage points.

Usage: python -B analyse/extract_metrics.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import Counter
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "docs" / "experiment_metrics.xlsx"
SOURCES = (
    ("outputs/DirectionC", ROOT / "outputs" / "DirectionC", "DART-R"),
    ("outputs/DART-R-TS", ROOT / "outputs" / "DART-R-TS", "DART-R-TS"),
    ("outputs/RDT-CD/Run1", ROOT / "outputs" / "RDT-CD" / "Run1", "RDT-CD-Run1"),
    ("outputs/RDT-CD/Run2", ROOT / "outputs" / "RDT-CD" / "Run2", "RDT-CD-Run2"),
    ("outputs/RDT-CD/Run3", ROOT / "outputs" / "RDT-CD" / "Run3", "RDT-CD-Run3"),
    ("outputs/SCTC/Run1", ROOT / "outputs" / "SCTC" / "Run1", "SCTC-Run1"),
    ("outputs/FA-SCRD/Run1", ROOT / "outputs" / "FA-SCRD" / "Run1", "FA-SCRD-Run1"),
    ("outputs/SAGE-CD/Run1", ROOT / "outputs" / "SAGE-CD" / "Run1", "SAGE-CD-Run1"),
    ("outputs/SAGE-CD/Run2", ROOT / "outputs" / "SAGE-CD" / "Run2", "SAGE-CD-Run2"),
    ("outputs/CATA-CD", ROOT / "outputs" / "CATA-CD", "CATA-CD"),
)
METRICS = ("Recall", "Precision", "OA", "F1", "IoU", "Kappa")
REQUIRED = ("Dataset", "Experiment", "Train Params", "Infer Params", "FLOPs", *METRICS)
CONFIG_KEYS = {
    "experiment", "experiment_id", "experiment_name", "dataset_name", "auxiliary_mode",
    "batch_size", "max_steps", "seed", "lr", "lr_mode", "weight_decay",
    "backbone_lr_mult", "dice_reduction", "hsd_lambda", "hsd_max_ratio",
    "temporal_calibration_mode", "mechanism",
    "joint_bn", "scf", "gate", "use_teacher", "routing",
    "dca_mode", "teacher_package",
}
COLUMNS = (
    "Run", "Stage", "Experiment ID", "Experiment Name", "Dataset",
    "Configuration", "Batch Size", "Max Steps", "Seed", "Learning Rate",
    "LR Mode", "Weight Decay", "Backbone LR Mult", "Dice Reduction",
    "HSD Lambda", "HSD Max Ratio", "Auxiliary Mode",
    "Train Params (M)", "Infer Params (M)", "FLOPs (G)",
    *METRICS, "Auxiliary Toggle Error", "Deploy Error", "Total Time",
    "Source Group", "Source Log",
)
BLOCK_RE = re.compile(
    r"^=== TEST RESULTS ===[ \t]*\r?$\n(?P<body>.*?)"
    r"^=== END TEST RESULTS ===[ \t]*\r?$", re.MULTILINE | re.DOTALL,
)
KV_RE = re.compile(r"^([^:\r\n]+):[ \t]*(.*?)\r?$", re.MULTILINE)
DATASET_RE = re.compile(r"^(?:CDD|LEVIR|SYSU|WHU)-CD-256$")


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "gbk", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def pairs(text: str) -> dict[str, str]:
    return {m.group(1).strip(): m.group(2).strip() for m in KV_RE.finditer(text)}


def last_test(text: str) -> dict[str, str]:
    blocks = list(BLOCK_RE.finditer(text))
    return pairs(blocks[-1].group("body")) if blocks else {}


def config_values(text: str) -> dict[str, str]:
    result = {}
    for match in KV_RE.finditer(text):
        key = match.group(1).strip()
        if key in CONFIG_KEYS:
            result[key] = match.group(2).strip()
    return result


def numeric(value: str | None, unit: str = "") -> float | None:
    if value is None or value.lower() == "unavailable":
        return None
    value = value.strip()
    if unit and value.upper().endswith(unit.upper()):
        value = value[:-len(unit)].strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
    return float(match.group()) if match else None


def integer(value: str | None) -> int | None:
    value = numeric(value)
    return int(value) if value is not None else None


def natural(value: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", value))


def dataset_name(log: Path, alias: str | None) -> str:
    for part in reversed(log.parts):
        if DATASET_RE.fullmatch(part):
            return part
    mapping = {"CDD": "CDD-CD-256", "LEVIR": "LEVIR-CD-256",
               "SYSU": "SYSU-CD-256", "WHU": "WHU-CD-256"}
    return mapping.get((alias or "").upper(), alias or "Unknown")


def experiment_names(test: dict[str, str], config: dict[str, str]) -> tuple[str, str]:
    value = test.get("Experiment", "")
    if "/" in value:
        return tuple(part.strip() for part in value.split("/", 1))
    return (
        value or config.get("experiment", "Unknown"),
        config.get("experiment_name", value or "Unknown"),
    )


def config_summary(config: dict[str, str]) -> str:
    labels = (
        ("mode", "auxiliary_mode"), ("batch", "batch_size"), ("steps", "max_steps"),
        ("seed", "seed"), ("lr", "lr"), ("lr_mode", "lr_mode"),
        ("wd", "weight_decay"), ("backbone_lr", "backbone_lr_mult"),
        ("dice", "dice_reduction"), ("hsd_lambda", "hsd_lambda"),
        ("hsd_cap", "hsd_max_ratio"), ("temporal", "temporal_calibration_mode"),
        ("joint_bn", "joint_bn"), ("scf", "scf"), ("gate", "gate"),
        ("teacher", "use_teacher"), ("routing", "routing"),
        ("dca", "dca_mode"), ("teacher_pkg", "teacher_package"),
    )
    return "; ".join(f"{label}={config[key]}" for label, key in labels if key in config)


def parse_log(label: str, source_root: Path, fixed_run: str | None,
              log: Path) -> tuple[dict | None, str]:
    text = read_text(log)
    test = last_test(text)
    if not test:
        return None, "missing complete TEST RESULTS block"
    missing = [key for key in REQUIRED if key not in test]
    if missing:
        return None, "missing test fields: " + ", ".join(missing)
    parsed_numeric = {
        "Train Params": numeric(test.get("Train Params"), "M"),
        "Infer Params": numeric(test.get("Infer Params"), "M"),
        "FLOPs": numeric(test.get("FLOPs"), "G"),
        **{metric: numeric(test.get(metric)) for metric in METRICS},
    }
    invalid = [key for key, value in parsed_numeric.items() if value is None]
    if invalid:
        return None, "non-numeric test fields: " + ", ".join(invalid)
    config = config_values(text)
    relative_source = log.relative_to(source_root)
    run = fixed_run or (relative_source.parts[0] if relative_source.parts else "Unknown")
    exp_id, exp_name = experiment_names(test, config)
    steps = integer(config.get("max_steps"))
    row = {
        "Run": run,
        "Stage": "Stage1" if ((run == "Run3" and steps == 12000) or run == "Run4") else "Full",
        "Experiment ID": exp_id,
        "Experiment Name": exp_name,
        "Dataset": dataset_name(log, test.get("Dataset")),
        "Configuration": config_summary(config),
        "Batch Size": integer(config.get("batch_size")),
        "Max Steps": steps,
        "Seed": integer(config.get("seed")),
        "Learning Rate": numeric(config.get("lr")),
        "LR Mode": config.get("lr_mode"),
        "Weight Decay": numeric(config.get("weight_decay")),
        "Backbone LR Mult": numeric(config.get("backbone_lr_mult")),
        "Dice Reduction": config.get("dice_reduction"),
        "HSD Lambda": numeric(config.get("hsd_lambda")),
        "HSD Max Ratio": numeric(config.get("hsd_max_ratio")),
        "Auxiliary Mode": config.get("auxiliary_mode"),
        "Train Params (M)": parsed_numeric["Train Params"],
        "Infer Params (M)": parsed_numeric["Infer Params"],
        "FLOPs (G)": parsed_numeric["FLOPs"],
        "Auxiliary Toggle Error": numeric(test.get("Auxiliary toggle max error")),
        "Deploy Error": numeric(test.get("Deploy max error")),
        "Total Time": test.get("Total time"),
        "Source Group": label,
        "Source Log": log.relative_to(ROOT).as_posix(),
    }
    for metric in METRICS:
        row[metric] = parsed_numeric[metric] * 100.0
    return row, "complete"


def collect() -> tuple[list[dict], list[dict]]:
    results, audit = [], []
    for label, source_root, fixed_run in SOURCES:
        if not source_root.is_dir():
            audit.append({"Source": label, "Log": "", "Status": "missing root"})
            continue
        logs = sorted(source_root.rglob("train_log.txt"), key=lambda p: natural(p.as_posix()))
        for log in logs:
            row, status = parse_log(label, source_root, fixed_run, log)
            audit.append({
                "Source": label, "Log": log.relative_to(ROOT).as_posix(), "Status": status,
            })
            if row:
                results.append(row)
    results.sort(key=lambda row: (
        natural(row["Run"]), natural(row["Experiment ID"]), natural(row["Dataset"]),
        row["Max Steps"] or 0, row["Seed"] or 0,
    ))
    paths = [row["Source Log"] for row in results]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate source logs detected")
    return results, audit


NAVY = PatternFill("solid", fgColor="1F4E78")
RUN3 = PatternFill("solid", fgColor="E2F0D9")
WHITE = Font(name="Arial", size=10, bold=True, color="FFFFFF")
BODY = Font(name="Arial", size=10, color="000000")
TITLE = Font(name="Arial", size=15, bold=True, color="1F4E78")
BOTTOM = Border(bottom=Side(style="thin", color="D9E2F3"))


def style_results(ws, count: int) -> None:
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "F2"
    positions = {name: index + 1 for index, name in enumerate(COLUMNS)}
    last_col = get_column_letter(len(COLUMNS))
    for cell in ws[1]:
        cell.fill, cell.font = NAVY, WHITE
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 34
    for row_index in range(2, count + 2):
        current_run3 = ws.cell(row_index, positions["Run"]).value == "Run3"
        for cell in ws[row_index]:
            cell.font, cell.border = BODY, BOTTOM
            cell.alignment = Alignment(vertical="center")
            if current_run3:
                cell.fill = RUN3
        for field in ("Configuration", "Source Log"):
            ws.cell(row_index, positions[field]).alignment = Alignment(
                vertical="center", wrap_text=True,
            )
        ws.row_dimensions[row_index].height = 30
    formats = {
        "Train Params (M)": "0.0000", "Infer Params (M)": "0.0000",
        "FLOPs (G)": "0.0000", "Learning Rate": "0.000000",
        "Weight Decay": "0.000000", "HSD Lambda": "0.000000",
        "HSD Max Ratio": "0.000000", "Auxiliary Toggle Error": "0.00E+00",
        "Deploy Error": "0.00E+00", **{name: "0.0000" for name in METRICS},
    }
    for field, number_format in formats.items():
        column = get_column_letter(positions[field])
        for cell in ws[column][1:]:
            cell.number_format = number_format
    f1 = get_column_letter(positions["F1"])
    ws.conditional_formatting.add(
        f"{f1}2:{f1}{count + 1}",
        ColorScaleRule(
            start_type="min", start_color="F8696B",
            mid_type="percentile", mid_value=50, mid_color="FFEB84",
            end_type="max", end_color="63BE7B",
        ),
    )
    widths = {
        "Run": 9, "Stage": 10, "Experiment ID": 14, "Experiment Name": 32,
        "Dataset": 17, "Configuration": 64, "Batch Size": 11, "Max Steps": 12,
        "Seed": 10, "Learning Rate": 14, "LR Mode": 10, "Weight Decay": 13,
        "Backbone LR Mult": 17, "Dice Reduction": 14, "HSD Lambda": 12,
        "HSD Max Ratio": 14, "Auxiliary Mode": 18, "Train Params (M)": 17,
        "Infer Params (M)": 17, "FLOPs (G)": 12, "Recall": 11,
        "Precision": 11, "OA": 11, "F1": 11, "IoU": 11, "Kappa": 11,
        "Auxiliary Toggle Error": 20, "Deploy Error": 15, "Total Time": 18,
        "Source Group": 25, "Source Log": 70,
    }
    for index, field in enumerate(COLUMNS, 1):
        ws.column_dimensions[get_column_letter(index)].width = widths[field]
    table = Table(displayName="ExperimentResultsTable", ref=f"A1:{last_col}{count + 1}")
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
        showRowStripes=True, showColumnStripes=False,
    )
    ws.add_table(table)


def build_workbook(results: list[dict], audit: list[dict], output: Path) -> None:
    workbook = Workbook()
    readme = workbook.active
    readme.title = "README"
    readme.sheet_view.showGridLines = False
    readme["A1"] = "LS-Rep_BCD_RSML_3 — Experiment Metrics (DART-R / RDT-CD / SCTC / FA-SCRD / SAGE-CD)"
    readme["A1"].font = TITLE
    counts = Counter(row["Run"] for row in results)
    metadata = (
        ("Generated", dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")),
        ("Extraction rule", "Last complete formal TEST RESULTS block per train_log.txt"),
        ("Metric unit", "Percentage points; source log fractions multiplied by 100"),
        ("Sources", "outputs/DirectionC (DART-R) + outputs/DART-R-TS + outputs/RDT-CD + outputs/SCTC + outputs/SAGE-CD; launcher logs excluded"),
        ("Completed rows", len(results)),
        ("Rows by run", ", ".join(f"{run}={counts[run]}" for run in sorted(counts, key=natural))),
        ("DART-R scope", "DART-R=C0 gradient routing (failed); DART-R-TS=C1 fixed teacher (failed, B0 only); RDT-CD-Run1=D1 dynamic teacher (B0/D1/D2); RDT-CD-Run2=SCGR (B0/D1NG/D1/D2, ineffective). All 40k, seed 2333 — not a multi-seed paper result."),
        ("SCTC scope", "SCTC-Run1: S0=clean A2Net, S1=full-pixel symmetric calibration, S2=unchanged-aware SCTC. S2-S0 = WHU +0.43 / SYSU -0.25 / CDD -0.20 / LEVIR -0.10 → WHU-only single-dataset trick, H4 failed. 40k, seed 2333 only."),
        ("FA-SCRD scope", "FA-SCRD-Run1: C0=clean, C1=joint temporal BN, A1=fixed DINOv3 teacher relation KD, M1=+failure-aware weighting. A1-C1 = SYSU -0.33 / CDD -0.33 → teacher transfer negative (not a trick, a straight loss). WHU/LEVIR stopped early (C0 only). 40k, seed 2333 only."),
        ("SAGE-CD scope", "SAGE-CD-Run1 (Safe-DINO gate): G0=clean, G1=joint temporal BN, G2=full-res DINO semantic target + aux head + logit standardization + gradient budget. G2-G1 = WHU +0.46 / SYSU -0.25 → gate FAIL (SYSU), WHU-only again; but KD mechanism fixed (kd 12-15→0.6, grad budget ~25%). 40k, seed 2333 only."),
        ("SAGE-CD Run2 scope", "SAGE-CD-Run2 (failure-type routed native-scale dual-expert distillation + symmetric context fusion): R0=clean A2Net, R1=+joint temporal BN, R2=+SCF + identity-centered decoder gate, R3=+DINO semantic (1/16) + SAM boundary (1/4) uniform, R4=+failure-type reliability routing (main method). R4-R2 (main gate, needs >=+0.30 on both) = SYSU -0.44 / WHU +0.18 → FAIL. R4-R3 = SYSU +0.19 / WHU +0.02. All 10 runs completed 40000 steps, swap/deploy error 0, deploy 2.9131M / 2.7676G. 40k, seed 2333 only."),
        ("CATA-CD scope", "CATA-CD (Capability-Validated Adaptive Teacher Agent): C0=clean A2Net, C1=C0+DCA (moe128, deploy ~3.28M), TV-*=C1+one teacher package (sam2/dinov2/dinov3_lvd/dinov3_sat/remoteclip/mars Wave-A; anysat/universat/radio Wave-B). Teacher utility = TV-* - C1 per dataset. Deploy params <5M. 40k, seed 2333 only; teacher-dataset fit decided by experiment, not assumption."),
    )
    for row, (label, value) in enumerate(metadata, 3):
        readme.cell(row, 1, label).font = Font(
            name="Arial", size=10, bold=True, color="1F4E78",
        )
        readme.cell(row, 2, value).font = BODY
        readme.cell(row, 2).alignment = Alignment(wrap_text=True, vertical="top")
    readme.column_dimensions["A"].width = 22
    readme.column_dimensions["B"].width = 100

    sheet = workbook.create_sheet("Experiment Results")
    sheet.append(COLUMNS)
    for result in results:
        sheet.append([result.get(field) for field in COLUMNS])
    style_results(sheet, len(results))

    audit_sheet = workbook.create_sheet("Extraction Audit")
    audit_sheet.append(("Source", "Log", "Status"))
    for entry in audit:
        audit_sheet.append((entry["Source"], entry["Log"], entry["Status"]))
    audit_sheet.sheet_view.showGridLines = False
    audit_sheet.freeze_panes = "A2"
    for cell in audit_sheet[1]:
        cell.fill, cell.font = NAVY, WHITE
    for row in audit_sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font, cell.border = BODY, BOTTOM
    audit_sheet.column_dimensions["A"].width = 28
    audit_sheet.column_dimensions["B"].width = 100
    audit_sheet.column_dimensions["C"].width = 44
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)


def verify(output: Path, expected_rows: int) -> None:
    workbook = load_workbook(output, data_only=False)
    if workbook.sheetnames != ["README", "Experiment Results", "Extraction Audit"]:
        raise ValueError(f"Unexpected sheets: {workbook.sheetnames}")
    sheet = workbook["Experiment Results"]
    if sheet.max_row != expected_rows + 1 or sheet.max_column != len(COLUMNS):
        raise ValueError(f"Unexpected results shape: {sheet.max_row}x{sheet.max_column}")
    if tuple(cell.value for cell in sheet[1]) != COLUMNS:
        raise ValueError("Result headers do not match schema")
    excel_errors = {"#REF!", "#DIV/0!", "#VALUE!", "#N/A", "#NAME?"}
    errors = [
        f"{ws.title}!{cell.coordinate}"
        for ws in workbook.worksheets for row in ws.iter_rows() for cell in row
        if cell.value in excel_errors
    ]
    workbook.close()
    if errors:
        raise ValueError("Excel errors: " + ", ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    results, audit = collect()
    rejected = [entry for entry in audit if entry["Status"] != "complete"]
    if rejected:
        # In-progress runs are expected while a campaign is still training; skip
        # them with a warning instead of failing the whole extraction.
        print(f"[warn] {len(rejected)} incomplete/invalid logs skipped:")
        for entry in rejected[:15]:
            print(f"  - {entry['Log'] or entry['Source']}: {entry['Status']}")
    if not results:
        raise SystemExit("No complete SAM-HSD test results found")
    build_workbook(results, audit, output)
    verify(output, len(results))
    counts = Counter(row["Run"] for row in results)
    print(f"Saved {len(results)} completed results to: {output}")
    print("Rows by run: " + ", ".join(
        f"{run}={counts[run]}" for run in sorted(counts, key=natural)
    ))


if __name__ == "__main__":
    main()
