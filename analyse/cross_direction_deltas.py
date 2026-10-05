#!/usr/bin/env python3
"""Cross-direction delta table: every mechanism arm vs its clean anchor.

Purpose: check whether the "one dataset up / the other down" pattern seen in
SAGE-CD Run2 is specific to Run2 or has been recurring across all archived
directions (RDT-CD / SCTC / FA-SCRD / SAGE-CD).

Reads docs/experiment_metrics.xlsx ("Experiment Results" sheet, percentage
points) and prints SYSU / WHU F1 deltas with the sign pattern.

Usage: python -B analyse/cross_direction_deltas.py
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "docs" / "experiment_metrics.xlsx"

# (run, high arm, low arm, variable label)
PAIRS = (
    ("RDT-CD-Run3", "R3", "B0", "full BT-SAM-RDT"),
    ("RDT-CD-Run3", "R3A", "B0", "SAM only"),
    ("RDT-CD-Run3", "R3O", "B0", "OV only"),
    ("SCTC-Run1", "S1", "S0", "full-pixel symmetric calibration"),
    ("SCTC-Run1", "S2", "S0", "SCTC (main method)"),
    ("FA-SCRD-Run1", "A1", "C1", "fixed teacher relation KD"),
    ("FA-SCRD-Run1", "A1", "C0", "fixed teacher vs clean"),
    ("FA-SCRD-Run1", "M1", "C1", "FA-SCRD (main method)"),
    ("SAGE-CD-Run1", "G1", "G0", "joint temporal BN"),
    ("SAGE-CD-Run1", "G2", "G0", "Safe-DINO"),
    ("SAGE-CD-Run1", "G2", "G1", "Safe-DINO gate"),
    ("SAGE-CD-Run2", "R1", "R0", "joint temporal BN"),
    ("SAGE-CD-Run2", "R2", "R0", "SCF + identity gate"),
    ("SAGE-CD-Run2", "R3", "R0", "dual teacher"),
    ("SAGE-CD-Run2", "R4", "R0", "full method"),
    ("SAGE-CD-Run2", "R4", "R2", "MAIN gate: full vs student reform"),
    ("SAGE-CD-Run2", "R4", "R3", "failure-type routing"),
)


def pattern(sysu: float, whu: float) -> str:
    if sysu < 0 < whu:
        return "SYSU-/WHU+"
    if sysu > 0 > whu:
        return "SYSU+/WHU-"
    if sysu > 0 and whu > 0:
        return "both+"
    if sysu < 0 and whu < 0:
        return "both-"
    return "mixed"


def main() -> None:
    sheet = openpyxl.load_workbook(XLSX, data_only=True)["Experiment Results"]
    rows = list(sheet.iter_rows(values_only=True))
    header = [str(c) for c in rows[0]]
    run_i, exp_i = header.index("Run"), header.index("Experiment ID")
    ds_i, f1_i = header.index("Dataset"), header.index("F1")
    table = {(r[run_i], r[exp_i], r[ds_i]): float(r[f1_i]) for r in rows[1:]}

    print(f"{'Run':16}{'comparison':44}{'dSYSU':>8}{'dWHU':>8}   pattern")
    for run, high, low, label in PAIRS:
        sysu = table.get((run, high, "SYSU-CD-256"))
        sysu_lo = table.get((run, low, "SYSU-CD-256"))
        whu = table.get((run, high, "WHU-CD-256"))
        whu_lo = table.get((run, low, "WHU-CD-256"))
        d_sysu = f"{sysu - sysu_lo:+.2f}" if None not in (sysu, sysu_lo) else "n/a"
        d_whu = f"{whu - whu_lo:+.2f}" if None not in (whu, whu_lo) else "n/a"
        if d_sysu == "n/a" or d_whu == "n/a":
            note = "SYSU only (WHU not run)"
        else:
            note = pattern(float(d_sysu), float(d_whu))
        print(f"{run:16}{f'{label} ({high}-{low})':44}{d_sysu:>8}{d_whu:>8}   {note}")


if __name__ == "__main__":
    main()
