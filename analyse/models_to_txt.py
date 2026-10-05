#!/usr/bin/env python3
"""
Export models/ code + experiment metrics into text files.

Usage:
    python analyse/models_to_txt.py                    # Upload version: Code + Excel metrics → Upload_Chatgpt/
    python analyse/models_to_txt.py --run3             # Current code + unified metrics → temporary/Run3 snapshot
    python analyse/models_to_txt.py --run4             # Current code + unified metrics → temporary/Run4 snapshot
    python analyse/models_to_txt.py --dartr            # Current code + unified metrics → temporary/DART-R Run1 snapshot
    python analyse/models_to_txt.py --rdtcd            # Current code + unified metrics → temporary/RDT-CD Run2 snapshot
    python analyse/models_to_txt.py --rdtcd3           # Current code + unified metrics → temporary/RDT-CD Run3 snapshot
    python analyse/models_to_txt.py --archive          # Run2 archive: Code + raw train logs → temporary/
    python analyse/models_to_txt.py --code-only        # Code only (upload dir)
    python analyse/models_to_txt.py --output name.txt  # Custom output name
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
METRICS_XLSX = PROJECT_ROOT / 'docs' / 'experiment_metrics.xlsx'
SAM_HSD_RUN2 = PROJECT_ROOT / 'saved_models' / 'SAM-HSD' / 'Run2'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'Upload_Chatgpt'
ARCHIVE_DIR = PROJECT_ROOT / 'docs' / 'temporary'
RUN3_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_SAM-HSD_Run3.txt'
RUN4_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_SAM-HSD_Run4.txt'
DARTR_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_DART-R_Run1.txt'
RDTCD_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_RDT-CD_Run2.txt'
RDTCD3_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_RDT-CD_Run3.txt'
SCTC1_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_SCTC_Run1.txt'
FASCRD1_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_FA-SCRD_Run1.txt'
SAGE1_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_SAGE-CD_Run1.txt'
SAGE2_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_SAGE-CD_Run2.txt'
CATA_OUTPUT = ARCHIVE_DIR / 'models_and_metrics_CATA-CD_Run1.txt'


def read_file(path: Path) -> str:
    """Try multiple encodings to read a file."""
    for enc in ('utf-8', 'gbk', 'latin-1', 'cp1252'):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_bytes().decode('utf-8', errors='replace')


def separator(title: str) -> str:
    """Build a separator banner."""
    return f'\n\n{"=" * 68}\n  {title}\n{"=" * 68}\n\n'


def extract_sheet_table(ws) -> str:
    """Extract a worksheet as lossless, machine-readable TSV."""
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return '(empty sheet)\n'
    headers = [str(value or '') for value in rows[0]]

    def format_value(value, header=''):
        if value is None:
            return ''
        if not isinstance(value, float):
            return str(value).replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
        if header in {'Recall', 'Precision', 'OA', 'F1', 'IoU', 'Kappa'}:
            return f'{value:.4f}'
        if header in {'Train Params (M)', 'Infer Params (M)', 'FLOPs (G)'}:
            return f'{value:.4f}'
        if 'Error' in header:
            return f'{value:.8e}'
        return f'{value:.8g}'

    lines = ['\t'.join(headers)]
    for row in rows[1:]:
        lines.append('\t'.join(
            format_value(value, headers[index] if index < len(headers) else '')
            for index, value in enumerate(row)
        ))
    return '\n'.join(lines) + '\n'


def append_models_code(parts: list[str]) -> int:
    """Append every Python source file below models/ in deterministic order."""
    if not MODELS_DIR.is_dir():
        parts.append('(models/ directory not found)\n')
        return 0
    py_files = sorted(MODELS_DIR.rglob('*.py'))
    for py_file in py_files:
        rel = py_file.relative_to(PROJECT_ROOT)
        parts.append(separator(f'{rel.parent.name}  |  {rel.as_posix()}'))
        parts.append(read_file(py_file))
        parts.append('\n\n')
    parts.append(f'\n(Total: {len(py_files)} Python files)\n\n')
    return len(py_files)


def append_excel_metrics(parts: list[str]) -> int:
    """Append every sheet from the unified metrics workbook."""
    parts.append(separator('UNIFIED EXPERIMENT METRICS'))
    if not METRICS_XLSX.is_file():
        parts.append('(docs/experiment_metrics.xlsx not found; run analyse/extract_metrics.py first)\n')
        return 0
    import openpyxl
    workbook = openpyxl.load_workbook(str(METRICS_XLSX), data_only=False, read_only=True)
    result_rows = 0
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        parts.append(f'\n--- Sheet: {sheet_name} ---\n')
        parts.append(extract_sheet_table(worksheet))
        if sheet_name == 'Experiment Results':
            result_rows = max(worksheet.max_row - 1, 0)
    workbook.close()
    return result_rows


def build_run3_version():
    """Build the requested Run3 research snapshot from current code and all metrics."""
    parts = [separator('SAM-HSD RUN3 CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --run3\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('Run3 note: current downloaded results are 12k Stage1 unless the workbook says otherwise.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    RUN3_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = RUN3_OUTPUT.stat().st_size / 1024
    print(
        f'Exported {code_count} model files + {result_count} metric rows '
        f'→ {RUN3_OUTPUT} ({size_kb:.1f} KB)'
    )
    return RUN3_OUTPUT


def build_run4_version():
    """Build the Run4 (CR-SRD) research snapshot from current code and all metrics."""
    parts = [separator('SAM-HSD RUN4 (CR-SRD) CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --run4\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('Run4 note: 200-epoch Stage1 mechanism screening; not a 40k final result.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    RUN4_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = RUN4_OUTPUT.stat().st_size / 1024
    print(
        f'Exported {code_count} model files + {result_count} metric rows '
        f'→ {RUN4_OUTPUT} ({size_kb:.1f} KB)'
    )
    return RUN4_OUTPUT


def build_dartr_version():
    """Build the DART-R (direction C) Run1 research snapshot from current code and all metrics."""
    parts = [separator('DART-R (DIRECTION C) RUN1 CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --dartr\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('DART-R note: B0=clean baseline, C0=DART-R (difficulty routing + reject). 40k, seed 2333 only; not a multi-seed paper result.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    DARTR_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = DARTR_OUTPUT.stat().st_size / 1024
    print(
        f'Exported {code_count} model files + {result_count} metric rows '
        f'→ {DARTR_OUTPUT} ({size_kb:.1f} KB)'
    )
    return DARTR_OUTPUT


def build_rdtcd_version():
    """Build the RDT-CD (direction C) Run2 research snapshot from current code and all metrics."""
    parts = [separator('RDT-CD + SCGR RUN2 CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --rdtcd\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('RDT-CD Run2 note: B0=clean baseline, D1NG=SCGR no grad-gate, D1=full SCGR, D2=no-cache. SCGR found ineffective vs B0. 40k, seed 2333 only; not a multi-seed paper result.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    RDTCD_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = RDTCD_OUTPUT.stat().st_size / 1024
    print(
        f'Exported {code_count} model files + {result_count} metric rows '
        f'→ {RDTCD_OUTPUT} ({size_kb:.1f} KB)'
    )
    return RDTCD_OUTPUT


def build_rdtcd3_version():
    """Build the RDT-CD Run3 (BT-SAM-RDT) snapshot from current code and all metrics."""
    parts = [separator('RDT-CD RUN3 (BT-SAM-RDT) CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --rdtcd3\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('RDT-CD Run3 note: B0=clean baseline, R3A=BT-SAM only, R3=BT-SAM+OV. 40k, seed 2333 only; not a multi-seed paper result.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    RDTCD3_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = RDTCD3_OUTPUT.stat().st_size / 1024
    print(
        f'Exported {code_count} model files + {result_count} metric rows '
        f'→ {RDTCD3_OUTPUT} ({size_kb:.1f} KB)'
    )
    return RDTCD3_OUTPUT


def build_sctc1_version():
    """Build the SCTC Run1 (unchanged-aware cross-temporal calibration) snapshot."""
    parts = [separator('SCTC RUN1 CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --sctc1\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('SCTC Run1 note: S0=clean A2Net, S1=full-pixel symmetric calibration, S2=unchanged-aware SCTC. S2-S0 = WHU +0.43 / SYSU -0.25 / CDD -0.20 / LEVIR -0.10 → WHU-only single-dataset trick, H4 failed. 40k, seed 2333 only; not a multi-seed paper result.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    SCTC1_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = SCTC1_OUTPUT.stat().st_size / 1024
    print(
        f'Exported {code_count} model files + {result_count} metric rows '
        f'→ {SCTC1_OUTPUT} ({size_kb:.1f} KB)'
    )
    return SCTC1_OUTPUT


def build_fascrd1_version():
    """Build the FA-SCRD Run1 snapshot from current code and all metrics."""
    parts = [separator('FA-SCRD RUN1 CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --fascrd1\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('FA-SCRD Run1 note: C0=clean, C1=joint temporal BN, A1=fixed DINOv3 teacher relation KD, M1=+failure-aware weighting. A1-C1 = SYSU -0.33 / CDD -0.33 -> teacher transfer negative. WHU/LEVIR stopped early (C0 only). 40k, seed 2333 only.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    FASCRD1_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = FASCRD1_OUTPUT.stat().st_size / 1024
    print(f'Exported {code_count} model files + {result_count} metric rows '
          f'-> {FASCRD1_OUTPUT} ({size_kb:.1f} KB)')
    return FASCRD1_OUTPUT


def build_sage1_version():
    """Build the SAGE-CD Run1 snapshot from current code and all metrics."""
    parts = [separator('SAGE-CD RUN1 CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --sage1\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('SAGE-CD Run1 note: G0=clean, G1=joint temporal BN, G2=full-res DINO semantic target + aux head + logit standardization + gradient budget. G2-G1 = WHU +0.46 / SYSU -0.25 -> gate FAIL (SYSU), WHU-only; but KD mechanism fixed (kd 12-15->0.6, grad budget ~25%). 40k, seed 2333 only.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    SAGE1_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = SAGE1_OUTPUT.stat().st_size / 1024
    print(f'Exported {code_count} model files + {result_count} metric rows '
          f'-> {SAGE1_OUTPUT} ({size_kb:.1f} KB)')
    return SAGE1_OUTPUT


def build_sage2_version():
    """Build the SAGE-CD Run2 snapshot from current code and all metrics."""
    parts = [separator('SAGE-CD RUN2 CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --sage2\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('SAGE-CD Run2 note: R0=clean A2Net, R1=+joint temporal BN, R2=+SCF + identity-centered decoder gate, R3=+DINO semantic (1/16) + SAM boundary (1/4) uniform, R4=+failure-type reliability routing (main method). R4-R2 = SYSU -0.44 / WHU +0.18 -> FAIL (needs >=+0.30 both); R4-R3 = SYSU +0.19 / WHU +0.02. All 10 runs 40000 steps, swap/deploy error 0, deploy 2.9131M / 2.7676G. 40k, seed 2333 only.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    SAGE2_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = SAGE2_OUTPUT.stat().st_size / 1024
    print(f'Exported {code_count} model files + {result_count} metric rows '
          f'-> {SAGE2_OUTPUT} ({size_kb:.1f} KB)')
    return SAGE2_OUTPUT


def build_cata_version():
    """Build the CATA-CD v2 snapshot from current code and all metrics."""
    parts = [separator('CATA-CD V2 CODE + UNIFIED METRICS SNAPSHOT')]
    parts.append('Generated by: analyse/models_to_txt.py --cata\n')
    parts.append('Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Metrics source: docs/experiment_metrics.xlsx\n')
    parts.append('CATA-CD v2 note: C0=clean A2Net, C1=C0+DCA (moe128), TV-*=C1+teacher package (Wave-A: sam2/dinov2/dinov3_lvd/dinov3_sat/remoteclip/mars; Wave-B: anysat/universat/radio). Teacher utility = TV-* - C1 per dataset; teacher-dataset fit decided by 40K experiment. Deploy params <5M (C0=2.9131M, C1=3.2803M). 40k, seed 2333 only.\n')
    parts.append(separator('MODELS CODE'))
    code_count = append_models_code(parts)
    result_count = append_excel_metrics(parts)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    CATA_OUTPUT.write_text(''.join(parts), encoding='utf-8')
    size_kb = CATA_OUTPUT.stat().st_size / 1024
    print(f'Exported {code_count} model files + {result_count} metric rows '
          f'-> {CATA_OUTPUT} ({size_kb:.1f} KB)')
    return CATA_OUTPUT


def extract_train_log_summary(log_path: Path) -> str:
    """Extract config header + final training + test results from train_log.txt."""
    try:
        text = read_file(log_path)
    except Exception as e:
        return f'(failed to read: {e})\n'

    lines = text.splitlines()
    parts = []

    # Config header (first ~30 lines before first --- separator)
    parts.append('--- Configuration ---\n')
    for i, line in enumerate(lines[:35]):
        if line.strip().startswith('---'):
            break
        parts.append(line + '\n')

    # Last 60 lines (final epochs + test results block)
    parts.append('\n--- Final Training + Test Results ---\n')
    for line in lines[-60:]:
        parts.append(line + '\n')

    return ''.join(parts)


def build_upload_version(code_only=False, output_name=None):
    """Build upload version: models/ code + Excel metrics summary."""
    output_file = OUTPUT_DIR / (output_name or 'models_and_metrics.txt')
    parts = []

    parts.append(separator('MODELS CODE + EXPERIMENT METRICS'))
    parts.append('Generated by: analyse/models_to_txt.py\n')
    parts.append(f'Project: LS-Rep_BCD_RSML_3\n\n')

    # ---- Part 1: All .py files in models/ ----
    parts.append('\n')
    if MODELS_DIR.is_dir():
        py_files = sorted(MODELS_DIR.rglob('*.py'))
        for py_file in py_files:
            rel = py_file.relative_to(PROJECT_ROOT)
            parts.append(separator(f'{rel.parent.name}  |  {rel.as_posix()}'))
            parts.append(read_file(py_file))
            parts.append('\n\n')
        parts.append(f'\n(Total: {len(py_files)} Python files)\n\n')
    else:
        parts.append('(models/ directory not found)\n')

    # ---- Part 2: Experiment metrics from Excel ----
    if not code_only:
        parts.append(separator('EXPERIMENT METRICS'))
        if METRICS_XLSX.is_file():
            import openpyxl
            wb = openpyxl.load_workbook(str(METRICS_XLSX))
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                parts.append(f'\n--- Sheet: {sheet_name} ---\n')
                parts.append(extract_sheet_table(ws))
            wb.close()
        else:
            parts.append('(experiment_metrics.xlsx not found — run analyse/extract_metrics.py first)\n')

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file.write_text(''.join(parts), encoding='utf-8')
    py_count = len(list(MODELS_DIR.rglob('*.py'))) if MODELS_DIR.is_dir() else 0
    size_kb = output_file.stat().st_size / 1024
    print(f'Exported {py_count} files + metrics → {output_file}  ({size_kb:.1f} KB)')
    return output_file


def build_archive_version():
    """Build archival version: models/ code + raw train logs from SAM-HSD Run2."""
    output_file = ARCHIVE_DIR / 'models_and_metrics_SAM-HSD_Run2.txt'
    parts = []

    parts.append(separator('SAM-HSD RUN2 ARCHIVAL SNAPSHOT'))
    parts.append('Generated by: analyse/models_to_txt.py --archive\n')
    parts.append(f'Project: LS-Rep_BCD_RSML_3\n')
    parts.append('Content: models/ code + saved_models/SAM-HSD/Run2/ train logs\n\n')

    # ---- Part 1: All .py files in models/ ----
    parts.append('\n')
    parts.append(separator('MODELS CODE'))
    parts.append('\n')
    if MODELS_DIR.is_dir():
        py_files = sorted(MODELS_DIR.rglob('*.py'))
        for py_file in py_files:
            rel = py_file.relative_to(PROJECT_ROOT)
            parts.append(separator(f'{rel.parent.name}  |  {rel.as_posix()}'))
            parts.append(read_file(py_file))
            parts.append('\n\n')
        parts.append(f'\n(Total: {len(py_files)} Python files)\n\n')
    else:
        parts.append('(models/ directory not found)\n')

    # ---- Part 2: Train logs from SAM-HSD Run2 ----
    parts.append('\n')
    parts.append(separator('SAM-HSD RUN2 TRAIN LOGS'))
    parts.append('\n')

    if SAM_HSD_RUN2.is_dir():
        log_files = sorted(SAM_HSD_RUN2.rglob('train_log.txt'))
        for log_file in log_files:
            rel = log_file.relative_to(SAM_HSD_RUN2)
            parts.append(separator(f'{rel.parent}  |  train_log.txt'))
            parts.append(extract_train_log_summary(log_file))
            parts.append('\n\n')
        parts.append(f'\n(Total: {len(log_files)} train logs)\n')
    else:
        parts.append(f'(SAM-HSD Run2 directory not found: {SAM_HSD_RUN2})\n')

    # Write
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    output_file.write_text(''.join(parts), encoding='utf-8')
    size_kb = output_file.stat().st_size / 1024
    print(f'Archived → {output_file}  ({size_kb:.1f} KB)')
    return output_file


def main():
    run3_mode = '--run3' in sys.argv
    run4_mode = '--run4' in sys.argv
    dartr_mode = '--dartr' in sys.argv
    rdtcd_mode = '--rdtcd' in sys.argv
    rdtcd3_mode = '--rdtcd3' in sys.argv
    sctc1_mode = '--sctc1' in sys.argv
    fascrd1_mode = '--fascrd1' in sys.argv
    sage1_mode = '--sage1' in sys.argv
    sage2_mode = '--sage2' in sys.argv
    cata_mode = '--cata' in sys.argv
    archive_mode = '--archive' in sys.argv
    code_only = '--code-only' in sys.argv
    output_name = None

    for i, arg in enumerate(sys.argv):
        if arg == '--output' and i + 1 < len(sys.argv):
            output_name = sys.argv[i + 1]
            break

    if run3_mode:
        build_run3_version()
    elif run4_mode:
        build_run4_version()
    elif dartr_mode:
        build_dartr_version()
    elif rdtcd_mode:
        build_rdtcd_version()
    elif rdtcd3_mode:
        build_rdtcd3_version()
    elif sctc1_mode:
        build_sctc1_version()
    elif fascrd1_mode:
        build_fascrd1_version()
    elif sage1_mode:
        build_sage1_version()
    elif sage2_mode:
        build_sage2_version()
    elif cata_mode:
        build_cata_version()
    elif archive_mode:
        build_archive_version()
    else:
        build_upload_version(code_only=code_only, output_name=output_name)


if __name__ == '__main__':
    main()
