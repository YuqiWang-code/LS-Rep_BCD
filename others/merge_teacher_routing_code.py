#!/usr/bin/env python3
"""Merge every retained Python source file in others/ into one text file.

Retained sources are the key-innovation .py files kept from three repos:
  - unic-main            (UNIC, ECCV 2024)
  - U-Know-DiffPAN-main  (U-Know-DiffPAN, CVPR 2025)
  - pytorch-change-models-main (Segment Any Change, NeurIPS 2024)

Usage:
    python -B others/merge_teacher_routing_code.py
    python -B others/merge_teacher_routing_code.py --output custom_name.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


SCRIPT = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT.parent
DEFAULT_SOURCE = SCRIPT_DIR
DEFAULT_OUTPUT_NAME = "teacher_routing_code.txt"
SEPARATOR = "=" * 88


def read_source(path: Path) -> str:
    """Read source code while tolerating common encodings in downloaded projects."""
    for encoding in ("utf-8-sig", "utf-8", "gbk", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def natural_key(path: Path) -> tuple[str, ...]:
    """Provide a deterministic, case-insensitive order across platforms."""
    return tuple(part.casefold() for part in path.parts)


def merge_python_files(source_dir: Path, output_file: Path) -> int:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    python_files = sorted(
        (p for p in source_dir.rglob("*.py") if p.resolve() != SCRIPT),
        key=lambda p: natural_key(p.relative_to(source_dir)),
    )
    if not python_files:
        raise FileNotFoundError(f"No Python files found below: {source_dir}")

    parts = [
        "TEACHER ROUTING — KEY INNOVATION PYTHON SOURCE SNAPSHOT\n",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}\n",
        f"Source directory: {source_dir}\n",
        f"Python file count: {len(python_files)}\n",
    ]

    for source_file in python_files:
        relative_path = source_file.relative_to(source_dir).as_posix()
        code = read_source(source_file)
        parts.extend(
            (
                f"\n\n{SEPARATOR}\n",
                f"FILE: {relative_path}\n",
                f"{SEPARATOR}\n\n",
                code,
            )
        )
        if not code.endswith("\n"):
            parts.append("\n")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary_file.write_text("".join(parts), encoding="utf-8", newline="\n")
    temporary_file.replace(output_file)
    return len(python_files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_NAME,
        help="Output filename or path (default: others/teacher_routing_code.txt)",
    )
    args = parser.parse_args()

    output_arg = Path(args.output)
    output_file = output_arg if output_arg.is_absolute() else SCRIPT_DIR / output_arg
    count = merge_python_files(DEFAULT_SOURCE, output_file)
    print(f"Merged {count} Python files into: {output_file}")


if __name__ == "__main__":
    main()
