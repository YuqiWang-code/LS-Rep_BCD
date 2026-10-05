#!/usr/bin/env python3
"""
Extract text from PDF references under ``docs/参考文献/`` and combine
into a single txt file.

By default it scans ALL PDFs recursively under ``docs/参考文献/`` and writes
``docs/参考文献/参考文献_all.txt``. Pass a subdirectory name to scan only that
subdirectory and write ``docs/参考文献/参考文献_<subdir>.txt`` instead.

Uses PyMuPDF (fitz) for best Chinese text extraction; falls back to
PyPDF2, then pdfplumber.

Usage:
    python analyse/pdf_references_to_txt.py                 # all references
    python analyse/pdf_references_to_txt.py 文献调研0920     # one subdirectory
    python analyse/pdf_references_to_txt.py 文献调研0920 --output 自定义名.txt
"""

import argparse
import os
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCES_DIR = PROJECT_ROOT / 'docs' / '参考文献'

# Try readers in preference order
_readers = []

try:
    import pymupdf as fitz  # PyMuPDF; `fitz` is the deprecated alias, import `pymupdf` directly
    _readers.append(('pymupdf', fitz))
except ImportError:
    pass

try:
    from PyPDF2 import PdfReader
    _readers.append(('pypdf2', PdfReader))
except ImportError:
    pass

try:
    import pdfplumber
    _readers.append(('pdfplumber', pdfplumber))
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text from PDF references under docs/参考文献/ into a single txt."
        )
    )
    parser.add_argument(
        "subdir",
        nargs="?",
        default=None,
        help=(
            "optional subdirectory of docs/参考文献 to scan "
            "(default: all PDFs recursively)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "optional output filename inside docs/参考文献 "
            "(default: 参考文献_all.txt, or 参考文献_<subdir>.txt)"
        ),
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace):
    """Return (scan_dir, output_file) for the given CLI arguments."""
    if args.subdir:
        scan_dir = REFERENCES_DIR / args.subdir
        default_name = f'参考文献_{args.subdir}.txt'
    else:
        scan_dir = REFERENCES_DIR
        default_name = '参考文献_all.txt'

    output_file = REFERENCES_DIR / (args.output or default_name)
    return scan_dir, output_file


def extract_pymupdf(path: Path) -> str:
    """Extract text using PyMuPDF (best for CJK)."""
    doc = fitz.open(str(path))
    pages: list[str] = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text and text.strip():
            pages.append(f'--- Page {i + 1} ---\n{text}')
    doc.close()
    return '\n'.join(pages) if pages else '⚠ No extractable text.\n'


def extract_pypdf2(path: Path) -> str:
    """Extract text using PyPDF2."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(f'--- Page {i + 1} ---\n{text}')
    return '\n'.join(pages) if pages else '⚠ No extractable text.\n'


def extract_pdfplumber(path: Path) -> str:
    """Extract text using pdfplumber."""
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f'--- Page {i + 1} ---\n{text}')
    return '\n'.join(pages) if pages else '⚠ No extractable text.\n'


def extract_text(path: Path) -> str:
    """Try each available reader; return the first successful result."""
    for name, _ in _readers:
        try:
            if name == 'pymupdf':
                text = extract_pymupdf(path)
            elif name == 'pypdf2':
                text = extract_pypdf2(path)
            elif name == 'pdfplumber':
                text = extract_pdfplumber(path)
            else:
                continue
            if text.strip():
                return text
        except Exception:
            continue
    return '⚠ All readers failed to extract text from this PDF.\n'


def separator(title: str) -> str:
    bar = '━' * 78
    return f'\n{bar}\n  {title}\n{bar}\n'


def main():
    args = parse_args()
    scan_dir, output_file = resolve_paths(args)

    if not _readers:
        print('ERROR: No PDF library available.  Install one of:')
        print('  pip install PyMuPDF   (recommended)')
        print('  pip install PyPDF2')
        print('  pip install pdfplumber')
        return

    reader_names = ', '.join(name for name, _ in _readers)
    print(f'PDF readers available: {reader_names}')
    print(f'Scan dir      : {scan_dir}')
    print(f'Output file   : {output_file}')

    if not scan_dir.is_dir():
        print(f'ERROR: {scan_dir} does not exist.')
        return

    # Collect PDFs recursively under the scan directory
    pdf_files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(scan_dir):
        for fname in sorted(filenames):
            if fname.lower().endswith('.pdf'):
                pdf_files.append(Path(dirpath) / fname)

    if not pdf_files:
        print(f'No PDF files found under {scan_dir}')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text('⚠ No PDF files found.\n', encoding='utf-8')
        return

    print(f'Found {len(pdf_files)} PDF(s)')

    parts: list[str] = []
    parts.append(separator(f'参考文献合集  —  {len(pdf_files)} papers'))
    parts.append(
        f'Source directory: {scan_dir}\n'
        f'Generated by     : analyse/pdf_references_to_txt.py\n'
        f'Reader           : {reader_names}\n\n'
    )

    for pdf_path in pdf_files:
        rel = pdf_path.relative_to(PROJECT_ROOT)
        print(f'  → {rel}')
        parts.append(separator(f'▌ {rel}'))
        parts.append(extract_text(pdf_path))

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(parts)
    output_file.write_text(content, encoding='utf-8')

    size_kb = output_file.stat().st_size / 1024
    print(f'\nDone.  {output_file}  ({size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
