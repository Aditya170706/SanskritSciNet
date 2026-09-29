"""
Module 2 — Data Cleaning Pipeline
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Normalizes Unicode, removes artifacts, saves cleaned text to data/cleaned/
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DIR, CLEANED_DIR, LOGS_DIR
from scripts.logger import setup_logger, JSONLogger

logger = setup_logger("cleaning", LOGS_DIR)
audit = JSONLogger(LOGS_DIR / "cleaning_audit.jsonl")


# ---------------------------------------------------------------------------
# Cleaning patterns
# ---------------------------------------------------------------------------

# Page numbers: lines consisting only of digits (possibly surrounded by whitespace or dashes)
_RE_PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", re.MULTILINE)

# Common header/footer patterns (English and Devanagari numbers/markers)
_RE_HEADER_FOOTER = re.compile(
    r"^\s*(page|chapter|section|verse|hymn|sūkta|ṛgveda|shloka)\s*\d*\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Consecutive blank lines → single blank line
_RE_MULTI_BLANK = re.compile(r"\n{3,}")

# Trailing/leading whitespace per line
_RE_LINE_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)

# Soft hyphens and zero-width characters
_RE_INVISIBLE = re.compile(r"[\u00ad\u200b\u200c\u200d\ufeff\u2060]")

# Repetitive dashes (often used as decorative separators)
_RE_DASH_SEPARATOR = re.compile(r"[-–—]{3,}")

# Roman numeral lines (e.g. "I.", "IV.", "XLII") used as section markers
_RE_ROMAN_NUMERAL_LINE = re.compile(
    r"^\s*(?:M{0,4})(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\.?\s*$",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization."""
    return unicodedata.normalize("NFC", text)


def remove_invisible_chars(text: str) -> str:
    """Remove soft hyphens, zero-width chars, BOM, etc."""
    return _RE_INVISIBLE.sub("", text)


def remove_page_numbers(text: str) -> str:
    """Remove standalone numeric lines (page numbers)."""
    return _RE_PAGE_NUMBER.sub("", text)


def remove_headers_footers(text: str) -> str:
    """Remove common header/footer lines."""
    return _RE_HEADER_FOOTER.sub("", text)


def remove_dash_separators(text: str) -> str:
    """Remove decorative dash lines."""
    return _RE_DASH_SEPARATOR.sub("", text)


def remove_roman_numeral_lines(text: str) -> str:
    """Remove standalone roman numeral section markers."""
    return _RE_ROMAN_NUMERAL_LINE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """
    - Remove trailing spaces on each line
    - Collapse 3+ consecutive blank lines to 2
    - Strip overall leading/trailing whitespace
    """
    text = _RE_LINE_WHITESPACE.sub("", text)
    text = _RE_MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def ensure_utf8(text: str) -> str:
    """
    Re-encode via UTF-8 round-trip to drop any non-encodable characters.
    All Python strings are Unicode internally; this removes surrogates etc.
    """
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def clean_text(text: str) -> str:
    """
    Apply full cleaning pipeline to a raw text string.

    Steps:
    1. Remove invisible/zero-width characters
    2. NFC Unicode normalization
    3. UTF-8 safety
    4. Remove page numbers
    5. Remove headers/footers
    6. Remove dash separators
    7. Remove roman numeral lines
    8. Normalize whitespace
    """
    text = remove_invisible_chars(text)
    text = normalize_unicode(text)
    text = ensure_utf8(text)
    text = remove_page_numbers(text)
    text = remove_headers_footers(text)
    text = remove_dash_separators(text)
    text = remove_roman_numeral_lines(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# File-level cleaning
# ---------------------------------------------------------------------------

def clean_file(raw_path: str | Path, output_path: Optional[str | Path] = None) -> Path:
    """
    Clean a single raw text file and save the result.

    Args:
        raw_path: Path to the raw text file
        output_path: Destination path (defaults to data/cleaned/<same name>)

    Returns:
        Path to the cleaned file
    """
    raw_path = Path(raw_path)
    if output_path is None:
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = CLEANED_DIR / raw_path.name
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cleaning: {raw_path.name}")

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    original_chars = len(raw_text)

    cleaned = clean_text(raw_text)
    cleaned_chars = len(cleaned)

    output_path.write_text(cleaned, encoding="utf-8")

    reduction_pct = (1 - cleaned_chars / max(original_chars, 1)) * 100
    logger.info(
        f"  {raw_path.name}: {original_chars:,} → {cleaned_chars:,} chars "
        f"({reduction_pct:.1f}% reduction)"
    )
    audit.log(
        "file_cleaned",
        file=raw_path.name,
        original_chars=original_chars,
        cleaned_chars=cleaned_chars,
        reduction_pct=round(reduction_pct, 2),
    )
    return output_path


def clean_directory(
    raw_dir: str | Path = None,
    output_dir: str | Path = None,
    extensions: tuple = (".txt",),
) -> list[Path]:
    """
    Clean all text files in raw_dir and write results to output_dir.

    Returns:
        List of output file paths
    """
    raw_dir = Path(raw_dir or RAW_DIR)
    output_dir = Path(output_dir or CLEANED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in raw_dir.iterdir() if f.suffix.lower() in extensions]
    logger.info(f"Found {len(files)} files to clean in {raw_dir}")

    results = []
    for i, f in enumerate(files, 1):
        try:
            out = clean_file(f, output_dir / f.name)
            results.append(out)
        except Exception as exc:
            logger.error(f"[{i}/{len(files)}] Failed to clean {f.name}: {exc}")
            audit.log("clean_error", file=str(f), error=str(exc))

    logger.info(f"Cleaning complete. {len(results)}/{len(files)} files processed.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean Sanskrit text files")
    parser.add_argument("path", help="File or directory to clean")
    parser.add_argument("--output", default=None, help="Output file or directory")
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        clean_directory(p, args.output)
    else:
        clean_file(p, args.output)
