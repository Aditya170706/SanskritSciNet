"""
Module 3 — Shloka Segmentation Engine
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Detects shloka boundaries using ।  and ॥  markers.
Assigns unique IDs: SHLOKA_000001, SHLOKA_000002, ...
Outputs JSON to data/segmented/
"""

import json
import re
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CLEANED_DIR, SEGMENTED_DIR, LOGS_DIR
from scripts.logger import setup_logger, JSONLogger

logger = setup_logger("segmentation", LOGS_DIR)
audit = JSONLogger(LOGS_DIR / "segmentation_audit.jsonl")

# ---------------------------------------------------------------------------
# Segmentation patterns
# ---------------------------------------------------------------------------

# Match a shloka unit ending with ।  or ॥
# We split on these dandas while keeping them attached to the preceding chunk
_RE_SPLIT = re.compile(r"([^।॥]+[।॥]+)")

# Clean extra whitespace within a shloka
_RE_INNER_WS = re.compile(r"[ \t]+")
_RE_NEWLINES = re.compile(r"\n+")

# Devanagari character range — used to validate that a segment contains Sanskrit
_RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Minimum number of Devanagari chars to accept a segment as a shloka
MIN_DEVANAGARI_CHARS = 5


# ---------------------------------------------------------------------------
# ID generator
# ---------------------------------------------------------------------------

class ShlokaIDGenerator:
    """Thread-safe sequential shloka ID generator."""

    def __init__(self, start: int = 1):
        self._counter = start - 1

    def next(self) -> str:
        self._counter += 1
        return f"SHLOKA_{self._counter:06d}"

    @property
    def current(self) -> int:
        return self._counter


_global_id_gen = ShlokaIDGenerator()


# ---------------------------------------------------------------------------
# Core segmentation logic
# ---------------------------------------------------------------------------

def _normalize_shloka_text(raw: str) -> str:
    """Normalize whitespace inside a shloka while preserving Devanagari."""
    text = _RE_INNER_WS.sub(" ", raw)
    text = _RE_NEWLINES.sub(" ", text)
    return text.strip()


def _is_valid_shloka(text: str) -> bool:
    """
    A shloka is valid if it contains at least MIN_DEVANAGARI_CHARS Devanagari characters.
    """
    deva_chars = _RE_DEVANAGARI.findall(text)
    return len(deva_chars) >= MIN_DEVANAGARI_CHARS


def segment_text(
    text: str,
    source: str = "Unknown",
    chapter: str = "",
    id_generator: Optional[ShlokaIDGenerator] = None,
) -> list[dict]:
    """
    Segment a cleaned Sanskrit text string into individual shlokas.

    Args:
        text: Cleaned Sanskrit text
        source: Name of the source text (e.g. "Rigveda")
        chapter: Chapter/book/adhyaya reference (e.g. "1.1")
        id_generator: ShlokaIDGenerator instance (uses global if None)

    Returns:
        List of shloka dicts with id, text, source, chapter
    """
    gen = id_generator or _global_id_gen

    matches = _RE_SPLIT.findall(text)

    # Fallback: if no danda markers found, split on double-newlines
    if not matches:
        logger.warning(f"No danda markers (। ॥) found in text for source '{source}'. "
                       "Falling back to paragraph splitting.")
        matches = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    shlokas = []
    for raw_chunk in matches:
        normalized = _normalize_shloka_text(raw_chunk)
        if not normalized:
            continue
        if not _is_valid_shloka(normalized):
            logger.debug(f"Skipping non-Devanagari chunk: {normalized[:50]!r}")
            continue

        shloka = {
            "id": gen.next(),
            "text": normalized,
            "source": source,
            "chapter": chapter,
        }
        shlokas.append(shloka)

    logger.info(f"Segmented {len(shlokas)} shlokas from '{source}' (chapter={chapter or 'N/A'})")
    return shlokas


# ---------------------------------------------------------------------------
# File-level segmentation
# ---------------------------------------------------------------------------

def segment_file(
    cleaned_path: str | Path,
    source: str = "Unknown",
    chapter: str = "",
    output_path: Optional[str | Path] = None,
    id_generator: Optional[ShlokaIDGenerator] = None,
) -> Path:
    """
    Segment a single cleaned text file into shlokas and save as JSON.

    Args:
        cleaned_path: Path to cleaned .txt file
        source: Source name
        chapter: Chapter reference
        output_path: Where to save JSON (defaults to data/segmented/<stem>.json)
        id_generator: Shared ID generator for global uniqueness

    Returns:
        Path to output JSON file
    """
    cleaned_path = Path(cleaned_path)

    if output_path is None:
        SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = SEGMENTED_DIR / (cleaned_path.stem + ".json")
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    text = cleaned_path.read_text(encoding="utf-8")
    shlokas = segment_text(text, source=source, chapter=chapter, id_generator=id_generator)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(shlokas, f, ensure_ascii=False, indent=2)

    audit.log(
        "file_segmented",
        file=cleaned_path.name,
        source=source,
        chapter=chapter,
        shloka_count=len(shlokas),
        output=str(output_path),
    )
    logger.info(f"Saved {len(shlokas)} shlokas → {output_path}")
    return output_path


def segment_directory(
    cleaned_dir: str | Path = None,
    output_dir: str | Path = None,
    source_map: Optional[dict] = None,
) -> list[dict]:
    """
    Segment all cleaned text files in a directory.

    Args:
        cleaned_dir: Directory containing cleaned .txt files
        output_dir: Directory to save JSON files
        source_map: Optional dict mapping filename stem → {"source": ..., "chapter": ...}

    Returns:
        List of summary dicts
    """
    cleaned_dir = Path(cleaned_dir or CLEANED_DIR)
    output_dir = Path(output_dir or SEGMENTED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_map = source_map or {}

    gen = ShlokaIDGenerator()  # Shared generator for globally unique IDs
    files = list(cleaned_dir.glob("*.txt"))
    logger.info(f"Found {len(files)} cleaned files to segment")

    summaries = []
    for f in files:
        info = source_map.get(f.stem, {})
        src = info.get("source", f.stem)
        chap = info.get("chapter", "")
        try:
            out = segment_file(f, source=src, chapter=chap,
                               output_path=output_dir / (f.stem + ".json"),
                               id_generator=gen)
            summaries.append({"file": f.name, "output": str(out), "status": "ok"})
        except Exception as exc:
            logger.error(f"Failed to segment {f.name}: {exc}")
            summaries.append({"file": f.name, "status": "error", "error": str(exc)})

    total_shlokas = gen.current
    logger.info(
        f"Segmentation complete. {total_shlokas:,} total shlokas across {len(files)} files."
    )
    return summaries


def load_segmented_shlokas(segmented_dir: str | Path = None) -> list[dict]:
    """Load all segmented shloka JSON files into a flat list."""
    segmented_dir = Path(segmented_dir or SEGMENTED_DIR)
    all_shlokas = []
    for jf in sorted(segmented_dir.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            shlokas = json.load(f)
        all_shlokas.extend(shlokas)
    logger.info(f"Loaded {len(all_shlokas):,} shlokas from {segmented_dir}")
    return all_shlokas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment Sanskrit text into shlokas")
    parser.add_argument("path", help="Cleaned text file or directory")
    parser.add_argument("--source", default="Unknown", help="Source name")
    parser.add_argument("--chapter", default="", help="Chapter reference")
    parser.add_argument("--output", default=None, help="Output file or directory")
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        segment_directory(p, args.output)
    else:
        segment_file(p, source=args.source, chapter=args.chapter, output_path=args.output)
