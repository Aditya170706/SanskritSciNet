"""
Module 1 — Data Ingestion Pipeline
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Accepts: PDF, TXT, XML, HTML, CSV
Outputs: raw files in data/raw/, metadata in metadata/source_registry.json
"""

import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DIR, METADATA_DIR, SOURCE_REGISTRY_FILE, SUPPORTED_FORMATS
from scripts.logger import setup_logger, JSONLogger

logger = setup_logger("ingestion", Path(__file__).parent.parent / "logs")
audit = JSONLogger(Path(__file__).parent.parent / "logs" / "ingestion_audit.jsonl")


# ---------------------------------------------------------------------------
# Extractor helpers
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        raise
    except Exception as exc:
        logger.error(f"PDF extraction failed for {path}: {exc}")
        raise


def _extract_txt(path: Path) -> str:
    """Extract text from plain text file, auto-detecting encoding."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "iso-8859-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} with known encodings")


def _extract_xml(path: Path) -> str:
    """Extract text from XML/TEI files."""
    from bs4 import BeautifulSoup
    content = path.read_bytes()
    soup = BeautifulSoup(content, "lxml-xml")
    return soup.get_text(separator="\n")


def _extract_html(path: Path) -> str:
    """Extract text from HTML files."""
    from bs4 import BeautifulSoup
    content = path.read_bytes()
    soup = BeautifulSoup(content, "lxml")
    # Remove script/style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _extract_csv(path: Path) -> str:
    """Extract text column from CSV (assumes a 'text' or 'shloka' column)."""
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8")
    text_cols = [c for c in df.columns if c.lower() in ("text", "shloka", "content", "sanskrit")]
    if not text_cols:
        # Fall back to all string columns
        text_cols = df.select_dtypes(include="object").columns.tolist()
    return "\n".join(df[text_cols[0]].dropna().astype(str).tolist())


EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".txt": _extract_txt,
    ".xml": _extract_xml,
    ".html": _extract_html,
    ".htm": _extract_html,
    ".csv": _extract_csv,
}


# ---------------------------------------------------------------------------
# Source registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if SOURCE_REGISTRY_FILE.exists():
        with open(SOURCE_REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"sources": []}


def _save_registry(registry: dict):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCE_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_file(
    source_path: str | Path,
    source_name: str,
    author: str = "Unknown",
    publication_year: Optional[int] = None,
    license_type: str = "Public Domain",
    citation_reference: str = "",
) -> dict:
    """
    Ingest a single source file into the raw data directory and register metadata.

    Args:
        source_path: Path to input file
        source_name: Human-readable source name (e.g. "Rigveda Book 1")
        author: Author/compiler of the text
        publication_year: Year of publication or transcription
        license_type: License string (default "Public Domain")
        citation_reference: Full bibliographic citation

    Returns:
        Metadata dict for this source entry
    """
    source_path = Path(source_path)
    suffix = source_path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS and suffix not in EXTRACTORS:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {SUPPORTED_FORMATS}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Copy raw file preserving original name (add timestamp to avoid collisions)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    raw_filename = f"{ts}_{source_path.name}"
    raw_dest = RAW_DIR / raw_filename
    shutil.copy2(source_path, raw_dest)
    logger.info(f"Copied raw file → {raw_dest}")

    # Extract text
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        raise ValueError(f"No extractor for {suffix}")

    logger.info(f"Extracting text from {source_path} (format={suffix})")
    extracted_text = extractor(source_path)
    char_count = len(extracted_text)
    logger.info(f"Extracted {char_count:,} characters from {source_path.name}")

    # Build metadata entry
    sha256 = _file_sha256(source_path)
    metadata_entry = {
        "raw_filename": raw_filename,
        "original_filename": source_path.name,
        "source_name": source_name,
        "author": author,
        "publication_year": publication_year,
        "license_type": license_type,
        "citation_reference": citation_reference,
        "format": suffix,
        "sha256": sha256,
        "char_count": char_count,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "status": "ingested",
    }

    # Update registry
    registry = _load_registry()
    # Avoid duplicate entries (same sha256)
    existing_hashes = {s["sha256"] for s in registry["sources"]}
    if sha256 in existing_hashes:
        logger.warning(f"Duplicate file detected (sha256={sha256}). Skipping registry update.")
    else:
        registry["sources"].append(metadata_entry)
        _save_registry(registry)
        logger.info(f"Registered source: {source_name}")

    audit.log("file_ingested", source=source_name, sha256=sha256, chars=char_count)
    return metadata_entry


def ingest_directory(
    directory: str | Path,
    source_name_prefix: str = "Vedic Text",
    **metadata_kwargs,
) -> list[dict]:
    """
    Batch-ingest all supported files from a directory.

    Returns:
        List of metadata dicts for each ingested file
    """
    directory = Path(directory)
    results = []
    files = [f for f in directory.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]
    logger.info(f"Found {len(files)} supported files in {directory}")

    for i, f in enumerate(files, 1):
        try:
            source_name = f"{source_name_prefix} — {f.stem}"
            logger.info(f"[{i}/{len(files)}] Ingesting: {f.name}")
            entry = ingest_file(f, source_name=source_name, **metadata_kwargs)
            results.append(entry)
        except Exception as exc:
            logger.error(f"Failed to ingest {f.name}: {exc}")
            audit.log("ingest_error", file=str(f), error=str(exc))

    logger.info(f"Batch ingestion complete. {len(results)}/{len(files)} files succeeded.")
    return results


def list_registry() -> list[dict]:
    """Return all registered source entries."""
    return _load_registry().get("sources", [])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Sanskrit source files")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--source-name", default="Sanskrit Text", help="Human-readable source name")
    parser.add_argument("--author", default="Unknown")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--license", default="Public Domain")
    parser.add_argument("--citation", default="")
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        ingest_directory(p, source_name_prefix=args.source_name,
                         author=args.author, publication_year=args.year,
                         license_type=args.license, citation_reference=args.citation)
    else:
        ingest_file(p, source_name=args.source_name,
                    author=args.author, publication_year=args.year,
                    license_type=args.license, citation_reference=args.citation)
