"""
Module 4 — Annotation Engine
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System

Supports manual and AI-assisted annotation of segmented shlokas.
Saves annotated data to data/annotated/
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SEGMENTED_DIR, ANNOTATED_DIR, LOGS_DIR, ANNOTATION_FIELDS
from scripts.logger import setup_logger, JSONLogger

logger = setup_logger("annotation", LOGS_DIR)
audit = JSONLogger(LOGS_DIR / "annotation_audit.jsonl")


# ---------------------------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------------------------

VEDA_OPTIONS = ["Rigveda", "Samaveda", "Yajurveda", "Atharvaveda", "Other", "Unknown"]

DOMAIN_OPTIONS = [
    "Mathematics", "Astronomy", "Grammar", "Philosophy", "Ritual",
    "Medicine", "Linguistics", "Cosmology", "Ethics", "Unknown"
]

BRANCH_OPTIONS = [
    "Arithmetic", "Geometry", "Algebra", "Combinatorics", "Number Theory",
    "Kalpa", "Jyotisha", "Shiksha", "Chandas", "Nirukta", "Vyakarana",
    "Vedanta", "Mimamsa", "Unknown"
]

LANGUAGE_OPTIONS = ["Sanskrit", "Prakrit", "Mixed", "Unknown"]


def _validate_annotation(annotation: dict) -> list[str]:
    """
    Validate annotation fields. Returns list of warning messages (empty = valid).
    """
    warnings = []
    for field in ANNOTATION_FIELDS:
        if field not in annotation:
            warnings.append(f"Missing field: {field}")
    if "keywords" in annotation and not isinstance(annotation["keywords"], list):
        warnings.append("'keywords' must be a list")
    return warnings


def _default_annotation() -> dict:
    """Return a blank annotation template."""
    return {
        "veda": "Unknown",
        "domain": "Unknown",
        "branch": "Unknown",
        "formula": "",
        "source": "",
        "language": "Sanskrit",
        "keywords": [],
        "annotated_by": "manual",
        "annotated_at": None,
        "confidence": 1.0,
        "notes": "",
    }


# ---------------------------------------------------------------------------
# AI-assisted annotation
# ---------------------------------------------------------------------------

def ai_annotate_shloka(shloka: dict) -> dict:
    """
    Generate AI-assisted annotation for a single shloka using heuristic rules.

    This module uses rule-based heuristics to classify shlokas without
    requiring external API calls. For production, replace with an LLM call.

    Returns:
        Annotation dict
    """
    text = shloka.get("text", "")
    source = shloka.get("source", "")

    annotation = _default_annotation()
    annotation["source"] = source
    annotation["annotated_by"] = "ai_heuristic"
    annotation["annotated_at"] = datetime.utcnow().isoformat() + "Z"
    annotation["confidence"] = 0.6  # Heuristic confidence

    # --- Veda detection ---
    source_lower = source.lower()
    if "rigveda" in source_lower or "rig veda" in source_lower:
        annotation["veda"] = "Rigveda"
    elif "samaveda" in source_lower or "sama veda" in source_lower:
        annotation["veda"] = "Samaveda"
    elif "yajurveda" in source_lower or "yajur veda" in source_lower:
        annotation["veda"] = "Yajurveda"
    elif "atharvaveda" in source_lower or "atharva veda" in source_lower:
        annotation["veda"] = "Atharvaveda"
    else:
        annotation["veda"] = "Unknown"

    # --- Domain detection (keyword matching in Devanagari + transliteration) ---
    math_keywords = [
        "गणित", "संख्या", "क्षेत्र", "कोण", "वर्ग", "घन",
        "ganita", "sankhya", "kshetra", "kona", "varga", "ghana"
    ]
    astro_keywords = [
        "ज्योतिष", "ग्रह", "नक्षत्र", "चन्द्र", "सूर्य",
        "jyotisha", "graha", "nakshatra", "chandra", "surya"
    ]
    grammar_keywords = [
        "व्याकरण", "धातु", "सूत्र", "पाणिनि",
        "vyakarana", "dhatu", "sutra", "panini"
    ]
    philosophy_keywords = [
        "ब्रह्म", "आत्म", "मोक्ष", "धर्म",
        "brahma", "atma", "moksha", "dharma"
    ]

    text_lower = text.lower()
    if any(kw in text for kw in math_keywords) or any(kw in text_lower for kw in math_keywords):
        annotation["domain"] = "Mathematics"
        annotation["branch"] = "Arithmetic"
    elif any(kw in text for kw in astro_keywords) or any(kw in text_lower for kw in astro_keywords):
        annotation["domain"] = "Astronomy"
        annotation["branch"] = "Jyotisha"
    elif any(kw in text for kw in grammar_keywords) or any(kw in text_lower for kw in grammar_keywords):
        annotation["domain"] = "Grammar"
        annotation["branch"] = "Vyakarana"
    elif any(kw in text for kw in philosophy_keywords) or any(kw in text_lower for kw in philosophy_keywords):
        annotation["domain"] = "Philosophy"
        annotation["branch"] = "Vedanta"
    else:
        annotation["domain"] = "Unknown"
        annotation["branch"] = "Unknown"

    # --- Formula detection ---
    # Check for numeric patterns or mathematical terms
    import re
    if re.search(r"[\d०-९]+", text):  # ASCII or Devanagari digits
        annotation["formula"] = "contains_numeric_expression"
    elif "सूत्र" in text or "sutra" in text.lower():
        annotation["formula"] = "sutric_formula"
    else:
        annotation["formula"] = ""

    # --- Keywords ---
    keywords = []
    if annotation["domain"] != "Unknown":
        keywords.append(annotation["domain"].lower())
    if annotation["veda"] != "Unknown":
        keywords.append(annotation["veda"].lower())
    if annotation["branch"] != "Unknown":
        keywords.append(annotation["branch"].lower())
    annotation["keywords"] = keywords

    annotation["language"] = "Sanskrit"
    return annotation


# ---------------------------------------------------------------------------
# Annotation of files
# ---------------------------------------------------------------------------

def annotate_shloka(shloka: dict, annotation: Optional[dict] = None, use_ai: bool = False) -> dict:
    """
    Annotate a single shloka dict.

    Args:
        shloka: Shloka dict (id, text, source, chapter)
        annotation: Manual annotation dict (optional)
        use_ai: If True and no manual annotation provided, run AI heuristics

    Returns:
        Annotated shloka dict
    """
    result = dict(shloka)  # Copy

    if annotation:
        ann = {**_default_annotation(), **annotation}
        ann["annotated_by"] = annotation.get("annotated_by", "manual")
        ann["annotated_at"] = annotation.get("annotated_at", datetime.utcnow().isoformat() + "Z")
    elif use_ai:
        ann = ai_annotate_shloka(shloka)
    else:
        ann = _default_annotation()
        ann["source"] = shloka.get("source", "")

    # Validate
    warnings = _validate_annotation(ann)
    if warnings:
        logger.warning(f"Annotation warnings for {shloka.get('id')}: {warnings}")

    result["annotation"] = ann
    return result


def annotate_file(
    segmented_path: str | Path,
    output_path: Optional[str | Path] = None,
    use_ai: bool = True,
    manual_annotations: Optional[dict] = None,
) -> Path:
    """
    Annotate all shlokas in a segmented JSON file.

    Args:
        segmented_path: Path to segmented JSON file
        output_path: Destination (defaults to data/annotated/<stem>.json)
        use_ai: Use AI-assisted heuristic annotation as default
        manual_annotations: Dict mapping shloka_id → annotation dict for manual overrides

    Returns:
        Path to annotated JSON file
    """
    segmented_path = Path(segmented_path)
    manual_annotations = manual_annotations or {}

    if output_path is None:
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ANNOTATED_DIR / segmented_path.name
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(segmented_path, encoding="utf-8") as f:
        shlokas = json.load(f)

    annotated = []
    ai_count = 0
    manual_count = 0

    for shloka in shlokas:
        sid = shloka.get("id", "")
        manual_ann = manual_annotations.get(sid)
        if manual_ann:
            result = annotate_shloka(shloka, annotation=manual_ann)
            manual_count += 1
        else:
            result = annotate_shloka(shloka, use_ai=use_ai)
            if use_ai:
                ai_count += 1

        annotated.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotated, f, ensure_ascii=False, indent=2)

    audit.log(
        "file_annotated",
        file=segmented_path.name,
        total=len(annotated),
        manual=manual_count,
        ai=ai_count,
        output=str(output_path),
    )
    logger.info(
        f"Annotated {len(annotated)} shlokas → {output_path} "
        f"(manual={manual_count}, ai={ai_count})"
    )
    return output_path


def annotate_directory(
    segmented_dir: str | Path = None,
    output_dir: str | Path = None,
    use_ai: bool = True,
) -> list[Path]:
    """
    Annotate all segmented JSON files in a directory.
    """
    segmented_dir = Path(segmented_dir or SEGMENTED_DIR)
    output_dir = Path(output_dir or ANNOTATED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(segmented_dir.glob("*.json"))
    logger.info(f"Found {len(files)} segmented files to annotate")

    results = []
    for f in files:
        try:
            out = annotate_file(f, output_dir / f.name, use_ai=use_ai)
            results.append(out)
        except Exception as exc:
            logger.error(f"Failed to annotate {f.name}: {exc}")
            audit.log("annotation_error", file=str(f), error=str(exc))

    logger.info(f"Annotation complete. {len(results)}/{len(files)} files processed.")
    return results


def load_annotated_shlokas(annotated_dir: str | Path = None) -> list[dict]:
    """Load all annotated JSON files into a flat list."""
    annotated_dir = Path(annotated_dir or ANNOTATED_DIR)
    all_shlokas = []
    for jf in sorted(annotated_dir.glob("*.json")):
        with open(jf, encoding="utf-8") as f:
            shlokas = json.load(f)
        all_shlokas.extend(shlokas)
    logger.info(f"Loaded {len(all_shlokas):,} annotated shlokas")
    return all_shlokas


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate segmented shlokas")
    parser.add_argument("path", help="Segmented JSON file or directory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-ai", action="store_true", help="Disable AI-assisted annotation")
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        annotate_directory(p, args.output, use_ai=not args.no_ai)
    else:
        annotate_file(p, args.output, use_ai=not args.no_ai)
