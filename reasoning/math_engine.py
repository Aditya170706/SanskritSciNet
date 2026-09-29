"""
reasoning/math_detector.py
reasoning/formula_extractor.py
reasoning/formula_generator.py
──────────────────────────────
Modules 5, 6, 7 — Mathematical Content Detection, Formula Extraction,
and Symbolic Formula Generation via SymPy.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_config import (
    MATH_DOMAINS, MATH_BRANCHES, MATH_KEYWORDS_DEVANAGARI,
    MATH_KEYWORDS_LATIN, KNOWN_FORMULAS,
)
from retrieval.p3_logger import get_logger

logger = get_logger("math_engine")


# ═════════════════════════════════════════════════════════════════════════════
# Module 5 — Mathematical Content Detector
# ═════════════════════════════════════════════════════════════════════════════

class MathContentDetector:
    """
    Rule-based detector that decides whether a shloka contains
    mathematical content.

    Decision hierarchy:
    1. Phase 2 domain prediction → if domain in MATH_DOMAINS → True
    2. Phase 2 branch prediction → if branch in MATH_BRANCHES → True
    3. Devanagari keyword scan
    4. Latin transliteration keyword scan
    5. Numeric pattern scan (Sanskrit/Devanagari digits)
    """

    # Devanagari digits range
    _RE_DIGITS = re.compile(r"[०-९0-9]")
    # Common mathematical operator transliterations
    _RE_MATH_OP = re.compile(
        r"\b(varga|ghana|mul|bhag|yoga|vyavakalana|kshetra|paridhi|vyasa)\b",
        re.IGNORECASE
    )

    def detect(
        self,
        text: str,
        domain: str = "Unknown",
        branch: str = "Unknown",
        phase2_probs: Optional[dict] = None,
    ) -> dict:
        """
        Determine if the shloka is mathematical.

        Returns:
            {
              "math_flag": bool,
              "confidence": float,
              "triggers":   list of triggered rules
            }
        """
        triggers  = []
        score     = 0.0

        # Rule 1 — Phase 2 domain
        if domain in MATH_DOMAINS:
            triggers.append(f"Domain '{domain}' is mathematical")
            score = max(score, 0.85)

        # Rule 2 — Phase 2 branch
        if branch in MATH_BRANCHES:
            triggers.append(f"Branch '{branch}' is mathematical")
            score = max(score, 0.80)

        # Rule 3 — Phase 2 probability mass on math domains
        if phase2_probs:
            domain_probs = phase2_probs.get("domain", {})
            math_mass    = domain_probs.get("Mathematics", 0)
            if math_mass > 0.25:
                triggers.append(
                    f"Phase 2 probability mass on Mathematics: {math_mass:.2f}"
                )
                score = max(score, math_mass)

        # Rule 4 — Devanagari keywords
        for kw in MATH_KEYWORDS_DEVANAGARI:
            if kw in text:
                triggers.append(f"Devanagari math keyword: '{kw}'")
                score = max(score, 0.70)
                break

        # Rule 5 — Latin transliteration keywords
        text_lower = text.lower()
        for kw in MATH_KEYWORDS_LATIN:
            if kw in text_lower:
                triggers.append(f"Latin math keyword: '{kw}'")
                score = max(score, 0.65)
                break

        # Rule 6 — Numeric patterns
        if self._RE_DIGITS.search(text):
            triggers.append("Numeric content detected")
            score = max(score, 0.60)

        # Rule 7 — Math operator patterns
        if self._RE_MATH_OP.search(text):
            triggers.append("Mathematical operation term detected")
            score = max(score, 0.65)

        math_flag = len(triggers) > 0 and score >= 0.60
        logger.debug(f"Math detection: flag={math_flag} score={score:.2f}  "
                     f"triggers={len(triggers)}")
        return {
            "math_flag":  math_flag,
            "confidence": round(score, 4),
            "triggers":   triggers,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Module 6 — Formula Extractor
# ═════════════════════════════════════════════════════════════════════════════

class FormulaExtractor:
    """
    Searches the knowledge graph and KNOWN_FORMULAS dictionary for a formula
    associated with the input shloka.
    """

    # Keyword → formula_key mapping for heuristic matching
    _HINT_MAP = {
        # Devanagari
        "कर्ण":     "pythagorean",
        "भुज":      "pythagorean",
        "वर्गमूल":  "pythagorean",
        "क्षेत्रफल":"triangle_area",
        "त्रिभुज":  "triangle_area",
        "वृत्त":    "circle_area",
        "व्यास":    "circle_area",
        "वर्ग":     "square_area",
        "समान्तर":  "arithmetic_sum",
        "द्विघात":  "quadratic",
        # Latin transliterations
        "karna":     "pythagorean",
        "bhuja":     "pythagorean",
        "tribhuja":  "triangle_area",
        "vritta":    "circle_area",
        "varga":     "square_area",
        "samanta":   "arithmetic_sum",
        "dvighata":  "quadratic",
        "diagonal":  "pythagorean",
        "hypotenuse":"pythagorean",
        "circle":    "circle_area",
        "triangle":  "triangle_area",
        "square":    "square_area",
        "pythagoras":"pythagorean",
    }

    def __init__(self, kg_engine=None):
        self._kg = kg_engine

    def extract(
        self,
        text: str,
        shloka_id: Optional[str] = None,
        domain: str = "Unknown",
        branch: str = "Unknown",
    ) -> Optional[dict]:
        """
        Attempt to find a matching formula.

        Priority:
        1. Knowledge graph formula edge (if shloka_id provided)
        2. KNOWN_FORMULAS keyword match
        3. None (triggers Module 7 generation)

        Returns:
            Formula dict or None
        """
        # Step 1 — KG lookup
        if shloka_id and self._kg:
            kg_formulas = self._kg.find_formulas(shloka_id)
            for f_name in kg_formulas:
                f_name_norm = f_name.lower().replace(" ", "_")
                if f_name_norm in KNOWN_FORMULAS:
                    result = dict(KNOWN_FORMULAS[f_name_norm])
                    result["match_method"] = "knowledge_graph"
                    logger.info(f"Formula from KG: {f_name}")
                    return result
                # Return raw name if not in registry
                logger.info(f"KG formula (unregistered): {f_name}")
                return {
                    "name":         f_name,
                    "expr":         f_name,
                    "latex":        f_name,
                    "source":       "Knowledge Graph",
                    "match_method": "knowledge_graph_raw",
                }

        # Step 2 — Keyword heuristic
        text_lower = text.lower()
        for kw, formula_key in self._HINT_MAP.items():
            if kw in text or kw in text_lower:
                result = dict(KNOWN_FORMULAS[formula_key])
                result["match_method"] = "keyword_hint"
                result["trigger_kw"]   = kw
                logger.info(f"Formula by keyword '{kw}': {formula_key}")
                return result

        # Step 3 — Domain/branch heuristic
        domain_formula_map = {
            "Geometry":      "pythagorean",
            "Algebra":       "quadratic",
            "Arithmetic":    "arithmetic_sum",
            "Number Theory": "arithmetic_sum",
            "Astronomy":     "circle_area",
        }
        key = domain_formula_map.get(branch) or domain_formula_map.get(domain)
        if key:
            result = dict(KNOWN_FORMULAS[key])
            result["match_method"] = "domain_heuristic"
            logger.info(f"Formula by domain/branch heuristic: {key}")
            return result

        return None


# ═════════════════════════════════════════════════════════════════════════════
# Module 7 — Formula Generator (SymPy)
# ═════════════════════════════════════════════════════════════════════════════

class FormulaGenerator:
    """
    Generates a symbolic mathematical expression when no formula is found.

    Strategy:
    1. Parse the shloka text for numeric relationships and variable names
    2. Build a SymPy expression from detected pattern
    3. Validate it symbolically
    4. Return a structured formula dict
    """

    # Regex patterns to identify potential variable tokens (single letters)
    _RE_VARS = re.compile(r"\b([a-zA-Z])\b")
    # Detect equality-like relationships
    _RE_EQ   = re.compile(r"=|equals|sama|tulyam|समान|तुल्य")
    # Detect sum/product relationships
    _RE_SUM  = re.compile(r"\+|plus|yoga|योग|sum")
    _RE_PROD = re.compile(r"\*|times|guna|गुण|product")
    _RE_SQR  = re.compile(r"square|varga|वर्ग|\^2")
    _RE_SQRT = re.compile(r"sqrt|root|mula|मूल")

    def generate(
        self,
        text: str,
        domain: str = "Unknown",
        branch: str = "Unknown",
    ) -> dict:
        """
        Generate a plausible symbolic expression for the shloka.

        Returns:
            {
              "expr":         str (SymPy-compatible),
              "latex":        str,
              "sympy_valid":  bool,
              "source":       "generated",
              "method":       str,
            }
        """
        import sympy as sp

        text_lower = text.lower()
        result = {
            "source":      "generated",
            "sympy_valid": False,
            "method":      "pattern_recognition",
        }

        # ── Pattern matching ───────────────────────────────────────────────
        expr_str  = None
        latex_str = None

        # 1. Square relationship (a² + b² = c²)
        if self._RE_SQR.search(text_lower) and self._RE_SUM.search(text_lower):
            expr_str  = "a**2 + b**2 - c**2"
            latex_str = "a^2 + b^2 = c^2"
            result["method"] = "square_sum_pattern"

        # 2. Square root
        elif self._RE_SQRT.search(text_lower):
            expr_str  = "sqrt(a**2 + b**2)"
            latex_str = "\\sqrt{a^2 + b^2}"
            result["method"] = "sqrt_pattern"

        # 3. Product relationship
        elif self._RE_PROD.search(text_lower):
            expr_str  = "a * b"
            latex_str = "a \\cdot b"
            result["method"] = "product_pattern"

        # 4. Sum relationship
        elif self._RE_SUM.search(text_lower):
            expr_str  = "a + b"
            latex_str = "a + b"
            result["method"] = "sum_pattern"

        # 5. Domain-based fallback
        else:
            branch_defaults = {
                "Geometry":      ("pi * r**2",          "\\pi r^2"),
                "Algebra":       ("a*x**2 + b*x + c",   "ax^2 + bx + c"),
                "Arithmetic":    ("n * (a + l) / 2",     "\\frac{n(a+l)}{2}"),
                "Trigonometry":  ("sin(x)**2 + cos(x)**2 - 1",
                                  "\\sin^2 x + \\cos^2 x = 1"),
                "Number Theory": ("a**2 - b**2",         "a^2 - b^2"),
            }
            domain_defaults = {
                "Mathematics":  ("a + b",  "a + b"),
                "Astronomy":    ("2 * pi * r", "2\\pi r"),
            }
            pair = (branch_defaults.get(branch)
                    or domain_defaults.get(domain)
                    or ("x + y", "x + y"))
            expr_str, latex_str = pair
            result["method"] = "domain_fallback"

        # ── SymPy validation ───────────────────────────────────────────────
        try:
            parsed = sp.sympify(expr_str)
            # Simplify and convert back to canonical form
            simplified = sp.simplify(parsed)
            expr_str   = str(simplified)
            result["sympy_valid"] = True
            result["sympy_repr"] = sp.latex(simplified)
            logger.info(f"Generated formula: {expr_str}  (valid SymPy)")
        except Exception as exc:
            logger.warning(f"SymPy validation failed for '{expr_str}': {exc}")

        result["expr"]  = expr_str
        result["latex"] = latex_str
        return result
