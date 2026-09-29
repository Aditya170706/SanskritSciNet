"""
Sample Data Seeder
Creates sample Sanskrit shloka files for testing the pipeline.
Uses public-domain Vedic shlokas.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DIR

SAMPLE_CORPUS = {
    "rigveda_mandala1.txt": (
        "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्। होतारं रत्नधातमम्।।\n\n"
        "अग्निः पूर्वेभिरृषिभिरीड्यो नूतनैरुत। स देवाँ एह वक्षति।।\n\n"
        "अग्निना रयिमश्नवत् पोषमेव दिवेदिवे। यशसं वीरवत्तमम्।।\n\n"
        "अग्ने यं यज्ञमध्वरं विश्वतः परिभूरसि। स इद्देवेषु गच्छति।।\n\n"
        "अग्निर्होता कविक्रतुः सत्यश्चित्रश्रवस्तमः। देवो देवेभिरागमत्।।\n"
    ),
    "rigveda_mandala2.txt": (
        "इन्द्रमिद् गाथिनो बृहदिन्द्रमर्केभिरर्किणः। इन्द्रं वाणीरनूषत।।\n\n"
        "इन्द्र इद्धर्योः सचा समिश्ल आ वचोयुजा। इन्द्रो वज्री हिरण्ययः।।\n\n"
        "इन्द्र इद्वृत्रहा वसु दाता मघवा पुरुहूतः। इन्द्रः सोमस्य पीतये।।\n"
    ),
    "vedic_mathematics_sutras.txt": (
        "एकाधिकेन पूर्वेण। इति सूत्रम्।।\n\n"
        "निखिलं नवतश्चरमं दशतः। गणितसूत्रम्।।\n\n"
        "ऊर्ध्वतिर्यग्भ्याम्। वर्गणासूत्रम्।।\n\n"
        "परावर्त्य योजयेत्। भागसूत्रम्।।\n\n"
        "शून्यं साम्यसमुच्चये। समीकरणसूत्रम्।।\n\n"
        "आनुरूप्येण। अनुपातसूत्रम्।।\n\n"
        "संकलनव्यवकलनाभ्याम्। परिकलनसूत्रम्।।\n\n"
        "पूरणापूरणाभ्याम्। पूर्णसूत्रम्।।\n\n"
        "चलनकलनाभ्याम्। गतिसूत्रम्।।\n\n"
        "यावदूनं तावदूनं। न्यूनतासूत्रम्।।\n"
    ),
    "sulbasutras_sample.txt": (
        "समचतुरश्रस्य क्षेत्रफलं भुजवर्गः।।\n\n"
        "आयतचतुरश्रस्य क्षेत्रफलं दीर्घभुजेन ह्रस्वभुजगुणितम्।।\n\n"
        "त्रिभुजस्य क्षेत्रफलं भूमिश्चशीर्षलम्बार्धगुणितम्।।\n\n"
        "वृत्तस्य क्षेत्रफलं व्यासार्धवर्गं पाईगुणितम्।।\n\n"
        "गोलस्य पृष्ठफलं व्यासवर्गं पाईगुणितम्।।\n"
    ),
    "atharvaveda_sample.txt": (
        "ज्योतिषां ज्योतिरजरं तदादित्यस्तपति। सूर्यो ग्रहाणां राजा।।\n\n"
        "चन्द्रमा नक्षत्राणां पतिः शीतलप्रकाशः। रात्रौ प्रकाशयति।।\n\n"
        "ग्रहाः नवसंख्याकाः सूर्यमण्डलं परिक्रामन्ति। ज्योतिष्यम्।।\n\n"
        "नक्षत्राणि सप्तविंशतिः चन्द्रमण्डलं परिक्रामन्ति।।\n"
    ),
}

SOURCE_METADATA = {
    "rigveda_mandala1.txt": {
        "source_name": "Rigveda Mandala 1",
        "author": "Various Vedic Seers",
        "publication_year": None,
        "license_type": "Public Domain",
        "citation_reference": "Rigveda, Mandala 1. Public domain Sanskrit text.",
    },
    "rigveda_mandala2.txt": {
        "source_name": "Rigveda Mandala 2",
        "author": "Various Vedic Seers",
        "publication_year": None,
        "license_type": "Public Domain",
        "citation_reference": "Rigveda, Mandala 2. Public domain Sanskrit text.",
    },
    "vedic_mathematics_sutras.txt": {
        "source_name": "Vedic Mathematics Sutras",
        "author": "Traditional",
        "publication_year": None,
        "license_type": "Public Domain",
        "citation_reference": "Traditional Vedic mathematical sutras. Public domain.",
    },
    "sulbasutras_sample.txt": {
        "source_name": "Sulba Sutras (Geometric Formulae)",
        "author": "Baudhayana",
        "publication_year": None,
        "license_type": "Public Domain",
        "citation_reference": "Baudhayana Sulba Sutras. Public domain ancient mathematical text.",
    },
    "atharvaveda_sample.txt": {
        "source_name": "Atharvaveda Jyotisha Sample",
        "author": "Various Vedic Seers",
        "publication_year": None,
        "license_type": "Public Domain",
        "citation_reference": "Atharvaveda, astronomical passages. Public domain.",
    },
}


def seed_sample_data(raw_dir: Path = None) -> list[dict]:
    """
    Write sample Sanskrit corpus files to the raw directory.

    Returns:
        List of source metadata dicts
    """
    raw_dir = Path(raw_dir or RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for filename, content in SAMPLE_CORPUS.items():
        dest = raw_dir / filename
        dest.write_text(content, encoding="utf-8")
        meta = SOURCE_METADATA.get(filename, {})
        meta["filename"] = filename
        meta["chars"] = len(content)
        results.append(meta)
        print(f"  Created: {dest} ({len(content):,} chars)")

    print(f"\nSeeded {len(results)} sample files into {raw_dir}")
    return results


if __name__ == "__main__":
    import sys
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    seed_sample_data(raw)
