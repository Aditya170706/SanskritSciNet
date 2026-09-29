"""
Vedic Shloka Intelligence and Mathematical Knowledge Extraction System
Phase 1 — Data and Database Infrastructure Configuration
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
SEGMENTED_DIR = DATA_DIR / "segmented"
ANNOTATED_DIR = DATA_DIR / "annotated"
TRAINING_DIR = DATA_DIR / "training"

DATABASE_DIR = PROJECT_ROOT / "database"
KNOWLEDGE_GRAPH_DIR = DATABASE_DIR / "knowledge_graph"
SEARCH_INDEX_DIR = DATABASE_DIR / "search_index"

METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Metadata files
SOURCE_REGISTRY_FILE = METADATA_DIR / "source_registry.json"

# Neo4j configuration (defaults, override via .env)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Elasticsearch configuration
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "vedic_shlokas")

# Dataset split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Shloka segmentation markers
SHLOKA_MARKERS = ["।", "॥"]

# Supported input formats
SUPPORTED_FORMATS = [".pdf", ".txt", ".xml", ".html", ".csv"]

# Annotation fields
ANNOTATION_FIELDS = ["veda", "domain", "branch", "formula", "source", "language", "keywords"]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "system.log"

# Allowed public-domain sources
PUBLIC_DOMAIN_SOURCES = [
    "Digital Corpus of Sanskrit (DCS)",
    "GRETIL Sanskrit Text Repository",
    "Sanskrit Documents Archive",
    "Muktabodha Digital Library",
    "Public-domain Vedic texts",
    "Public-domain mathematical Sanskrit texts",
]
