"""
Phase 2 structured logger — thin wrapper around Phase 1 logger.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.logger import setup_logger, JSONLogger   # re-use Phase 1 utility

__all__ = ["setup_logger", "JSONLogger", "get_logger", "get_audit"]


def get_logger(name: str):
    from phase2_config import LOGS_DIR
    return setup_logger(name, LOGS_DIR)


def get_audit(name: str):
    from phase2_config import LOGS_DIR
    return JSONLogger(LOGS_DIR / f"{name}_audit.jsonl")
