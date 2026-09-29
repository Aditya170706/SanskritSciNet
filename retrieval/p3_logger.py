"""Phase 3 structured logger."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.logger import setup_logger, JSONLogger

def get_logger(name: str):
    from phase3_config import LOGS_DIR
    return setup_logger(name, LOGS_DIR)

def get_audit(name: str):
    from phase3_config import LOGS_DIR
    return JSONLogger(LOGS_DIR / f"{name}_audit.jsonl")
