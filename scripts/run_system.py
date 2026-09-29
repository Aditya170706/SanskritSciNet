"""
scripts/run_system.py
─────────────────────
Module 6 — System launcher.

Starts the FastAPI server and Streamlit UI as separate subprocesses.

Usage
  python scripts/run_system.py           # start both
  python scripts/run_system.py --api     # API only
  python scripts/run_system.py --ui      # UI only
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from app_config.settings import SERVER_HOST, SERVER_PORT, UI_PORT, LOG_DIR
from scripts.logger import setup_logger

logger = setup_logger("run_system", LOG_DIR)

_procs: list[subprocess.Popen] = []


def _kill_all(sig=None, frame=None):
    logger.info("Shutting down all processes…")
    for p in _procs:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            p.kill()
    sys.exit(0)


def start_api() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--workers", "1",
    ]
    logger.info(f"Starting API server on {SERVER_HOST}:{SERVER_PORT}")
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    _procs.append(proc)
    return proc


def start_ui() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "ui" / "app.py"),
        "--server.port", str(UI_PORT),
        "--server.address", "localhost",
        "--server.headless", "true",
    ]
    logger.info(f"Starting Streamlit UI on http://localhost:{UI_PORT}")
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    _procs.append(proc)
    return proc


def wait_for_api(timeout: int = 60) -> bool:
    """Poll /health until the API is ready or timeout."""
    import urllib.request
    url = f"http://localhost:{SERVER_PORT}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Vedic Shloka System")
    parser.add_argument("--api",    action="store_true", help="Start API only")
    parser.add_argument("--ui",     action="store_true", help="Start UI only")
    parser.add_argument("--no-wait",action="store_true", help="Skip health wait")
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  _kill_all)
    signal.signal(signal.SIGTERM, _kill_all)

    run_api = args.api or (not args.api and not args.ui)
    run_ui  = args.ui  or (not args.api and not args.ui)

    print("\n" + "=" * 56)
    print("  Vedic Shloka Intelligence System — Launcher")
    print("=" * 56)

    if run_api:
        start_api()
        if not args.no_wait:
            print("Waiting for API to be ready…", end="", flush=True)
            ready = wait_for_api(timeout=90)
            print(" ✓ ready" if ready else " ✗ timeout")

    if run_ui:
        start_ui()
        time.sleep(2)

    print(f"\n  API : http://localhost:{SERVER_PORT}")
    print(f"  Docs: http://localhost:{SERVER_PORT}/docs")
    if run_ui:
        print(f"  UI  : http://localhost:{UI_PORT}")
    print("\n  Press Ctrl+C to stop all services.\n")

    # Keep alive
    try:
        while True:
            time.sleep(5)
            # Restart crashed processes
            for p in list(_procs):
                if p.poll() is not None:
                    logger.warning(f"Process {p.pid} died — restart manually if needed.")
                    _procs.remove(p)
    except KeyboardInterrupt:
        _kill_all()
