"""
api/metrics_collector.py
────────────────────────
Module 9 — Performance monitoring.
Tracks request latency, CPU, memory and persists to metrics/.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

import psutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app_config.settings import METRICS_DIR, METRICS_ENABLED


class MetricsCollector:
    """
    Thread-safe in-memory metrics store with periodic JSON flush.
    """

    def __init__(self, flush_interval: int = 60):
        self._lock          = threading.Lock()
        self._latencies:    deque = deque(maxlen=1000)
        self._total         = 0
        self._success       = 0
        self._failed        = 0
        self._cache_hits    = 0
        self._start_time    = time.time()
        self._flush_interval= flush_interval

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        self._metrics_file  = METRICS_DIR / "requests.json"

        if METRICS_ENABLED and flush_interval > 0:
            self._start_flush_thread()

    # ── Recording ─────────────────────────────────────────────────────────

    def record(self, latency_ms: float, success: bool, cache_hit: bool = False):
        with self._lock:
            self._total    += 1
            self._latencies.append(latency_ms)
            if success:
                self._success += 1
            else:
                self._failed  += 1
            if cache_hit:
                self._cache_hits += 1

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            lats = sorted(self._latencies)
            n    = len(lats)
            avg  = sum(lats) / n if n else 0.0
            p95  = lats[int(n * 0.95)] if n >= 20 else (max(lats) if lats else 0.0)
            proc = psutil.Process()
            return {
                "total_requests":  self._total,
                "successful":      self._success,
                "failed":          self._failed,
                "avg_latency_ms":  round(avg, 2),
                "p95_latency_ms":  round(p95, 2),
                "cache_hit_rate":  round(self._cache_hits / max(self._total, 1), 3),
                "cpu_percent":     round(psutil.cpu_percent(interval=None), 1),
                "memory_mb":       round(proc.memory_info().rss / 1_048_576, 1),
                "uptime_seconds":  round(time.time() - self._start_time, 1),
            }

    # ── Flush ─────────────────────────────────────────────────────────────

    def flush(self):
        snap = self.snapshot()
        snap["flushed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self._metrics_file, "w") as f:
            json.dump(snap, f, indent=2)

    def _start_flush_thread(self):
        def _loop():
            while True:
                time.sleep(self._flush_interval)
                try:
                    self.flush()
                except Exception:
                    pass
        t = threading.Thread(target=_loop, daemon=True)
        t.start()


# Singleton
_collector: MetricsCollector | None = None


def get_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
