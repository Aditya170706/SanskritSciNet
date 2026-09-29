"""
cache/redis_cache.py
────────────────────
Module 10 — Retrieval Cache

Primary:  Redis (reduces repeated computation for identical queries)
Fallback: In-process LRU dict cache (no Redis install required)

Cache key = SHA-256 hash of normalised query text.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_config import REDIS_HOST, REDIS_PORT, REDIS_DB, CACHE_TTL
from retrieval.p3_logger import get_logger

logger = get_logger("cache")


# ─────────────────────────────────────────────────────────────────────────────
# In-process LRU fallback
# ─────────────────────────────────────────────────────────────────────────────

class _LRUCache:
    """Thread-unsafe LRU cache for single-process use."""

    def __init__(self, maxsize: int = 512, ttl: int = CACHE_TTL):
        self._store:   OrderedDict[str, tuple] = OrderedDict()
        self._maxsize  = maxsize
        self._ttl      = ttl
        self.hits      = 0
        self.misses    = 0

    def get(self, key: str) -> Optional[dict]:
        if key not in self._store:
            self.misses += 1
            return None
        value, ts = self._store[key]
        if time.time() - ts > self._ttl:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: dict) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time())
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        return {
            "backend":  "lru_dict",
            "size":     len(self._store),
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": round(
                self.hits / max(self.hits + self.misses, 1), 3
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Redis backend
# ─────────────────────────────────────────────────────────────────────────────

class _RedisBackend:
    def __init__(self):
        try:
            import redis
            self._r = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                socket_connect_timeout=1, decode_responses=True,
            )
            self._r.ping()
            self.available = True
            logger.info(f"Redis connected at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as exc:
            logger.warning(f"Redis unavailable ({exc}) — using LRU fallback")
            self._r        = None
            self.available = False

    def get(self, key: str) -> Optional[dict]:
        raw = self._r.get(key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value: dict, ttl: int = CACHE_TTL) -> None:
        self._r.setex(key, ttl, json.dumps(value, ensure_ascii=False))

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def clear(self) -> None:
        self._r.flushdb()

    def stats(self) -> dict:
        info = self._r.info("stats")
        return {
            "backend":       "redis",
            "hits":          info.get("keyspace_hits",   0),
            "misses":        info.get("keyspace_misses", 0),
            "connected_clients": info.get("connected_clients", 0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public cache class
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalCache:
    """
    Unified cache — tries Redis, falls back to in-process LRU.

    Usage
    -----
    cache = RetrievalCache()
    key   = cache.make_key(text)
    hit   = cache.get(key)
    if hit is None:
        result = expensive_computation(text)
        cache.set(key, result)
    """

    def __init__(self, ttl: int = CACHE_TTL):
        self._ttl   = ttl
        self._redis = _RedisBackend()
        self._lru   = _LRUCache(ttl=ttl)
        self._backend_name = "redis" if self._redis.available else "lru_dict"

    @staticmethod
    def make_key(text: str) -> str:
        """Stable SHA-256 key for a query string."""
        return "vshloka:" + hashlib.sha256(
            text.strip().lower().encode("utf-8")
        ).hexdigest()[:32]

    # ── Read-through ───────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[dict]:
        if self._redis.available:
            return self._redis.get(key)
        return self._lru.get(key)

    def set(self, key: str, value: dict) -> None:
        if self._redis.available:
            self._redis.set(key, value, ttl=self._ttl)
        else:
            self._lru.set(key, value)

    def delete(self, key: str) -> None:
        if self._redis.available:
            self._redis.delete(key)
        else:
            self._lru.delete(key)

    def clear(self) -> None:
        if self._redis.available:
            self._redis.clear()
        self._lru.clear()

    def stats(self) -> dict:
        if self._redis.available:
            return self._redis.stats()
        return self._lru.stats()

    @property
    def backend(self) -> str:
        return self._backend_name
