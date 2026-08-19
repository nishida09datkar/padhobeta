"""
In-memory routing cache with query normalization.

Caches routing decisions to avoid repeated LLM orchestrator calls
for the same or similar queries.
"""
import re
import time
import hashlib
import logging
from collections import OrderedDict

logger = logging.getLogger("routing_cache")


class RoutingCache:
    def __init__(self, max_size: int = 512, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _normalize(self, query: str) -> str:
        q = query.strip().lower()
        q = re.sub(r"[?.!,;:]+", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _make_key(self, query: str) -> str:
        normalized = self._normalize(query)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    def get(self, query: str) -> dict | None:
        key = self._make_key(query)
        if key in self._cache:
            decision, stored_at = self._cache[key]
            if time.time() - stored_at < self.ttl_seconds:
                self._hits += 1
                self._cache.move_to_end(key)
                logger.info(
                    "[ROUTING_CACHE] HIT key=%s model=%s",
                    key[:8], decision.get("recommended_model"),
                )
                return decision
            else:
                del self._cache[key]

        self._misses += 1
        return None

    def put(self, query: str, decision: dict) -> None:
        key = self._make_key(query)
        self._cache[key] = (decision, time.time())
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        logger.info(
            "[ROUTING_CACHE] STORE key=%s model=%s",
            key[:8], decision.get("recommended_model"),
        )

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(hit_rate, 4),
            "size": len(self._cache),
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0


routing_cache = RoutingCache()
