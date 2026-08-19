"""
Performance metrics tracker for the orchestrator pipeline.

Tracks routing latency, model latency, TTFT, cache hits, fallbacks, etc.
"""
import time
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("performance")


@dataclass
class RequestMetrics:
    routing_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    selected_model: str = ""
    model_tier: str = ""
    difficulty: str = ""
    cache_hit: bool = False
    route_source: str = ""
    fallback_count: int = 0
    escalated: bool = False
    llm_calls: int = 0

    def to_dict(self) -> dict:
        return {
            "routing_latency_ms": round(self.routing_latency_ms, 1),
            "model_latency_ms": round(self.model_latency_ms, 1),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "selected_model": self.selected_model,
            "model_tier": self.model_tier,
            "difficulty": self.difficulty,
            "cache_hit": self.cache_hit,
            "route_source": self.route_source,
            "fallback_count": self.fallback_count,
            "escalated": self.escalated,
            "llm_calls": self.llm_calls,
        }


class PerformanceTracker:
    def __init__(self):
        self._request_count = 0
        self._total_routing_ms = 0.0
        self._total_model_ms = 0.0
        self._total_llm_calls = 0
        self._cache_hits = 0
        self._fallbacks = 0
        self._tier_counts: dict[str, int] = {"lower": 0, "average": 0, "higher": 0}

    def record(self, metrics: RequestMetrics) -> None:
        self._request_count += 1
        self._total_routing_ms += metrics.routing_latency_ms
        self._total_model_ms += metrics.model_latency_ms
        self._total_llm_calls += metrics.llm_calls
        if metrics.cache_hit:
            self._cache_hits += 1
        if metrics.fallback_count > 0:
            self._fallbacks += 1
        tier = metrics.model_tier
        if tier in self._tier_counts:
            self._tier_counts[tier] += 1

        logger.info(
            "[PERF] routing=%.0fms model=%.0fms total=%.0fms tier=%s source=%s cache=%s fallbacks=%d",
            metrics.routing_latency_ms,
            metrics.model_latency_ms,
            metrics.total_latency_ms,
            metrics.model_tier,
            metrics.route_source,
            "hit" if metrics.cache_hit else "miss",
            metrics.fallback_count,
        )

    def get_stats(self) -> dict:
        n = self._request_count or 1
        return {
            "total_requests": self._request_count,
            "avg_routing_latency_ms": round(self._total_routing_ms / n, 1),
            "avg_model_latency_ms": round(self._total_model_ms / n, 1),
            "total_llm_calls": self._total_llm_calls,
            "avg_llm_calls_per_request": round(self._total_llm_calls / n, 2),
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(self._cache_hits / n, 4),
            "fallbacks": self._fallbacks,
            "fallback_rate": round(self._fallbacks / n, 4),
            "tier_distribution": dict(self._tier_counts),
        }


perf_tracker = PerformanceTracker()
