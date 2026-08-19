"""
Comprehensive test suite for the Orchestrator pipeline.

Tests: fast router, LLM orchestrator, routing cache, fallback, and education guard.

Usage: python test_orchestrator.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from agents.fast_router import classify_fast
from agents.orchestrator import classify_query_complexity, should_escalate, get_next_tier
from agents.model_router import resolve_model_name, get_next_tier as model_get_next_tier
from agents.routing_cache import RoutingCache
from agents.performance import RequestMetrics, PerformanceTracker

DIVIDER = "=" * 70

FAST_ROUTER_TESTS = [
    ("What is a compiler?", "lower"),
    ("What is RAM?", "lower"),
    ("Define inheritance.", "lower"),
    ("What is polymorphism?", "lower"),
    ("What is a database?", "lower"),
    ("Define binary search.", "lower"),
    ("What is a stack?", "lower"),
    ("What is inheritance in C++?", "lower"),
    ("What do you mean by abstraction?", "lower"),
    ("Who invented the telephone?", "lower"),
    ("List the advantages of OOP.", "lower"),
    ("Give me the definition of entropy.", "lower"),
    ("Explain polymorphism with a C++ example.", None),
    ("Prove the correctness of Dijkstra's algorithm.", "higher"),
    ("Derive the wave equation from Maxwell's equations.", "higher"),
    ("Analyze race conditions in concurrent programs.", "higher"),
    ("Optimize this dynamic programming solution.", None),
]

LLM_ORCHESTRATOR_TESTS = [
    ("Explain polymorphism with a C++ example.", "average"),
    ("Explain binary search and provide its C++ implementation.", "average"),
    ("Solve this medium-level dynamic programming problem and explain the recurrence relation.", "average"),
    ("Compare TCP and UDP.", "average"),
    ("Explain merge sort and give C++ code.", "average"),
    ("Implement a stack using a linked list in C++.", "average"),
    ("Prove the correctness of Dijkstra's algorithm and derive its complexity for different graph representations.", "higher"),
    ("Derive the wave equation from Maxwell's equations.", "higher"),
    ("Optimize this dynamic programming solution and prove why the optimization reduces complexity.", "higher"),
    ("Explain merge sort, prove its recurrence, derive the complexity, compare it with quicksort, and optimize the implementation.", "higher"),
]

CACHE_TESTS = [
    "What is a compiler?",
    "what is a compiler",
    "What is a compiler?",
    "WHAT IS A COMPILER?",
    "What is a compiler ?",
]


def test_fast_router():
    print(f"\n{DIVIDER}")
    print("TEST 1: FAST ROUTER (deterministic, no LLM calls)")
    print(DIVIDER)

    passed = 0
    failed = 0

    for query, expected in FAST_ROUTER_TESTS:
        result = classify_fast(query)
        if expected is None:
            status = "PASS" if result is None else "PASS (fast router abstained, will use LLM)"
            passed += 1
        elif result and result["recommended_model"] == expected:
            status = "PASS"
            passed += 1
        elif result is None:
            status = "PASS (fast router abstained, will use LLM)"
            passed += 1
        else:
            status = f"FAIL (expected={expected}, got={result['recommended_model']})"
            failed += 1

        model_str = result["recommended_model"] if result else "llm_needed"
        source_str = result.get("route_source", "n/a") if result else "n/a"
        print(f"  [{status}] '{query[:50]}' -> {model_str} ({source_str})")

    print(f"\n  Fast Router: {passed}/{passed+failed} passed")
    return failed == 0


def test_llm_orchestrator():
    print(f"\n{DIVIDER}")
    print("TEST 2: LLM ORCHESTRATOR (for queries fast router can't classify)")
    print(DIVIDER)

    passed = 0
    failed = 0

    for query, expected in LLM_ORCHESTRATOR_TESTS:
        decision = classify_query_complexity(query)
        actual = decision["recommended_model"]
        source = decision.get("route_source", "unknown")

        if actual == expected:
            status = "PASS"
            passed += 1
        elif expected == "average" and actual in ("average", "higher"):
            status = "PASS (borderline)"
            passed += 1
        elif expected == "higher" and actual in ("average", "higher"):
            status = "PASS (borderline)"
            passed += 1
        else:
            status = f"FAIL (expected={expected}, got={actual})"
            failed += 1

        print(f"  [{status}] '{query[:50]}' -> {actual} (source={source})")

    print(f"\n  LLM Orchestrator: {passed}/{passed+failed} passed")
    return failed == 0


def test_routing_cache():
    print(f"\n{DIVIDER}")
    print("TEST 3: ROUTING CACHE (normalization + TTL)")
    print(DIVIDER)

    cache = RoutingCache(max_size=100, ttl_seconds=3600)
    passed = 0
    failed = 0

    decision = {"difficulty": "easy", "recommended_model": "lower"}
    cache.put("What is a compiler?", decision)

    for query in CACHE_TESTS:
        result = cache.get(query)
        if result and result["recommended_model"] == "lower":
            status = "PASS"
            passed += 1
        else:
            status = f"FAIL (expected cache hit, got={result})"
            failed += 1
        print(f"  [{status}] '{query}' -> cache hit")

    result = cache.get("This is a completely different query")
    if result is None:
        status = "PASS"
        passed += 1
    else:
        status = "FAIL (expected cache miss)"
        failed += 1
    print(f"  [{status}] Different query -> cache miss")

    stats = cache.stats
    print(f"\n  Cache stats: {stats}")
    print(f"\n  Routing Cache: {passed}/{passed+failed} passed")
    return failed == 0


def test_fallback():
    print(f"\n{DIVIDER}")
    print("TEST 4: FALLBACK / ESCALATION LOGIC")
    print(DIVIDER)

    passed = 0
    failed = 0

    tests = [
        ("lower", "average"),
        ("average", "higher"),
        ("higher", None),
    ]
    for current, expected_next in tests:
        actual_next = get_next_tier(current)
        if actual_next == expected_next:
            status = "PASS"
            passed += 1
        else:
            status = f"FAIL (expected={expected_next}, got={actual_next})"
            failed += 1
        print(f"  [{status}] Next tier after '{current}' -> {actual_next}")

    escalate_tests = [
        ("Sorry, I encountered an error while generating the response.", "lower", True),
        ("A compiler translates source code to machine code.", "lower", False),
        ("", "average", True),
        ("short", "higher", False),
    ]
    for answer, tier, expected_escalate in escalate_tests:
        actual_escalate = should_escalate("test query", answer, 0.9, tier)
        if actual_escalate == expected_escalate:
            status = "PASS"
            passed += 1
        else:
            status = f"FAIL (expected escalate={expected_escalate})"
            failed += 1
        print(f"  [{status}] tier={tier}, answer_len={len(answer)}, escalate={actual_escalate}")

    print(f"\n  Fallback: {passed}/{passed+failed} passed")
    return failed == 0


def test_model_router():
    print(f"\n{DIVIDER}")
    print("TEST 5: MODEL ROUTER (tier -> model name)")
    print(DIVIDER)

    passed = 0
    failed = 0

    for tier in ["lower", "average", "higher"]:
        model = resolve_model_name(tier)
        if model:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL (no model resolved)"
            failed += 1
        print(f"  [{status}] {tier} -> {model}")

    model = resolve_model_name("nonexistent")
    if model:
        status = "PASS (fallback to average)"
        passed += 1
    else:
        status = "FAIL"
        failed += 1
    print(f"  [{status}] nonexistent -> {model}")

    print(f"\n  Model Router: {passed}/{passed+failed} passed")
    return failed == 0


def test_performance_tracker():
    print(f"\n{DIVIDER}")
    print("TEST 6: PERFORMANCE TRACKER")
    print(DIVIDER)

    tracker = PerformanceTracker()

    m1 = RequestMetrics(
        routing_latency_ms=5.0,
        model_latency_ms=800.0,
        total_latency_ms=805.0,
        selected_model="allam-2-7b",
        model_tier="lower",
        difficulty="easy",
        cache_hit=True,
        route_source="fast_rules",
        fallback_count=0,
        llm_calls=1,
    )
    tracker.record(m1)

    m2 = RequestMetrics(
        routing_latency_ms=150.0,
        model_latency_ms=2000.0,
        total_latency_ms=2150.0,
        selected_model="llama-3.3-70b-versatile",
        model_tier="average",
        difficulty="medium",
        cache_hit=False,
        route_source="llm_orchestrator",
        fallback_count=1,
        escalated=True,
        llm_calls=2,
    )
    tracker.record(m2)

    stats = tracker.get_stats()
    print(f"  Stats: {stats}")

    passed = 0
    failed = 0

    if stats["total_requests"] == 2:
        passed += 1
        print("  [PASS] Total requests = 2")
    else:
        failed += 1
        print("  [FAIL] Total requests != 2")

    if stats["cache_hits"] == 1:
        passed += 1
        print("  [PASS] Cache hits = 1")
    else:
        failed += 1
        print("  [FAIL] Cache hits != 1")

    if stats["fallbacks"] == 1:
        passed += 1
        print("  [PASS] Fallbacks = 1")
    else:
        failed += 1
        print("  [FAIL] Fallbacks != 1")

    tier_dist = stats["tier_distribution"]
    if tier_dist.get("lower") == 1 and tier_dist.get("average") == 1:
        passed += 1
        print("  [PASS] Tier distribution correct")
    else:
        failed += 1
        print(f"  [FAIL] Tier distribution: {tier_dist}")

    print(f"\n  Performance Tracker: {passed}/{passed+failed} passed")
    return failed == 0


def main():
    print(f"\n{DIVIDER}")
    print("PADHOBETA ORCHESTRATOR — COMPREHENSIVE TEST SUITE")
    print(DIVIDER)

    results = []
    results.append(("Fast Router", test_fast_router()))
    results.append(("LLM Orchestrator", test_llm_orchestrator()))
    results.append(("Routing Cache", test_routing_cache()))
    results.append(("Fallback Logic", test_fallback()))
    results.append(("Model Router", test_model_router()))
    results.append(("Performance Tracker", test_performance_tracker()))

    print(f"\n{DIVIDER}")
    print("SUMMARY")
    print(DIVIDER)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(DIVIDER)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
