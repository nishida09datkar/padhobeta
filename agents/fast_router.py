"""
Fast deterministic router — Stage 1 of two-stage routing.

Classifies obviously simple/hard queries using lightweight rules
without calling any LLM, saving latency and cost.
"""
import re
import logging

logger = logging.getLogger("fast_router")

SIMPLE_PATTERNS = [
    r"^what\s+(is|are|was|were)\s+(a|an|the)?\s*\w+(\s+\w+){0,2}\??$",
    r"^define\s+\w+(\s+\w+){0,2}\??$",
    r"^what\s+do\s+you\s+mean\s+by\s+\w+(\s+\w+){0,2}\??$",
    r"^who\s+(is|are|was|were)\s+\w+(\s+\w+){0,2}\??$",
    r"^when\s+(is|are|was|were)\s+\w+(\s+\w+){0,2}\??$",
    r"^where\s+(is|are|was|were)\s+\w+(\s+\w+){0,2}\??$",
    r"^list\s+(the\s+)?\w+(\s+\w+){0,2}\??$",
    r"^name\s+(the\s+)?\w+(\s+\w+){0,2}\??$",
    r"^give\s+(me\s+)?(the\s+)?definition\s+of\s+\w+(\s+\w+){0,2}\??$",
]

SIMPLE_KEYWORD_PHRASES = [
    "what is", "what are", "define ", "definition of",
    "what do you mean by", "give me the definition",
    "who invented", "when was", "where is",
    "list the", "name the",
]

COMPLEXITY_BOOSTERS = [
    "prove", "derive", "derivation", "optimize", "analyze",
    "explain why", "explain how", "compare and contrast",
    "analyze the", "race condition", "concurrent",
    "dynamic programming", "recurrence relation", "complexity analysis",
    "implement and", "write a program", "debug this",
    "multi-step", "step by step proof", "mathematical proof",
    "instantiate", "interconnected", "architectures",
    "why does", "why is", "justify", "reasoning behind",
]

MEDIUM_INDICATORS = [
    "explain", "with example", "with code", "implement",
    "write", "show me", "difference between", "compare",
    "solve", "calculate", "algorithm", "function",
    "how does", "how do", "code for", "program to",
    "debug", "fix this", "what happens when",
]


def _normalize_query(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r"[?.!,;:]+", "", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _word_count(query: str) -> int:
    return len(query.split())


def classify_fast(query: str) -> dict | None:
    normalized = _normalize_query(query)
    words = _word_count(normalized)
    wc = words

    for pattern in SIMPLE_PATTERNS:
        if re.match(pattern, normalized):
            logger.info("[FAST_ROUTER] Pattern match -> lower (query='%s')", normalized[:60])
            return {
                "difficulty": "easy",
                "reasoning_required": False,
                "technical_depth": "low",
                "recommended_model": "lower",
                "route_source": "fast_rules",
                "reason": f"Matched simple query pattern ({wc} words)",
            }

    has_simple_phrase = any(phrase in normalized for phrase in SIMPLE_KEYWORD_PHRASES)
    has_complex_booster = any(booster in normalized for booster in COMPLEXITY_BOOSTERS)
    has_medium_indicator = any(indicator in normalized for indicator in MEDIUM_INDICATORS)

    if has_simple_phrase and not has_complex_booster and not has_medium_indicator and wc <= 8:
        logger.info("[FAST_ROUTER] Simple keyword match -> lower (query='%s')", normalized[:60])
        return {
            "difficulty": "easy",
            "reasoning_required": False,
            "technical_depth": "low",
            "recommended_model": "lower",
            "route_source": "fast_rules",
            "reason": f"Simple keyword phrase with {wc} words, no complexity indicators",
        }

    if has_complex_booster:
        if any(h in normalized for h in [
            "prove", "derive", "derivation", "mathematical proof",
            "complexity analysis", "recurrence relation",
            "concurrent", "race condition",
        ]):
            logger.info("[FAST_ROUTER] Complex booster -> higher (query='%s')", normalized[:60])
            return {
                "difficulty": "hard",
                "reasoning_required": True,
                "technical_depth": "high",
                "recommended_model": "higher",
                "route_source": "fast_rules",
                "reason": f"Detected advanced complexity keywords ({wc} words)",
            }

    return None
