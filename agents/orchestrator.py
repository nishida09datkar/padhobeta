"""
Two-stage orchestrator with fast routing, LLM classification, and caching.

Stage 1: Fast deterministic rule-based router (no LLM call)
Stage 2: LLM orchestrator for ambiguous queries (only when needed)

Both stages use routing cache to avoid repeated work.
"""
import re
import json
import time
import logging
from groq import Groq
from config import settings
from agents.fast_router import classify_fast
from agents.routing_cache import routing_cache

logger = logging.getLogger("orchestrator")

client = Groq(api_key=settings.GROQ_API_KEY)

ORCHESTRATOR_MODEL = settings.ORCHESTRATOR_MODEL

MAX_ESCALATIONS = settings.MAX_MODEL_FALLBACKS

MODEL_FALLBACK_CHAIN = ["lower", "average", "higher"]

ORCHESTRATOR_SYSTEM_PROMPT = """You are a routing model for an education-focused AI assistant.

Your ONLY task is to classify the user's educational query and select the appropriate model tier.

Available tiers:
- lower: Simple definitions, basic concepts, straightforward explanations, simple calculations, beginner programming concepts
- average: Moderate explanations, normal DSA, programming questions, moderate mathematics, debugging, comparisons, multi-step but manageable academic questions
- higher: Advanced reasoning, proofs, difficult mathematics, advanced algorithms, difficult debugging, complex multi-part technical questions

Evaluate: reasoning difficulty, technical depth, number of steps, mathematical complexity, programming complexity, ambiguity, number of subtasks.

Do not classify based only on length. Do not answer the question.

Return ONLY valid JSON:
{"difficulty":"easy|medium|hard","reasoning_required":true|false,"technical_depth":"low|medium|high","recommended_model":"lower|average|higher"}"""


def _llm_classify(query: str, conversation_context: str = "") -> dict:
    start = time.time()

    user_content = query
    if conversation_context:
        user_content = f"Previous conversation context:\n{conversation_context}\n\nCurrent user query: {query}"

    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        decision = json.loads(cleaned)

        valid_difficulties = {"easy", "medium", "hard"}
        valid_models = {"lower", "average", "higher"}
        valid_depths = {"low", "medium", "high"}

        if decision.get("difficulty") not in valid_difficulties:
            decision["difficulty"] = "medium"
        if decision.get("recommended_model") not in valid_models:
            decision["recommended_model"] = "average"
        if decision.get("technical_depth") not in valid_depths:
            decision["technical_depth"] = "medium"
        if "reasoning_required" not in decision:
            decision["reasoning_required"] = decision["difficulty"] != "easy"

        elapsed = (time.time() - start) * 1000
        logger.info(
            "[LLM_ORCHESTRATOR] difficulty=%s model=%s time=%.0fms",
            decision["difficulty"], decision["recommended_model"], elapsed,
        )
        return decision

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("[LLM_ORCHESTRATOR] Parse error, defaulting to average: %s", e)
        return {
            "difficulty": "medium",
            "reasoning_required": True,
            "technical_depth": "medium",
            "recommended_model": "average",
        }

    except Exception as e:
        logger.error("[LLM_ORCHESTRATOR] Error: %s", e)
        return {
            "difficulty": "medium",
            "reasoning_required": True,
            "technical_depth": "medium",
            "recommended_model": "average",
        }


def classify_query_complexity(query: str, conversation_context: str = "") -> dict:
    cached = routing_cache.get(query)
    if cached:
        cached["route_source"] = "cache"
        return cached

    fast_result = classify_fast(query)
    if fast_result:
        routing_cache.put(query, fast_result)
        return fast_result

    llm_result = _llm_classify(query, conversation_context)
    llm_result["route_source"] = "llm_orchestrator"
    routing_cache.put(query, llm_result)
    return llm_result


def should_escalate(query: str, answer: str, confidence: float, current_tier: str) -> bool:
    if current_tier == "higher":
        return False

    if confidence < 0.3:
        return True

    error_indicators = [
        "sorry, i encountered an error",
        "i couldn't generate",
        "unable to process",
        "i'm not sure",
        "i don't have enough",
    ]
    answer_lower = answer.lower()
    if any(indicator in answer_lower for indicator in error_indicators):
        return True

    if len(answer.strip()) < 10:
        return True

    return False


def get_next_tier(current_tier: str) -> str | None:
    try:
        idx = MODEL_FALLBACK_CHAIN.index(current_tier)
        if idx < len(MODEL_FALLBACK_CHAIN) - 1:
            return MODEL_FALLBACK_CHAIN[idx + 1]
    except ValueError:
        pass
    return None
