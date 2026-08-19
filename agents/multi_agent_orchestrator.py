import time
import logging
from agents.researcher_agent import research as doc_research
from agents.web_agent import research as web_research
from agents.synthesizer_agent import synthesize

logger = logging.getLogger("multi_orchestrator")


def run(
    store,
    query: str,
    doc_id: str | None = None,
    model: str | None = None,
) -> dict:
    """
    Multi-Agent Orchestrator — coordinates Researcher, Web, and Synthesizer agents.

    Flow:
    1. Launch Researcher Agent (document) and Web Agent (internet)
    2. Evaluate results from both
    3. Send to Synthesizer Agent for final answer with source attribution
    4. Return answer + explicit source info + injection report
    """
    total_start = time.time()
    agents_used = []
    timing = {}
    total_injections_blocked = 0

    # Stage 1: Researcher Agent
    doc_start = time.time()
    researcher_result = doc_research(store, query, doc_id)
    timing["researcher_ms"] = (time.time() - doc_start) * 1000

    doc_has = researcher_result.get("has_answer", False)
    doc_conf = researcher_result.get("confidence", 0.0)
    doc_injections = researcher_result.get("injections_blocked", 0)
    total_injections_blocked += doc_injections
    if doc_has:
        agents_used.append("Researcher Agent")
    logger.info(
        "[ORCHESTRATOR] Researcher done: has_answer=%s confidence=%.2f injections=%d (%.0fms)",
        doc_has, doc_conf, doc_injections, timing["researcher_ms"],
    )

    # Stage 2: Web Agent
    web_start = time.time()
    web_result = web_research(query)
    timing["web_ms"] = (time.time() - web_start) * 1000

    web_has = web_result.get("has_answer", False)
    web_conf = web_result.get("confidence", 0.0)
    web_injections = web_result.get("injections_blocked", 0)
    total_injections_blocked += web_injections
    if web_has:
        agents_used.append("Web Agent")
    logger.info(
        "[ORCHESTRATOR] Web Agent done: has_answer=%s confidence=%.2f injections=%d (%.0fms)",
        web_has, web_conf, web_injections, timing["web_ms"],
    )

    # Stage 3: Synthesize
    synth_start = time.time()
    answer, confidence = synthesize(
        query=query,
        researcher_result=researcher_result,
        web_result=web_result,
        model=model,
    )
    timing["synthesizer_ms"] = (time.time() - synth_start) * 1000
    agents_used.append("Synthesizer Agent")
    logger.info(
        "[ORCHESTRATOR] Synthesizer done: confidence=%.2f (%.0fms)",
        confidence, timing["synthesizer_ms"],
    )

    # Build source attribution
    sources_used = []

    if doc_has and doc_conf >= 0.3:
        for src in researcher_result.get("doc_sources", []):
            sources_used.append({
                "type": "document",
                "name": src["filename"],
                "location": f"Page/Section: {src['page_hint']}" if src.get("page_hint") else "Document content",
                "relevance_score": round(src["score"], 3),
            })

    if web_has and web_conf >= 0.3:
        for src in web_result.get("web_sources", []):
            sources_used.append({
                "type": "web",
                "name": src["name"],
                "url": src["url"],
            })

    total_ms = (time.time() - total_start) * 1000
    timing["total_ms"] = total_ms

    logger.info(
        "[ORCHESTRATOR] Complete: agents=%s sources=%d injections_blocked=%d total=%.0fms",
        agents_used, len(sources_used), total_injections_blocked, total_ms,
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "sources_used": sources_used,
        "agents_used": agents_used,
        "timing": timing,
        "injections_blocked": total_injections_blocked,
        "doc_agent": {
            "has_answer": doc_has,
            "confidence": doc_conf,
            "summary": researcher_result.get("summary", ""),
            "injections_blocked": doc_injections,
        },
        "web_agent": {
            "has_answer": web_has,
            "confidence": web_conf,
            "summary": web_result.get("summary", ""),
            "injections_blocked": web_injections,
        },
    }
