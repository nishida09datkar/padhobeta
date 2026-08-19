import os
import uuid
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from models.schemas import (
    ChatRequest,
    ChatResponse,
    UploadResponse,
    DocumentInfo,
    DocumentsList,
    HealthResponse,
    SessionCreateRequest,
    SessionResponse,
    SessionListResponse,
    SessionMessagesResponse,
    SessionSummaryResponse,
    DAGStructureResponse,
    CrossSessionContextResponse,
)
from agents.document_parser import parse_document, detect_file_type
from agents.query_classifier import classify_query, detect_casual, get_casual_response
from agents.multi_agent_orchestrator import run as multiagent_run
from agents.response_generator import generate_rejection_response
from agents.performance import perf_tracker, RequestMetrics
from agents.routing_cache import routing_cache
from vector_store.store import DocumentStore
from session_manager import (
    create_new_session,
    link_to_previous_session,
    canary_summary,
    get_cross_session_context,
    get_session_history,
    list_all_sessions,
    get_dag_structure,
    end_session,
)
import session_db as db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("orchestrator")

_ai_ready = False


def _warmup_ai():
    global _ai_ready
    if not settings.AI_WARMUP_ENABLED:
        _ai_ready = True
        logger.info("[WARMUP] Skipped (disabled)")
        return

    logger.info("[WARMUP] Starting AI service warm-up...")
    start = time.time()

    try:
        from groq import Groq
        client = Groq(api_key=settings.GROQ_API_KEY)
        models_to_check = [
            settings.ORCHESTRATOR_MODEL,
            settings.LOWER_MODEL,
            settings.AVERAGE_MODEL,
            settings.HIGHER_MODEL,
        ]
        unique_models = list(dict.fromkeys(models_to_check))

        for model in unique_models:
            try:
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=5,
                    temperature=0.0,
                )
                logger.info("[WARMUP] Model '%s' OK", model)
            except Exception as e:
                logger.warning("[WARMUP] Model '%s' check failed: %s (will retry on first use)", model, str(e)[:80])

    except Exception as e:
        logger.warning("[WARMUP] Warm-up error: %s", e)

    elapsed = time.time() - start
    _ai_ready = True
    logger.info("[WARMUP] AI service ready (%.1fs)", elapsed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warmup_ai()
    yield


app = FastAPI(
    title="Padhobeta",
    description="AI Educational Chatbot — Upload academic documents and ask questions!",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = None


def get_store():
    global store
    if store is None:
        store = DocumentStore()
    return store


@app.get("/health", response_model=HealthResponse)
def health_check():
    status = "ok" if _ai_ready else "warming_up"
    message = "Padhobeta is running!" if _ai_ready else "Padhobeta is warming up..."
    return HealthResponse(status=status, message=message)


@app.get("/metrics")
def get_metrics():
    return {
        "performance": perf_tracker.get_stats(),
        "routing_cache": routing_cache.stats,
        "ai_ready": _ai_ready,
        "config": {
            "lower_model": settings.LOWER_MODEL,
            "average_model": settings.AVERAGE_MODEL,
            "higher_model": settings.HIGHER_MODEL,
            "orchestrator_model": settings.ORCHESTRATOR_MODEL,
            "cache_enabled": settings.ROUTING_CACHE_ENABLED,
            "warmup_enabled": settings.AI_WARMUP_ENABLED,
        },
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    doc_type = detect_file_type(file.filename)
    if not doc_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS.keys())}",
        )

    doc_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1]
    save_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}{ext}")

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB",
                )
            f.write(content)

        parsed, chunks = parse_document(save_path, file.filename)

        get_store().add_document(
            doc_id=doc_id,
            chunks=chunks,
            filename=file.filename,
            doc_type=doc_type,
        )

        return UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            doc_type=doc_type,
            chunk_count=len(chunks),
            page_count=parsed.page_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    total_start = time.time()

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_id = request.session_id
    if not session_id:
        session_result = create_new_session(request.document_id)
        session_id = session_result["session_id"]
        if request.parent_session_id:
            link_to_previous_session(session_id, request.parent_session_id)
    elif not db.get_session(session_id):
        session_result = create_new_session(request.document_id)
        session_id = session_result["session_id"]

    casual_category = detect_casual(request.query)
    if casual_category:
        answer = get_casual_response(casual_category)
        db.add_message(session_id, "user", request.query)
        db.add_message(session_id, "assistant", answer)
        summary_result = canary_summary(session_id)
        return ChatResponse(
            answer=answer,
            is_educational=True,
            sources=[],
            confidence=1.0,
            session_id=session_id,
            session_summary=summary_result["summary"],
        )

    is_educational, classifier_confidence = classify_query(request.query)

    if not is_educational:
        answer, confidence = generate_rejection_response()
        db.add_message(session_id, "user", request.query)
        db.add_message(session_id, "assistant", answer)
        summary_result = canary_summary(session_id)
        return ChatResponse(
            answer=answer,
            is_educational=False,
            sources=[],
            confidence=confidence,
            session_id=session_id,
            session_summary=summary_result["summary"],
        )

    metrics = RequestMetrics()

    logger.info("[CHAT] query='%s' launching multiagent system", request.query[:80])

    result = multiagent_run(
        store=get_store(),
        query=request.query,
        doc_id=request.document_id,
    )

    answer = result["answer"]
    confidence = result["confidence"]
    sources_used = result["sources_used"]
    agents_used = result["agents_used"]
    timing = result["timing"]

    metrics.total_latency_ms = timing.get("total_ms", 0)
    metrics.llm_calls = 3
    perf_tracker.record(metrics)

    source_strings = []
    for src in sources_used:
        if src["type"] == "document":
            s = src["name"]
            if src.get("location"):
                s += f" ({src['location']})"
            source_strings.append(s)
        elif src["type"] == "web":
            source_strings.append(f"{src['name']} [Web]")

    db.add_message(session_id, "user", request.query)
    db.add_message(session_id, "assistant", answer)
    summary_result = canary_summary(session_id)

    from_web = any(s["type"] == "web" for s in sources_used)
    web_sources = [
        {"name": s["name"], "url": s.get("url", "")}
        for s in sources_used if s["type"] == "web"
    ]

    doc_agent_report = result.get("doc_agent")
    web_agent_report = result.get("web_agent")

    response_kwargs = dict(
        answer=answer,
        is_educational=True,
        sources=source_strings,
        confidence=confidence,
        session_id=session_id,
        session_summary=summary_result["summary"],
        from_web=from_web,
        web_sources=web_sources,
        sources_used=sources_used,
        agents_used=agents_used,
        injections_blocked=result.get("injections_blocked", 0),
    )

    if doc_agent_report:
        response_kwargs["doc_agent"] = {
            "has_answer": doc_agent_report.get("has_answer", False),
            "confidence": doc_agent_report.get("confidence", 0.0),
            "summary": doc_agent_report.get("summary", ""),
            "injections_blocked": doc_agent_report.get("injections_blocked", 0),
        }
    if web_agent_report:
        response_kwargs["web_agent"] = {
            "has_answer": web_agent_report.get("has_answer", False),
            "confidence": web_agent_report.get("confidence", 0.0),
            "summary": web_agent_report.get("summary", ""),
            "injections_blocked": web_agent_report.get("injections_blocked", 0),
        }

    logger.info(
        "[CHAT] complete: agents=%s sources=%d confidence=%.2f total=%.0fms",
        agents_used, len(sources_used), confidence, timing.get("total_ms", 0),
    )

    return ChatResponse(**response_kwargs)


@app.post("/sessions", response_model=SessionResponse)
def create_session(request: SessionCreateRequest):
    result = create_new_session(request.document_id)
    if request.parent_session_id:
        link_to_previous_session(result["session_id"], request.parent_session_id)
    session = db.get_session(result["session_id"])
    return SessionResponse(
        session_id=session["session_id"],
        created_at=session["created_at"],
        document_id=session.get("document_id"),
    )


@app.get("/sessions", response_model=SessionListResponse)
def list_sessions():
    sessions = list_all_sessions()
    return SessionListResponse(
        sessions=[
            SessionResponse(
                session_id=s["session_id"],
                created_at=s["created_at"],
                document_id=s.get("document_id"),
            )
            for s in sessions
        ]
    )


@app.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
def get_session_messages(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = get_session_history(session_id)
    return SessionMessagesResponse(
        session_id=session_id,
        messages=messages,
    )


@app.post("/sessions/{session_id}/summary", response_model=SessionSummaryResponse)
def generate_summary(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    result = canary_summary(session_id)
    return SessionSummaryResponse(**result)


@app.post("/sessions/{session_id}/end")
def end_session_endpoint(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    result = end_session(session_id)
    return {"success": True, "message": "Session ended.", "final_summary": result["summary"]}


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    client = db.get_client()
    client.table("dag_edges").delete().or_(
        f"parent_session_id.eq.{session_id},child_session_id.eq.{session_id}"
    ).execute()
    client.table("dag_nodes").delete().eq("session_id", session_id).execute()
    client.table("messages").delete().eq("session_id", session_id).execute()
    client.table("sessions").delete().eq("session_id", session_id).execute()
    return {"success": True, "message": f"Session {session_id} deleted."}


@app.get("/sessions/{session_id}/context", response_model=CrossSessionContextResponse)
def get_context(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    context = get_cross_session_context(session_id)
    ancestors = db.get_all_ancestor_summaries(session_id)
    return CrossSessionContextResponse(
        session_id=session_id,
        context_string=context,
        ancestor_count=len(ancestors),
    )


@app.get("/dag", response_model=DAGStructureResponse)
def get_dag():
    structure = get_dag_structure()
    return DAGStructureResponse(**structure)


@app.get("/documents", response_model=DocumentsList)
def list_documents():
    docs = get_store().list_documents()
    document_list = [
        DocumentInfo(
            id=d["id"],
            filename=d["filename"],
            doc_type=d["doc_type"],
            chunk_count=d["chunk_count"],
            uploaded_at=datetime.now().isoformat(),
        )
        for d in docs
    ]
    return DocumentsList(documents=document_list)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    success = get_store().remove_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")

    for f in os.listdir(settings.UPLOAD_DIR):
        if f.startswith(doc_id):
            os.remove(os.path.join(settings.UPLOAD_DIR, f))

    return {"success": True, "message": f"Document {doc_id} deleted."}


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
