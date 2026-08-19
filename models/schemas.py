from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class ChatRequest(BaseModel):
    query: str
    document_id: str | None = None
    session_id: str | None = None
    parent_session_id: str | None = None


class WebSource(BaseModel):
    name: str
    url: str


class SourceAttribution(BaseModel):
    type: str  # "document" or "web"
    name: str
    url: str | None = None
    location: str | None = None
    relevance_score: float | None = None


class AgentReport(BaseModel):
    has_answer: bool
    confidence: float
    summary: str
    injections_blocked: int = 0


class ChatResponse(BaseModel):
    answer: str
    is_educational: bool
    sources: list[str]
    confidence: float
    session_id: str
    session_summary: str
    model_used: str | None = None
    difficulty: str | None = None
    routing_reason: str | None = None
    performance: dict | None = None
    from_web: bool = False
    web_sources: list[WebSource] = []
    # Multiagent fields
    sources_used: list[SourceAttribution] = []
    agents_used: list[str] = []
    doc_agent: AgentReport | None = None
    web_agent: AgentReport | None = None
    injections_blocked: int = 0


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    doc_type: str
    chunk_count: int
    page_count: int


class DocumentInfo(BaseModel):
    id: str
    filename: str
    doc_type: str
    chunk_count: int
    uploaded_at: str


class DocumentsList(BaseModel):
    documents: list[DocumentInfo]


class ParsedDocument(BaseModel):
    content: str
    title: str | None = None
    doc_type: str
    page_count: int = 1
    metadata: dict = {}


class HealthResponse(BaseModel):
    status: str
    message: str


class SessionCreateRequest(BaseModel):
    document_id: str | None = None
    parent_session_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    document_id: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[dict]


class SessionSummaryResponse(BaseModel):
    session_id: str
    summary: str
    message_count: int
    timestamp: str


class DAGStructureResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


class CrossSessionContextResponse(BaseModel):
    session_id: str
    context_string: str
    ancestor_count: int
