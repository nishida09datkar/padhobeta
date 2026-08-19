from datetime import datetime
from typing import Optional
from supabase import create_client, Client
from config import settings

_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _client


# --- Session CRUD ---

def create_session(session_id: str, document_id: Optional[str] = None) -> dict:
    client = get_client()
    now = datetime.utcnow().isoformat()
    client.table("sessions").insert({
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "document_id": document_id,
        "is_active": True,
    }).execute()
    return {"session_id": session_id, "created_at": now, "document_id": document_id}


def get_session(session_id: str) -> Optional[dict]:
    client = get_client()
    result = client.table("sessions").select("*").eq("session_id", session_id).execute()
    if result.data:
        return result.data[0]
    return None


def update_session(session_id: str):
    client = get_client()
    client.table("sessions").update({"updated_at": datetime.utcnow().isoformat()}).eq("session_id", session_id).execute()


def end_session(session_id: str):
    client = get_client()
    client.table("sessions").update({
        "is_active": False,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("session_id", session_id).execute()


def list_sessions(active_only: bool = False) -> list[dict]:
    client = get_client()
    query = client.table("sessions").select("*")
    if active_only:
        query = query.eq("is_active", True)
    result = query.order("created_at", desc=True).execute()
    return result.data or []


# --- Messages ---

def add_message(session_id: str, role: str, content: str) -> int:
    client = get_client()
    now = datetime.utcnow().isoformat()
    result = client.table("messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": now,
    }).execute()
    update_session(session_id)
    return result.data[0]["id"] if result.data else 0


def get_messages(session_id: str, limit: int = 50) -> list[dict]:
    client = get_client()
    result = (
        client.table("messages")
        .select("*")
        .eq("session_id", session_id)
        .order("timestamp", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_recent_messages(session_id: str, limit: int = 10) -> list[dict]:
    client = get_client()
    result = (
        client.table("messages")
        .select("*")
        .eq("session_id", session_id)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    messages = result.data or []
    messages.reverse()
    return messages


# --- DAG Nodes ---

def create_dag_node(session_id: str, summary: str, message_count: int = 0) -> dict:
    client = get_client()
    now = datetime.utcnow().isoformat()
    existing = get_dag_node(session_id)
    if existing:
        client.table("dag_nodes").update({
            "summary": summary,
            "message_count": message_count,
        }).eq("session_id", session_id).execute()
    else:
        client.table("dag_nodes").insert({
            "session_id": session_id,
            "summary": summary,
            "message_count": message_count,
            "created_at": now,
        }).execute()
    return {"session_id": session_id, "summary": summary, "message_count": message_count}


def update_dag_node(session_id: str, summary: str, message_count: int):
    client = get_client()
    client.table("dag_nodes").update({
        "summary": summary,
        "message_count": message_count,
    }).eq("session_id", session_id).execute()


def get_dag_node(session_id: str) -> Optional[dict]:
    client = get_client()
    result = client.table("dag_nodes").select("*").eq("session_id", session_id).execute()
    if result.data:
        return result.data[0]
    return None


# --- DAG Edges ---

def add_dag_edge(parent_session_id: str, child_session_id: str, edge_type: str = "context_flow"):
    client = get_client()
    now = datetime.utcnow().isoformat()
    client.table("dag_edges").upsert({
        "parent_session_id": parent_session_id,
        "child_session_id": child_session_id,
        "edge_type": edge_type,
        "created_at": now,
    }, on_conflict="parent_session_id,child_session_id").execute()


def get_parents(session_id: str) -> list[dict]:
    client = get_client()
    result = (
        client.table("dag_edges")
        .select("parent_session_id")
        .eq("child_session_id", session_id)
        .execute()
    )
    if not result.data:
        return []
    parent_ids = [e["parent_session_id"] for e in result.data]
    parents = []
    for pid in parent_ids:
        node = get_dag_node(pid)
        if node:
            parents.append(node)
    return parents


def get_children(session_id: str) -> list[dict]:
    client = get_client()
    fallback = (
        client.table("dag_edges")
        .select("child_session_id")
        .eq("parent_session_id", session_id)
        .execute()
    )
    if not fallback.data:
        return []
    child_ids = [e["child_session_id"] for e in fallback.data]
    children = []
    for cid in child_ids:
        node = get_dag_node(cid)
        if node:
            children.append(node)
    return children


def get_all_ancestor_summaries(session_id: str, max_depth: int = 10) -> list[dict]:
    visited = set()
    result = []
    queue = [session_id]
    depth = 0

    while queue and depth < max_depth:
        next_queue = []
        for sid in queue:
            if sid in visited:
                continue
            visited.add(sid)
            parents = get_parents(sid)
            for p in parents:
                result.append(p)
                next_queue.append(p["session_id"])
        queue = next_queue
        depth += 1

    return result


def get_full_dag() -> dict:
    client = get_client()
    nodes_result = client.table("dag_nodes").select("*").order("created_at", desc=False).execute()
    edges_result = client.table("dag_edges").select("*").order("created_at", desc=False).execute()
    return {
        "nodes": nodes_result.data or [],
        "edges": edges_result.data or [],
    }
