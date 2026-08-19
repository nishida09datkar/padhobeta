import json
from datetime import datetime
from typing import Optional
import session_db as db


class ContextDAG:
    def __init__(self):
        pass

    def create_session(self, session_id: str, document_id: Optional[str] = None) -> dict:
        return db.create_session(session_id, document_id)

    def add_interaction(self, session_id: str, question: str, answer: str) -> dict:
        db.add_message(session_id, "user", question)
        db.add_message(session_id, "assistant", answer)
        return {"session_id": session_id, "question": question, "answer": answer}

    def set_summary(self, session_id: str, summary: str):
        messages = db.get_messages(session_id)
        db.create_dag_node(session_id, summary, len(messages))

    def update_summary(self, session_id: str, summary: str):
        messages = db.get_messages(session_id)
        db.update_dag_node(session_id, summary, len(messages))

    def link_sessions(self, parent_session_id: str, child_session_id: str, edge_type: str = "context_flow"):
        db.add_dag_edge(parent_session_id, child_session_id, edge_type)

    def get_context_from_ancestors(self, session_id: str, max_depth: int = 3) -> list[dict]:
        ancestors = db.get_all_ancestor_summaries(session_id, max_depth)
        return ancestors

    def get_session_messages(self, session_id: str, limit: int = 50) -> list[dict]:
        return db.get_messages(session_id, limit)

    def get_session_summary(self, session_id: str) -> Optional[dict]:
        return db.get_dag_node(session_id)

    def build_context_string(self, session_id: str) -> str:
        parts = []

        ancestors = self.get_context_from_ancestors(session_id)
        if ancestors:
            parts.append("=== Previous Session Context ===")
            for a in reversed(ancestors):
                parts.append(f"\n[Session {a['session_id']}]: {a['summary']}")
            parts.append("")

        messages = db.get_recent_messages(session_id, limit=10)
        if messages:
            parts.append("=== Current Session Recent Messages ===")
            for m in messages:
                role = "User" if m["role"] == "user" else "Assistant"
                parts.append(f"[{role}]: {m['content'][:300]}")
            parts.append("")

        return "\n".join(parts)

    def get_all_sessions(self) -> list[dict]:
        return db.list_sessions()

    def get_dag_visualization(self) -> dict:
        return db.get_full_dag()

    def end_session(self, session_id: str):
        db.end_session(session_id)


dag = ContextDAG()
