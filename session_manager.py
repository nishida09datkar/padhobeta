import re
from groq import Groq
from config import settings
from context_dag import dag
import session_db as db

client = Groq(api_key=settings.GROQ_API_KEY)

SUMMARY_SYSTEM_PROMPT = """Summarize this educational chat conversation in 2-3 sentences. Capture the main topics, key questions, and concepts discussed. Reply with ONLY the summary text, nothing else.

/no_think"""

CONTEXT_AUGMENT_PROMPT = """You are Padhobeta, an AI educational tutor and study buddy.

You have access to context from previous conversation sessions. Use this context to:
1. Provide continuity in the learning journey
2. Reference previously discussed topics when relevant
3. Build upon earlier explanations
4. Avoid repeating information already covered

Previous session context:
{previous_context}

Current session messages:
{current_messages}

Rules:
- Answer questions based on document context AND previous session context
- Maintain conversation continuity
- Reference earlier discussions when they are relevant
- Be clear, concise, and student-friendly
- Do NOT include thinking or reasoning steps in your output
- Give the final answer directly"""


def _strip_thinking_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<思考>.*?</思考>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if not cleaned:
        for pattern in [r"<think>(.*?)</think>", r"<reasoning>(.*?)</reasoning>", r"<思考>(.*?)</思考>"]:
            match = re.search(pattern, text, flags=re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                break
    return cleaned


def generate_session_summary(session_id: str) -> str:
    messages = db.get_messages(session_id)
    if not messages:
        return "Empty session with no messages."

    conversation = []
    for m in messages[-10:]:
        role = "User" if m["role"] == "user" else "Assistant"
        content = m["content"][:500]
        conversation.append(f"[{role}]: {content}")

    conversation_text = "\n".join(conversation)

    try:
        response = client.chat.completions.create(
            model="allam-2-7b",
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation:\n\n{conversation_text}"},
            ],
            temperature=0.3,
            max_tokens=128,
        )

        summary = _strip_thinking_tags(response.choices[0].message.content.strip())
        return summary if summary else f"Discussion covering {len(messages)} messages."

    except Exception as e:
        return f"Session with {len(messages)} messages discussing educational content."


def canary_summary(session_id: str) -> dict:
    summary = generate_session_summary(session_id)
    existing = db.get_dag_node(session_id)
    if existing:
        dag.update_summary(session_id, summary)
    else:
        dag.set_summary(session_id, summary)

    messages = db.get_messages(session_id)
    return {
        "session_id": session_id,
        "summary": summary,
        "message_count": len(messages),
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
    }


def create_new_session(document_id: str = None) -> dict:
    import uuid
    session_id = str(uuid.uuid4())[:12]
    dag.create_session(session_id, document_id)
    return {"session_id": session_id, "document_id": document_id}


def link_to_previous_session(child_session_id: str, parent_session_id: str):
    dag.link_sessions(parent_session_id, child_session_id, "context_flow")


def get_cross_session_context(session_id: str) -> str:
    return dag.build_context_string(session_id)


def get_session_history(session_id: str) -> list[dict]:
    return dag.get_session_messages(session_id)


def list_all_sessions() -> list[dict]:
    return dag.get_all_sessions()


def get_dag_structure() -> dict:
    return dag.get_dag_visualization()


def end_session(session_id: str) -> dict:
    summary_result = canary_summary(session_id)
    dag.end_session(session_id)
    return summary_result
