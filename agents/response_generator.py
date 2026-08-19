import re
import logging
from groq import Groq
from config import settings

logger = logging.getLogger("response_generator")

client = Groq(api_key=settings.GROQ_API_KEY)

RESPONSE_SYSTEM_PROMPT = """You are Padhobeta, an AI educational tutor and study buddy.

Rules:
1. Answer questions ONLY based on the provided document context.
2. If the context doesn't contain enough information, say so clearly.
3. Be clear, concise, and student-friendly in your explanations.
4. Use examples from the document when available.
5. Format your answers nicely with proper structure.
6. Cite sources using [Source: filename] notation when referencing specific parts.
7. If a concept is complex, break it down into simpler parts.
8. End with a brief summary or key takeaway when appropriate.

You are helpful, encouraging, and focused on making learning easy. Do NOT include thinking or reasoning steps in your output. Just give the final answer directly."""


def _strip_thinking_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<思考>.*?</思考>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _is_truncated(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    if text.endswith((".", "!", "?")):
        return False
    if text.endswith((":", ",", ";", "-", "–", "—")):
        return True
    last_line = text.split("\n")[-1].strip()
    if last_line and not last_line.endswith((".", "!", "?", ")", "]", "}", "`")):
        if len(last_line) > 20:
            return True
    return False


def _generate_single(
    messages: list[dict],
    model: str,
    max_tokens: int = 4096,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def generate_response(
    query: str,
    context: str,
    sources: list[dict],
    model: str | None = None,
) -> tuple[str, float]:
    user_message = f"""Document Context:
{context}

---
User Question: {query}

Please answer the question based on the document context above. If the context doesn't contain relevant information, mention that clearly. Do NOT include thinking steps. Give the final answer directly."""

    use_model = model or settings.LLM_MODEL
    messages = [
        {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        raw_answer = _generate_single(messages, use_model, max_tokens=4096)
        answer = _strip_thinking_tags(raw_answer)

        if _is_truncated(answer):
            logger.info("[RESPONSE] Response truncated, requesting continuation (len=%d)", len(answer))
            continuation_messages = messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Your response was cut off. Please continue from where you stopped. Do not repeat what you already said."},
            ]
            try:
                continuation = _generate_single(continuation_messages, use_model, max_tokens=4096)
                continuation = _strip_thinking_tags(continuation)
                if continuation and not continuation.lower().startswith(("sorry", "i already", "as i mentioned")):
                    answer = answer.rstrip() + "\n\n" + continuation
            except Exception as e:
                logger.warning("[RESPONSE] Continuation failed: %s", e)

        if not answer:
            answer = "I apologize, but I couldn't generate a response. Please try rephrasing your question."

        confidence = 0.9
        answer_lower = answer.lower()
        low_confidence_phrases = [
            "not found", "not available", "there is no information",
            "i cannot answer", "i can't answer", "i do not have",
            "i don't have", "no information about", "cannot be determined",
            "not mentioned", "not provided", "no relevant",
            "does not contain", "doesn't contain", "not enough information",
            "based on the provided document context, there is no",
            "i cannot provide", "unable to answer", "not present in",
            "the document does not", "context does not",
        ]
        if any(phrase in answer_lower for phrase in low_confidence_phrases):
            confidence = 0.3
        elif "based on the" in answer_lower or "according to" in answer_lower:
            confidence = 0.95

        return answer, confidence

    except Exception as e:
        return f"Sorry, I encountered an error while generating the response: {str(e)}", 0.0


def generate_rejection_response() -> tuple[str, float]:
    return (
        "Hey! I appreciate the message, but I'm Padhobeta — your study buddy! 🎓\n\n"
        "I'm best at helping with academic and educational questions. "
        "You can:\n"
        "• Upload a document and ask me anything about it\n"
        "• Ask me about concepts, definitions, or explanations\n"
        "• I can also search the web if your document doesn't have the answer!\n\n"
        "What would you like to learn today?",
        1.0,
    )
