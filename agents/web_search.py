import logging
from ddgs import DDGS

logger = logging.getLogger("web_search")

WEB_SEARCH_SYSTEM_PROMPT = """You are Padhobeta, an AI educational tutor and study buddy.

Rules:
1. Answer the student's question using ONLY the web search results provided below.
2. Be clear, concise, and student-friendly.
3. Cite sources inline using [Source: Title] notation where relevant.
4. Do NOT add a "Web Sources Used" section or list URLs at the end. Sources are tracked separately.
5. If the web results don't contain enough info, say so clearly.
6. Do NOT include thinking or reasoning steps. Give the final answer directly."""


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo. Returns list of {title, url, snippet}."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", "Untitled"),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                })
            logger.info("[WEB_SEARCH] query='%s' results=%d", query[:60], len(formatted))
            return formatted
    except Exception as e:
        logger.error("[WEB_SEARCH] Search failed: %s", e)
        return []


def format_web_results(results: list[dict]) -> str:
    """Format web search results into context string for LLM."""
    if not results:
        return "No web results found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"--- Web Result {i}: {r['title']} ---\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['snippet']}"
        )
    return "\n\n".join(parts)


def format_web_sources(results: list[dict]) -> list[dict]:
    """Format web results into source list for API response."""
    return [
        {"name": r["title"], "url": r["url"]}
        for r in results if r.get("url")
    ]


def generate_web_response(
    query: str,
    web_results: list[dict],
    model: str,
    client,
) -> tuple[str, list[dict]]:
    """Generate response using web search results."""
    web_context = format_web_results(web_results)
    sources = format_web_sources(web_results)

    user_message = f"""Web Search Results:
{web_context}

---
User Question: {query}

Answer the question using the web results above. Cite sources inline where relevant.
Do NOT add a sources list or URLs section at the end."""

    messages = [
        {"role": "system", "content": WEB_SEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
        )
        answer = response.choices[0].message.content.strip()
        return answer, sources
    except Exception as e:
        logger.error("[WEB_SEARCH] Response generation failed: %s", e)
        return (
            "I searched the web but encountered an error generating the response. "
            "Please try rephrasing your question.",
            [],
        )
