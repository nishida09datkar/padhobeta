from vector_store.store import DocumentStore


def retrieve_context(
    store: DocumentStore,
    query: str,
    doc_id: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    results = store.search(query, top_k=top_k, doc_id=doc_id)
    return results


def format_context(results: list[dict]) -> str:
    if not results:
        return "No relevant document context found."

    parts = []
    for i, r in enumerate(results, 1):
        source = r["filename"]
        page = r.get("page_hint", "")
        source_ref = f"[Source: {source}"
        if page:
            source_ref += f" | {page}"
        source_ref += "]"

        parts.append(f"--- Chunk {i} {source_ref} ---\n{r['content']}")

    return "\n\n".join(parts)
