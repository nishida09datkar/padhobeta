from config import settings


def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[dict]:
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""
    current_page_hint = ""

    for para in paragraphs:
        if para.startswith("--- Slide") or para.startswith("<!-- Page"):
            current_page_hint = para

        if len(current_chunk) + len(para) + 2 > chunk_size:
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "page_hint": current_page_hint,
                })

            overlap_text = ""
            if chunk_overlap > 0 and current_chunk:
                words = current_chunk.split()
                overlap_words = words[-chunk_overlap // 5:] if len(words) > chunk_overlap // 5 else words
                overlap_text = " ".join(overlap_words) + "\n\n"

            current_chunk = overlap_text + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "page_hint": current_page_hint,
        })

    if not chunks and text.strip():
        words = text.split()
        for i in range(0, len(words), chunk_size // 5):
            chunk_words = words[i:i + chunk_size // 5]
            chunks.append({
                "content": " ".join(chunk_words),
                "page_hint": "",
            })

    return chunks
