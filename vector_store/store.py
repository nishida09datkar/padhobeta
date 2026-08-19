import numpy as np
import logging

logger = logging.getLogger("vector_store")


def _lazy_imports():
    import faiss
    from fastembed import TextEmbedding
    return faiss, TextEmbedding


# Model dimension mapping
MODEL_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class DocumentStore:
    def __init__(self):
        from config import settings
        faiss, TextEmbedding = _lazy_imports()

        model_name = settings.EMBEDDING_MODEL
        if model_name == "all-MiniLM-L6-v2":
            model_name = "BAAI/bge-small-en-v1.5"

        logger.info("[STORE] Loading embedding model: %s", model_name)
        self.model = TextEmbedding(model_name)
        self.faiss = faiss
        dim = MODEL_DIMS.get(model_name, 384)
        self.index = faiss.IndexFlatIP(dim)
        self.documents: dict[str, dict] = {}
        self.chunk_metadata: list[dict] = []
        self._id_counter = 0
        logger.info("[STORE] Ready (dim=%d)", dim)

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings, dtype=np.float32)

    def add_document(self, doc_id: str, chunks: list[dict], filename: str, doc_type: str):
        texts = [c["content"] for c in chunks]
        if not texts:
            return

        embeddings = self._encode(texts)
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            global_id = self._id_counter
            self._id_counter += 1
            self.chunk_metadata.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "page_hint": chunk.get("page_hint", ""),
                "filename": filename,
                "doc_type": doc_type,
                "content": chunk["content"],
            })

        self.documents[doc_id] = {
            "filename": filename,
            "doc_type": doc_type,
            "chunk_count": len(chunks),
            "metadata": chunks[0] if chunks else {},
        }

    def search(self, query: str, top_k: int | None = None, doc_id: str | None = None) -> list[dict]:
        from config import settings
        if top_k is None:
            top_k = settings.TOP_K_CHUNKS

        if self.index.ntotal == 0:
            return []

        query_embedding = self._encode([query])

        search_k = min(top_k * 3, self.index.ntotal) if doc_id else min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.chunk_metadata[idx]
            if doc_id and meta["doc_id"] != doc_id:
                continue
            results.append({
                "content": meta["content"],
                "score": float(score),
                "filename": meta["filename"],
                "page_hint": meta["page_hint"],
                "doc_id": meta["doc_id"],
            })
            if len(results) >= top_k:
                break

        return results

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False

        chunks_to_remove = [
            i for i, m in enumerate(self.chunk_metadata) if m["doc_id"] == doc_id
        ]

        if not chunks_to_remove:
            return False

        remaining = [
            m for i, m in enumerate(self.chunk_metadata) if i not in set(chunks_to_remove)
        ]

        self.index.reset()
        self.chunk_metadata = []
        self._id_counter = 0

        if remaining:
            texts = [m["content"] for m in remaining]
            embeddings = self._encode(texts)
            self.index.add(embeddings)
            self.chunk_metadata = remaining
            self._id_counter = len(remaining)

        del self.documents[doc_id]
        return True

    def get_document_info(self, doc_id: str) -> dict | None:
        return self.documents.get(doc_id)

    def list_documents(self) -> list[dict]:
        result = []
        for doc_id, info in self.documents.items():
            result.append({
                "id": doc_id,
                "filename": info["filename"],
                "doc_type": info["doc_type"],
                "chunk_count": info["chunk_count"],
            })
        return result
