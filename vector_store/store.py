import numpy as np
import faiss
import logging
from groq import Groq
from config import settings

logger = logging.getLogger("vector_store")


class DocumentStore:
    """Embedding store using Groq API embeddings (no local model needed)."""

    # Groq embedding dimensions for known models
    EMBED_DIM = 384  # nomic-embed-text-v1.5 returns 768, but we use a subset
    # We'll use a simple char n-gram fallback if API fails

    def __init__(self):
        self.client = None  # Lazy init
        self.index = None
        self.documents: dict[str, dict] = {}
        self.chunk_metadata: list[dict] = []
        self._id_counter = 0
        self._dim = None
        self._use_fallback = False

    def _get_client(self):
        if self.client is None:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
        return self.client

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Get embeddings via Groq API, fall back to TF-IDF if unavailable."""
        try:
            client = self._get_client()
            result = client.embeddings.create(
                model="nomic-embed-text-v1.5",
                input=texts,
            )
            embeddings = [item.embedding for item in result.data]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.warning("[STORE] Groq embedding failed (%s), using TF-IDF fallback", str(e)[:60])
            return self._tfidf_embed(texts)

    def _tfidf_embed(self, texts: list[str]) -> np.ndarray:
        """Simple TF-IDF fallback — no external API or model needed."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not hasattr(self, '_vectorizer') or self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                max_features=512,
                analyzer="char_wb",
                ngram_range=(2, 4),
            )
            # Fit on stored documents + new texts
            all_texts = [m["content"] for m in self.chunk_metadata] + texts
            if len(all_texts) < 2:
                all_texts = texts + ["placeholder"]
            self._vectorizer.fit(all_texts)

        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)

    def _ensure_index(self, dim: int):
        if self.index is None or self._dim != dim:
            self.index = faiss.IndexFlatIP(dim)
            self._dim = dim

    def add_document(self, doc_id: str, chunks: list[dict], filename: str, doc_type: str):
        texts = [c["content"] for c in chunks]
        if not texts:
            return

        embeddings = self._embed(texts)
        self._ensure_index(embeddings.shape[1])
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
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
        if top_k is None:
            top_k = settings.TOP_K_CHUNKS

        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self._embed([query])
        faiss.normalize_L2(query_embedding)

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
            embeddings = self._embed(texts)
            faiss.normalize_L2(embeddings)
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
