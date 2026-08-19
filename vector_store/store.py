import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import settings


class DocumentStore:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index = faiss.IndexFlatIP(self.model.get_embedding_dimension())
        self.documents: dict[str, dict] = {}
        self.chunk_metadata: list[dict] = []
        self._id_counter = 0

    def add_document(self, doc_id: str, chunks: list[dict], filename: str, doc_type: str):
        texts = [c["content"] for c in chunks]
        if not texts:
            return

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

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
        if top_k is None:
            top_k = settings.TOP_K_CHUNKS

        if self.index.ntotal == 0:
            return []

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

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
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)
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
