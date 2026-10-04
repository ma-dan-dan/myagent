from __future__ import annotations

from app.rag.embedding import EmbeddingAdapter
from app.rag.models import RagSourceType, RetrievalResult, SchemaCandidate
from app.rag.vector_store import LanceVectorStore


class SchemaRetriever:
    def __init__(self, embedding: EmbeddingAdapter, vector_store: LanceVectorStore, rrf_k: int = 60) -> None:
        self.embedding = embedding
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def search(self, query: str, namespace: str, limit: int) -> RetrievalResult:
        vector = self.embedding.embed([query])[0]
        hits = [
            *self.vector_store.search(RagSourceType.DDL, vector, namespace, limit),
            *self.vector_store.search(RagSourceType.SAMPLE_VALUE, vector, namespace, limit),
        ]
        grouped: dict[str, dict] = {}
        for hit in hits:
            item = grouped.setdefault(hit.document.table_id, {"table_name": hit.document.table_name, "ddl": "", "columns": {}, "sources": [], "score": 0.0})
            item["score"] += 1 / (self.rrf_k + hit.rank)
            if hit.document.source_type not in item["sources"]:
                item["sources"].append(hit.document.source_type)
            if hit.document.source_type is RagSourceType.DDL:
                item["ddl"] = hit.document.text
            elif hit.document.column_name:
                item["columns"].setdefault(hit.document.column_name, []).append(hit.document.text)
        candidates = [
            SchemaCandidate(table_id=table_id, table_name=item["table_name"], ddl=item["ddl"], matched_columns=sorted(item["columns"]), sample_values=item["columns"], score=item["score"], matched_by=item["sources"])
            for table_id, item in grouped.items()
        ]
        candidates.sort(key=lambda value: (-value.score, value.table_id))
        candidates = candidates[:limit]
        if not candidates:
            return RetrievalResult(status="empty")
        if len(candidates) > 1 and candidates[0].score - candidates[1].score < 0.001:
            return RetrievalResult(status="ambiguous", candidates=candidates)
        return RetrievalResult(status="ok", candidates=candidates)
