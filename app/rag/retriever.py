from __future__ import annotations

from app.rag.embedding import EmbeddingAdapter
from collections.abc import Sequence

from app.rag.models import RagSourceType, RetrievalResult, SchemaCandidate
from app.rag.vector_store import LanceVectorStore


class SchemaRetriever:
    def __init__(self, embedding: EmbeddingAdapter, vector_store: LanceVectorStore, rrf_k: int = 60, min_hit_score: float = 0.2, ambiguity_margin: float = 0.001) -> None:
        self.embedding = embedding
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.min_hit_score = min_hit_score
        self.ambiguity_margin = ambiguity_margin

    def search(
        self,
        query: str,
        namespace: str,
        limit: int,
        sources: Sequence[RagSourceType] | None = None,
    ) -> RetrievalResult:
        if not query.strip() or limit < 1:
            return RetrievalResult(status="empty")
        return self.search_with_vector(self.embed_query(query), namespace, limit, sources=sources)

    def embed_query(self, query: str) -> list[float]:
        return self.embedding.embed([query])[0]

    def search_with_vector(
        self,
        vector: list[float],
        namespace: str,
        limit: int,
        sources: Sequence[RagSourceType] | None = None,
    ) -> RetrievalResult:
        if limit < 1:
            return RetrievalResult(status="empty")
        active_sources = tuple(sources) if sources is not None else (RagSourceType.DDL, RagSourceType.SAMPLE_VALUE)
        hits = [hit for source in active_sources for hit in self.vector_store.search(source, vector, namespace, limit)]
        hits = [hit for hit in hits if hit.score >= self.min_hit_score]
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
        if len(candidates) > 1 and candidates[0].score - candidates[1].score < self.ambiguity_margin:
            return RetrievalResult(status="ambiguous", candidates=candidates)
        return RetrievalResult(status="ok", candidates=candidates)
