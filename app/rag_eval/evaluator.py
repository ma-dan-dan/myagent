from __future__ import annotations

import time
from collections import defaultdict
from typing import Protocol, Sequence

from app.rag.models import RagSourceType, RetrievalResult
from app.rag_eval.models import AggregateMetric, BirdCase, GoldTableSet, RetrievalCaseMetric


class EvaluationRetriever(Protocol):
    rrf_k: int
    min_hit_score: float
    ambiguity_margin: float

    def search(self, query: str, namespace: str, limit: int, sources: Sequence[RagSourceType] | None = None) -> RetrievalResult:
        """Return candidates from one or both RAG sources."""


class SchemaRagEvaluator:
    BRANCHES = {
        "ddl_only": (RagSourceType.DDL,),
        "sample_value_only": (RagSourceType.SAMPLE_VALUE,),
        "fusion_rrf": None,
    }

    def __init__(self, retriever: EvaluationRetriever) -> None:
        self.retriever = retriever

    def evaluate_case(
        self,
        case: BirdCase,
        gold: GoldTableSet,
        database_table_count: int,
        top_ks: Sequence[int],
    ) -> list[RetrievalCaseMetric]:
        if case.question_id != gold.question_id or case.db_id != gold.db_id:
            raise ValueError("评测题目与 Gold Tables 不匹配。")
        if database_table_count < 1:
            raise ValueError("database_table_count 必须大于 0。")
        if not top_ks or any(value < 1 for value in top_ks):
            raise ValueError("top_ks 必须包含正整数。")
        query_vector = self._embed_query(case.question)
        results: list[RetrievalCaseMetric] = []
        for branch, sources in self.BRANCHES.items():
            for top_k in sorted(set(top_ks)):
                started = time.perf_counter()
                retrieval = self._search(case.question, query_vector, case.db_id, top_k, sources)
                elapsed_ms = (time.perf_counter() - started) * 1000
                results.append(self._metric(case, gold, branch, top_k, retrieval, database_table_count, elapsed_ms))
        return results

    def aggregate(self, metrics: Sequence[RetrievalCaseMetric]) -> list[AggregateMetric]:
        grouped: dict[tuple[str, int], list[RetrievalCaseMetric]] = defaultdict(list)
        for metric in metrics:
            grouped[(metric.branch, metric.top_k)].append(metric)
        reports: list[AggregateMetric] = []
        for (branch, top_k), values in sorted(grouped.items()):
            count = len(values)
            matched_total = sum(len(item.matched_tables) for item in values)
            gold_total = sum(len(item.gold_tables) for item in values)
            reports.append(
                AggregateMetric(
                    branch=branch,
                    top_k=top_k,
                    case_count=count,
                    macro_recall=sum(item.recall for item in values) / count,
                    micro_recall=matched_total / gold_total,
                    full_recall_rate=sum(item.full_recall for item in values) / count,
                    macro_precision=sum(item.precision for item in values) / count,
                    empty_rate=sum(item.empty for item in values) / count,
                    ambiguous_rate=sum(item.ambiguous for item in values) / count,
                    average_false_positive_count=sum(item.false_positive_count for item in values) / count,
                    average_elapsed_ms=sum(item.elapsed_ms for item in values) / count,
                    average_database_coverage=sum(item.database_coverage for item in values) / count,
                )
            )
        return reports

    def aggregate_by_database(self, metrics: Sequence[RetrievalCaseMetric]) -> dict[str, list[AggregateMetric]]:
        grouped: dict[str, list[RetrievalCaseMetric]] = defaultdict(list)
        for metric in metrics:
            grouped[metric.db_id].append(metric)
        return {db_id: self.aggregate(values) for db_id, values in sorted(grouped.items())}

    def _embed_query(self, query: str):
        embed = getattr(self.retriever, "embed_query", None)
        return embed(query) if callable(embed) else None

    def _search(self, query: str, vector, namespace: str, top_k: int, sources):
        search_with_vector = getattr(self.retriever, "search_with_vector", None)
        if vector is not None and callable(search_with_vector):
            return search_with_vector(vector, namespace, top_k, sources=sources)
        return self.retriever.search(query, namespace, top_k, sources=sources)

    @staticmethod
    def _metric(case, gold, branch, top_k, retrieval, database_table_count, elapsed_ms) -> RetrievalCaseMetric:
        gold_tables = set(gold.tables)
        retrieved_tables = list(dict.fromkeys(candidate.table_id for candidate in retrieval.candidates))
        retrieved_names = {candidate.table_name.lower() for candidate in retrieval.candidates}
        matched = sorted(gold_tables & retrieved_names)
        return RetrievalCaseMetric(
            question_id=case.question_id,
            question_id_source=case.question_id_source,
            db_id=case.db_id,
            branch=branch,
            top_k=top_k,
            gold_tables=sorted(gold_tables),
            retrieved_tables=retrieved_tables,
            matched_tables=matched,
            recall=len(matched) / len(gold_tables),
            full_recall=gold_tables.issubset(retrieved_names),
            precision=len(matched) / len(retrieved_tables) if retrieved_tables else 0.0,
            database_coverage=len(retrieved_tables) / database_table_count,
            empty=retrieval.status == "empty",
            ambiguous=retrieval.status == "ambiguous",
            false_positive_count=len(set(retrieved_tables)) - len(matched),
            elapsed_ms=elapsed_ms,
        )
