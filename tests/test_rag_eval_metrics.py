from app.rag.models import RagDocument, RagSourceType, RetrievalResult, SchemaCandidate, VectorHit
from app.rag.retriever import SchemaRetriever
from app.rag_eval.evaluator import SchemaRagEvaluator
from app.rag_eval.models import BirdCase, GoldTableSet


class FakeRetriever:
    rrf_k = 60
    min_hit_score = 0.2
    ambiguity_margin = 0.001

    def search(self, query, namespace, limit, sources=None):
        source_key = tuple(source.value for source in sources) if sources else ("fusion",)
        table_ids = {
            ("ddl",): ["shop.orders"],
            ("sample_value",): ["shop.users"],
            ("fusion",): ["shop.orders", "shop.users", "shop.products"],
        }[source_key][:limit]
        return RetrievalResult(
            status="ambiguous" if source_key == ("sample_value",) else "ok",
            candidates=[
                SchemaCandidate(table_id=table_id, table_name=table_id.rsplit(".", 1)[-1], score=1.0)
                for table_id in table_ids
            ],
        )


def test_evaluator_calculates_three_branches_and_gold_metrics_without_database_coverage_confusion(tmp_path):
    case = BirdCase(question_id="q1", db_id="shop", question="查询订单用户", gold_sql="SELECT 1", database_path=tmp_path / "shop.sqlite")
    report = SchemaRagEvaluator(FakeRetriever()).evaluate_case(
        case,
        GoldTableSet(question_id="q1", db_id="shop", tables=["orders", "users"]),
        database_table_count=4,
        top_ks=[1, 3],
    )

    fusion_at_three = next(item for item in report if item.branch == "fusion_rrf" and item.top_k == 3)
    assert fusion_at_three.recall == 1.0
    assert fusion_at_three.full_recall is True
    assert fusion_at_three.precision == 2 / 3
    assert fusion_at_three.database_coverage == 3 / 4
    assert fusion_at_three.false_positive_count == 1
    assert any(item.branch == "sample_value_only" and item.ambiguous for item in report)


def test_evaluator_reuses_one_query_embedding_for_all_branches_and_top_k_values(tmp_path):
    class CountingEmbedding:
        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(texts)
            return [[1.0] for _ in texts]

    class Store:
        def search(self, source, vector, namespace, limit):
            document = RagDocument(
                document_id=f"{source.value}:shop:orders",
                source_type=source,
                namespace=namespace,
                table_id="shop.orders",
                table_name="orders",
                column_name="status" if source is RagSourceType.SAMPLE_VALUE else None,
                text="orders",
                content_hash=source.value,
            )
            return [VectorHit(document=document, score=0.9, rank=1)]

    embedding = CountingEmbedding()
    evaluator = SchemaRagEvaluator(SchemaRetriever(embedding, Store()))
    case = BirdCase(question_id="q1", db_id="shop", question="查询订单", gold_sql="SELECT 1", database_path=tmp_path / "shop.sqlite")
    gold = GoldTableSet(question_id="q1", db_id="shop", tables=["orders"])

    evaluator.evaluate_case(case, gold, database_table_count=1, top_ks=[1, 3, 5, 10])

    assert embedding.calls == [["查询订单"]]
