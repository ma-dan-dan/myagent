import pytest

from app.rag.context_packer import ContextPacker, RagContextBudgetExceeded
from app.rag.embedding import FakeEmbedding
from app.rag.indexer import RagColumnRecord, RagSchemaRecord, SchemaIndexer
from app.rag.models import RagDocument, RagSourceType, VectorHit
from app.rag.retriever import SchemaRetriever
from app.rag.vector_store import LanceVectorStore


def make_record(table_id, column_name="value", samples=None, namespace="default"):
    return RagSchemaRecord(namespace=namespace, table_id=table_id, table_name=table_id, ddl=f"CREATE TABLE {table_id} ({column_name} TEXT);", columns=[RagColumnRecord(column_name=column_name, data_type="text", sample_values=samples or [])])


def test_default_records_without_samples_only_create_ddl_index(tmp_path):
    store = LanceVectorStore(tmp_path)
    SchemaIndexer(FakeEmbedding([1.0, 0.0]), store).index([make_record("orders")])

    assert store.search(RagSourceType.DDL, [1.0, 0.0], "default", 10)
    assert store.search(RagSourceType.SAMPLE_VALUE, [1.0, 0.0], "default", 10) == []


def test_explicit_non_sensitive_samples_are_deduplicated_and_bounded(tmp_path):
    store = LanceVectorStore(tmp_path)
    values = ["ORD-001", "ORD-001", "x" * 100, "ORD-003", "ORD-004"]
    SchemaIndexer(FakeEmbedding([1.0, 0.0]), store).index([make_record("orders", samples=values)])

    hit = store.search(RagSourceType.SAMPLE_VALUE, [1.0, 0.0], "default", 10)[0]
    assert "ORD-001" in hit.document.text
    assert "x" * 80 in hit.document.text
    assert "ORD-003" in hit.document.text
    assert "ORD-004" not in hit.document.text


@pytest.mark.parametrize("column_name", ["password", "api_key", "secret_token", "session_cookie", "authorization"])
def test_sensitive_sample_values_are_not_indexed(tmp_path, column_name):
    store = LanceVectorStore(tmp_path)
    SchemaIndexer(FakeEmbedding([1.0, 0.0]), store).index([make_record("accounts", column_name, ["super-secret", "Bearer abc", "token-value"])])

    assert store.search(RagSourceType.SAMPLE_VALUE, [1.0, 0.0], "default", 10) == []


def test_index_sync_removes_stale_documents(tmp_path):
    store = LanceVectorStore(tmp_path)
    indexer = SchemaIndexer(FakeEmbedding([1.0, 0.0]), store)
    indexer.index([make_record("old", samples=["OLD"]), make_record("new", samples=["NEW"])])
    indexer.index([make_record("new", samples=["NEW"])])

    assert [hit.document.table_id for hit in store.search(RagSourceType.SAMPLE_VALUE, [1.0, 0.0], "default", 10)] == ["new"]


def test_index_sync_clears_stale_sample_values_when_samples_are_removed(tmp_path):
    store = LanceVectorStore(tmp_path)
    indexer = SchemaIndexer(FakeEmbedding([1.0, 0.0]), store)
    indexer.index([make_record("accounts", samples=["active"])])
    indexer.index([make_record("accounts")])

    assert store.search(RagSourceType.SAMPLE_VALUE, [1.0, 0.0], "default", 10) == []


def test_namespace_and_document_id_with_quotes_do_not_break_sync_or_search(tmp_path):
    store = LanceVectorStore(tmp_path)
    indexer = SchemaIndexer(FakeEmbedding([1.0, 0.0]), store)
    namespace = "team'o"
    indexer.index([make_record("order's", samples=["ORD-001"], namespace=namespace)])

    assert store.search(RagSourceType.SAMPLE_VALUE, [1.0, 0.0], namespace, 10)[0].document.table_id == "order's"
    indexer.index([make_record("order's", namespace=namespace)])
    assert store.search(RagSourceType.SAMPLE_VALUE, [1.0, 0.0], namespace, 10) == []


@pytest.mark.parametrize("value", ["password=abc", "super-secret", "api_key=abc", "token-value", "Bearer abc", "sk-test-key"])
def test_sensitive_sample_values_are_not_indexed_when_field_name_is_not_sensitive(tmp_path, value):
    store = LanceVectorStore(tmp_path)
    SchemaIndexer(FakeEmbedding([1.0, 0.0]), store).index([make_record("accounts", "note", [value])])

    assert store.search(RagSourceType.SAMPLE_VALUE, [1.0, 0.0], "default", 10) == []


def test_low_similarity_returns_empty():
    class Store:
        def search(self, source, vector, namespace, limit):
            document = RagDocument(document_id="ddl:default:a", source_type=RagSourceType.DDL, namespace="default", table_id="a", table_name="a", text="DDL", content_hash="x")
            return [VectorHit(document=document, score=0.01, rank=1)]

    assert SchemaRetriever(FakeEmbedding([1.0]), Store(), min_hit_score=0.2).search("none", "default", 3).status == "empty"


def test_close_top_candidates_return_ambiguous_and_invalid_input_is_empty():
    class Store:
        def search(self, source, vector, namespace, limit):
            if source is RagSourceType.SAMPLE_VALUE:
                return []
            return [
                VectorHit(document=RagDocument(document_id="ddl:default:a", source_type=RagSourceType.DDL, namespace="default", table_id="a", table_name="a", text="DDL A", content_hash="a"), score=0.9, rank=1),
                VectorHit(document=RagDocument(document_id="ddl:default:b", source_type=RagSourceType.DDL, namespace="default", table_id="b", table_name="b", text="DDL B", content_hash="b"), score=0.9, rank=1),
            ]

    retriever = SchemaRetriever(FakeEmbedding([1.0]), Store(), ambiguity_margin=0.001)
    assert retriever.search("业务对象", "default", 3).status == "ambiguous"
    assert retriever.search("", "default", 3).status == "empty"
    assert retriever.search("业务对象", "default", 0).status == "empty"


def test_context_budget_exceeded_is_explicit():
    class TokenManager:
        def estimate(self, messages): return 100

    from app.rag.models import SchemaCandidate
    with pytest.raises(RagContextBudgetExceeded):
        ContextPacker(TokenManager()).pack([SchemaCandidate(table_id="a", table_name="a", ddl="DDL", score=1)], 10)
