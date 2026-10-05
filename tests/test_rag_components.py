import litellm
import pytest

from app.rag.context_packer import ContextPacker
from app.rag.embedding import FakeEmbedding, QwenEmbeddingAdapter, RagConfigurationError
from app.rag.models import RagDocument, RagSourceType, SchemaCandidate
from app.rag.retriever import SchemaRetriever
from app.rag.vector_store import LanceVectorStore


def ddl_document():
    return RagDocument(
        document_id="ddl:default:production_output",
        source_type=RagSourceType.DDL,
        namespace="default",
        table_id="production_output",
        table_name="production_output",
        text="CREATE TABLE production_output (output_date DATE, line_code TEXT);",
        content_hash="ddl-hash",
    )


def sample_document():
    return RagDocument(
        document_id="sample:default:production_output:line_code",
        source_type=RagSourceType.SAMPLE_VALUE,
        namespace="default",
        table_id="production_output",
        table_name="production_output",
        column_name="line_code",
        text="production_output line_code TEXT samples: LINE-A, LINE-B",
        content_hash="sample-hash",
    )


def test_qwen_embedding_uses_fixed_config(monkeypatch):
    calls = []
    monkeypatch.setattr(
        litellm,
        "embedding",
        lambda **kwargs: calls.append(kwargs) or {"data": [{"embedding": [0.1, 0.2]}]},
    )
    monkeypatch.setenv("QWEN_API_KEY", "rag-key")

    vectors = QwenEmbeddingAdapter.from_env().embed(["产量"])

    assert vectors == [[0.1, 0.2]]
    assert calls[0]["model"] == "dashscope/text-embedding-v3"
    assert calls[0]["api_base"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_embedding_requires_qwen_key(monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)

    with pytest.raises(RagConfigurationError, match="QWEN_API_KEY"):
        QwenEmbeddingAdapter.from_env().embed(["产量"])


def test_vector_store_keeps_ddl_and_sample_indexes_separate(tmp_path):
    store = LanceVectorStore(tmp_path)
    store.upsert(RagSourceType.DDL, [(ddl_document(), [1.0, 0.0])])
    store.upsert(RagSourceType.SAMPLE_VALUE, [(sample_document(), [0.0, 1.0])])

    assert [hit.document.document_id for hit in store.search(RagSourceType.DDL, [1.0, 0.0], "default", 3)] == [
        "ddl:default:production_output"
    ]
    assert [hit.document.document_id for hit in store.search(RagSourceType.SAMPLE_VALUE, [0.0, 1.0], "default", 3)] == [
        "sample:default:production_output:line_code"
    ]


def test_retriever_fuses_two_sources_by_table_and_column(tmp_path):
    store = LanceVectorStore(tmp_path)
    store.upsert(RagSourceType.DDL, [(ddl_document(), [1.0, 0.0])])
    store.upsert(RagSourceType.SAMPLE_VALUE, [(sample_document(), [1.0, 0.0])])
    retriever = SchemaRetriever(FakeEmbedding([1.0, 0.0]), store)

    result = retriever.search("产量", "default", limit=3)

    assert result.status == "ok"
    assert result.candidates[0].table_id == "production_output"
    assert set(result.candidates[0].matched_by) == {RagSourceType.DDL, RagSourceType.SAMPLE_VALUE}
    assert result.candidates[0].matched_columns == ["line_code"]


def test_retriever_supports_source_selection_without_changing_default_fusion(tmp_path):
    store = LanceVectorStore(tmp_path)
    store.upsert(RagSourceType.DDL, [(ddl_document(), [1.0, 0.0])])
    store.upsert(RagSourceType.SAMPLE_VALUE, [(sample_document(), [1.0, 0.0])])
    retriever = SchemaRetriever(FakeEmbedding([1.0, 0.0]), store)

    ddl_only = retriever.search("产量", "default", limit=3, sources=[RagSourceType.DDL])
    default_fusion = retriever.search("产量", "default", limit=3)

    assert ddl_only.candidates[0].matched_by == [RagSourceType.DDL]
    assert set(default_fusion.candidates[0].matched_by) == {RagSourceType.DDL, RagSourceType.SAMPLE_VALUE}


def test_context_packer_respects_token_budget_and_keeps_schema_evidence():
    class TokenManager:
        def estimate(self, messages):
            return sum(len(message["content"]) for message in messages)

    packed = ContextPacker(TokenManager()).pack(
        [
            SchemaCandidate(
                table_id="production_output",
                table_name="production_output",
                ddl="CREATE TABLE production_output (output_date DATE, line_code TEXT);",
                matched_columns=["line_code"],
                sample_values={"line_code": ["LINE-A"]},
                score=0.8,
                matched_by=[RagSourceType.DDL],
            )
        ],
        budget=1000,
    )

    assert "production_output" in packed
