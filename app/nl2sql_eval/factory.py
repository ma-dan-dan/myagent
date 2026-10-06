from __future__ import annotations

from pathlib import Path

from app.agent.adapter import LLMAdapter, LLMResponse, ToolSpec
from app.memory.token_manager import TokenManager
from app.nl2sql.executor import SQLExecutor
from app.nl2sql.graph import NL2SQLGraphService
from app.nl2sql_eval.models import PredictionInput
from app.rag.context_packer import ContextPacker
from app.rag.embedding import EmbeddingAdapter
from app.rag.indexer import SchemaIndexer
from app.rag.models import RagSourceType
from app.rag.retriever import SchemaRetriever
from app.rag.service import SchemaLinkingService
from app.rag.vector_store import LanceVectorStore
from app.rag_eval.sqlite_catalog import SQLiteCatalog


class CountingLLMAdapter:
    def __init__(self, delegate: LLMAdapter) -> None:
        self.delegate = delegate
        self.call_count = 0

    def complete(self, messages: list[dict], tools: list[ToolSpec]) -> LLMResponse:
        self.call_count += 1
        return self.delegate.complete(messages, tools)


class BirdGraphFactory:
    def __init__(
        self,
        llm: LLMAdapter,
        embedding: EmbeddingAdapter,
        index_path: str | Path,
        *,
        dialect: str,
        top_k: int,
        max_attempts: int,
        max_reflections: int,
        max_rows: int,
        max_sql_length: int,
        rebuild_index: bool = False,
        sources: tuple[RagSourceType, ...] | None = None,
    ) -> None:
        self.llm = llm
        self.store = LanceVectorStore(index_path)
        self.indexer = SchemaIndexer(embedding, self.store)
        self.retriever = SchemaRetriever(embedding, self.store)
        self.catalog = SQLiteCatalog()
        self.dialect = dialect
        self.top_k = top_k
        self.max_attempts = max_attempts
        self.max_reflections = max_reflections
        self.max_rows = max_rows
        self.max_sql_length = max_sql_length
        self.rebuild_index = rebuild_index
        self.sources = sources
        self._schema_services: dict[str, SchemaLinkingService] = {}
        self._database_paths: dict[str, Path] = {}

    def create(self, prediction_input: PredictionInput, executor: SQLExecutor) -> NL2SQLGraphService:
        schema_service = self._schema_service(prediction_input)
        counted_llm = CountingLLMAdapter(self.llm)
        return NL2SQLGraphService(
            schema_service,
            counted_llm,
            None,
            executor,
            dialect=self.dialect,
            max_attempts=self.max_attempts,
            max_reflections=self.max_reflections,
        )

    def _schema_service(self, prediction_input: PredictionInput) -> SchemaLinkingService:
        database_path = prediction_input.database_path.resolve()
        previous_path = self._database_paths.setdefault(prediction_input.db_id, database_path)
        if previous_path != database_path:
            raise ValueError(f"db_id {prediction_input.db_id} 指向了多个数据库文件。")
        service = self._schema_services.get(prediction_input.db_id)
        if service is not None:
            return service
        runtime_provider = getattr(self.llm, "provider", "unknown")
        runtime_model = getattr(self.llm, "model", "unknown")
        litellm_model = getattr(self.llm, "litellm_model_name", runtime_model)
        packer = ContextPacker(TokenManager(runtime_provider, runtime_model, litellm_model_name=litellm_model))
        service = SchemaLinkingService(
            self.indexer,
            self.retriever,
            packer,
            self.catalog.load(prediction_input.db_id, database_path),
            namespace=prediction_input.db_id,
            limit=self.top_k,
            sources=self.sources,
        )
        if not self.rebuild_index and required_namespaces_ready(self.store, prediction_input.db_id, self.sources):
            service._indexed = True
        self._schema_services[prediction_input.db_id] = service
        return service


def required_namespaces_ready(
    store: LanceVectorStore,
    namespace: str,
    sources: tuple[RagSourceType, ...] | None = None,
) -> bool:
    required_sources = sources or (RagSourceType.DDL, RagSourceType.SAMPLE_VALUE)
    return all(store.has_namespace(source, namespace) for source in required_sources)
