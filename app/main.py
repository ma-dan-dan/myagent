from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.agent.adapter import LLMAdapter
from app.agent.chat_agent import ChatAgent
from app.agent.llm_factory import LLMAdapterFactory
from app.agent.tool_registry import ToolRegistry
from app.api.chat import router as chat_router
from app.config import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_RAG_INDEX_PATH,
    DEFAULT_WEB_PATH,
    INTENT_MIN_CONFIDENCE,
    get_nl2sql_runtime_config,
)
from app.intent.classifier import IntentClassificationError, IntentClassifier
from app.intent.factory import IntentClassifierFactory, UnavailableIntentClassifier
from app.intent.router import IntentRouter
from app.memory.context_manager import ContextManager
from app.memory.long_term_memory import LongTermMemoryStore
from app.memory.project_context import ProjectContextLoader
from app.memory.summary_service import SummaryService
from app.memory.token_manager import TokenManager
from app.memory.models import ContextPolicy
from app.services.chat_service import ChatService
from app.storage.schema_catalog import SchemaCatalog
from app.storage.session_service import SessionService
from app.tools.schema_search import SchemaSearchTool
from app.rag.context_packer import ContextPacker
from app.rag.embedding import QwenEmbeddingAdapter
from app.rag.indexer import RagColumnRecord, RagSchemaRecord, SchemaIndexer
from app.rag.retriever import SchemaRetriever
from app.rag.service import SchemaLinkingService
from app.rag.vector_store import LanceVectorStore
from app.nl2sql.executor import ReadOnlySQLiteExecutor, UnavailableSQLExecutor
from app.nl2sql.graph import NL2SQLGraphService
from app.nl2sql.validator import SQLValidator


def create_app(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    web_path: str | Path = DEFAULT_WEB_PATH,
    llm_adapter: LLMAdapter | None = None,
    workspace_root: str | Path | None = None,
    context_policy: ContextPolicy | None = None,
    token_manager: TokenManager | None = None,
    intent_classifier: IntentClassifier | None = None,
    intent_llm_adapter: LLMAdapter | None = None,
    schema_linking_service: SchemaLinkingService | None = None,
    rag_records: list[RagSchemaRecord] | None = None,
    rag_index_path: str | Path | None = None,
    nl2sql_graph_service: NL2SQLGraphService | None = None,
) -> FastAPI:

    # 1、 创建 FastAPI 应用实例，并确定工作空间的路径，如果没有提供 workspace_root，则使用当前文件的父目录作为默认路径。
    app = FastAPI(title="Minimal Agentic Chat V1")
    resolved_workspace = Path(workspace_root) if workspace_root is not None else Path(__file__).resolve().parents[1]


    # 3、创建一个 SchemaCatalog 实例，用于管理和访问模式目录。然后创建一个 ToolRegistry 实例，并将 SchemaSearchTool 添加到工具注册表中。接着创建一个 ChatAgent 实例，使用提供的 llm_adapter（如果有的话）或从环境变量中获取的 LLMAdapterFactory 来初始化。
    catalog = SchemaCatalog(catalog_path)
    registry = ToolRegistry([SchemaSearchTool(catalog)])
    agent = ChatAgent(llm_adapter if llm_adapter is not None else LLMAdapterFactory.from_env(), registry)


    # 4、上下文管理部分
    session_service = SessionService(db_path)
    policy = context_policy or ContextPolicy()
    manager = token_manager or TokenManager(
        str(getattr(agent.llm, "provider", "openai") or "openai"),
        str(getattr(agent.llm, "model", "") or "unknown"),
        policy,
        str(getattr(agent.llm, "litellm_model_name", "") or getattr(agent.llm, "model", "") or "unknown"),
    )
    memory = LongTermMemoryStore(resolved_workspace)
    summary = SummaryService(session_service, agent.llm, manager, policy)

    context = ContextManager(
        session_service,
        summary,
        ProjectContextLoader(resolved_workspace),
        memory,
        manager,
        policy,
    )
    if schema_linking_service is None:
        records = rag_records if rag_records is not None else [
            RagSchemaRecord(
                table_id=table.table_name,
                table_name=table.table_name,
                ddl="CREATE TABLE " + table.table_name + " (" + ", ".join(f"{column.column_name} {column.data_type}" for column in table.columns) + ");",
                columns=[RagColumnRecord(column_name=column.column_name, data_type=column.data_type) for column in table.columns],
            )
            for table in catalog.tables
        ]
        embedding = QwenEmbeddingAdapter.from_env()
        store = LanceVectorStore(rag_index_path or DEFAULT_RAG_INDEX_PATH)
        schema_linking_service = SchemaLinkingService(SchemaIndexer(embedding, store), SchemaRetriever(embedding, store), ContextPacker(manager), records)

    if intent_classifier is None:
        try:
            intent_classifier = IntentClassifierFactory.from_env(intent_llm_adapter)
        except IntentClassificationError as exc:
            intent_classifier = UnavailableIntentClassifier(str(exc))
    intent_router = IntentRouter(intent_classifier, min_confidence=INTENT_MIN_CONFIDENCE)

    if nl2sql_graph_service is None:
        nl2sql_config = get_nl2sql_runtime_config()
        executor = (
            ReadOnlySQLiteExecutor(
                nl2sql_config.database_url,
                nl2sql_config.max_rows,
                nl2sql_config.max_columns,
                nl2sql_config.query_timeout_seconds,
            )
            if nl2sql_config.database_url
            else UnavailableSQLExecutor()
        )
        nl2sql_graph_service = NL2SQLGraphService(
            schema_linking_service,
            agent.llm,
            SQLValidator(nl2sql_config.dialect, nl2sql_config.max_rows, nl2sql_config.max_sql_length),
            executor,
            nl2sql_config.dialect,
            nl2sql_config.max_attempts,
            nl2sql_config.max_reflections,
        )


    # 5、将 ChatService 实例和 web_path 添加到 FastAPI 应用的状态中，以便在应用的其他部分访问。然后将 chat_router 包含到应用中，以处理与聊天相关的 API 路由。最后，定义一个根路径的 GET 请求处理函数 index，用于返回 web_path 指定的 HTML 文件作为响应。
    app.state.chat_service = ChatService(session_service, agent, context, memory, intent_router, schema_linking_service, nl2sql_graph_service)
    app.state.web_path = Path(web_path)
    app.include_router(chat_router)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(app.state.web_path, media_type="text/html")

    return app


app = create_app()
