import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.adapter import LLMResponse
from app.intent.classifier import IntentClassificationError
from app.intent.fake_classifier import FakeIntentClassifier
from app.main import create_app
from app.memory.models import ContextPolicy, TokenUsage
from app.rag.context_packer import RagContextBudgetExceeded
from app.rag.indexer import RagSchemaRecord
from app.rag.service import SchemaLinkingResult


class FakeLLM:
    provider = "fake"
    model = "fake-chat-v1"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        return self.responses.pop(0)


def make_client(tmp_path, classifier, responses=None):
    return TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=FakeLLM(responses or []),
            intent_classifier=classifier,
            workspace_root=tmp_path,
        )
    )


def test_chat_intent_keeps_existing_agent_path(tmp_path):
    classifier = FakeIntentClassifier.for_result("你好", "chat", None, 1.0)
    client = make_client(tmp_path, classifier, [LLMResponse.message("你好，我是助手。")])

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "你好"})

    assert response.status_code == 200
    assert response.json()["message"] == "你好，我是助手。"
    assert response.json()["intent_decision"]["intent"] == "chat"


def test_create_app_uses_injected_intent_llm_separately_from_chat_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("INTENT_PROVIDER", "llm")
    chat_llm = FakeLLM([LLMResponse.message("主聊天模型回复")])
    intent_llm = FakeLLM(
        [LLMResponse.message('{"intent":"chat","data_action":null,"confidence":0.95}')]
    )
    chat_llm.provider = "qwen"
    chat_llm.model = "chat-model"
    intent_llm.provider = "deepseek"
    intent_llm.model = "intent-model"
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=chat_llm,
            intent_llm_adapter=intent_llm,
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "你好"})

    assert response.status_code == 200
    assert response.json()["message"] == "主聊天模型回复"
    assert response.json()["intent_decision"]["provider"] == "deepseek"
    assert response.json()["intent_decision"]["model"] == "intent-model"


def test_data_operation_create_returns_stable_placeholder(tmp_path):
    classifier = FakeIntentClassifier.for_result("新增产量", "data_operation", "create", 1.0)
    client = make_client(tmp_path, classifier)

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "新增产量"})

    assert response.status_code == 200
    assert response.json()["intent_decision"]["data_action"] == "create"
    assert "数据查询意图" in response.json()["message"]


def test_data_read_uses_schema_linking_context_without_tools(tmp_path):
    class FakeSchemaLinkingService:
        def search(self, message):
            return SchemaLinkingResult(
                status="ok",
                context="<schema_evidence table=\"production_output\">CREATE TABLE production_output ();</schema_evidence>",
            )

    classifier = FakeIntentClassifier.for_result("查产量", "data_operation", "read", 1.0)
    chat_llm = FakeLLM(
        [
            LLMResponse.message(
                "production_output 包含产量字段。",
                usage=TokenUsage(requests=1, input_tokens=12, output_tokens=4, total_tokens=16),
            )
        ]
    )
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=chat_llm,
            intent_classifier=classifier,
            schema_linking_service=FakeSchemaLinkingService(),
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "查产量"})

    assert response.status_code == 200
    assert response.json()["message"] == "production_output 包含产量字段。"
    assert response.json()["usage"]["current_turn"]["total_tokens"] == 16
    assert chat_llm.calls[0][1] == []
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert connection.execute("SELECT operation, total_tokens FROM token_usages").fetchall() == [("chat", 16)]


def test_data_read_returns_413_without_calling_chat_model_when_schema_budget_is_exceeded(tmp_path):
    class BudgetExceededSchemaLinkingService:
        def search(self, message):
            raise RagContextBudgetExceeded("Schema evidence exceeds the available context budget.")

    classifier = FakeIntentClassifier.for_result("查产量", "data_operation", "read", 1.0)
    chat_llm = FakeLLM([])
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=chat_llm,
            intent_classifier=classifier,
            schema_linking_service=BudgetExceededSchemaLinkingService(),
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "查产量"})

    assert response.status_code == 413
    assert "Schema 证据过长" in response.json()["detail"]
    assert chat_llm.calls == []
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_data_read_default_rag_wiring_uses_mocked_embedding_and_no_tools(tmp_path, monkeypatch):
    def fake_embedding(**kwargs):
        return {"data": [{"embedding": [1.0, 0.0]} for _ in kwargs["input"]]}

    monkeypatch.setenv("QWEN_API_KEY", "test-key-not-real")
    monkeypatch.setattr("app.rag.embedding.litellm.embedding", fake_embedding)
    classifier = FakeIntentClassifier.for_result("查订单", "data_operation", "read", 1.0)
    chat_llm = FakeLLM([LLMResponse.message("orders 是订单表。")])
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=chat_llm,
            intent_classifier=classifier,
            rag_records=[
                RagSchemaRecord(
                    table_id="orders",
                    table_name="orders",
                    ddl="CREATE TABLE orders (order_id TEXT);",
                )
            ],
            rag_index_path=tmp_path / "rag",
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "查订单"})

    assert response.status_code == 200
    assert response.json()["message"] == "orders 是订单表。"
    assert chat_llm.calls[0][1] == []


def test_data_read_reuses_summary_memory_and_usage_finalization(tmp_path):
    class FakeSchemaLinkingService:
        def search(self, message):
            return SchemaLinkingResult(
                status="ok",
                context='<schema_evidence table="orders">CREATE TABLE orders ();</schema_evidence>',
            )

    classifier = FakeIntentClassifier.for_result("查订单", "data_operation", "read", 1.0)
    chat_llm = FakeLLM(
        [
            LLMResponse.message("历史摘要", usage=TokenUsage(requests=1, total_tokens=2)),
            LLMResponse.message('[{"content":"用户偏好按月查看订单"}]', usage=TokenUsage(requests=1, total_tokens=3)),
            LLMResponse.message("orders 是订单表。", usage=TokenUsage(requests=1, total_tokens=4)),
        ]
    )
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=chat_llm,
            intent_classifier=classifier,
            schema_linking_service=FakeSchemaLinkingService(),
            context_policy=ContextPolicy(recent_message_limit=2),
            workspace_root=tmp_path,
        )
    )
    service = client.app.state.chat_service
    session_id = service.session_service.create_session("alice")
    for index in range(3):
        service.session_service.append_message("alice", session_id, "user", f"历史消息 {index}")

    response = client.post(
        "/api/v1/chat",
        json={"user_id": "alice", "session_id": session_id, "message": "查订单"},
    )

    assert response.status_code == 200
    assert response.json()["usage"]["current_turn"]["total_tokens"] == 9
    assert chat_llm.calls[-1][1] == []
    assert "用户偏好按月查看订单" in (tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md").read_text(encoding="utf-8")
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert connection.execute("SELECT operation FROM token_usages ORDER BY id").fetchall() == [
            ("chat",),
            ("summary",),
            ("memory",),
        ]


def test_data_read_returns_503_when_qwen_embedding_key_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    classifier = FakeIntentClassifier.for_result("查产量", "data_operation", "read", 1.0)
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=FakeLLM([]),
            intent_classifier=classifier,
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "查产量"})

    assert response.status_code == 503
    assert "QWEN_API_KEY" in response.json()["detail"]


def test_nl2sql_returns_stable_placeholder(tmp_path):
    classifier = FakeIntentClassifier.for_result("生成 SQL", "nl2sql", None, 1.0)
    client = make_client(tmp_path, classifier)

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "生成 SQL"})

    assert response.status_code == 200
    assert "NL2SQL 意图" in response.json()["message"]


def test_low_confidence_fallback_uses_existing_agent_path(tmp_path):
    classifier = FakeIntentClassifier.for_result("帮我看看", "nl2sql", None, 0.2)
    client = make_client(tmp_path, classifier, [LLMResponse.message("普通回答")])

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "帮我看看"})

    assert response.status_code == 200
    assert response.json()["message"] == "普通回答"
    assert response.json()["intent_decision"]["intent"] == "nl2sql"
    assert response.json()["intent_decision"]["provider"] == "fake"
    assert response.json()["routed_intent"] == "chat"
    assert response.json()["fallback_reason"] == "low_confidence"


def test_low_confidence_is_counted_under_its_source_provider(tmp_path):
    classifier = FakeIntentClassifier.for_result("ambiguous", "nl2sql", None, 0.2)
    client = make_client(tmp_path, classifier, [LLMResponse.message("普通回答")])

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "ambiguous"})
    metrics = client.get("/api/v1/intent-metrics", params={"user_id": "alice", "provider": "fake"})

    assert response.status_code == 200
    assert response.json()["intent_decision"]["intent"] == "nl2sql"
    assert response.json()["routed_intent"] == "chat"
    assert metrics.json()["request_count"] == 1
    assert metrics.json()["classified_intent_counts"] == {"nl2sql": 1}
    assert metrics.json()["routed_intent_counts"] == {"chat": 1}


def test_intent_classification_is_persisted_and_metrics_are_aggregated(tmp_path):
    classifier = FakeIntentClassifier.for_result("查产量", "data_operation", "read", 0.9)
    client = make_client(tmp_path, classifier)

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "查产量"})
    metrics = client.get("/api/v1/intent-metrics", params={"user_id": "alice", "provider": "fake"})

    assert response.status_code == 200
    assert metrics.status_code == 200
    assert metrics.json()["request_count"] == 1
    assert metrics.json()["classified_intent_counts"] == {"data_operation": 1}
    assert metrics.json()["routed_intent_counts"] == {"data_operation": 1}
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        row = connection.execute(
            "SELECT provider, model, intent, routed_intent, data_action FROM intent_classifications"
        ).fetchone()
    assert row == ("fake", "fake-intent-v1", "data_operation", "data_operation", "read")


def test_intent_configuration_error_returns_503_without_persisting_messages(tmp_path):
    class BrokenClassifier:
        def classify(self, message):
            raise IntentClassificationError("分类服务不可用")

    client = make_client(tmp_path, BrokenClassifier())

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "你好"})

    assert response.status_code == 503
    assert response.json()["detail"] == "分类服务不可用"
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM intent_classifications").fetchone()[0] == 0


def test_missing_jev_key_returns_503_without_exposing_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("INTENT_PROVIDER", "jev")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=FakeLLM([]),
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "你好"})

    assert response.status_code == 503
    assert "TYPESAFE_API_KEY" in response.json()["detail"]
