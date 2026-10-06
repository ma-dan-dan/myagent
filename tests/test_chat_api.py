import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.adapter import LLMResponse
from app.intent.fake_classifier import FakeIntentClassifier
from app.intent.models import IntentDecision
from app.main import create_app
from app.memory.models import CompactionResult, ContextPolicy, MemoryEntry, TokenUsage
from app.nl2sql.models import NL2SQLState, QueryResult, SQLDraft, SQLValidationResult


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(messages)
        return self.responses.pop(0)


class OverBudgetTokenManager:
    policy = ContextPolicy()
    context_window = 100
    output_reserve = 90
    hard_input_budget = 10
    soft_input_budget = 8
    fixed_context_budget = 1

    def estimate(self, messages):
        if any(message.get("content") == "oversized" for message in messages):
            return 11
        return 1


def make_client(tmp_path, fake, context_policy=None):
    return TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=fake,
            intent_classifier=FakeIntentClassifier(),
            workspace_root=tmp_path,
            context_policy=context_policy,
        )
    )


def test_api_creates_session_and_preserves_history_for_next_message(tmp_path):
    fake = FakeLLM([LLMResponse.message("请补充时间范围。"), LLMResponse.message("好的，我记住了你的补充。")])
    client = make_client(tmp_path, fake)

    first = client.post("/api/v1/chat", json={"user_id": "alice", "message": "查产量"})
    session_id = first.json()["session_id"]
    second = client.post(
        "/api/v1/chat",
        json={"user_id": "alice", "session_id": session_id, "message": "本季度"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
    assert "查产量" in json.dumps(fake.calls[1], ensure_ascii=False)
    assert "请补充时间范围" in json.dumps(fake.calls[1], ensure_ascii=False)


def test_api_does_not_share_same_session_id_between_users(tmp_path):
    fake = FakeLLM([LLMResponse.message("A 的回复"), LLMResponse.message("B 的回复")])
    client = make_client(tmp_path, fake)

    client.post(
        "/api/v1/chat",
        json={"user_id": "alice", "session_id": "same-id", "message": "A 的私密问题"},
    )
    response = client.post(
        "/api/v1/chat",
        json={"user_id": "bob", "session_id": "same-id", "message": "B 的问题"},
    )

    assert response.status_code == 200
    assert "A 的私密问题" not in json.dumps(fake.calls[1], ensure_ascii=False)


def test_api_returns_safe_tool_trace(tmp_path):
    fake = FakeLLM(
        [
            LLMResponse.tool_call("search_schema", {"query": "产量", "limit": 5}),
            LLMResponse.message("找到产量结构。"),
        ]
    )
    client = make_client(tmp_path, fake)

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "有哪些产量表？"})

    assert response.status_code == 200
    assert response.json()["tool_events"] == [
        {
            "tool_name": "search_schema",
            "input": {"query": "产量", "limit": 5},
            "status": "ok",
            "summary": "找到 1 个匹配的 Schema。",
        }
    ]
    assert "production_output" not in response.json()["tool_events"][0]["summary"]


def test_api_rejects_blank_message(tmp_path):
    client = make_client(tmp_path, FakeLLM([LLMResponse.message("不会调用")]))

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": " "})

    assert response.status_code == 422


def test_api_serves_native_html_page(tmp_path):
    client = make_client(tmp_path, FakeLLM([LLMResponse.message("不会调用")]))

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Agentic Chat" in response.text
    assert "web-user" not in response.text
    assert "localStorage" in response.text


def test_api_returns_503_when_real_model_configuration_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "你好"})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_fake_llm_still_serves_api_without_real_model_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = make_client(tmp_path, FakeLLM([LLMResponse.message("Fake 回复")]))

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "你好"})

    assert response.status_code == 200
    assert response.json()["message"] == "Fake 回复"


def test_api_exposes_actual_and_estimated_usage_separately(tmp_path, monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kwargs: 17)
    fake = FakeLLM(
        [LLMResponse.message("Fake 回复", usage=TokenUsage(requests=1, input_tokens=12, output_tokens=4, total_tokens=16))]
    )
    response = make_client(tmp_path, fake).post("/api/v1/chat", json={"user_id": "alice", "message": "你好"})

    assert response.status_code == 200
    assert response.json()["usage"]["current_turn"] == {
        "requests": 1,
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
        "estimated_context_tokens": 17,
        "context_window": 16384,
    }


def test_api_writes_memory_after_successful_summary_and_reuses_whole_file_next_turn(tmp_path):
    fake = FakeLLM(
        [
            LLMResponse.message("第一轮"),
            LLMResponse.message("第二轮"),
            LLMResponse.message("摘要：项目长期使用 FakeLLM。", usage=TokenUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)),
            LLMResponse.message('[{"content":"项目长期使用 FakeLLM。"}]', usage=TokenUsage(requests=1, input_tokens=8, output_tokens=4, total_tokens=12)),
            LLMResponse.message("第三轮", usage=TokenUsage(requests=1, input_tokens=6, output_tokens=3, total_tokens=9)),
            LLMResponse.message("第四轮"),
        ]
    )
    client = make_client(tmp_path, fake, ContextPolicy(recent_message_limit=2))

    first = client.post("/api/v1/chat", json={"user_id": "alice", "message": "one"})
    session_id = first.json()["session_id"]
    client.post("/api/v1/chat", json={"user_id": "alice", "session_id": session_id, "message": "two"})
    third = client.post("/api/v1/chat", json={"user_id": "alice", "session_id": session_id, "message": "three"})

    memory_path = tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md"
    assert third.status_code == 200
    assert "项目长期使用 FakeLLM。" in memory_path.read_text(encoding="utf-8")
    assert third.json()["usage"]["current_turn"]["requests"] == 3
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        operations = {row[0] for row in connection.execute("SELECT operation FROM token_usages")}
    assert operations == {"chat", "summary", "memory"}
    assert "<long_term_memory>" not in json.dumps(fake.calls[4], ensure_ascii=False)

    fourth = client.post("/api/v1/chat", json={"user_id": "alice", "session_id": session_id, "message": "unrelated"})

    assert fourth.status_code == 200
    assert "项目长期使用 FakeLLM。" in json.dumps(fake.calls[5], ensure_ascii=False)


def test_api_returns_413_when_required_rolling_summary_fails(tmp_path):
    fake = FakeLLM(
        [
            LLMResponse.message("第一轮"),
            LLMResponse.message("第二轮"),
            LLMResponse.message(""),
            LLMResponse.message("第三轮"),
        ]
    )
    client = make_client(tmp_path, fake, ContextPolicy(recent_message_limit=2))

    first = client.post("/api/v1/chat", json={"user_id": "alice", "message": "one"})
    session_id = first.json()["session_id"]
    client.post("/api/v1/chat", json={"user_id": "alice", "session_id": session_id, "message": "two"})
    third = client.post("/api/v1/chat", json={"user_id": "alice", "session_id": session_id, "message": "three"})

    assert third.status_code == 413
    assert "缩短" in third.json()["detail"]
    assert not (tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md").exists()


def test_api_returns_413_when_minimum_prompt_exceeds_hard_budget(tmp_path):
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=FakeLLM([LLMResponse.message("should not run")]),
            workspace_root=tmp_path,
            token_manager=OverBudgetTokenManager(),
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "oversized"})

    assert response.status_code == 413
    assert "缩短" in response.json()["detail"]
    assert "Agent.md" not in response.json()["detail"]
    assert "MEMORY.md" not in response.json()["detail"]
    assert not (tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md").exists()
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_api_routes_nl2sql_to_injected_graph_without_agent_loop(tmp_path):
    class FakeGraph:
        def __init__(self):
            self.calls = []

        def invoke(self, user_id, session_id, user_message):
            self.calls.append((user_id, session_id, user_message))
            return NL2SQLState(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                status="ok",
                final_message="查询完成，共返回 1 行结果。",
                sql_draft=SQLDraft(status="ok", sql="SELECT output_quantity FROM production_output", tables=["production_output"], parameters={}, explanation="查询产量"),
                validation=SQLValidationResult(ok=True, normalized_sql="SELECT output_quantity FROM production_output LIMIT 100"),
                query_result=QueryResult(columns=["output_quantity"], rows=[[10]], row_count=1, truncated=False),
                usage=TokenUsage(requests=2, input_tokens=10, output_tokens=4, total_tokens=14),
                estimated_context_tokens=23,
                context_window=1024,
                maintenance=CompactionResult(
                    compacted=True,
                    memory_entries=[MemoryEntry(content="产量查询只允许使用业务数据库。")],
                    summary_usage=TokenUsage(requests=1, input_tokens=3, output_tokens=2, total_tokens=5),
                    memory_usage=TokenUsage(requests=1, input_tokens=2, output_tokens=1, total_tokens=3),
                ),
            )

    graph = FakeGraph()
    fake = FakeLLM([])
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=fake,
            intent_classifier=FakeIntentClassifier.for_result("生成 SQL", "nl2sql", None, 1.0),
            nl2sql_graph_service=graph,
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "生成 SQL"})

    assert response.status_code == 200
    assert graph.calls
    assert fake.calls == []
    assert response.json()["nl2sql_status"] == "ok"
    assert response.json()["sql"] == "SELECT output_quantity FROM production_output LIMIT 100"
    assert response.json()["query_result"]["rows"] == [[10]]
    assert response.json()["usage"]["current_turn"]["requests"] == 4
    assert response.json()["usage"]["current_turn"]["estimated_context_tokens"] == 23
    assert response.json()["usage"]["current_turn"]["context_window"] == 1024
    memory_path = tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md"
    assert "产量查询只允许使用业务数据库。" in memory_path.read_text(encoding="utf-8")
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert {row[0] for row in connection.execute("SELECT operation FROM token_usages")} == {"chat", "summary", "memory"}


def test_api_returns_503_when_nl2sql_business_database_is_not_configured(tmp_path):
    class FakeSchemaLinkingService:
        def search(self, message):
            from app.rag.models import SchemaCandidate
            from app.rag.service import SchemaLinkingResult

            return SchemaLinkingResult(
                status="ok",
                context="<schema_evidence>CREATE TABLE production_output (output_quantity INTEGER);</schema_evidence>",
                candidates=[SchemaCandidate(table_id="production_output", table_name="production_output", ddl="CREATE TABLE production_output (output_quantity INTEGER);", score=1.0)],
            )

    fake = FakeLLM([LLMResponse.message('{"status":"ok","sql":"SELECT output_quantity FROM production_output","tables":["production_output"],"parameters":{},"explanation":"查询产量"}')])
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=fake,
            intent_classifier=FakeIntentClassifier.for_result("生成 SQL", "nl2sql", None, 1.0),
            schema_linking_service=FakeSchemaLinkingService(),
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "生成 SQL"})

    assert response.status_code == 503
    assert "业务数据库未配置" in response.json()["detail"]


def test_api_returns_422_and_persists_stable_message_for_unsafe_nl2sql(tmp_path):
    class UnsafeGraph:
        def invoke(self, user_id, session_id, user_message):
            return NL2SQLState(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                status="failed",
                final_message="SQL 校验失败。",
                validation_error="SQL 引用了未授权的字段。",
                maintenance=CompactionResult(
                    compacted=True,
                    memory_entries=[MemoryEntry(content="不应写入失败的 SQL 结论。")],
                ),
            )

    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=FakeLLM([]),
            intent_classifier=FakeIntentClassifier.for_result("生成 SQL", "nl2sql", None, 1.0),
            nl2sql_graph_service=UnsafeGraph(),
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "生成 SQL"})

    assert response.status_code == 422
    assert not (tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md").exists()
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert [row[0] for row in connection.execute("SELECT role FROM messages ORDER BY id")] == ["user", "assistant"]


def test_api_returns_503_without_calling_dependencies_when_nl2sql_is_disabled(tmp_path, monkeypatch):
    class CountingSchemaLinkingService:
        def __init__(self):
            self.calls = []

        def search(self, message):
            self.calls.append(message)
            raise AssertionError("disabled NL2SQL must not link schema")

    monkeypatch.setenv("NL2SQL_ENABLED", "false")
    schema_linking_service = CountingSchemaLinkingService()
    fake = FakeLLM([])
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=fake,
            intent_classifier=FakeIntentClassifier.for_result("生成 SQL", "nl2sql", None, 1.0),
            schema_linking_service=schema_linking_service,
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "生成 SQL"})

    assert response.status_code == 503
    assert "未启用" in response.json()["detail"]
    assert schema_linking_service.calls == []
    assert fake.calls == []


def test_api_returns_503_when_configured_nl2sql_database_file_is_missing(tmp_path, monkeypatch):
    class FakeSchemaLinkingService:
        def search(self, message):
            from app.rag.models import SchemaCandidate
            from app.rag.service import SchemaLinkingResult

            return SchemaLinkingResult(
                status="ok",
                context="<schema_evidence>CREATE TABLE production_output (output_quantity INTEGER);</schema_evidence>",
                candidates=[SchemaCandidate(table_id="production_output", table_name="production_output", ddl="CREATE TABLE production_output (output_quantity INTEGER);", score=1.0)],
            )

    monkeypatch.setenv("NL2SQL_DATABASE_URL", str(tmp_path / "missing-business.sqlite3"))
    fake = FakeLLM([LLMResponse.message('{"status":"ok","sql":"SELECT output_quantity FROM production_output","tables":["production_output"],"parameters":{},"explanation":"查询产量"}')])
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=fake,
            intent_classifier=FakeIntentClassifier.for_result("生成 SQL", "nl2sql", None, 1.0),
            schema_linking_service=FakeSchemaLinkingService(),
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "生成 SQL"})

    assert response.status_code == 503
    assert "业务数据库不可用" in response.json()["detail"]


def test_default_nl2sql_wiring_uses_dedicated_temporary_sqlite_database(tmp_path, monkeypatch):
    class FakeSchemaLinkingService:
        def search(self, message):
            from app.rag.models import SchemaCandidate
            from app.rag.service import SchemaLinkingResult

            return SchemaLinkingResult(
                status="ok",
                context="<schema_evidence>CREATE TABLE production_output (output_quantity INTEGER);</schema_evidence>",
                candidates=[SchemaCandidate(table_id="production_output", table_name="production_output", ddl="CREATE TABLE production_output (output_quantity INTEGER);", score=1.0)],
            )

    business_database = tmp_path / "business.sqlite3"
    with sqlite3.connect(business_database) as connection:
        connection.execute("CREATE TABLE production_output (output_quantity INTEGER)")
        connection.executemany("INSERT INTO production_output VALUES (?)", [(10,), (20,)])
    monkeypatch.setenv("NL2SQL_ENABLED", "true")
    monkeypatch.setenv("NL2SQL_DATABASE_URL", f"sqlite:///{business_database.as_posix()}")
    fake = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT output_quantity FROM production_output","tables":["production_output"],"parameters":{},"explanation":"查询产量"}'),
            LLMResponse.message('{"decision":"pass","reason":"结果符合问题"}'),
        ]
    )
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=fake,
            intent_classifier=FakeIntentClassifier.for_result("生成 SQL", "nl2sql", None, 1.0),
            schema_linking_service=FakeSchemaLinkingService(),
            workspace_root=tmp_path,
        )
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "生成 SQL"})

    assert response.status_code == 200
    assert response.json()["nl2sql_status"] == "ok"
    assert response.json()["query_result"]["columns"] == ["output_quantity"]
    assert response.json()["query_result"]["rows"] == [[10], [20]]
    assert response.json()["sql"] == "SELECT output_quantity FROM production_output"
    assert len(fake.calls) == 2


def test_default_nl2sql_keeps_the_five_node_graph_separate_from_chat_context_maintenance(tmp_path, monkeypatch):
    class FakeSchemaLinkingService:
        def search(self, message):
            from app.rag.models import SchemaCandidate
            from app.rag.service import SchemaLinkingResult

            return SchemaLinkingResult(
                status="ok",
                context="<schema_evidence>CREATE TABLE production_output (output_quantity INTEGER);</schema_evidence>",
                candidates=[SchemaCandidate(table_id="production_output", table_name="production_output", ddl="CREATE TABLE production_output (output_quantity INTEGER);", score=1.0)],
            )

    business_database = tmp_path / "business.sqlite3"
    with sqlite3.connect(business_database) as connection:
        connection.execute("CREATE TABLE production_output (output_quantity INTEGER)")
        connection.execute("INSERT INTO production_output VALUES (10)")
    monkeypatch.setenv("NL2SQL_ENABLED", "true")
    monkeypatch.setenv("NL2SQL_DATABASE_URL", str(business_database))
    fake = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT output_quantity FROM production_output","tables":["production_output"],"parameters":{},"explanation":"查询产量"}'),
            LLMResponse.message('{"decision":"pass","reason":"结果符合问题"}'),
            LLMResponse.message('{"status":"ok","sql":"SELECT output_quantity FROM production_output","tables":["production_output"],"parameters":{},"explanation":"查询产量"}'),
            LLMResponse.message('{"decision":"pass","reason":"结果符合问题"}'),
            LLMResponse.message('{"status":"ok","sql":"SELECT output_quantity FROM production_output","tables":["production_output"],"parameters":{},"explanation":"查询产量"}'),
            LLMResponse.message('{"decision":"pass","reason":"结果符合问题"}'),
        ]
    )
    client = TestClient(
        create_app(
            db_path=tmp_path / "chat.sqlite3",
            catalog_path=Path(__file__).parents[1] / "data" / "schema_catalog.json",
            llm_adapter=fake,
            intent_classifier=FakeIntentClassifier(
                {
                    message: IntentDecision(intent="nl2sql", confidence=1.0, provider="fake", model="fake-intent-v1", latency_ms=0)
                    for message in ("查询产量", "那按日期呢", "再查一次")
                }
            ),
            schema_linking_service=FakeSchemaLinkingService(),
            workspace_root=tmp_path,
            context_policy=ContextPolicy(recent_message_limit=2),
        )
    )

    first = client.post("/api/v1/chat", json={"user_id": "alice", "message": "查询产量"})
    session_id = first.json()["session_id"]
    second = client.post("/api/v1/chat", json={"user_id": "alice", "session_id": session_id, "message": "那按日期呢"})
    third = client.post("/api/v1/chat", json={"user_id": "alice", "session_id": session_id, "message": "再查一次"})

    assert first.status_code == second.status_code == third.status_code == 200
    assert "查询产量" not in json.dumps(fake.calls[2], ensure_ascii=False)
    assert third.json()["usage"]["current_turn"]["requests"] == 0
    memory_path = tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md"
    assert memory_path.exists() is False
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        assert {row[0] for row in connection.execute("SELECT operation FROM token_usages")} == {"chat"}
