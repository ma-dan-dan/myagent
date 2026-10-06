from app.agent.adapter import LLMResponse
from app.nl2sql.executor import SQLExecutionError
from app.nl2sql.graph import NL2SQLGraphService
from app.nl2sql.models import QueryResult
from app.nl2sql.validator import SQLValidator
from app.rag.models import SchemaCandidate
from app.rag.service import SchemaLinkingResult


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        return self.responses.pop(0)


class FakeSchemaLinkingService:
    def __init__(self, status="ok"):
        self.status = status

    def search(self, message):
        candidate = SchemaCandidate(
            table_id="production_output",
            table_name="production_output",
            ddl="CREATE TABLE production_output (output_quantity INTEGER);",
            score=1.0,
        )
        return SchemaLinkingResult(
            status=self.status,
            context="<schema_evidence>CREATE TABLE production_output (output_quantity INTEGER);</schema_evidence>",
            candidates=[candidate] if self.status == "ok" else [],
        )


class FakeSQLExecutor:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def execute(self, sql, parameters):
        self.calls.append((sql, parameters))
        result = self.result.pop(0) if isinstance(self.result, list) else self.result
        if isinstance(result, Exception):
            raise result
        return result


def build_graph(llm, executor, schema=None, max_attempts=3, max_reflections=2):
    return NL2SQLGraphService(
        schema or FakeSchemaLinkingService(),
        llm,
        SQLValidator(max_rows=100),
        executor,
        dialect="sqlite",
        max_attempts=max_attempts,
        max_reflections=max_reflections,
    )


def test_graph_success_path_calls_executor_and_outputs_result():
    llm = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT output_quantity FROM production_output","tables":["production_output"],"parameters":{},"explanation":"查询产量"}'),
            LLMResponse.message('{"decision":"pass","reason":"结果符合问题"}'),
        ]
    )
    executor = FakeSQLExecutor(QueryResult(columns=["output_quantity"], rows=[[10]], row_count=1, truncated=False))

    result = build_graph(llm, executor).invoke("u1", "s1", "查询产量")

    assert result.status == "ok"
    assert result.query_result.rows == [[10]]
    assert executor.calls
    assert [call[1] for call in llm.calls] == [[], []]


def test_graph_contains_only_the_five_nl2sql_workflow_nodes():
    graph = build_graph(FakeLLM([]), FakeSQLExecutor(QueryResult()))

    nodes = set(graph.graph.get_graph().nodes)

    assert {"schema_linking", "gen_sql", "execute_sql", "reflection", "output"} <= nodes
    assert "context_prepare" not in nodes
    assert "validate_sql" not in nodes


def test_execution_error_is_sent_to_reflection_without_validate_node():
    llm = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT missing FROM production_output","tables":["production_output"],"parameters":{},"explanation":"错误"}'),
            LLMResponse.message('{"decision":"regenerate","reason":"执行失败"}'),
            LLMResponse.message('{"status":"ok","sql":"SELECT output_quantity FROM production_output","tables":["production_output"],"parameters":{},"explanation":"查询产量"}'),
            LLMResponse.message('{"decision":"pass","reason":"符合"}'),
        ]
    )
    executor = FakeSQLExecutor(
        [
            SQLExecutionError("no such column: missing"),
            QueryResult(columns=["output_quantity"], rows=[[10]], row_count=1, truncated=False),
        ]
    )

    result = build_graph(llm, executor).invoke("u1", "s1", "查询产量")

    assert result.status == "ok"
    assert len(executor.calls) == 2
    assert executor.calls[0][0] == "SELECT missing FROM production_output"
    assert len(llm.calls) == 4
    assert "执行失败" in str(llm.calls[2][0])


def test_empty_schema_does_not_call_llm_or_executor():
    llm = FakeLLM([])
    executor = FakeSQLExecutor(QueryResult())

    result = build_graph(llm, executor, schema=FakeSchemaLinkingService("empty")).invoke("u1", "s1", "查询产量")

    assert result.status == "empty"
    assert llm.calls == []
    assert executor.calls == []


def test_attempt_limit_ends_without_infinite_regeneration():
    llm = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"DELETE FROM production_output","tables":["production_output"],"parameters":{},"explanation":"错误"}'),
            LLMResponse.message('{"decision":"regenerate","reason":"重试"}'),
        ]
    )
    executor = FakeSQLExecutor(QueryResult())

    result = build_graph(llm, executor, max_attempts=1).invoke("u1", "s1", "查询产量")

    assert result.status == "failed"
    assert len(executor.calls) == 1
    assert len(llm.calls) == 2
