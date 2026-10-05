import sqlite3

from app.agent.adapter import LLMResponse
from app.nl2sql.executor import ReadOnlySQLiteExecutor
from app.nl2sql.graph import NL2SQLGraphService
from app.nl2sql.validator import SQLValidator
from app.nl2sql_eval.evaluator import NL2SQLWorkflowEvaluator
from app.nl2sql_eval.models import BirdEvaluationCase
from app.nl2sql_eval.reference_executor import GoldReferenceExecutor
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
    def search(self, message):
        candidate = SchemaCandidate(
            table_id="shop.items",
            table_name="items",
            ddl="CREATE TABLE items (value INTEGER);",
            score=1.0,
        )
        return SchemaLinkingResult(status="ok", context="<schema_evidence>items(value INTEGER)</schema_evidence>", candidates=[candidate])


class GraphFactory:
    def __init__(self, llm, max_attempts=2, max_reflections=2):
        self.llm = llm
        self.max_attempts = max_attempts
        self.max_reflections = max_reflections

    def create(self, prediction_input, executor):
        return NL2SQLGraphService(
            FakeSchemaLinkingService(),
            self.llm,
            SQLValidator(max_rows=10),
            executor,
            dialect="sqlite",
            max_attempts=self.max_attempts,
            max_reflections=self.max_reflections,
        )


def make_case_and_database(tmp_path):
    database_path = tmp_path / "shop.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE items (value INTEGER)")
        connection.execute("INSERT INTO items VALUES (7)")
    return BirdEvaluationCase(
        question_id="bird-1",
        db_id="shop",
        question="查询商品值",
        gold_sql="SELECT value FROM items",
        database_path=database_path,
        gold_tables=["items"],
    )


def test_workflow_reads_final_prediction_from_state_and_does_not_leak_gold_sql(tmp_path):
    case = make_case_and_database(tmp_path)
    llm = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT value AS result FROM items","tables":["items"],"parameters":{},"explanation":"查询"}'),
            LLMResponse.message('{"decision":"pass","reason":"结果符合问题"}'),
        ]
    )
    evaluator = NL2SQLWorkflowEvaluator(
        graph_factory=GraphFactory(llm),
        executor_factory=lambda item: ReadOnlySQLiteExecutor(item.database_path, 10, 10, 1),
        reference_executor_factory=lambda item: GoldReferenceExecutor(
            ReadOnlySQLiteExecutor(item.database_path, 10, 10, 1)
        ),
    )

    result = evaluator.evaluate_case(case)

    assert result.predicted_sql == "SELECT value AS result FROM items"
    assert result.final_execution_accuracy is True
    assert result.initial_execution_accuracy is True
    assert result.llm_calls == 2
    assert all(case.gold_sql not in str(messages) for messages, _ in llm.calls)


def test_workflow_records_reflection_recovery_and_clarify_without_extra_generation(tmp_path):
    case = make_case_and_database(tmp_path)
    recover_llm = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT value FROM items WHERE value = 999","tables":["items"],"parameters":{},"explanation":"错误"}'),
            LLMResponse.message('{"decision":"regenerate","reason":"字段不存在"}'),
            LLMResponse.message('{"status":"ok","sql":"SELECT value FROM items","tables":["items"],"parameters":{},"explanation":"修复"}'),
            LLMResponse.message('{"decision":"pass","reason":"正确"}'),
        ]
    )
    clarify_llm = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT value FROM items","tables":["items"],"parameters":{},"explanation":"查询"}'),
            LLMResponse.message('{"decision":"clarify","reason":"需要范围"}'),
        ]
    )

    def evaluate(llm):
        return NL2SQLWorkflowEvaluator(
            graph_factory=GraphFactory(llm),
            executor_factory=lambda item: ReadOnlySQLiteExecutor(item.database_path, 10, 10, 1),
            reference_executor_factory=lambda item: GoldReferenceExecutor(
                ReadOnlySQLiteExecutor(item.database_path, 10, 10, 1)
            ),
        ).evaluate_case(case)

    recovered = evaluate(recover_llm)
    clarified = evaluate(clarify_llm)

    assert recovered.reflection_recovery is True
    assert recovered.attempts == 2
    assert clarified.final_status == "clarify"
    assert clarified.attempts == 1
    assert clarified.final_execution_accuracy is False


def test_workflow_records_reject_and_stops_after_the_attempt_limit(tmp_path):
    case = make_case_and_database(tmp_path)
    reject_llm = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT value FROM items","tables":["items"],"parameters":{},"explanation":"查询"}'),
            LLMResponse.message('{"decision":"reject","reason":"请求不安全"}'),
        ]
    )
    limit_llm = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT value FROM items WHERE value = 999","tables":["items"],"parameters":{},"explanation":"查询"}'),
            LLMResponse.message('{"decision":"regenerate","reason":"重新生成"}'),
        ]
    )

    def evaluate(llm, **bounds):
        return NL2SQLWorkflowEvaluator(
            graph_factory=GraphFactory(llm, **bounds),
            executor_factory=lambda item: ReadOnlySQLiteExecutor(item.database_path, 10, 10, 1),
            reference_executor_factory=lambda item: GoldReferenceExecutor(
                ReadOnlySQLiteExecutor(item.database_path, 10, 10, 1)
            ),
        ).evaluate_case(case)

    rejected = evaluate(reject_llm)
    limited = evaluate(limit_llm, max_attempts=1)

    assert rejected.final_status == "rejected"
    assert rejected.attempts == 1
    assert limited.final_status == "failed"
    assert limited.max_attempt_reached is True
    assert limited.attempts == 1


def test_workflow_does_not_count_a_later_execution_as_initial_when_validation_regenerates(tmp_path):
    case = make_case_and_database(tmp_path)
    llm = FakeLLM(
        [
            LLMResponse.message('{"status":"ok","sql":"SELECT missing FROM items","tables":["items"],"parameters":{},"explanation":"错误"}'),
            LLMResponse.message('{"decision":"regenerate","reason":"字段不存在"}'),
            LLMResponse.message('{"status":"ok","sql":"SELECT value FROM items","tables":["items"],"parameters":{},"explanation":"修复"}'),
            LLMResponse.message('{"decision":"pass","reason":"正确"}'),
        ]
    )
    result = NL2SQLWorkflowEvaluator(
        graph_factory=GraphFactory(llm),
        executor_factory=lambda item: ReadOnlySQLiteExecutor(item.database_path, 10, 10, 1),
        reference_executor_factory=lambda item: GoldReferenceExecutor(
            ReadOnlySQLiteExecutor(item.database_path, 10, 10, 1)
        ),
    ).evaluate_case(case)

    assert result.initial_generation_valid is False
    assert result.initial_execution_success is False
    assert result.initial_execution_accuracy is False
    assert result.final_execution_accuracy is True
