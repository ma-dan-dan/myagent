from __future__ import annotations

import time
from collections import defaultdict
from typing import Protocol

from app.nl2sql.executor import SQLExecutionError, SQLExecutor
from app.nl2sql.models import NL2SQLState, QueryResult
from app.nl2sql_eval.models import BirdEvaluationCase, PredictionInput, WorkflowCaseResult, WorkflowMetricSummary
from app.nl2sql_eval.reference_executor import GoldReferenceExecutor
from app.nl2sql_eval.result_comparator import ResultComparator


class EvaluationGraph(Protocol):
    def invoke(self, user_id: str, session_id: str, user_message: str) -> NL2SQLState:
        """Run the existing LangGraph workflow."""


class EvaluationGraphFactory(Protocol):
    def create(self, prediction_input: PredictionInput, executor: SQLExecutor) -> EvaluationGraph:
        """Create a graph bound to the case database without Gold information."""


class EvaluationExecutorFactory(Protocol):
    def __call__(self, case: BirdEvaluationCase) -> SQLExecutor:
        """Create an isolated read-only executor for one BIRD database."""


class ReferenceExecutorFactory(Protocol):
    def __call__(self, case: BirdEvaluationCase) -> GoldReferenceExecutor:
        """Create the side-channel Gold executor for one BIRD database."""


class RecordedExecution:
    def __init__(self, result: QueryResult | None, error: str | None) -> None:
        self.result = result
        self.error = error


class RecordingSQLExecutor:
    def __init__(self, executor: SQLExecutor) -> None:
        self.executor = executor
        self.executions: list[RecordedExecution] = []

    def execute(self, sql: str, parameters: dict[str, object]) -> QueryResult:
        try:
            result = self.executor.execute(sql, parameters)
        except SQLExecutionError as exc:
            self.executions.append(RecordedExecution(None, str(exc)))
            raise
        self.executions.append(RecordedExecution(result, None))
        return result


class NL2SQLWorkflowEvaluator:
    def __init__(
        self,
        graph_factory: EvaluationGraphFactory,
        executor_factory: EvaluationExecutorFactory,
        reference_executor_factory: ReferenceExecutorFactory,
        comparator: ResultComparator | None = None,
        rag_branch: str = "fusion_rrf",
    ) -> None:
        self.graph_factory = graph_factory
        self.executor_factory = executor_factory
        self.reference_executor_factory = reference_executor_factory
        self.comparator = comparator or ResultComparator()
        self.rag_branch = rag_branch

    def evaluate_case(self, case: BirdEvaluationCase) -> WorkflowCaseResult:
        reference = self.reference_executor_factory(case).execute(case.gold_sql)
        started = time.perf_counter()
        recording_executor = RecordingSQLExecutor(self.executor_factory(case))
        state: NL2SQLState | None = None
        trace: list[tuple[str, NL2SQLState]] = []
        graph_error: str | None = None
        graph: EvaluationGraph | None = None
        try:
            graph = self.graph_factory.create(case.prediction_input(), recording_executor)
            invoke_with_trace = getattr(graph, "invoke_with_trace", None)
            if callable(invoke_with_trace):
                state, trace = invoke_with_trace(f"bird-eval:{case.db_id}", f"bird-eval:{case.question_id}", case.question)
            else:
                state = graph.invoke(f"bird-eval:{case.db_id}", f"bird-eval:{case.question_id}", case.question)
        except Exception as exc:
            graph_error = str(exc)
        latency_ms = (time.perf_counter() - started) * 1000
        initial, initial_generation_valid, initial_sql = _initial_cycle(trace, recording_executor.executions, predicted_sql=None)
        predicted_result = state.query_result if state is not None else None
        predicted_sql = state.sql_draft.sql if state and state.sql_draft else None
        comparison = self.comparator.compare(
            predicted_result,
            reference.result,
            predicted_sql=predicted_sql,
            gold_sql=case.gold_sql,
            prediction_error=graph_error or (state.execution_error if state else None),
            gold_error=reference.error,
        )
        initial_comparison = self.comparator.compare(
            initial.result if initial else None,
            reference.result,
            predicted_sql=initial_sql,
            gold_sql=case.gold_sql,
            prediction_error=initial.error if initial else "first execution was not reached",
            gold_error=reference.error,
        )
        retrieved_tables = [candidate.table_name for candidate in state.schema_candidates] if state else []
        schema_status = "ok" if retrieved_tables else (state.status if state else "failed")
        matched_tables = set(name.lower() for name in retrieved_tables) & set(name.lower() for name in case.gold_tables)
        schema_recall = len(matched_tables) / len(case.gold_tables)
        final_accuracy = (
            comparison.execution_accuracy
            if reference.executed and not (reference.result and reference.result.truncated) and state and state.status == "ok"
            else None if not reference.executed or (reference.result and reference.result.truncated) else False
        )
        initial_accuracy = initial_comparison.execution_accuracy if reference.executed and not (reference.result and reference.result.truncated) else None
        reflection_decision = state.reflection.decision if state and state.reflection else None
        attempts = state.attempt if state else 0
        reflection_count = state.reflection_count if state else 0
        max_attempt_reached = bool(
            state
            and (
                state.attempt >= state.max_attempts
                or (state.reflection_count >= state.max_reflections and reflection_decision == "regenerate")
            )
        )
        return WorkflowCaseResult(
            question_id=case.question_id,
            db_id=case.db_id,
            question=case.question,
            rag_branch=self.rag_branch,
            gold_tables=sorted(case.gold_tables),
            retrieved_tables=retrieved_tables,
            schema_status=schema_status,
            schema_recall=schema_recall,
            gold_executed=reference.executed,
            gold_row_count=reference.result.row_count if reference.result else None,
            gold_truncated=bool(reference.result and reference.result.truncated),
            gold_error_kind=reference.error_kind,
            predicted_sql=predicted_sql,
            initial_generation_valid=initial_generation_valid,
            initial_execution_success=bool(initial and initial.error is None),
            initial_execution_accuracy=initial_accuracy,
            final_status=state.status if state else "failed",
            final_execution_accuracy=final_accuracy,
            reflection_recovery=initial_accuracy is False and final_accuracy is True,
            reflection_decision=reflection_decision,
            attempts=attempts,
            reflection_count=reflection_count,
            max_attempt_reached=max_attempt_reached,
            latency_ms=latency_ms,
            llm_calls=_llm_call_count(graph),
            comparison=comparison,
            error=graph_error,
        )


def aggregate_workflow_metrics(
    results: list[WorkflowCaseResult],
    *,
    float_abs_tolerance: float = 1e-2,
) -> WorkflowMetricSummary:
    count = len(results)
    if not count:
        return WorkflowMetricSummary(
            case_count=0,
            gold_valid_rate=0,
            initial_generation_valid_rate=0,
            initial_execution_success_rate=0,
            initial_execution_accuracy=0,
            final_execution_accuracy=0,
            exact_sql_match_rate=0,
            reflection_recovery_rate=0,
            clarify_rate=0,
            reject_rate=0,
            max_attempt_rate=0,
            average_latency_ms=0,
            average_llm_calls=0,
            average_attempts=0,
            average_reflections=0,
            average_schema_recall=0,
            float_abs_tolerance=float_abs_tolerance,
        )
    valid = [item for item in results if item.gold_executed and not item.gold_truncated]
    initially_wrong = [item for item in valid if item.initial_execution_accuracy is False]
    return WorkflowMetricSummary(
        case_count=count,
        gold_valid_rate=len(valid) / count,
        initial_generation_valid_rate=sum(item.initial_generation_valid for item in results) / count,
        initial_execution_success_rate=sum(item.initial_execution_success for item in results) / count,
        initial_execution_accuracy=sum(item.initial_execution_accuracy is True for item in valid) / len(valid) if valid else 0,
        final_execution_accuracy=sum(item.final_execution_accuracy is True for item in valid) / len(valid) if valid else 0,
        exact_sql_match_rate=sum(bool(item.comparison and item.comparison.exact_sql_match) for item in valid) / len(valid) if valid else 0,
        reflection_recovery_rate=sum(item.reflection_recovery for item in initially_wrong) / len(initially_wrong) if initially_wrong else 0,
        clarify_rate=sum(item.final_status == "clarify" for item in results) / count,
        reject_rate=sum(item.final_status == "rejected" for item in results) / count,
        max_attempt_rate=sum(item.max_attempt_reached for item in results) / count,
        average_latency_ms=sum(item.latency_ms for item in results) / count,
        average_llm_calls=sum(item.llm_calls for item in results) / count,
        average_attempts=sum(item.attempts for item in results) / count,
        average_reflections=sum(item.reflection_count for item in results) / count,
        average_schema_recall=sum(item.schema_recall or 0 for item in results) / count,
        float_abs_tolerance=float_abs_tolerance,
    )


def aggregate_by_database(results: list[WorkflowCaseResult], *, float_abs_tolerance: float = 1e-2) -> dict[str, WorkflowMetricSummary]:
    grouped: dict[str, list[WorkflowCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.db_id].append(result)
    return {
        db_id: aggregate_workflow_metrics(values, float_abs_tolerance=float_abs_tolerance)
        for db_id, values in sorted(grouped.items())
    }


def _llm_call_count(graph: EvaluationGraph | None) -> int:
    llm = getattr(graph, "llm", None)
    if isinstance(getattr(llm, "call_count", None), int):
        return llm.call_count
    calls = getattr(llm, "calls", None)
    return len(calls) if isinstance(calls, list) else 0


def _initial_cycle(
    trace: list[tuple[str, NL2SQLState]],
    executions: list[RecordedExecution],
    *,
    predicted_sql: str | None,
) -> tuple[RecordedExecution | None, bool, str | None]:
    if not trace:
        first = executions[0] if executions else None
        return first, first is not None, predicted_sql
    first_sql: str | None = None
    first_execute_seen = False
    for node_name, node_state in trace:
        if node_name == "gen_sql" and node_state.sql_draft and node_state.sql_draft.status == "ok":
            first_sql = node_state.sql_draft.sql
        if node_name == "execute_sql":
            first_execute_seen = True
            result = node_state.query_result
            return RecordedExecution(result, node_state.execution_error), True, first_sql
        if node_name == "reflection":
            break
    return None, first_execute_seen, first_sql
