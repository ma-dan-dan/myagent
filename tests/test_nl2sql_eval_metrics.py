import json

from app.nl2sql_eval.evaluator import aggregate_workflow_metrics
from app.nl2sql_eval.models import WorkflowCaseResult
from app.nl2sql_eval.report import write_workflow_reports


def result(question_id, *, initial, final, recovered=False, status="ok", max_reached=False):
    return WorkflowCaseResult(
        question_id=question_id,
        db_id="shop",
        question="查询",
        gold_tables=["items"],
        retrieved_tables=["items"],
        schema_status="ok",
        gold_executed=True,
        gold_row_count=1,
        gold_truncated=False,
        predicted_sql="SELECT value FROM items",
        initial_generation_valid=True,
        initial_execution_success=True,
        initial_execution_accuracy=initial,
        final_status=status,
        final_execution_accuracy=final,
        reflection_recovery=recovered,
        reflection_decision="pass" if status == "ok" else status,
        attempts=2 if recovered else 1,
        reflection_count=1 if recovered else 0,
        max_attempt_reached=max_reached,
        latency_ms=10,
        llm_calls=2,
    )


def test_metrics_distinguish_initial_final_accuracy_and_reflection_recovery():
    summary = aggregate_workflow_metrics(
        [
            result("1", initial=True, final=True),
            result("2", initial=False, final=True, recovered=True),
            result("3", initial=False, final=False, status="clarify", max_reached=True),
        ],
        float_abs_tolerance=1e-2,
    )

    assert summary.initial_execution_accuracy == 1 / 3
    assert summary.final_execution_accuracy == 2 / 3
    assert summary.reflection_recovery_rate == 1 / 2
    assert summary.clarify_rate == 1 / 3
    assert summary.max_attempt_rate == 1 / 3
    assert summary.exact_sql_match_rate == 0


def test_report_writes_jsonl_json_and_csv_without_query_result_rows(tmp_path):
    paths = write_workflow_reports(
        [result("1", initial=True, final=True)],
        output_dir=tmp_path,
        metadata={"dataset": "fixture", "float_abs_tolerance": 1e-2},
    )

    assert all(path.is_file() for path in paths)
    case = json.loads(paths[0].read_text(encoding="utf-8").strip())
    summary = json.loads(paths[1].read_text(encoding="utf-8"))
    assert "query_result" not in case
    assert summary["metrics"]["final_execution_accuracy"] == 1.0
