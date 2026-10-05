import json

from app.rag_eval.evaluator import SchemaRagEvaluator
from app.rag_eval.models import RetrievalCaseMetric
from app.rag_eval.report import write_reports


class EmptyRetriever:
    rrf_k = 60
    min_hit_score = 0.2
    ambiguity_margin = 0.001


def test_write_reports_creates_jsonl_json_and_csv_without_sensitive_runtime_configuration(tmp_path):
    metric = RetrievalCaseMetric(
        question_id="q1",
        question_id_source="dataset",
        db_id="shop",
        branch="ddl_only",
        top_k=1,
        gold_tables=["orders"],
        retrieved_tables=["shop.orders"],
        matched_tables=["orders"],
        recall=1,
        full_recall=True,
        precision=1,
        database_coverage=0.5,
        empty=False,
        ambiguous=False,
        false_positive_count=0,
        elapsed_ms=1.5,
    )

    cases_path, summary_path, csv_path = write_reports(
        [metric],
        SchemaRagEvaluator(EmptyRetriever()),
        tmp_path / "reports",
        {"dataset": "bird-dev", "rrf_k": 60},
    )

    assert json.loads(cases_path.read_text(encoding="utf-8"))[
        "retrieved_tables"
    ] == ["shop.orders"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"][0]["macro_recall"] == 1
    assert summary["by_database"]["shop"][0]["micro_recall"] == 1
    assert "ddl_only" in csv_path.read_text(encoding="utf-8")
