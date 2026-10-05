from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.nl2sql_eval.evaluator import aggregate_by_database, aggregate_workflow_metrics
from app.nl2sql_eval.models import WorkflowCaseResult


def write_workflow_reports(
    results: list[WorkflowCaseResult],
    *,
    output_dir: str | Path,
    metadata: dict[str, Any],
    float_abs_tolerance: float = 1e-2,
) -> tuple[Path, Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cases_path = destination / "cases.jsonl"
    summary_path = destination / "summary.json"
    csv_path = destination / "summary.csv"
    with cases_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")
    summary = aggregate_workflow_metrics(results, float_abs_tolerance=float_abs_tolerance)
    by_database = aggregate_by_database(results, float_abs_tolerance=float_abs_tolerance)
    by_rag_branch = {
        branch: aggregate_workflow_metrics(values, float_abs_tolerance=float_abs_tolerance)
        for branch, values in _group_by_rag_branch(results).items()
    }
    summary_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "metrics": summary.model_dump(mode="json"),
                "by_database": {db_id: value.model_dump(mode="json") for db_id, value in by_database.items()},
                "by_rag_branch": {branch: value.model_dump(mode="json") for branch, value in by_rag_branch.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    row = summary.model_dump(mode="json")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return cases_path, summary_path, csv_path


def _group_by_rag_branch(results: list[WorkflowCaseResult]) -> dict[str, list[WorkflowCaseResult]]:
    grouped: dict[str, list[WorkflowCaseResult]] = {}
    for result in results:
        grouped.setdefault(result.rag_branch, []).append(result)
    return grouped
