from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from app.rag_eval.evaluator import SchemaRagEvaluator
from app.rag_eval.models import RetrievalCaseMetric


def write_reports(
    metrics: Sequence[RetrievalCaseMetric],
    evaluator: SchemaRagEvaluator,
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cases_path = destination / "cases.jsonl"
    summary_path = destination / "summary.json"
    csv_path = destination / "summary.csv"
    with cases_path.open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric.model_dump(mode="json"), ensure_ascii=False) + "\n")
    aggregate = evaluator.aggregate(metrics)
    by_database = evaluator.aggregate_by_database(metrics)
    summary_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "metrics": [item.model_dump(mode="json") for item in aggregate],
                "by_database": {
                    db_id: [item.model_dump(mode="json") for item in values]
                    for db_id, values in by_database.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        rows = [item.model_dump(mode="json") for item in aggregate]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["branch", "top_k"])
        writer.writeheader()
        writer.writerows(rows)
    return cases_path, summary_path, csv_path
