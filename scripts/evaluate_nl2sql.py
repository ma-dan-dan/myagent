from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.llm_factory import LLMAdapterFactory
from app.config import get_nl2sql_runtime_config
from app.nl2sql.executor import ReadOnlySQLiteExecutor
from app.nl2sql_eval.evaluator import NL2SQLWorkflowEvaluator
from app.nl2sql_eval.factory import BirdGraphFactory
from app.nl2sql_eval.models import BirdEvaluationCase, WorkflowCaseResult
from app.nl2sql_eval.reference_executor import GoldReferenceExecutor
from app.nl2sql_eval.report import write_workflow_reports
from app.nl2sql_eval.result_comparator import ResultComparator
from app.rag.embedding import QwenEmbeddingAdapter
from app.rag.models import RagSourceType
from app.rag_eval.bird_loader import load_bird_cases
from app.rag_eval.gold_tables import extract_gold_tables


WORKFLOW_DESCRIPTION = "SchemaLinking -> GenSQL -> Execute -> Reflection -> Output"


def build_parser() -> argparse.ArgumentParser:
    runtime = get_nl2sql_runtime_config()
    parser = argparse.ArgumentParser(description="Evaluate the existing LangGraph NL2SQL workflow against BIRD Dev.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rag-branch", choices=["ddl_only", "sample_value_only", "fusion_rrf"], default="fusion_rrf")
    parser.add_argument("--max-attempts", type=int, default=runtime.max_attempts)
    parser.add_argument("--max-reflections", type=int, default=runtime.max_reflections)
    parser.add_argument("--float-abs-tolerance", type=float, default=1e-2)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.top_k < 1:
        raise ValueError("top_k 必须大于 0。")
    if args.max_attempts < 1 or args.max_reflections < 0:
        raise ValueError("max_attempts/max_reflections 配置无效。")
    cases = load_bird_cases(
        args.question_file,
        args.database_root,
        limit=args.limit,
        question_ids=set(args.question_id) or None,
    )
    runtime = get_nl2sql_runtime_config()
    llm = LLMAdapterFactory.from_env()
    embedding = QwenEmbeddingAdapter.from_env()
    graph_factory = BirdGraphFactory(
        llm,
        embedding,
        args.index_path,
        dialect=runtime.dialect,
        top_k=args.top_k,
        max_attempts=args.max_attempts,
        max_reflections=args.max_reflections,
        max_rows=runtime.max_rows,
        max_sql_length=runtime.max_sql_length,
        rebuild_index=args.rebuild_index,
        sources=_sources_for_branch(args.rag_branch),
    )
    evaluator = build_workflow_evaluator(args, runtime=runtime, graph_factory=graph_factory)
    results: list[WorkflowCaseResult] = []
    for bird_case in cases:
        try:
            evaluation_case = BirdEvaluationCase(
                question_id=bird_case.question_id,
                db_id=bird_case.db_id,
                question=bird_case.question,
                gold_sql=bird_case.gold_sql,
                database_path=bird_case.database_path,
                gold_tables=extract_gold_tables(bird_case.gold_sql),
            )
            results.append(evaluator.evaluate_case(evaluation_case))
        except Exception as exc:
            results.append(
                WorkflowCaseResult(
                    question_id=bird_case.question_id,
                    db_id=bird_case.db_id,
                    question=bird_case.question,
                    rag_branch=args.rag_branch,
                    gold_tables=[],
                    retrieved_tables=[],
                    schema_status="failed",
                    gold_executed=False,
                    gold_error_kind="setup_error",
                    initial_generation_valid=False,
                    initial_execution_success=False,
                    final_status="failed",
                    attempts=0,
                    reflection_count=0,
                    latency_ms=0,
                    llm_calls=0,
                    error=str(exc),
                )
            )
    metadata = {
        "dataset": args.dataset_root.name if args.dataset_root else args.question_file.name,
        "question_file_sha256": hashlib.sha256(args.question_file.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "top_k": args.top_k,
        "rag_branch": args.rag_branch,
        "max_attempts": args.max_attempts,
        "max_reflections": args.max_reflections,
        "float_abs_tolerance": args.float_abs_tolerance,
        "workflow": WORKFLOW_DESCRIPTION,
    }
    return write_workflow_reports(
        results,
        output_dir=args.output_dir,
        metadata=metadata,
        float_abs_tolerance=args.float_abs_tolerance,
    )


def build_workflow_evaluator(
    args: argparse.Namespace,
    *,
    runtime: object,
    graph_factory: BirdGraphFactory,
) -> NL2SQLWorkflowEvaluator:
    return NL2SQLWorkflowEvaluator(
        graph_factory=graph_factory,
        executor_factory=lambda case: ReadOnlySQLiteExecutor(
            case.database_path,
            runtime.max_rows,
            runtime.max_columns,
            runtime.query_timeout_seconds,
        ),
        reference_executor_factory=lambda case: GoldReferenceExecutor(
            ReadOnlySQLiteExecutor(
                case.database_path,
                runtime.max_rows,
                runtime.max_columns,
                runtime.query_timeout_seconds,
            )
        ),
        comparator=ResultComparator(float_abs_tolerance=args.float_abs_tolerance),
        rag_branch=args.rag_branch,
    )


def _sources_for_branch(branch: str) -> tuple[RagSourceType, ...] | None:
    if branch == "ddl_only":
        return (RagSourceType.DDL,)
    if branch == "sample_value_only":
        return (RagSourceType.SAMPLE_VALUE,)
    return None


def main() -> int:
    paths = run(build_parser().parse_args())
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
