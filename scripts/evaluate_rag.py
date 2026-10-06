from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embedding import QwenEmbeddingAdapter
from app.rag.indexer import SchemaIndexer
from app.rag.models import RagSourceType
from app.rag.retriever import SchemaRetriever
from app.rag.vector_store import LanceVectorStore
from app.rag_eval.bird_loader import load_bird_cases
from app.rag_eval.evaluator import SchemaRagEvaluator
from app.rag_eval.gold_tables import extract_gold_tables
from app.rag_eval.models import GoldTableSet
from app.rag_eval.report import write_reports
from app.rag_eval.sqlite_catalog import SQLiteCatalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the offline Schema RAG retriever against BIRD Dev.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cases = load_bird_cases(
        args.question_file,
        args.database_root,
        limit=args.limit,
        question_ids=set(args.question_id) or None,
    )
    embedding = QwenEmbeddingAdapter.from_env()
    store = LanceVectorStore(args.index_path)
    indexer = SchemaIndexer(embedding, store)
    catalog = SQLiteCatalog()
    records_by_db: dict[str, list] = {}
    for case in cases:
        records_by_db.setdefault(case.db_id, catalog.load(case.db_id, case.database_path))
    for db_id, records in records_by_db.items():
        if args.rebuild_index or not required_namespaces_ready(store, db_id):
            indexer.index(records)
    retriever = SchemaRetriever(embedding, store)
    evaluator = SchemaRagEvaluator(retriever)
    metrics = []
    failures = []
    for case in cases:
        try:
            gold = GoldTableSet(
                question_id=case.question_id,
                db_id=case.db_id,
                tables=extract_gold_tables(case.gold_sql),
            )
            metrics.extend(evaluator.evaluate_case(case, gold, len(records_by_db[case.db_id]), args.top_k))
        except ValueError as exc:
            failures.append({"question_id": case.question_id, "db_id": case.db_id, "reason": str(exc)})
    metadata = {
        "dataset": args.dataset_root.name if args.dataset_root else args.question_file.name,
        "question_file": args.question_file.name,
        "database_root": args.database_root.name,
        "case_count": len(cases),
        "top_k": sorted(set(args.top_k)),
        "rrf_k": retriever.rrf_k,
        "min_hit_score": retriever.min_hit_score,
        "ambiguity_margin": retriever.ambiguity_margin,
        "failed_cases": failures,
    }
    return write_reports(metrics, evaluator, args.output_dir, metadata)


def main() -> int:
    args = build_parser().parse_args()
    paths = run(args)
    print("\n".join(str(path) for path in paths))
    return 0


def required_namespaces_ready(
    store: LanceVectorStore,
    namespace: str,
    sources: tuple[RagSourceType, ...] | None = None,
) -> bool:
    required_sources = sources or (RagSourceType.DDL, RagSourceType.SAMPLE_VALUE)
    return all(store.has_namespace(source, namespace) for source in required_sources)


if __name__ == "__main__":
    raise SystemExit(main())
