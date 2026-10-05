from scripts.evaluate_rag import build_parser


def test_evaluation_cli_accepts_explicit_paths_and_benchmark_controls():
    args = build_parser().parse_args(
        [
            "--question-file", "dev.json",
            "--database-root", "dev_databases",
            "--index-path", "artifacts/index",
            "--output-dir", "artifacts/reports",
            "--top-k", "1", "3", "5", "10",
            "--limit", "2",
            "--question-id", "q1",
            "--rebuild-index",
        ]
    )

    assert args.top_k == [1, 3, 5, 10]
    assert args.limit == 2
    assert args.question_id == ["q1"]
    assert args.rebuild_index is True
