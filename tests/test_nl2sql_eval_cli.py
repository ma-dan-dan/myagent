from scripts.evaluate_nl2sql import WORKFLOW_DESCRIPTION, build_parser, build_workflow_evaluator


def test_nl2sql_evaluation_cli_accepts_bird_paths_limits_and_workflow_bounds():
    args = build_parser().parse_args(
        [
            "--question-file",
            "D:/bird/dev.json",
            "--database-root",
            "D:/bird/dev_databases",
            "--index-path",
            "artifacts/nl2sql-eval/lancedb",
            "--output-dir",
            "artifacts/nl2sql-eval/reports",
            "--limit",
            "20",
            "--question-id",
            "bird-1",
            "--top-k",
            "5",
            "--rag-branch",
            "ddl_only",
            "--max-attempts",
            "4",
            "--max-reflections",
            "2",
            "--rebuild-index",
        ]
    )

    assert args.limit == 20
    assert args.question_id == ["bird-1"]
    assert args.top_k == 5
    assert args.rag_branch == "ddl_only"
    assert args.max_attempts == 4
    assert args.max_reflections == 2
    assert args.rebuild_index is True


def test_cli_wires_float_tolerance_into_result_comparator():
    args = build_parser().parse_args(
        [
            "--question-file", "D:/bird/dev.json",
            "--database-root", "D:/bird/dev_databases",
            "--index-path", "artifacts/nl2sql-eval/lancedb",
            "--output-dir", "artifacts/nl2sql-eval/reports",
            "--float-abs-tolerance", "0.1",
        ]
    )

    evaluator = build_workflow_evaluator(args, runtime=object(), graph_factory=object())

    assert evaluator.comparator.float_abs_tolerance == 0.1


def test_cli_metadata_describes_only_the_five_node_workflow():
    assert WORKFLOW_DESCRIPTION == "SchemaLinking -> GenSQL -> Execute -> Reflection -> Output"
    assert "ContextPrepare" not in WORKFLOW_DESCRIPTION
    assert "ValidateSQL" not in WORKFLOW_DESCRIPTION
    assert "SQLGlot" not in WORKFLOW_DESCRIPTION
