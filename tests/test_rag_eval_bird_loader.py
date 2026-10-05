import json

import pytest

from app.rag_eval.bird_loader import BirdDatasetError, load_bird_cases


def test_load_bird_cases_maps_db_id_and_preserves_question_text(tmp_path):
    database = tmp_path / "dev_databases" / "shop" / "shop.sqlite"
    database.parent.mkdir(parents=True)
    database.touch()
    questions = tmp_path / "dev.json"
    questions.write_text(
        json.dumps([{"question_id": "q-1", "db_id": "shop", "question": "查询订单", "SQL": "SELECT * FROM orders"}]),
        encoding="utf-8",
    )

    cases = load_bird_cases(questions, tmp_path / "dev_databases")

    assert cases[0].database_path == database
    assert cases[0].question == "查询订单"
    assert cases[0].gold_sql == "SELECT * FROM orders"


def test_load_bird_cases_fails_when_database_is_missing(tmp_path):
    questions = tmp_path / "dev.json"
    questions.write_text(json.dumps([{"db_id": "missing", "question": "查询", "SQL": "SELECT 1"}]), encoding="utf-8")

    with pytest.raises(BirdDatasetError, match="missing"):
        load_bird_cases(questions, tmp_path / "dev_databases")


def test_load_bird_cases_filters_question_id_and_generates_missing_ids(tmp_path):
    root = tmp_path / "dev_databases"
    for db_id in ("shop", "sales"):
        database = root / db_id / f"{db_id}.sqlite"
        database.parent.mkdir(parents=True)
        database.touch()
    questions = tmp_path / "dev.json"
    questions.write_text(
        json.dumps(
            [
                {"question_id": "keep", "db_id": "shop", "question": "查询订单", "sql": "SELECT * FROM orders"},
                {"db_id": "sales", "question": "查询销售", "SQL": "SELECT * FROM sales"},
            ]
        ),
        encoding="utf-8",
    )

    selected = load_bird_cases(questions, root, question_ids={"keep"})
    generated = load_bird_cases(questions, root, question_ids={"record-1"})

    assert [case.question_id for case in selected] == ["keep"]
    assert generated[0].question_id_source == "generated"
