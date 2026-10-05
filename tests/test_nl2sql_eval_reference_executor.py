import sqlite3

from app.nl2sql.executor import ReadOnlySQLiteExecutor
from app.nl2sql_eval.reference_executor import GoldReferenceExecutor


def make_database(tmp_path):
    path = tmp_path / "bird.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items (value INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])
    return path


def test_gold_reference_executor_uses_readonly_executor_and_returns_result(tmp_path):
    executor = ReadOnlySQLiteExecutor(make_database(tmp_path), max_rows=10, max_columns=10, timeout_seconds=1)

    outcome = GoldReferenceExecutor(executor).execute("SELECT value FROM items ORDER BY value")

    assert outcome.executed is True
    assert outcome.result is not None
    assert outcome.result.rows == [[1], [2]]


def test_gold_reference_executor_classifies_missing_database_and_rejects_writes(tmp_path):
    missing = GoldReferenceExecutor(
        ReadOnlySQLiteExecutor(tmp_path / "missing.sqlite", max_rows=10, max_columns=10, timeout_seconds=1)
    ).execute("SELECT 1")
    write = GoldReferenceExecutor(
        ReadOnlySQLiteExecutor(make_database(tmp_path), max_rows=10, max_columns=10, timeout_seconds=1)
    ).execute("DELETE FROM items")

    assert missing.executed is False
    assert missing.error_kind == "unavailable"
    assert write.executed is False
    assert write.error_kind == "execution_error"
