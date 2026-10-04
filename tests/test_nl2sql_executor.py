import sqlite3

import pytest

from app.nl2sql.executor import (
    ReadOnlySQLiteExecutor,
    SQLExecutionError,
    SQLExecutorUnavailable,
    UnavailableSQLExecutor,
)


def make_database(tmp_path):
    path = tmp_path / "business.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE production_output (output_quantity INTEGER, note TEXT)")
        connection.executemany("INSERT INTO production_output VALUES (?, ?)", [(10, "a"), (20, "b"), (30, "c")])
    return path


def test_readonly_executor_returns_bounded_rows(tmp_path):
    executor = ReadOnlySQLiteExecutor(make_database(tmp_path), max_rows=2, max_columns=1, timeout_seconds=1)

    result = executor.execute("SELECT output_quantity, note FROM production_output", {})

    assert result.columns == ["output_quantity"]
    assert result.rows == [[10], [20]]
    assert result.truncated is True


def test_readonly_executor_rejects_write_sql(tmp_path):
    executor = ReadOnlySQLiteExecutor(make_database(tmp_path), max_rows=2, max_columns=2, timeout_seconds=1)

    with pytest.raises(SQLExecutionError, match="只读"):
        executor.execute("DELETE FROM production_output", {})


def test_readonly_executor_uses_bound_parameters(tmp_path):
    executor = ReadOnlySQLiteExecutor(make_database(tmp_path), max_rows=2, max_columns=2, timeout_seconds=1)

    result = executor.execute("SELECT output_quantity FROM production_output WHERE note = :note", {"note": "b"})

    assert result.rows == [[20]]


def test_unavailable_executor_has_stable_configuration_error():
    with pytest.raises(SQLExecutorUnavailable, match="业务数据库"):
        UnavailableSQLExecutor().execute("SELECT 1", {})


def test_readonly_executor_interrupts_query_after_timeout(tmp_path):
    executor = ReadOnlySQLiteExecutor(make_database(tmp_path), max_rows=2, max_columns=2, timeout_seconds=0)

    with pytest.raises(SQLExecutionError, match="执行失败"):
        executor.execute("SELECT output_quantity FROM production_output", {})


def test_sqlite_url_is_normalized_to_a_file_path(tmp_path):
    database_path = make_database(tmp_path)

    executor = ReadOnlySQLiteExecutor.from_database_url(
        f"sqlite:///{database_path.as_posix()}",
        max_rows=100,
        max_columns=30,
        timeout_seconds=5,
    )

    assert executor.database_path == database_path


def test_unknown_database_url_scheme_is_rejected():
    with pytest.raises(SQLExecutorUnavailable, match="仅支持 SQLite"):
        ReadOnlySQLiteExecutor.from_database_url("postgresql://localhost/business", 100, 30, 5)


def test_missing_database_file_raises_unavailable(tmp_path):
    executor = ReadOnlySQLiteExecutor(tmp_path / "missing.sqlite3", 100, 30, 5)

    with pytest.raises(SQLExecutorUnavailable, match="不可用"):
        executor.execute("SELECT 1", {})
