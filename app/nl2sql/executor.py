from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Protocol

from app.nl2sql.models import QueryResult


class SQLExecutionError(RuntimeError):
    """Raised when a bounded read-only SQL execution cannot complete."""


class SQLExecutorUnavailable(SQLExecutionError):
    """Raised when no dedicated business database is configured."""


class SQLUnsafeQueryError(SQLExecutionError):
    """Raised when generated SQL fails deterministic safety validation."""


class SQLExecutor(Protocol):
    def execute(self, sql: str, parameters: dict[str, object]) -> QueryResult:
        """Execute a pre-validated SQL query using bound parameters."""


class UnavailableSQLExecutor:
    def execute(self, sql: str, parameters: dict[str, object]) -> QueryResult:
        raise SQLExecutorUnavailable("NL2SQL 业务数据库未配置。")


class ReadOnlySQLiteExecutor:
    def __init__(self, database_path: str | Path, max_rows: int, max_columns: int, timeout_seconds: int) -> None:
        self.database_path = Path(database_path)
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.timeout_seconds = timeout_seconds

    def execute(self, sql: str, parameters: dict[str, object]) -> QueryResult:
        normalized = (sql or "").strip()
        if not normalized.upper().startswith(("SELECT", "WITH")) or ";" in normalized.rstrip(";"):
            raise SQLExecutionError("Executor 只允许单条只读 SELECT/WITH SQL。")
        try:
            connection = sqlite3.connect(
                f"file:{self.database_path.resolve().as_posix()}?mode=ro",
                uri=True,
                timeout=self.timeout_seconds,
            )
            connection.set_authorizer(self._authorizer)
            deadline = time.monotonic() + self.timeout_seconds
            connection.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 1)
            cursor = connection.execute(normalized, parameters)
            rows = cursor.fetchmany(self.max_rows + 1)
            source_columns = [column[0] for column in cursor.description or []]
        except sqlite3.Error as exc:
            raise SQLExecutionError("SQL 查询执行失败。") from exc
        finally:
            if "connection" in locals():
                connection.close()
        truncated = len(rows) > self.max_rows or len(source_columns) > self.max_columns
        bounded_rows = [list(row[: self.max_columns]) for row in rows[: self.max_rows]]
        return QueryResult(
            columns=source_columns[: self.max_columns],
            rows=bounded_rows,
            row_count=len(bounded_rows),
            truncated=truncated,
        )

    @staticmethod
    def _authorizer(action: int, parameter1: str | None, parameter2: str | None, database: str | None, trigger: str | None) -> int:
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
