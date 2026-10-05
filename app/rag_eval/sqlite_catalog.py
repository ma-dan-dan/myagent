from __future__ import annotations

import sqlite3
from pathlib import Path

from app.rag.indexer import RagColumnRecord, RagSchemaRecord, SchemaIndexer


class SQLiteCatalogError(ValueError):
    pass


class SQLiteCatalog:
    def __init__(self, sample_limit: int = 3) -> None:
        if sample_limit < 1:
            raise ValueError("sample_limit 必须大于 0。")
        self.sample_limit = sample_limit

    def load(self, db_id: str, database_path: str | Path) -> list[RagSchemaRecord]:
        path = Path(database_path)
        if not path.is_file():
            raise SQLiteCatalogError(f"SQLite 数据库不存在：{path}")
        try:
            connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
            rows = connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            records = [self._record(connection, db_id, str(name), str(ddl or "")) for name, ddl in rows if ddl]
        except sqlite3.Error as exc:
            raise SQLiteCatalogError("无法读取 SQLite Schema Catalog。") from exc
        finally:
            if "connection" in locals():
                connection.close()
        return records

    def _record(self, connection: sqlite3.Connection, db_id: str, table_name: str, ddl: str) -> RagSchemaRecord:
        columns: list[RagColumnRecord] = []
        for _, column_name, data_type, *_ in connection.execute(f"PRAGMA table_info({_quote_identifier(table_name)})"):
            values = self._sample_values(connection, table_name, str(column_name))
            columns.append(RagColumnRecord(column_name=str(column_name), data_type=str(data_type or "TEXT"), sample_values=values))
        return RagSchemaRecord(
            namespace=db_id,
            table_id=f"{db_id}.{table_name}",
            table_name=table_name,
            ddl=ddl,
            columns=columns,
        )

    def _sample_values(self, connection: sqlite3.Connection, table_name: str, column_name: str) -> list[str]:
        query = f"SELECT {_quote_identifier(column_name)} FROM {_quote_identifier(table_name)} WHERE {_quote_identifier(column_name)} IS NOT NULL LIMIT ?"
        values = [str(row[0]) for row in connection.execute(query, (self.sample_limit,)).fetchall()]
        return SchemaIndexer._safe_values(column_name, values)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
