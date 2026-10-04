from __future__ import annotations

import json
from pathlib import Path

from app.schemas.chat import SchemaTable


class SchemaCatalog:
    """Loads and searches de-identified schema metadata only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            raw_items = json.load(handle)
        self._tables = [SchemaTable.model_validate(item) for item in raw_items]

    def search(self, query: str, limit: int) -> list[SchemaTable]:
        needle = query.casefold()
        matches: list[SchemaTable] = []
        for table in self._tables:
            searchable = " ".join(
                [
                    table.table_name,
                    table.description,
                    *(
                        value
                        for column in table.columns
                        for value in (column.column_name, column.data_type, column.description)
                    ),
                ]
            ).casefold()
            if needle in searchable:
                matches.append(table)
            if len(matches) >= limit:
                break
        return matches

    @property
    def tables(self) -> list[SchemaTable]:
        return list(self._tables)
