from __future__ import annotations

from pathlib import Path

import lancedb

from app.rag.models import RagDocument, RagSourceType, VectorHit


class LanceVectorStore:
    _TABLES = {RagSourceType.DDL: "ddl_documents", RagSourceType.SAMPLE_VALUE: "sample_value_documents"}

    def __init__(self, path: str | Path) -> None:
        self.db = lancedb.connect(str(path))

    def _has_table(self, name: str) -> bool:
        return name in self.db.list_tables().tables

    @staticmethod
    def _quote_filter_value(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def upsert(self, source_type: RagSourceType, entries: list[tuple[RagDocument, list[float]]]) -> None:
        if not entries:
            return
        name = self._TABLES[source_type]
        rows = [
            {
                "document_id": document.document_id,
                "source_type": document.source_type.value,
                "namespace": document.namespace,
                "table_id": document.table_id,
                "table_name": document.table_name,
                "column_name": document.column_name or "",
                "text": document.text,
                "content_hash": document.content_hash,
                "metadata": document.metadata,
                "vector": vector,
            }
            for document, vector in entries
        ]
        if self._has_table(name):
            table = self.db.open_table(name)
            ids = ",".join(self._quote_filter_value(row["document_id"]) for row in rows)
            table.delete(f"document_id IN ({ids})")
            table.add(rows)
        else:
            self.db.create_table(name, data=rows)

    def replace_namespace(self, source_type: RagSourceType, namespace: str, entries: list[tuple[RagDocument, list[float]]]) -> None:
        name = self._TABLES[source_type]
        if self._has_table(name):
            self.db.open_table(name).delete(f"namespace = {self._quote_filter_value(namespace)}")
        if entries:
            self.upsert(source_type, entries)

    def search(self, source_type: RagSourceType, vector: list[float], namespace: str, limit: int) -> list[VectorHit]:
        name = self._TABLES[source_type]
        if not self._has_table(name) or limit < 1:
            return []
        rows = self.db.open_table(name).search(vector).where(f"namespace = {self._quote_filter_value(namespace)}").limit(limit).to_list()
        hits: list[VectorHit] = []
        for rank, row in enumerate(rows, start=1):
            document = RagDocument(
                document_id=row["document_id"], source_type=row["source_type"], namespace=row["namespace"],
                table_id=row["table_id"], table_name=row["table_name"], column_name=row["column_name"] or None,
                text=row["text"], content_hash=row["content_hash"], metadata=row.get("metadata") or {},
            )
            hits.append(VectorHit(document=document, score=max(0.0, 1.0 - float(row.get("_distance", 0.0))), rank=rank))
        return hits
