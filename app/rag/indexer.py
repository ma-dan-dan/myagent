from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from app.rag.embedding import EmbeddingAdapter
from app.rag.models import RagDocument, RagSourceType
from app.rag.vector_store import LanceVectorStore


class RagColumnRecord(BaseModel):
    column_name: str
    data_type: str
    sample_values: list[str] = Field(default_factory=list)


class RagSchemaRecord(BaseModel):
    namespace: str = "default"
    table_id: str
    table_name: str
    ddl: str
    columns: list[RagColumnRecord] = Field(default_factory=list)


class SchemaIndexer:
    def __init__(self, embedding: EmbeddingAdapter, vector_store: LanceVectorStore) -> None:
        self.embedding = embedding
        self.vector_store = vector_store

    def index(self, records: list[RagSchemaRecord]) -> None:
        ddl_documents: list[RagDocument] = []
        sample_documents: list[RagDocument] = []
        for record in records:
            if record.ddl.strip():
                ddl_documents.append(self._document(record, RagSourceType.DDL, None, record.ddl))
            for column in record.columns:
                values = [value.strip()[:80] for value in dict.fromkeys(column.sample_values) if value.strip()][:3]
                if values:
                    text = f"{record.table_name} {column.column_name} {column.data_type} samples: {', '.join(values)}"
                    sample_documents.append(self._document(record, RagSourceType.SAMPLE_VALUE, column.column_name, text))
        self._write(RagSourceType.DDL, ddl_documents)
        self._write(RagSourceType.SAMPLE_VALUE, sample_documents)

    def _write(self, source: RagSourceType, documents: list[RagDocument]) -> None:
        if documents:
            self.vector_store.upsert(source, list(zip(documents, self.embedding.embed([item.text for item in documents]))))

    @staticmethod
    def _document(record: RagSchemaRecord, source: RagSourceType, column: str | None, text: str) -> RagDocument:
        identity = f"{source.value}:{record.namespace}:{record.table_id}" + (f":{column}" if column else "")
        return RagDocument(document_id=identity, source_type=source, namespace=record.namespace, table_id=record.table_id, table_name=record.table_name, column_name=column, text=text, content_hash=hashlib.sha256(text.encode()).hexdigest())
