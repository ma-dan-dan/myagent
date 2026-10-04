from __future__ import annotations

import hashlib
import re

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
        namespaces = {record.namespace for record in records}
        for record in records:
            if record.ddl.strip():
                ddl_documents.append(self._document(record, RagSourceType.DDL, None, record.ddl))
            for column in record.columns:
                values = self._safe_values(column.column_name, column.sample_values)
                if values:
                    text = f"{record.table_name} {column.column_name} {column.data_type} samples: {', '.join(values)}"
                    sample_documents.append(self._document(record, RagSourceType.SAMPLE_VALUE, column.column_name, text))
        self._write(RagSourceType.DDL, ddl_documents, namespaces)
        self._write(RagSourceType.SAMPLE_VALUE, sample_documents, namespaces)

    def _write(
        self,
        source: RagSourceType,
        documents: list[RagDocument],
        namespaces: set[str],
    ) -> None:
        documents_by_namespace: dict[str, list[RagDocument]] = {}
        for document in documents:
            documents_by_namespace.setdefault(document.namespace, []).append(document)
        for namespace in namespaces:
            namespace_documents = documents_by_namespace.get(namespace, [])
            embeddings = self.embedding.embed([item.text for item in namespace_documents]) if namespace_documents else []
            self.vector_store.replace_namespace(source, namespace, list(zip(namespace_documents, embeddings)))

    @staticmethod
    def _safe_values(column_name: str, values: list[str]) -> list[str]:
        if re.search(r"password|passwd|secret|api[_-]?key|token|cookie|authorization|bearer|credential", column_name, re.I):
            return []
        sensitive_value = re.compile(
            r"password|passwd|secret|api[_-]?key|token|cookie|authorization|bearer|credential|\bsk-[\w-]+|^[\w-]+\.[\w-]+\.[\w-]+$",
            re.I,
        )
        return [
            value.strip()[:80]
            for value in dict.fromkeys(values)
            if value.strip() and not sensitive_value.search(value)
        ][:3]

    @staticmethod
    def _document(record: RagSchemaRecord, source: RagSourceType, column: str | None, text: str) -> RagDocument:
        identity = f"{source.value}:{record.namespace}:{record.table_id}" + (f":{column}" if column else "")
        return RagDocument(document_id=identity, source_type=source, namespace=record.namespace, table_id=record.table_id, table_name=record.table_name, column_name=column, text=text, content_hash=hashlib.sha256(text.encode()).hexdigest())
