from __future__ import annotations

from pydantic import BaseModel, Field

from app.rag.context_packer import ContextPacker
from app.rag.indexer import RagSchemaRecord, SchemaIndexer
from app.rag.retriever import SchemaRetriever
from app.rag.context_packer import RagContextBudgetExceeded
from app.rag.models import SchemaCandidate


class SchemaLinkingResult(BaseModel):
    status: str
    context: str = ""
    candidates: list[SchemaCandidate] = Field(default_factory=list)


class SchemaLinkingService:
    def __init__(self, indexer: SchemaIndexer, retriever: SchemaRetriever, packer: ContextPacker, records: list[RagSchemaRecord], namespace: str = "default") -> None:
        self.indexer = indexer
        self.retriever = retriever
        self.packer = packer
        self.records = records
        self.namespace = namespace
        self._indexed = False

    def search(self, message: str) -> SchemaLinkingResult:
        if not self._indexed:
            self.indexer.index(self.records)
            self._indexed = True
        result = self.retriever.search(message, self.namespace, limit=3)
        if result.status != "ok":
            return SchemaLinkingResult(status=result.status)
        context = self.packer.pack(result.candidates, budget=1200)
        if not context:
            raise RagContextBudgetExceeded("Schema evidence could not be packed safely.")
        return SchemaLinkingResult(status="ok", context=context, candidates=result.candidates)
