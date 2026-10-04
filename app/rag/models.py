from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RagSourceType(str, Enum):
    DDL = "ddl"
    SAMPLE_VALUE = "sample_value"


class RagDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    source_type: RagSourceType
    namespace: str = Field(min_length=1)
    table_id: str = Field(min_length=1)
    table_name: str = Field(min_length=1)
    column_name: str | None = None
    text: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_column(self) -> "RagDocument":
        if self.source_type is RagSourceType.SAMPLE_VALUE and not self.column_name:
            raise ValueError("sample_value documents require column_name")
        return self


class VectorHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: RagDocument
    score: float = Field(ge=0)
    rank: int = Field(ge=1)


class SchemaCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str = Field(min_length=1)
    table_name: str = Field(min_length=1)
    ddl: str = ""
    matched_columns: list[str] = Field(default_factory=list)
    sample_values: dict[str, list[str]] = Field(default_factory=dict)
    score: float = Field(ge=0)
    matched_by: list[RagSourceType] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    candidates: list[SchemaCandidate] = Field(default_factory=list)
