from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.memory.models import TokenUsage
from app.rag.models import SchemaCandidate


class SQLDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "clarify", "reject"]
    sql: str | None = Field(default=None, max_length=12000)
    tables: list[str] = Field(default_factory=list)
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    explanation: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_sql_for_status(self) -> "SQLDraft":
        if self.status == "ok" and not (self.sql or "").strip():
            raise ValueError("sql is required when status is ok")
        if self.status != "ok" and self.sql is not None:
            raise ValueError("sql must be empty when status is not ok")
        return self


class SQLValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    normalized_sql: str | None = None
    referenced_tables: list[str] = Field(default_factory=list)
    referenced_columns: list[str] = Field(default_factory=list)
    error: str | None = Field(default=None, max_length=500)


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    truncated: bool = False
    error: str | None = Field(default=None, max_length=500)


class ReflectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["pass", "regenerate", "clarify", "reject"]
    reason: str = Field(min_length=1, max_length=1000)


class NL2SQLState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    user_message: str = Field(min_length=1, max_length=4000)
    schema_candidates: list[SchemaCandidate] = Field(default_factory=list)
    schema_context: str = ""
    sql_draft: SQLDraft | None = None
    validation: SQLValidationResult | None = None
    query_result: QueryResult | None = None
    reflection: ReflectionDecision | None = None
    validation_error: str | None = None
    execution_error: str | None = None
    attempt: int = Field(default=0, ge=0)
    reflection_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    max_reflections: int = Field(default=2, ge=0, le=10)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    status: str = "running"
    final_message: str | None = None
