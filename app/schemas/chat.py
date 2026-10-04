from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.memory.models import SessionUsage
from app.intent.models import IntentDecision, IntentName


def _trimmed(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("value must not be blank")
    return value


class SearchSchemaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return _trimmed(value)


class SchemaColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(min_length=1, max_length=100)
    data_type: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)


class SchemaTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    columns: list[SchemaColumn]


class SearchSchemaOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "empty", "error"]
    message: str = Field(max_length=500)
    items: list[SchemaTable] = Field(default_factory=list)


class ToolEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal["search_schema"]
    input: SearchSchemaInput
    status: Literal["ok", "empty", "error"]
    summary: str = Field(max_length=500)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=100)
    session_id: str | None = Field(default=None, max_length=100)
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("user_id", "session_id", "message")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _trimmed(value)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    message: str
    tool_events: list[ToolEvent] = Field(default_factory=list)
    usage: SessionUsage = Field(default_factory=SessionUsage)
    intent_decision: IntentDecision
    routed_intent: IntentName
    fallback_reason: str | None = Field(default=None, max_length=300)


class IntentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    request_count: int = Field(ge=0)
    classified_intent_counts: dict[str, int]
    routed_intent_counts: dict[str, int]
    avg_latency_ms: float = Field(ge=0)
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)


class StoredMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    role: Literal["user", "assistant", "tool"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
