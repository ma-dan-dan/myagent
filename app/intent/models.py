from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.memory.models import TokenUsage


class IntentName(str, Enum):
    CHAT = "chat"
    DATA_OPERATION = "data_operation"
    NL2SQL = "nl2sql"


class DataAction(str, Enum):
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    UNKNOWN = "unknown"


class IntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentName
    data_action: DataAction | None = None
    confidence: float = Field(ge=0, le=1)
    provider: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=100)
    latency_ms: float = Field(ge=0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    fallback_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_data_action(self) -> "IntentDecision":
        if self.intent != IntentName.DATA_OPERATION:
            self.data_action = None
        return self


class IntentRouteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: IntentDecision
    routed_intent: IntentName
    fallback_reason: str | None = Field(default=None, max_length=300)
