from __future__ import annotations

from datetime import datetime, timezone

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recent_message_limit: int = Field(default=12, ge=2, le=50)
    compact_ratio: float = Field(default=0.80, gt=0.50, lt=0.95)
    fixed_context_ratio: float = Field(default=0.20, gt=0, lt=0.95)
    default_context_window: int = Field(default=16384, ge=4096)
    default_output_reserve: int = Field(default=2048, ge=256)
    summary_max_chars: int = Field(default=4000, ge=500, le=12000)
    minimum_summary_tokens: int = Field(default=512, ge=128)
    max_summary_calls_per_request: Literal[1, 2] = 2

    @model_validator(mode="after")
    def validate_ratios(self) -> "ContextPolicy":
        if self.fixed_context_ratio >= self.compact_ratio:
            raise ValueError("fixed_context_ratio must be smaller than compact_ratio")
        return self


class ContextBudgetExceeded(RuntimeError):
    def __init__(self, estimated_tokens: int, hard_input_budget: int, reason: str) -> None:
        self.estimated_tokens = estimated_tokens
        self.hard_input_budget = hard_input_budget
        self.reason = reason
        super().__init__(reason)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_context_tokens: int = Field(default=0, ge=0)
    context_window: int = Field(default=0, ge=0)

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_context_tokens=self.estimated_context_tokens + other.estimated_context_tokens,
            context_window=max(self.context_window, other.context_window),
        )


class SessionUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_turn: TokenUsage = Field(default_factory=TokenUsage)
    session_total: TokenUsage = Field(default_factory=TokenUsage)


class SummaryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=12000)
    summarized_through_message_id: int = Field(ge=1)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=300)


class CompactionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempted: bool = False
    compacted: bool = False
    summary: SummaryRecord | None = None
    summary_usage: TokenUsage = Field(default_factory=TokenUsage)
    memory_entries: list[MemoryEntry] = Field(default_factory=list)
    memory_usage: TokenUsage = Field(default_factory=TokenUsage)
