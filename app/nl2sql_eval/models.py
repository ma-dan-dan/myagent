from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.nl2sql.models import QueryResult


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    db_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    database_path: Path


class BirdEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    db_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_sql: str = Field(min_length=1)
    database_path: Path
    gold_tables: list[str] = Field(min_length=1)

    def prediction_input(self) -> PredictionInput:
        return PredictionInput(
            question_id=self.question_id,
            db_id=self.db_id,
            question=self.question,
            database_path=self.database_path,
        )


class ReferenceExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executed: bool
    result: QueryResult | None = None
    error_kind: str | None = None
    error: str | None = None


class ResultComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_accuracy: bool
    exact_sql_match: bool = False
    reason: str
    float_abs_tolerance: float = Field(ge=0)


class WorkflowCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    db_id: str
    question: str
    rag_branch: Literal["ddl_only", "sample_value_only", "fusion_rrf"] = "fusion_rrf"
    gold_tables: list[str]
    retrieved_tables: list[str]
    schema_status: str
    schema_recall: float | None = Field(default=None, ge=0, le=1)
    gold_executed: bool
    gold_row_count: int | None = Field(default=None, ge=0)
    gold_truncated: bool = False
    gold_error_kind: str | None = None
    predicted_sql: str | None = None
    initial_generation_valid: bool
    initial_execution_success: bool
    initial_execution_accuracy: bool | None = None
    final_status: str
    final_execution_accuracy: bool | None = None
    reflection_recovery: bool = False
    reflection_decision: str | None = None
    attempts: int = Field(ge=0)
    reflection_count: int = Field(ge=0)
    max_attempt_reached: bool = False
    latency_ms: float = Field(ge=0)
    llm_calls: int = Field(ge=0)
    comparison: ResultComparison | None = None
    error: str | None = None


class WorkflowMetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_count: int = Field(ge=0)
    gold_valid_rate: float = Field(ge=0, le=1)
    initial_generation_valid_rate: float = Field(ge=0, le=1)
    initial_execution_success_rate: float = Field(ge=0, le=1)
    initial_execution_accuracy: float = Field(ge=0, le=1)
    final_execution_accuracy: float = Field(ge=0, le=1)
    exact_sql_match_rate: float = Field(ge=0, le=1)
    reflection_recovery_rate: float = Field(ge=0, le=1)
    clarify_rate: float = Field(ge=0, le=1)
    reject_rate: float = Field(ge=0, le=1)
    max_attempt_rate: float = Field(ge=0, le=1)
    average_latency_ms: float = Field(ge=0)
    average_llm_calls: float = Field(ge=0)
    average_attempts: float = Field(ge=0)
    average_reflections: float = Field(ge=0)
    average_schema_recall: float = Field(ge=0, le=1)
    float_abs_tolerance: float = Field(ge=0)
