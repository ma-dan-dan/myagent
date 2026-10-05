from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BirdCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    db_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_sql: str = Field(min_length=1)
    database_path: Path
    question_id_source: Literal["dataset", "generated"] = "dataset"


class GoldTableSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    db_id: str = Field(min_length=1)
    tables: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tables(self) -> "GoldTableSet":
        normalized = sorted({table.strip().lower() for table in self.tables if table.strip()})
        if not normalized:
            raise ValueError("gold tables must not be empty")
        self.tables = normalized
        return self


class RetrievalCaseMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_id_source: Literal["dataset", "generated"]
    db_id: str
    branch: Literal["ddl_only", "sample_value_only", "fusion_rrf"]
    top_k: int = Field(ge=1)
    gold_tables: list[str]
    retrieved_tables: list[str]
    matched_tables: list[str]
    recall: float = Field(ge=0, le=1)
    full_recall: bool
    precision: float = Field(ge=0, le=1)
    database_coverage: float = Field(ge=0, le=1)
    empty: bool
    ambiguous: bool
    false_positive_count: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)


class AggregateMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branch: str
    top_k: int
    case_count: int
    macro_recall: float
    micro_recall: float
    full_recall_rate: float
    macro_precision: float
    empty_rate: float
    ambiguous_rate: float
    average_false_positive_count: float
    average_elapsed_ms: float
    average_database_coverage: float
