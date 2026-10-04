import pytest
from pydantic import ValidationError

from app.rag.models import RagDocument, RagSourceType, SchemaCandidate


def test_rag_document_requires_source_specific_identity():
    document = RagDocument(
        document_id="ddl:manufacturing:production_output",
        source_type=RagSourceType.DDL,
        namespace="manufacturing",
        table_id="production_output",
        table_name="production_output",
        text="CREATE TABLE production_output (output_date DATE);",
        content_hash="hash",
    )

    assert document.column_name is None


def test_sample_value_document_requires_column_name():
    with pytest.raises(ValidationError):
        RagDocument(
            document_id="sample:manufacturing:production_output:line_code",
            source_type=RagSourceType.SAMPLE_VALUE,
            namespace="manufacturing",
            table_id="production_output",
            table_name="production_output",
            text="line_code sample LINE-A",
            content_hash="hash",
        )


def test_schema_candidate_limits_packed_samples():
    candidate = SchemaCandidate(
        table_id="production_output",
        table_name="production_output",
        ddl="CREATE TABLE production_output (output_date DATE);",
        matched_columns=["line_code"],
        sample_values={"line_code": ["LINE-A"]},
        score=0.5,
        matched_by=[RagSourceType.DDL, RagSourceType.SAMPLE_VALUE],
    )

    assert candidate.table_name == "production_output"
