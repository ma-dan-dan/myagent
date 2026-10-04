import pytest

from app.nl2sql.validator import SQLValidator


def test_select_from_allowed_table_passes_and_gets_limit():
    result = SQLValidator(dialect="sqlite", max_rows=100).validate(
        "SELECT output_quantity FROM production_output",
        allowed_tables={"production_output"},
        allowed_columns={"production_output": {"output_quantity"}},
    )

    assert result.ok is True
    assert "LIMIT 100" in result.normalized_sql.upper()


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM production_output",
        "UPDATE production_output SET output_quantity = 0",
        "DROP TABLE production_output",
        "SELECT 1; DELETE FROM production_output",
    ],
)
def test_write_or_multi_statement_sql_is_rejected(sql):
    assert SQLValidator().validate(sql, {"production_output"}, {}).ok is False


def test_unknown_table_and_column_are_rejected():
    validator = SQLValidator()
    assert validator.validate("SELECT password FROM users", {"production_output"}, {}).ok is False
    assert validator.validate(
        "SELECT password FROM production_output",
        {"production_output"},
        {"production_output": {"output_quantity"}},
    ).ok is False


def test_existing_limit_is_capped():
    result = SQLValidator(max_rows=100).validate(
        "SELECT output_quantity FROM production_output LIMIT 1000",
        {"production_output"},
        {"production_output": {"output_quantity"}},
    )

    assert result.ok is True
    assert "LIMIT 100" in result.normalized_sql.upper()
