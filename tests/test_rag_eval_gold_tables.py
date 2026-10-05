import pytest

from app.rag_eval.gold_tables import GoldTableExtractionError, extract_gold_tables


def test_extract_gold_tables_excludes_cte_aliases_and_deduplicates_tables():
    tables = extract_gold_tables(
        """
        WITH recent_orders AS (
            SELECT * FROM orders
        )
        SELECT u.name
        FROM users AS u
        JOIN recent_orders AS r ON r.user_id = u.id
        JOIN orders AS o ON o.user_id = u.id
        """
    )

    assert tables == ["orders", "users"]


def test_extract_gold_tables_rejects_invalid_sql_instead_of_returning_empty():
    with pytest.raises(GoldTableExtractionError, match="Gold SQL"):
        extract_gold_tables("SELECT FROM")


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT * FROM `Orders`", ["orders"]),
        ("SELECT * FROM (SELECT * FROM payments) AS p JOIN users u ON 1 = 1", ["payments", "users"]),
    ],
)
def test_extract_gold_tables_normalizes_quoted_and_subquery_tables(sql, expected):
    assert extract_gold_tables(sql) == expected
