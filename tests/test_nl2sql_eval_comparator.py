from app.nl2sql.models import QueryResult
from app.nl2sql_eval.result_comparator import ResultComparator


def query_result(rows, columns=None, truncated=False):
    return QueryResult(
        columns=columns or ["value"],
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
    )


def test_comparator_accepts_different_sql_when_result_values_match():
    comparison = ResultComparator().compare(
        query_result([[10]]),
        query_result([[10]]),
        predicted_sql="SELECT value AS result FROM items",
        gold_sql="SELECT value FROM items",
    )

    assert comparison.execution_accuracy is True
    assert comparison.exact_sql_match is False
    assert comparison.reason == "result_values_equal"


def test_comparator_ignores_row_order_but_preserves_duplicate_counts():
    comparator = ResultComparator()

    equivalent = comparator.compare(query_result([[2], [1], [1]]), query_result([[1], [2], [1]]))
    duplicate_mismatch = comparator.compare(query_result([[2], [1]]), query_result([[1], [2], [1]]))

    assert equivalent.execution_accuracy is True
    assert duplicate_mismatch.execution_accuracy is False
    assert duplicate_mismatch.reason == "row_count_mismatch"


def test_comparator_distinguishes_null_from_string_and_supports_float_tolerance():
    comparator = ResultComparator(float_abs_tolerance=1e-2)

    close_float = comparator.compare(query_result([[1.001]]), query_result([[1.0]]))
    far_float = comparator.compare(query_result([[1.02]]), query_result([[1.0]]))
    null_mismatch = comparator.compare(query_result([[None]]), query_result([["NULL"]]))

    assert close_float.execution_accuracy is True
    assert far_float.execution_accuracy is False
    assert far_float.reason == "row_values_mismatch"
    assert null_mismatch.execution_accuracy is False


def test_comparator_handles_empty_results_columns_and_truncation_stably():
    comparator = ResultComparator()

    empty = comparator.compare(query_result([]), query_result([]))
    columns = comparator.compare(query_result([[1]], ["a"]), query_result([[1]], ["a", "b"]))
    truncated = comparator.compare(query_result([[1]], truncated=True), query_result([[1]]))
    missing_prediction = comparator.compare(None, query_result([[1]]), prediction_error="执行失败")

    assert empty.execution_accuracy is True
    assert columns.reason == "column_count_mismatch"
    assert truncated.reason == "prediction_result_truncated"
    assert missing_prediction.reason == "prediction_execution_error"
