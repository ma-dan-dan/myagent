from __future__ import annotations

import math
from numbers import Real

from app.nl2sql.models import QueryResult
from app.nl2sql_eval.models import ResultComparison


class ResultComparator:
    def __init__(self, float_abs_tolerance: float = 1e-2) -> None:
        if float_abs_tolerance < 0:
            raise ValueError("float_abs_tolerance 必须大于等于 0。")
        self.float_abs_tolerance = float_abs_tolerance

    def compare(
        self,
        predicted: QueryResult | None,
        gold: QueryResult | None,
        *,
        predicted_sql: str | None = None,
        gold_sql: str | None = None,
        prediction_error: str | None = None,
        gold_error: str | None = None,
    ) -> ResultComparison:
        exact_sql_match = _normalized_sql(predicted_sql) == _normalized_sql(gold_sql) if predicted_sql and gold_sql else False
        if gold_error or gold is None:
            return self._failure("gold_execution_error", exact_sql_match)
        if prediction_error or predicted is None:
            return self._failure("prediction_execution_error", exact_sql_match)
        if gold.truncated:
            return self._failure("gold_result_truncated", exact_sql_match)
        if predicted.truncated:
            return self._failure("prediction_result_truncated", exact_sql_match)
        if len(predicted.columns) != len(gold.columns):
            return self._failure("column_count_mismatch", exact_sql_match)
        if predicted.row_count != gold.row_count or len(predicted.rows) != len(gold.rows):
            return self._failure("row_count_mismatch", exact_sql_match)
        if not self._rows_match(predicted.rows, gold.rows):
            return self._failure("row_values_mismatch", exact_sql_match)
        return ResultComparison(
            execution_accuracy=True,
            exact_sql_match=exact_sql_match,
            reason="result_values_equal",
            float_abs_tolerance=self.float_abs_tolerance,
        )

    def _rows_match(self, predicted_rows: list[list[object]], gold_rows: list[list[object]]) -> bool:
        unmatched = list(gold_rows)
        for predicted in predicted_rows:
            for index, gold in enumerate(unmatched):
                if self._row_equal(predicted, gold):
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched

    def _row_equal(self, first: list[object], second: list[object]) -> bool:
        return len(first) == len(second) and all(self._value_equal(left, right) for left, right in zip(first, second))

    def _value_equal(self, first: object, second: object) -> bool:
        if first is None or second is None:
            return first is second
        if _is_number(first) and _is_number(second):
            return math.isclose(float(first), float(second), rel_tol=0, abs_tol=self.float_abs_tolerance)
        return type(first) is type(second) and first == second

    def _failure(self, reason: str, exact_sql_match: bool) -> ResultComparison:
        return ResultComparison(
            execution_accuracy=False,
            exact_sql_match=exact_sql_match,
            reason=reason,
            float_abs_tolerance=self.float_abs_tolerance,
        )


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _normalized_sql(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()
