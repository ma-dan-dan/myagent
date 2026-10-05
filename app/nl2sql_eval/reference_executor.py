from __future__ import annotations

from app.nl2sql.executor import SQLExecutionError, SQLExecutor, SQLExecutorUnavailable
from app.nl2sql_eval.models import ReferenceExecution


class GoldReferenceExecutor:
    def __init__(self, executor: SQLExecutor) -> None:
        self.executor = executor

    def execute(self, gold_sql: str) -> ReferenceExecution:
        try:
            result = self.executor.execute(gold_sql, {})
        except SQLExecutorUnavailable as exc:
            return ReferenceExecution(executed=False, error_kind="unavailable", error=str(exc))
        except SQLExecutionError as exc:
            return ReferenceExecution(executed=False, error_kind="execution_error", error=str(exc))
        return ReferenceExecution(executed=True, result=result)
