from __future__ import annotations

import sqlglot
from sqlglot import exp

from app.nl2sql.models import SQLValidationResult


class SQLValidator:
    def __init__(self, dialect: str = "sqlite", max_rows: int = 100, max_sql_length: int = 12000) -> None:
        self.dialect = dialect
        self.max_rows = max_rows
        self.max_sql_length = max_sql_length

    def validate(
        self,
        sql: str,
        allowed_tables: set[str],
        allowed_columns: dict[str, set[str]],
    ) -> SQLValidationResult:
        raw = (sql or "").strip()
        if not raw or len(raw) > self.max_sql_length:
            return self._failure("SQL 为空或长度超出限制。")
        try:
            statements = sqlglot.parse(raw, read=self.dialect)
        except sqlglot.errors.ParseError:
            return self._failure("SQL 语法无效。")
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            return self._failure("只允许单条只读 SELECT/WITH SQL。")

        statement = statements[0]
        if self._has_projection_wildcard(statement):
            return self._failure("必须明确选择授权字段。")
        cte_names = {cte.alias_or_name for cte in statement.find_all(exp.CTE)}
        tables = [table for table in statement.find_all(exp.Table) if table.name not in cte_names]
        referenced_tables = sorted({table.name for table in tables})
        if not referenced_tables or any(table not in allowed_tables for table in referenced_tables):
            return self._failure("SQL 引用了未授权的表。")

        aliases = {table.alias_or_name: table.name for table in tables}
        referenced_columns: list[str] = []
        for column in statement.find_all(exp.Column):
            if column.name == "*":
                continue
            table_name = aliases.get(column.table, column.table) if column.table else ""
            candidate_tables = [table_name] if table_name else referenced_tables
            if not any(column.name in allowed_columns.get(table, set()) for table in candidate_tables):
                return self._failure("SQL 引用了未授权的字段。")
            referenced_columns.append(f"{table_name + '.' if table_name else ''}{column.name}")

        limit = statement.args.get("limit")
        if limit is None:
            statement = statement.limit(self.max_rows)
        else:
            try:
                limit_value = int(limit.expression.name)
            except (TypeError, ValueError, AttributeError):
                return self._failure("LIMIT 必须是整数。")
            if limit_value > self.max_rows:
                statement = statement.limit(self.max_rows)
        return SQLValidationResult(
            ok=True,
            normalized_sql=statement.sql(dialect=self.dialect),
            referenced_tables=referenced_tables,
            referenced_columns=sorted(set(referenced_columns)),
        )

    @staticmethod
    def _failure(message: str) -> SQLValidationResult:
        return SQLValidationResult(ok=False, error=message)

    @staticmethod
    def _has_projection_wildcard(statement: exp.Select) -> bool:
        for select in statement.find_all(exp.Select):
            for expression in select.expressions:
                if isinstance(expression, exp.Star):
                    return True
                if isinstance(expression, exp.Column) and expression.name == "*":
                    return True
        return False
