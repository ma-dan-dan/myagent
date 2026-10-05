from __future__ import annotations

import sqlglot
from sqlglot import exp


class GoldTableExtractionError(ValueError):
    pass


def extract_gold_tables(gold_sql: str, dialect: str = "sqlite") -> list[str]:
    try:
        statements = sqlglot.parse((gold_sql or "").strip(), read=dialect)
    except sqlglot.errors.ParseError as exc:
        raise GoldTableExtractionError("Gold SQL 解析失败。") from exc
    if len(statements) != 1:
        raise GoldTableExtractionError("Gold SQL 必须是单条语句。")
    statement = statements[0]
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE) if cte.alias_or_name}
    tables = sorted({table.name.strip('`"[]').lower() for table in statement.find_all(exp.Table) if table.name and table.name.lower() not in cte_names})
    if not tables:
        raise GoldTableExtractionError("Gold SQL 未提取到基础表。")
    return tables
