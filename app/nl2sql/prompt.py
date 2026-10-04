from __future__ import annotations


def build_gensql_messages(
    user_message: str,
    schema_context: str,
    dialect: str,
    previous_error: str | None,
) -> list[dict[str, str]]:
    system = (
        "你是只读 SQL 生成器。只允许生成单条 SELECT/WITH SQL；只能使用给定 Schema；"
        "不能虚构表或字段；信息不足时返回 clarify；只输出 JSON，不要 Markdown 代码围栏。"
    )
    user = (
        f"方言：{dialect}\nSchema：{schema_context}\n问题：{user_message}\n"
        f"上次错误：{previous_error or '无'}\n"
        "JSON 字段：status、sql、tables、parameters、explanation。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_reflection_messages(
    user_message: str,
    schema_context: str,
    sql: str,
    validation_error: str | None,
    execution_summary: str | None,
    dialect: str,
) -> list[dict[str, str]]:
    system = "你是只读 SQL Reflection 器，只能返回 pass、regenerate、clarify 或 reject 的 JSON。"
    user = (
        f"方言：{dialect}\nSchema：{schema_context}\n问题：{user_message}\n"
        f"SQL：{sql}\n校验错误：{validation_error or '无'}\n"
        f"执行摘要：{execution_summary or '无'}\n"
        "JSON 字段：decision、reason。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
