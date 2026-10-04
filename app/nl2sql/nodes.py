from __future__ import annotations

import json
import re
from typing import Any

from app.agent.adapter import LLMAdapter, LLMResponse
from app.nl2sql.executor import SQLExecutionError, SQLExecutor, SQLExecutorUnavailable
from app.nl2sql.models import NL2SQLState, ReflectionDecision, SQLDraft
from app.nl2sql.prompt import build_gensql_messages, build_reflection_messages
from app.nl2sql.validator import SQLValidator
from app.rag.service import SchemaLinkingService


def schema_linking_node(state: dict[str, Any], schema_linking_service: SchemaLinkingService) -> dict[str, Any]:
    current = NL2SQLState.model_validate(state)
    linked = schema_linking_service.search(current.user_message)
    update = current.model_dump()
    update["schema_context"] = linked.context
    update["schema_candidates"] = [candidate.model_dump() for candidate in linked.candidates]
    update["status"] = "running" if linked.status == "ok" else linked.status
    if linked.status == "empty":
        update["final_message"] = "未找到匹配的 Schema，请补充表、字段或业务范围。"
    elif linked.status == "ambiguous":
        update["final_message"] = "匹配到多个可能的 Schema，请补充更具体的表、字段或业务范围。"
    return NL2SQLState.model_validate(update).model_dump()


def gen_sql_node(state: dict[str, Any], llm: LLMAdapter, dialect: str) -> dict[str, Any]:
    current = NL2SQLState.model_validate(state)
    update = current.model_dump()
    if current.attempt >= current.max_attempts:
        update.update(status="failed", final_message="SQL 生成已达到最大尝试次数。")
        return NL2SQLState.model_validate(update).model_dump()
    try:
        response = LLMResponse.model_validate(
            llm.complete(build_gensql_messages(current.user_message, current.schema_context, dialect, current.validation_error or current.execution_error), tools=[])
        )
        if response.kind != "message" or not response.content:
            raise ValueError("empty SQL draft")
        draft = SQLDraft.model_validate(json.loads(response.content))
    except Exception:
        update.update(status="failed", final_message="SQL 生成服务返回了无效结果。")
        return NL2SQLState.model_validate(update).model_dump()
    update["attempt"] = current.attempt + 1
    update["sql_draft"] = draft.model_dump()
    update["validation"] = None
    update["validation_error"] = None
    update["query_result"] = None
    update["execution_error"] = None
    update["reflection"] = None
    update["usage"] = current.usage.add(response.usage or current.usage.__class__()).model_dump()
    if draft.status != "ok":
        update.update(status=draft.status, final_message=draft.explanation)
    return NL2SQLState.model_validate(update).model_dump()


def validate_sql_node(state: dict[str, Any], validator: SQLValidator) -> dict[str, Any]:
    current = NL2SQLState.model_validate(state)
    update = current.model_dump()
    if current.sql_draft is None or current.sql_draft.status != "ok" or not current.sql_draft.sql:
        return update
    tables, columns = _allowed_schema(current)
    result = validator.validate(current.sql_draft.sql, tables, columns)
    update["validation"] = result.model_dump()
    update["validation_error"] = result.error
    return NL2SQLState.model_validate(update).model_dump()


def execute_sql_node(state: dict[str, Any], executor: SQLExecutor) -> dict[str, Any]:
    current = NL2SQLState.model_validate(state)
    update = current.model_dump()
    if not current.validation or not current.validation.ok or not current.validation.normalized_sql or not current.sql_draft:
        return update
    try:
        result = executor.execute(current.validation.normalized_sql, current.sql_draft.parameters)
    except SQLExecutorUnavailable:
        raise
    except SQLExecutionError as exc:
        update["execution_error"] = str(exc)
        return NL2SQLState.model_validate(update).model_dump()
    update["query_result"] = result.model_dump()
    return NL2SQLState.model_validate(update).model_dump()


def reflection_node(state: dict[str, Any], llm: LLMAdapter, dialect: str) -> dict[str, Any]:
    current = NL2SQLState.model_validate(state)
    update = current.model_dump()
    if current.reflection_count >= current.max_reflections or not current.sql_draft or not current.sql_draft.sql:
        update.update(status="failed", final_message="SQL 校验或执行未能在限制次数内完成。")
        return NL2SQLState.model_validate(update).model_dump()
    summary = None
    if current.query_result:
        summary = f"columns={','.join(current.query_result.columns)}; row_count={current.query_result.row_count}; truncated={current.query_result.truncated}"
    try:
        response = LLMResponse.model_validate(
            llm.complete(build_reflection_messages(current.user_message, current.schema_context, current.sql_draft.sql, current.validation_error, summary or current.execution_error, dialect), tools=[])
        )
        if response.kind != "message" or not response.content:
            raise ValueError("empty reflection")
        reflection = ReflectionDecision.model_validate(json.loads(response.content))
    except Exception:
        update.update(status="failed", final_message="SQL Reflection 服务返回了无效结果。")
        return NL2SQLState.model_validate(update).model_dump()
    update["reflection"] = reflection.model_dump()
    update["reflection_count"] = current.reflection_count + 1
    update["usage"] = current.usage.add(response.usage or current.usage.__class__()).model_dump()
    if reflection.decision == "clarify":
        update.update(status="clarify", final_message=reflection.reason)
    elif reflection.decision == "reject":
        update.update(status="rejected", final_message=reflection.reason)
    elif reflection.decision == "pass" and current.query_result is None:
        update.update(status="failed", final_message="SQL 查询没有可用结果。")
    return NL2SQLState.model_validate(update).model_dump()


def output_node(state: dict[str, Any]) -> dict[str, Any]:
    current = NL2SQLState.model_validate(state)
    update = current.model_dump()
    if current.reflection and current.reflection.decision == "pass" and current.query_result is not None:
        update.update(status="ok", final_message=f"查询完成，共返回 {current.query_result.row_count} 行结果。")
    elif current.status == "running":
        update.update(status="failed", final_message="SQL 查询未能完成。")
    return NL2SQLState.model_validate(update).model_dump()


def _allowed_schema(state: NL2SQLState) -> tuple[set[str], dict[str, set[str]]]:
    tables: set[str] = set()
    columns: dict[str, set[str]] = {}
    for candidate in state.schema_candidates:
        tables.add(candidate.table_name)
        column_names = set(candidate.matched_columns)
        match = re.search(r"\((.*)\)", candidate.ddl, re.S)
        if match:
            for definition in match.group(1).split(","):
                name = definition.strip().split()[0] if definition.strip() else ""
                if name:
                    column_names.add(name.strip('`"[]'))
        columns[candidate.table_name] = column_names
    return tables, columns
