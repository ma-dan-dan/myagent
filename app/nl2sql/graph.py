from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.adapter import LLMAdapter
from app.nl2sql.executor import SQLExecutor
from app.nl2sql.models import NL2SQLState
from app.nl2sql.nodes import (
    execute_sql_node,
    gen_sql_node,
    output_node,
    reflection_node,
    schema_linking_node,
    validate_sql_node,
)
from app.nl2sql.validator import SQLValidator
from app.rag.service import SchemaLinkingService


class GraphState(TypedDict, total=False):
    user_id: str
    session_id: str
    user_message: str
    schema_candidates: list[dict[str, Any]]
    schema_context: str
    sql_draft: dict[str, Any] | None
    validation: dict[str, Any] | None
    query_result: dict[str, Any] | None
    reflection: dict[str, Any] | None
    validation_error: str | None
    execution_error: str | None
    attempt: int
    reflection_count: int
    max_attempts: int
    max_reflections: int
    usage: dict[str, Any]
    status: str
    final_message: str | None


class NL2SQLGraphService:
    def __init__(self, schema_linking_service: SchemaLinkingService, llm: LLMAdapter, validator: SQLValidator, executor: SQLExecutor, dialect: str, max_attempts: int, max_reflections: int) -> None:
        self.schema_linking_service = schema_linking_service
        self.llm = llm
        self.validator = validator
        self.executor = executor
        self.dialect = dialect
        self.max_attempts = max_attempts
        self.max_reflections = max_reflections
        self.graph = self._build_graph()

    def invoke(self, user_id: str, session_id: str, user_message: str) -> NL2SQLState:
        initial = NL2SQLState(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            max_attempts=self.max_attempts,
            max_reflections=self.max_reflections,
        )
        result = self.graph.invoke(initial.model_dump())
        return NL2SQLState.model_validate(result)

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("schema_linking", lambda state: schema_linking_node(state, self.schema_linking_service))
        workflow.add_node("gen_sql", lambda state: gen_sql_node(state, self.llm, self.dialect))
        workflow.add_node("validate_sql", lambda state: validate_sql_node(state, self.validator))
        workflow.add_node("execute_sql", lambda state: execute_sql_node(state, self.executor))
        workflow.add_node("reflection", lambda state: reflection_node(state, self.llm, self.dialect))
        workflow.add_node("output", output_node)
        workflow.add_edge(START, "schema_linking")
        workflow.add_conditional_edges("schema_linking", self._after_schema, {"gen_sql": "gen_sql", "output": "output"})
        workflow.add_edge("gen_sql", "validate_sql")
        workflow.add_conditional_edges("validate_sql", self._after_validation, {"execute_sql": "execute_sql", "reflection": "reflection", "output": "output"})
        workflow.add_edge("execute_sql", "reflection")
        workflow.add_conditional_edges("reflection", self._after_reflection, {"gen_sql": "gen_sql", "output": "output"})
        workflow.add_edge("output", END)
        return workflow.compile()

    @staticmethod
    def _after_schema(state: GraphState) -> str:
        return "gen_sql" if state.get("status") == "running" and state.get("schema_candidates") else "output"

    @staticmethod
    def _after_validation(state: GraphState) -> str:
        if state.get("status") != "running":
            return "output"
        validation = state.get("validation") or {}
        return "execute_sql" if validation.get("ok") else "reflection"

    @staticmethod
    def _after_reflection(state: GraphState) -> str:
        reflection = state.get("reflection") or {}
        if reflection.get("decision") == "regenerate" and state.get("attempt", 0) < state.get("max_attempts", 0) and state.get("reflection_count", 0) < state.get("max_reflections", 0):
            return "gen_sql"
        return "output"
