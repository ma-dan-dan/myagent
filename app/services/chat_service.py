from __future__ import annotations

from typing import Any

from app.agent.chat_agent import ChatAgent
from app.intent.models import IntentDecision, IntentName, IntentRouteResult
from app.intent.router import IntentRouter
from app.memory.context_manager import ContextManager
from app.memory.long_term_memory import LongTermMemoryStore
from app.memory.models import TokenUsage
from app.schemas.chat import ChatResponse, IntentMetrics, ToolEvent
from app.storage.session_service import SessionService
from app.rag.service import SchemaLinkingService
from app.nl2sql.executor import SQLUnsafeQueryError
from app.nl2sql.graph import NL2SQLGraphService


class ChatService:
    """Orchestrates one chat request without knowing catalog internals."""

    def __init__(
        self,
        session_service: SessionService,
        agent: ChatAgent,
        context_manager: ContextManager,
        long_term_memory: LongTermMemoryStore,
        intent_router: IntentRouter,
        schema_linking_service: SchemaLinkingService | None = None,
        nl2sql_graph_service: NL2SQLGraphService | None = None,
    ) -> None:
        self.session_service = session_service
        self.agent = agent
        self.context_manager = context_manager
        self.long_term_memory = long_term_memory
        self.intent_router = intent_router
        self.schema_linking_service = schema_linking_service
        self.nl2sql_graph_service = nl2sql_graph_service

    def chat(self, user_id: str, session_id: str | None, message: str) -> ChatResponse:
        route_result = self.intent_router.route(message)
        active_session_id = session_id or self.session_service.create_session(user_id)
        self.session_service.ensure_session(user_id, active_session_id)
        self.session_service.record_intent_classification(
            user_id,
            active_session_id,
            route_result.decision,
            route_result.routed_intent,
            route_result.fallback_reason,
        )
        if route_result.routed_intent is IntentName.DATA_OPERATION and route_result.decision.data_action and route_result.decision.data_action.value == "read":
            return self._read_response(user_id, active_session_id, message, route_result)
        if route_result.routed_intent is IntentName.NL2SQL:
            return self._nl2sql_response(user_id, active_session_id, message, route_result)
        if route_result.routed_intent is not IntentName.CHAT:
            return self._placeholder_response(user_id, active_session_id, message, route_result)

        prepared = self.context_manager.prepare(user_id, active_session_id, message)
        self.session_service.append_message(user_id, active_session_id, "user", message)

        result = self.agent.run(prepared.messages)
        for event in result.tool_events:
            self.session_service.append_message(
                user_id,
                active_session_id,
                "tool",
                event.summary,
                metadata={
                    "tool_name": event.tool_name,
                    "input": event.input.model_dump(),
                    "status": event.status,
                },
            )
        self.session_service.append_message(user_id, active_session_id, "assistant", result.message)
        return self._finish_success(user_id, active_session_id, prepared, result, route_result)

    def _finish_success(self, user_id, session_id, prepared, result, route_result):
        chat_usage = result.usage.model_copy(update={"estimated_context_tokens": prepared.estimated_context_tokens, "context_window": prepared.context_window})
        model = str(getattr(self.agent.llm, "model", "unknown") or "unknown")
        self.session_service.record_usage(user_id, session_id, "chat", model, chat_usage)
        current_turn = chat_usage
        if prepared.maintenance.compacted:
            self.session_service.record_usage(
                user_id,
                session_id,
                "summary",
                model,
                prepared.maintenance.summary_usage,
            )
            self.session_service.record_usage(
                user_id,
                session_id,
                "memory",
                model,
                prepared.maintenance.memory_usage,
            )
            self.long_term_memory.upsert(prepared.maintenance.memory_entries)
            current_turn = current_turn.add(prepared.maintenance.summary_usage).add(prepared.maintenance.memory_usage)
        usage = self.session_service.get_session_usage(user_id, session_id, current_turn)
        return ChatResponse(
            session_id=session_id,
            message=result.message,
            tool_events=result.tool_events,
            usage=usage,
            intent_decision=route_result.decision,
            routed_intent=route_result.routed_intent,
            fallback_reason=route_result.fallback_reason,
        )

    def _read_response(self, user_id: str, session_id: str, message: str, route_result: IntentRouteResult) -> ChatResponse:
        if self.schema_linking_service is None:
            raise RuntimeError("RAG 服务未配置。")
        linked = self.schema_linking_service.search(message)
        if linked.status == "empty":
            return self._stable_response(user_id, session_id, message, route_result, "未找到匹配的 Schema，请补充业务对象或字段名称。")
        if linked.status == "ambiguous":
            return self._stable_response(user_id, session_id, message, route_result, "匹配到多个可能的 Schema，请补充更具体的表、字段或业务范围。")
        prepared = self.context_manager.prepare(user_id, session_id, message, extra_context=linked.context)
        self.session_service.append_message(user_id, session_id, "user", message)
        result = self.agent.run(prepared.messages, tools=[])
        self.session_service.append_message(user_id, session_id, "assistant", result.message)
        return self._finish_success(user_id, session_id, prepared, result, route_result)

    def _nl2sql_response(self, user_id: str, session_id: str, message: str, route_result: IntentRouteResult) -> ChatResponse:
        if self.nl2sql_graph_service is None:
            raise RuntimeError("NL2SQL 图服务未配置。")
        result = self.nl2sql_graph_service.invoke(user_id, session_id, message)
        if result.validation_error and result.status != "ok":
            self.session_service.append_message(user_id, session_id, "user", message)
            self.session_service.append_message(user_id, session_id, "assistant", "生成的 SQL 未通过安全校验。")
            raise SQLUnsafeQueryError("生成的 SQL 未通过安全校验。")
        assistant_message = result.final_message or "NL2SQL 查询未能完成。"
        self.session_service.append_message(user_id, session_id, "user", message)
        self.session_service.append_message(user_id, session_id, "assistant", assistant_message)
        model = str(getattr(self.agent.llm, "model", "unknown") or "unknown")
        usage = result.usage.model_copy(update={"context_window": self.context_manager.token_manager.context_window})
        self.session_service.record_usage(user_id, session_id, "chat", model, usage)
        sql = result.validation.normalized_sql if result.validation and result.validation.normalized_sql else (result.sql_draft.sql if result.sql_draft else None)
        return ChatResponse(
            session_id=session_id,
            message=assistant_message,
            usage=self.session_service.get_session_usage(user_id, session_id, usage),
            intent_decision=route_result.decision,
            routed_intent=route_result.routed_intent,
            fallback_reason=route_result.fallback_reason,
            nl2sql_status=result.status,
            sql=sql,
            query_result=result.query_result,
        )

    def _stable_response(self, user_id: str, session_id: str, message: str, route_result: IntentRouteResult, assistant_message: str) -> ChatResponse:
        self.session_service.append_message(user_id, session_id, "user", message)
        self.session_service.append_message(user_id, session_id, "assistant", assistant_message)
        return ChatResponse(session_id=session_id, message=assistant_message, usage=self.session_service.get_session_usage(user_id, session_id, TokenUsage()), intent_decision=route_result.decision, routed_intent=route_result.routed_intent, fallback_reason=route_result.fallback_reason)

    def intent_metrics(self, user_id: str, provider: str | None = None) -> IntentMetrics:
        return IntentMetrics.model_validate(self.session_service.get_intent_metrics(user_id, provider))

    def _placeholder_response(
        self,
        user_id: str,
        session_id: str,
        message: str,
        route_result: IntentRouteResult,
    ) -> ChatResponse:
        if route_result.routed_intent is IntentName.DATA_OPERATION:
            assistant_message = "已识别为数据查询意图，当前仅完成意图路由测试。"
        else:
            assistant_message = "已识别为 NL2SQL 意图，当前仅完成意图路由测试。"
        self.session_service.append_message(user_id, session_id, "user", message)
        self.session_service.append_message(user_id, session_id, "assistant", assistant_message)
        usage = self.session_service.get_session_usage(user_id, session_id, TokenUsage())
        return ChatResponse(
            session_id=session_id,
            message=assistant_message,
            usage=usage,
            intent_decision=route_result.decision,
            routed_intent=route_result.routed_intent,
            fallback_reason=route_result.fallback_reason,
        )
