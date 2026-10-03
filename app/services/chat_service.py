from __future__ import annotations

from typing import Any

from app.agent.chat_agent import ChatAgent
from app.intent.models import IntentName
from app.intent.router import IntentRouter
from app.memory.context_manager import ContextManager
from app.memory.long_term_memory import LongTermMemoryStore
from app.memory.models import TokenUsage
from app.schemas.chat import ChatResponse, IntentMetrics, ToolEvent
from app.storage.session_service import SessionService


class ChatService:
    """Orchestrates one chat request without knowing catalog internals."""

    def __init__(
        self,
        session_service: SessionService,
        agent: ChatAgent,
        context_manager: ContextManager,
        long_term_memory: LongTermMemoryStore,
        intent_router: IntentRouter,
    ) -> None:
        self.session_service = session_service
        self.agent = agent
        self.context_manager = context_manager
        self.long_term_memory = long_term_memory
        self.intent_router = intent_router

    def chat(self, user_id: str, session_id: str | None, message: str) -> ChatResponse:
        intent_decision = self.intent_router.route(message)
        active_session_id = session_id or self.session_service.create_session(user_id)
        self.session_service.ensure_session(user_id, active_session_id)
        self.session_service.record_intent_classification(user_id, active_session_id, intent_decision)
        if intent_decision.intent is not IntentName.CHAT:
            return self._placeholder_response(user_id, active_session_id, message, intent_decision)

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
        chat_usage = result.usage.model_copy(
            update={
                "estimated_context_tokens": prepared.estimated_context_tokens,
                "context_window": prepared.context_window,
            }
        )
        model = str(getattr(self.agent.llm, "model", "unknown") or "unknown")
        self.session_service.record_usage(user_id, active_session_id, "chat", model, chat_usage)
        current_turn = chat_usage
        if prepared.maintenance.compacted:
            self.session_service.record_usage(
                user_id,
                active_session_id,
                "summary",
                model,
                prepared.maintenance.summary_usage,
            )
            self.session_service.record_usage(
                user_id,
                active_session_id,
                "memory",
                model,
                prepared.maintenance.memory_usage,
            )
            self.long_term_memory.upsert(prepared.maintenance.memory_entries)
            current_turn = current_turn.add(prepared.maintenance.summary_usage).add(prepared.maintenance.memory_usage)
        usage = self.session_service.get_session_usage(user_id, active_session_id, current_turn)
        return ChatResponse(
            session_id=active_session_id,
            message=result.message,
            tool_events=result.tool_events,
            usage=usage,
            intent_decision=intent_decision,
        )

    def intent_metrics(self, user_id: str, provider: str | None = None) -> IntentMetrics:
        return IntentMetrics.model_validate(self.session_service.get_intent_metrics(user_id, provider))

    def _placeholder_response(
        self,
        user_id: str,
        session_id: str,
        message: str,
        intent_decision,
    ) -> ChatResponse:
        if intent_decision.intent is IntentName.DATA_OPERATION:
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
            intent_decision=intent_decision,
        )
