from __future__ import annotations

from typing import Any

from app.agent.chat_agent import ChatAgent
from app.memory.context_manager import ContextManager
from app.memory.long_term_memory import LongTermMemoryStore
from app.memory.models import TokenUsage
from app.schemas.chat import ChatResponse, ToolEvent
from app.storage.session_service import SessionService


class ChatService:
    """Orchestrates one chat request without knowing catalog internals."""

    def __init__(
        self,
        session_service: SessionService,
        agent: ChatAgent,
        context_manager: ContextManager,
        long_term_memory: LongTermMemoryStore,
    ) -> None:
        self.session_service = session_service
        self.agent = agent
        self.context_manager = context_manager
        self.long_term_memory = long_term_memory

    def chat(self, user_id: str, session_id: str | None, message: str) -> ChatResponse:
        active_session_id = session_id or self.session_service.create_session(user_id)
        self.session_service.ensure_session(user_id, active_session_id)
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
        )
