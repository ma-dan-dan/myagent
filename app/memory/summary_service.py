from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter

from app.agent.adapter import LLMAdapter, LLMResponse
from app.memory.models import CompactionResult, ContextPolicy, MemoryEntry, SummaryRecord, TokenUsage
from app.memory.token_manager import TokenManager
from app.storage.session_service import SessionService


class SummaryService:
    def __init__(
        self,
        session_service: SessionService,
        llm: LLMAdapter,
        token_manager: TokenManager,
        policy: ContextPolicy | None = None,
    ) -> None:
        self.session_service = session_service
        self.llm = llm
        self.token_manager = token_manager
        self.policy = policy or token_manager.policy

    def compact(
        self,
        user_id: str,
        session_id: str,
        old_summary: SummaryRecord | None,
        messages_to_summarize: list[Any],
        max_summary_chars: int | None = None,
        *,
        recompact_old_summary: bool = False,
    ) -> CompactionResult:
        if not messages_to_summarize and not recompact_old_summary:
            return CompactionResult(summary=old_summary)
        if not messages_to_summarize and old_summary is None:
            return CompactionResult(summary=old_summary)

        normalized_messages = [self._normalize_message(message) for message in messages_to_summarize]
        summary_messages = self._summary_messages(old_summary, normalized_messages)
        estimated = self.token_manager.estimate(summary_messages)
        try:
            response = LLMResponse.model_validate(self.llm.complete(summary_messages, tools=[]))
            if response.kind != "message" or not response.content or not response.content.strip():
                raise ValueError("summary response is empty")
            summary_limit = max_summary_chars or self.policy.summary_max_chars
            boundary = old_summary.summarized_through_message_id if old_summary else 0
            if normalized_messages:
                boundary = normalized_messages[-1][0] or boundary
            record = SummaryRecord(
                user_id=user_id,
                session_id=session_id,
                summary=response.content.strip()[:summary_limit],
                summarized_through_message_id=boundary,
            )
            self.session_service.save_summary(record)
            usage = self._usage(response.usage, estimated)
        except Exception:
            return CompactionResult(attempted=True, summary=old_summary)

        return CompactionResult(
            attempted=True,
            compacted=True,
            summary=record,
            summary_usage=usage,
        )

    def extract_long_term_memory(self, summary: str) -> tuple[list[MemoryEntry], TokenUsage]:
        messages = [
            {
                "role": "system",
                "content": "从本次新生成的摘要中提取安全、稳定、可复用的项目知识。只输出 JSON 数组，每个元素只能包含 content 字段，内容必须是一句不超过 300 字的事实。没有候选时输出空数组；拒绝密钥、Token、Cookie、密码、原始聊天、工具原始数据和未经确认的猜测。",
            },
            {"role": "user", "content": summary},
        ]
        estimated = self.token_manager.estimate(messages)
        try:
            response = LLMResponse.model_validate(self.llm.complete(messages, tools=[]))
            if response.kind != "message" or not response.content:
                raise ValueError("memory response is empty")
            raw: Any = json.loads(response.content)
            entries = TypeAdapter(list[MemoryEntry]).validate_python(raw)
            return entries, self._usage(response.usage, estimated)
        except Exception:
            return [], TokenUsage(estimated_context_tokens=estimated, context_window=self.token_manager.context_window)

    def _summary_messages(self, old_summary: SummaryRecord | None, moved: list[tuple[int | None, str, str]]) -> list[dict[str, str]]:
        prior = old_summary.summary if old_summary else "（无旧摘要）"
        transcript = "\n".join(f"{role}: {content}" for _, role, content in moved)
        return [
            {
                "role": "system",
                "content": "请将旧摘要和新增对话压缩为事实性摘要，保留目标、事实、约束、未解决事项和重要工具结论。不要编造。",
            },
            {"role": "user", "content": f"旧摘要：\n{prior}\n\n新增对话：\n{transcript}"},
        ]

    def _usage(self, actual: TokenUsage | None, estimated: int) -> TokenUsage:
        base = actual or TokenUsage(requests=0)
        return base.model_copy(
            update={"estimated_context_tokens": estimated, "context_window": self.token_manager.context_window}
        )

    @staticmethod
    def _normalize_message(message: Any) -> tuple[int | None, str, str]:
        if isinstance(message, tuple) and len(message) == 2:
            message_id, message = message
        elif isinstance(message, dict):
            message_id = message.get("id")
        else:
            message_id = getattr(message, "id", None)
        if isinstance(message, dict):
            return message_id, str(message.get("role", "user")), str(message.get("content", ""))
        return message_id, str(message.role), str(message.content)
