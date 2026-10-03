from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.memory.long_term_memory import LongTermMemoryStore
from app.memory.models import CompactionResult, ContextBudgetExceeded, ContextPolicy
from app.memory.project_context import ProjectContextLoader
from app.memory.summary_service import SummaryService
from app.memory.token_manager import TokenManager
from app.storage.session_service import SessionService


FIXED_SYSTEM_PROMPT = (
    "你是一个简洁、诚实的助手。只能使用 search_schema 查询脱敏 Schema 元数据，"
    "不能执行 SQL、读取数据行或调用其他工具。信息不足时直接用普通助手文字追问，结束本轮。"
)


@dataclass
class PreparedContext:
    messages: list[dict[str, Any]]
    estimated_context_tokens: int
    context_window: int
    maintenance: CompactionResult


class ContextManager:
    def __init__(
        self,
        session_service: SessionService,
        summary_service: SummaryService,
        project_context: ProjectContextLoader,
        long_term_memory: LongTermMemoryStore,
        token_manager: TokenManager,
        policy: ContextPolicy | None = None,
    ) -> None:
        self.session_service = session_service
        self.summary_service = summary_service
        self.project_context = project_context
        self.long_term_memory = long_term_memory
        self.token_manager = token_manager
        self.policy = policy or token_manager.policy

    def prepare(self, user_id: str, session_id: str, current_message: str) -> PreparedContext:
        summary = self.session_service.get_summary(user_id, session_id)
        rules = self.project_context.as_system_message()
        memory_message = self.long_term_memory.as_system_message()
        maintenance = CompactionResult(summary=summary)
        summary_calls = 0
        summary_succeeded = False

        while True:
            fixed_messages = self._build_fixed_messages(rules, memory_message, summary)
            fixed_messages = self._fit_fixed_context(fixed_messages)
            fixed_tokens = self.token_manager.estimate(fixed_messages)
            current_tokens = self.token_manager.estimate([{"role": "user", "content": current_message}])
            if fixed_tokens + current_tokens > self.token_manager.hard_input_budget:
                if summary is not None and summary_calls < self.policy.max_summary_calls_per_request:
                    result = self.summary_service.compact(
                        user_id,
                        session_id,
                        summary,
                        [],
                        self.policy.summary_max_chars,
                        recompact_old_summary=True,
                    )
                    summary_calls += 1
                    if result.compacted and result.summary is not None:
                        summary = result.summary
                        maintenance = self._merge_maintenance(maintenance, result)
                        summary_succeeded = True
                        continue
                raise ContextBudgetExceeded(
                    fixed_tokens + current_tokens,
                    self.token_manager.hard_input_budget,
                    "固定上下文和当前问题超过硬预算，请缩短当前问题。",
                )

            boundary = summary.summarized_through_message_id if summary else 0
            unsummarized = self.session_service.get_messages_after_id(user_id, session_id, boundary)
            recent, moved = self._select_recent_messages(unsummarized, fixed_tokens, current_tokens)
            messages = fixed_messages + [
                {"role": message.role, "content": message.content}
                for message in recent
                if message.role != "tool"
            ]
            messages.append({"role": "user", "content": current_message})
            estimated = self.token_manager.estimate(messages)
            fixed_over_soft_cap = fixed_tokens > self.token_manager.fixed_context_budget

            if (moved or fixed_over_soft_cap or estimated > self.token_manager.soft_input_budget) and summary_calls < self.policy.max_summary_calls_per_request:
                if moved:
                    result = self.summary_service.compact(
                        user_id,
                        session_id,
                        summary,
                        moved,
                        self.policy.summary_max_chars,
                    )
                elif summary is not None:
                    result = self.summary_service.compact(
                        user_id,
                        session_id,
                        summary,
                        [],
                        self.policy.summary_max_chars,
                        recompact_old_summary=True,
                    )
                else:
                    result = CompactionResult(summary=summary)
                if moved or summary is not None:
                    summary_calls += 1
                if result.compacted and result.summary is not None:
                    summary = result.summary
                    maintenance = self._merge_maintenance(maintenance, result)
                    summary_succeeded = True
                    continue
                if moved:
                    raise ContextBudgetExceeded(
                        estimated,
                        self.token_manager.hard_input_budget,
                        "历史消息无法安全摘要，请缩短当前问题后重试。",
                    )

            if moved:
                raise ContextBudgetExceeded(
                    estimated,
                    self.token_manager.hard_input_budget,
                    "历史消息无法在预算内保留，请缩短当前问题后重试。",
                )
            if fixed_over_soft_cap and summary is not None and summary_calls >= self.policy.max_summary_calls_per_request:
                raise ContextBudgetExceeded(
                    estimated,
                    self.token_manager.hard_input_budget,
                    "固定上下文无法在预算内压缩，请缩短当前问题。",
                )
            if estimated > self.token_manager.hard_input_budget:
                raise ContextBudgetExceeded(
                    estimated,
                    self.token_manager.hard_input_budget,
                    "完整上下文超过硬预算，请缩短当前问题。",
                )

            if summary_succeeded and summary is not None and not maintenance.memory_entries:
                entries, usage = self.summary_service.extract_long_term_memory(summary.summary)
                maintenance.memory_entries = entries
                if usage is not None:
                    maintenance.memory_usage = usage
            maintenance.summary = summary
            maintenance.compacted = summary_succeeded
            maintenance.attempted = summary_calls > 0
            return PreparedContext(messages, estimated, self.token_manager.context_window, maintenance)

    def _build_fixed_messages(
        self,
        rules: dict[str, str] | None,
        memory_message: dict[str, str] | None,
        summary: Any,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": FIXED_SYSTEM_PROMPT}]
        if rules:
            messages.append(rules)
        if memory_message:
            messages.append(memory_message)
        if summary:
            messages.append({"role": "system", "content": f"<conversation_summary>{summary.summary}</conversation_summary>"})
        return messages

    def _select_recent_messages(self, messages: list[Any], fixed_tokens: int, current_tokens: int) -> tuple[list[Any], list[Any]]:
        remaining = self.token_manager.soft_input_budget - fixed_tokens - current_tokens
        selected_reversed: list[Any] = []
        used = 0
        for message in reversed(messages):
            if len(selected_reversed) >= self.policy.recent_message_limit:
                break
            cost = 0 if message.role == "tool" else self.token_manager.estimate(
                [{"role": message.role, "content": message.content}]
            )
            if used + cost > max(0, remaining):
                break
            selected_reversed.append(message)
            used += cost
        selected = list(reversed(selected_reversed))
        return selected, messages[: len(messages) - len(selected)]

    def _fit_fixed_context(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.token_manager.estimate(messages) <= self.token_manager.fixed_context_budget:
            return messages
        for tag in ("long_term_memory", "project_rules"):
            index = next(
                (index for index, message in enumerate(messages) if f"<{tag}>" in message["content"]),
                None,
            )
            if index is None:
                continue
            messages[index] = self._shrink_tagged_message(messages, index, tag)
            if self.token_manager.estimate(messages) <= self.token_manager.fixed_context_budget:
                break
        return messages

    def _shrink_tagged_message(self, messages: list[dict[str, str]], index: int, tag: str) -> dict[str, str]:
        message = messages[index]
        content = message["content"]
        opening = f"<{tag}>"
        closing = f"</{tag}>"
        if opening not in content or closing not in content:
            return message
        inner = content[len(opening) : content.rfind(closing)]
        while inner and self.token_manager.estimate(messages) > self.token_manager.fixed_context_budget:
            inner = inner[: max(0, len(inner) // 2)]
            message = {**message, "content": f"{opening}{inner}{closing}"}
            messages[index] = message
        if self.token_manager.estimate(messages) > self.token_manager.fixed_context_budget:
            return {**message, "content": f"{opening}{closing}"}
        return message

    @staticmethod
    def _merge_maintenance(current: CompactionResult, result: CompactionResult) -> CompactionResult:
        current.attempted = True
        current.compacted = True
        current.summary = result.summary
        current.summary_usage = current.summary_usage.add(result.summary_usage)
        return current
