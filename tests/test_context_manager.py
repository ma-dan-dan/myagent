from app.memory.context_manager import FIXED_SYSTEM_PROMPT, ContextManager
from app.memory.long_term_memory import LongTermMemoryStore
from app.memory.models import CompactionResult, ContextPolicy, MemoryEntry, SummaryRecord
from app.memory.project_context import ProjectContextLoader
from app.memory.context_manager import ContextBudgetExceeded
from app.storage.session_service import SessionService


class DeterministicTokenManager:
    context_window = 150
    output_reserve = 50
    hard_input_budget = 100
    soft_input_budget = 80
    fixed_context_budget = 16

    def __init__(self, *, fixed_cost=5, current_cost=5, history_cost=1):
        self.policy = ContextPolicy(recent_message_limit=12)
        self.fixed_cost = fixed_cost
        self.current_cost = current_cost
        self.history_cost = history_cost

    def estimate(self, messages):
        total = 0
        for message in messages:
            content = str(message.get("content", ""))
            if content.startswith("history-"):
                total += self.history_cost
            elif content == "current" or content == "too-long-current":
                total += self.current_cost
            else:
                total += self.fixed_cost if message.get("role") == "system" else self.history_cost
        return total


class RecordingSummaryService:
    def __init__(self, session):
        self.session = session
        self.compact_calls = []
        self.memory_calls = []

    def compact(self, user_id, session_id, old_summary, messages_to_summarize, max_summary_chars, **kwargs):
        self.compact_calls.append((old_summary, list(messages_to_summarize), kwargs))
        if not messages_to_summarize and not kwargs.get("recompact_old_summary"):
            return CompactionResult(summary=old_summary)
        if kwargs.get("recompact_old_summary"):
            boundary = old_summary.summarized_through_message_id
        else:
            boundary = messages_to_summarize[-1].id
        summary = SummaryRecord(
            user_id=user_id,
            session_id=session_id,
            summary=f"summary-{len(self.compact_calls)}",
            summarized_through_message_id=boundary,
        )
        self.session.save_summary(summary)
        return CompactionResult(attempted=True, compacted=True, summary=summary)

    def extract_long_term_memory(self, summary):
        self.memory_calls.append(summary)
        return [], None


def build_manager(tmp_path, token_manager):
    session = SessionService(tmp_path / "chat.sqlite3")
    session_id = session.create_session("user-a")
    manager = ContextManager(
        session,
        RecordingSummaryService(session),
        ProjectContextLoader(tmp_path),
        LongTermMemoryStore(tmp_path),
        token_manager,
        token_manager.policy,
    )
    return manager, session, session_id


def test_context_manager_keeps_twelve_short_messages_without_summary(tmp_path):
    manager, session, session_id = build_manager(tmp_path, DeterministicTokenManager())
    for index in range(12):
        session.append_message("user-a", session_id, "user", f"history-{index}")

    prepared = manager.prepare("user-a", session_id, "current")

    assert not manager.summary_service.compact_calls
    assert [message["content"] for message in prepared.messages if message["role"] == "user"] == [
        *(f"history-{index}" for index in range(12)),
        "current",
    ]


def test_context_manager_moves_oldest_messages_when_twelve_messages_exceed_token_budget(tmp_path):
    manager, session, session_id = build_manager(
        tmp_path,
        DeterministicTokenManager(fixed_cost=5, current_cost=5, history_cost=10),
    )
    for index in range(12):
        session.append_message("user-a", session_id, "user", f"history-{index}")

    prepared = manager.prepare("user-a", session_id, "current")

    moved = manager.summary_service.compact_calls[0][1]
    assert [message.content for message in moved] == [f"history-{index}" for index in range(5)]
    assert [message["content"] for message in prepared.messages if message["content"].startswith("history-")] == [
        f"history-{index}" for index in range(6, 12)
    ]
    assert prepared.messages[-1]["content"] == "current"
    assert prepared.maintenance.summary.summarized_through_message_id == 6
    assert len(manager.summary_service.compact_calls) == 2


def test_context_manager_preserves_xml_tags_when_shrinking_fixed_context(tmp_path):
    (tmp_path / "Agent.md").write_text("project " * 100, encoding="utf-8")
    memory_path = tmp_path / ".myagent" / "memory" / "chat" / "MEMORY.md"
    memory_path.parent.mkdir(parents=True)
    memory_path.write_text("# 长期记忆\n- " + "memory " * 100, encoding="utf-8")
    manager, _, session_id = build_manager(tmp_path, DeterministicTokenManager(fixed_cost=10))

    prepared = manager.prepare("user-a", session_id, "current")

    joined = "\n".join(message["content"] for message in prepared.messages)
    assert FIXED_SYSTEM_PROMPT in joined
    assert joined.count("<long_term_memory>") == joined.count("</long_term_memory>")
    assert joined.count("<project_rules>") == joined.count("</project_rules>")


def test_context_manager_raises_when_minimum_prompt_exceeds_hard_budget(tmp_path):
    manager, _, session_id = build_manager(
        tmp_path,
        DeterministicTokenManager(fixed_cost=80, current_cost=30),
    )

    try:
        manager.prepare("user-a", session_id, "too-long-current")
    except ContextBudgetExceeded as exc:
        assert exc.estimated_tokens > exc.hard_input_budget
        assert "缩短" in exc.reason
    else:
        raise AssertionError("ContextBudgetExceeded was not raised")


def test_context_manager_rejects_schema_evidence_that_exceeds_fixed_context_budget(tmp_path):
    manager, _, session_id = build_manager(
        tmp_path,
        DeterministicTokenManager(fixed_cost=20, current_cost=5),
    )

    try:
        manager.prepare("user-a", session_id, "current", extra_context="<schema_evidence>DDL</schema_evidence>")
    except ContextBudgetExceeded as exc:
        assert "固定上下文" in exc.reason
    else:
        raise AssertionError("ContextBudgetExceeded was not raised")


def test_context_manager_limits_summary_calls_to_two(tmp_path):
    manager, session, session_id = build_manager(
        tmp_path,
        DeterministicTokenManager(fixed_cost=80, current_cost=5, history_cost=10),
    )
    for index in range(20):
        session.append_message("user-a", session_id, "user", f"history-{index}")

    try:
        manager.prepare("user-a", session_id, "current")
    except ContextBudgetExceeded:
        pass
    else:
        raise AssertionError("ContextBudgetExceeded was not raised")

    assert len(manager.summary_service.compact_calls) <= 2
