from app.agent.adapter import LLMResponse, LLMServiceUnavailable
from app.memory.models import ContextPolicy, SummaryRecord
from app.memory.summary_service import SummaryService
from app.memory.token_manager import TokenManager
from app.storage.session_service import SessionService


class FakeSummaryLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        return self.responses.pop(0)


def test_summary_service_compacts_only_explicitly_moved_messages(tmp_path, monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kwargs: 1)
    session = SessionService(tmp_path / "chat.sqlite3")
    session_id = session.create_session("user-a")
    for index in range(6):
        session.append_message("user-a", session_id, "user", f"message-{index}")
    old = SummaryRecord(user_id="user-a", session_id=session_id, summary="old summary", summarized_through_message_id=2)
    session.save_summary(old)
    messages = session.get_all_messages("user-a", session_id)
    fake = FakeSummaryLLM([LLMResponse.message("new rolling summary")])
    service = SummaryService(session, fake, TokenManager("openai", "gpt-4.1-mini"))

    result = service.compact("user-a", session_id, old, messages[2:4], max_summary_chars=1000)

    assert result.compacted is True
    assert result.summary is not None
    assert result.summary.summarized_through_message_id == 4
    assert len(session.get_all_messages("user-a", session_id)) == 6
    assert fake.calls[0][1] == []
    prompt = fake.calls[0][0][1]["content"]
    assert "old summary" in prompt
    assert "message-2" in prompt
    assert "message-3" in prompt
    assert "message-4" not in prompt
    assert "message-5" not in prompt


def test_summary_service_failure_keeps_old_summary_and_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda **kwargs: 1)
    session = SessionService(tmp_path / "chat.sqlite3")
    session_id = session.create_session("user-a")
    session.append_message("user-a", session_id, "user", "one")
    session.append_message("user-a", session_id, "user", "two")
    old = SummaryRecord(user_id="user-a", session_id=session_id, summary="old summary", summarized_through_message_id=1)
    session.save_summary(old)
    fake = FakeSummaryLLM([LLMResponse.message("")])
    service = SummaryService(session, fake, TokenManager("openai", "gpt-4.1-mini"))
    messages = session.get_all_messages("user-a", session_id)

    result = service.compact("user-a", session_id, old, messages[1:], max_summary_chars=1000)

    assert result.compacted is False
    assert result.summary == old
    assert session.get_summary("user-a", session_id) == old


def test_summary_service_recompacts_old_summary_without_moving_message_boundary(tmp_path):
    session = SessionService(tmp_path / "chat.sqlite3")
    session_id = session.create_session("user-a")
    old = SummaryRecord(user_id="user-a", session_id=session_id, summary="x" * 200, summarized_through_message_id=4)
    session.save_summary(old)
    fake = FakeSummaryLLM([LLMResponse.message("short summary")])
    service = SummaryService(session, fake, TokenManager("openai", "gpt-4.1-mini"))

    result = service.compact(
        "user-a",
        session_id,
        old,
        [],
        max_summary_chars=1000,
        recompact_old_summary=True,
    )

    assert result.compacted is True
    assert result.summary.summary == "short summary"
    assert result.summary.summarized_through_message_id == 4


def test_summary_service_model_failure_does_not_overwrite_old_summary(tmp_path):
    session = SessionService(tmp_path / "chat.sqlite3")
    session_id = session.create_session("user-a")
    old = SummaryRecord(user_id="user-a", session_id=session_id, summary="old", summarized_through_message_id=4)
    session.save_summary(old)

    class FailingLLM(FakeSummaryLLM):
        def complete(self, messages, tools):
            raise LLMServiceUnavailable("offline")

    service = SummaryService(session, FailingLLM([]), TokenManager("openai", "gpt-4.1-mini"))
    result = service.compact("user-a", session_id, old, [{"id": 5, "role": "user", "content": "new"}], 1000)

    assert result.compacted is False
    assert session.get_summary("user-a", session_id) == old


def test_summary_service_validates_long_term_memory_json(tmp_path):
    session = SessionService(tmp_path / "chat.sqlite3")
    fake = FakeSummaryLLM(
        [
            LLMResponse.message(
                '[{"content":"稳定项目规则"}]'
            )
        ]
    )
    service = SummaryService(session, fake, TokenManager("openai", "gpt-4.1-mini"))

    entries, usage = service.extract_long_term_memory("summary")

    assert entries[0].content == "稳定项目规则"
    assert usage.estimated_context_tokens == 1
    prompt = fake.calls[0][0][0]["content"]
    assert "只能包含 content 字段" in prompt


def test_summary_service_ignores_invalid_long_term_memory_json(tmp_path):
    session = SessionService(tmp_path / "chat.sqlite3")
    fake = FakeSummaryLLM([LLMResponse.message("not-json")])
    service = SummaryService(session, fake, TokenManager("openai", "gpt-4.1-mini"))

    entries, usage = service.extract_long_term_memory("summary")

    assert entries == []
    assert usage.estimated_context_tokens == 1


def test_summary_service_ignores_empty_or_extra_field_memory_candidates(tmp_path):
    session = SessionService(tmp_path / "chat.sqlite3")
    fake = FakeSummaryLLM(
        [
            LLMResponse.message("[]"),
            LLMResponse.message('[{"content":"valid","extra":"reject"}]'),
        ]
    )
    service = SummaryService(session, fake, TokenManager("openai", "gpt-4.1-mini"))

    empty_entries, _ = service.extract_long_term_memory("summary")
    invalid_entries, _ = service.extract_long_term_memory("summary")

    assert empty_entries == []
    assert invalid_entries == []
