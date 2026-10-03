from app.storage.session_service import SessionService
from app.memory.models import SummaryRecord, TokenUsage


def test_session_history_is_isolated_by_user_and_session(tmp_path):
    service = SessionService(tmp_path / "chat.sqlite3")

    service.append_message("user-a", "shared-session", "user", "我想看产量")
    service.append_message("user-a", "shared-session", "assistant", "请补充时间范围")
    service.append_message("user-b", "shared-session", "user", "这是另一个用户")

    assert [message.content for message in service.get_messages("user-a", "shared-session")] == [
        "我想看产量",
        "请补充时间范围",
    ]
    assert [message.content for message in service.get_messages("user-b", "shared-session")] == [
        "这是另一个用户",
    ]


def test_session_service_creates_and_returns_a_session_id(tmp_path):
    service = SessionService(tmp_path / "chat.sqlite3")

    session_id = service.create_session("user-a")

    assert session_id
    assert service.get_messages("user-a", session_id) == []


def test_session_service_persists_summary_and_reads_messages_after_boundary(tmp_path):
    service = SessionService(tmp_path / "chat.sqlite3")
    session_id = service.create_session("user-a")
    service.append_message("user-a", session_id, "user", "first")
    service.append_message("user-a", session_id, "assistant", "second")
    service.append_message("user-a", session_id, "user", "third")

    summary = SummaryRecord(
        user_id="user-a",
        session_id=session_id,
        summary="The first exchange is summarized.",
        summarized_through_message_id=2,
    )
    service.save_summary(summary)

    assert service.get_summary("user-a", session_id) == summary
    assert [message.content for message in service.get_messages_after_id("user-a", session_id, 2)] == ["third"]


def test_session_service_records_turn_and_session_usage(tmp_path):
    service = SessionService(tmp_path / "chat.sqlite3")
    session_id = service.create_session("user-a")
    usage = TokenUsage(
        requests=2,
        input_tokens=30,
        output_tokens=10,
        total_tokens=40,
        estimated_context_tokens=25,
        context_window=128000,
    )

    service.record_usage("user-a", session_id, "chat", "gpt-4.1-mini", usage)
    service.record_usage("user-a", session_id, "summary", "gpt-4.1-mini", usage.model_copy(update={"requests": 1}))

    session_usage = service.get_session_usage("user-a", session_id, usage)
    assert session_usage.current_turn == usage
    assert session_usage.session_total.requests == 3
    assert session_usage.session_total.input_tokens == 60
    assert session_usage.session_total.output_tokens == 20
    assert session_usage.session_total.total_tokens == 80
