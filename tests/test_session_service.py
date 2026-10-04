import sqlite3

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


def test_session_service_migrates_legacy_intent_classifications_table(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, session_id)
            );
            CREATE TABLE intent_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                intent TEXT NOT NULL,
                data_action TEXT NULL,
                confidence REAL NOT NULL,
                latency_ms REAL NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                fallback_reason TEXT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO sessions VALUES ('alice', 'session', datetime('now'));
            INSERT INTO intent_classifications(
                user_id, session_id, provider, model, intent, data_action,
                confidence, latency_ms, input_tokens, output_tokens, total_tokens,
                fallback_reason, created_at
            ) VALUES ('alice', 'session', 'jev', 'typesafe/jev-1.13', 'nl2sql', NULL,
                      0.9, 12.0, 3, 2, 5, NULL, datetime('now'));
            """
        )

    service = SessionService(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(intent_classifications)")}
        routed_intent = connection.execute(
            "SELECT routed_intent FROM intent_classifications WHERE user_id = 'alice'"
        ).fetchone()[0]
    metrics = service.get_intent_metrics("alice", "jev")

    assert "routed_intent" in columns
    assert routed_intent == "nl2sql"
    assert metrics["classified_intent_counts"] == {"nl2sql": 1}
    assert metrics["routed_intent_counts"] == {"nl2sql": 1}
