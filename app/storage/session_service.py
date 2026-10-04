from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from pathlib import Path

from app.intent.models import IntentDecision, IntentName
from app.memory.models import SessionUsage, SummaryRecord, TokenUsage
from app.schemas.chat import StoredMessage


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def _validate_identifier(value: str, field_name: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {field_name}")
    return value


class SessionService:
    """Stores bounded chat history and always scopes reads by user and session."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (user_id, session_id)
                        REFERENCES sessions(user_id, session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_scope
                    ON messages(user_id, session_id, id);
                CREATE TABLE IF NOT EXISTS session_summaries (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    summarized_through_message_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id),
                    FOREIGN KEY (user_id, session_id)
                        REFERENCES sessions(user_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS token_usages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    operation TEXT NOT NULL CHECK(operation IN ('chat', 'summary', 'memory')),
                    model TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_context_tokens INTEGER NOT NULL DEFAULT 0,
                    context_window INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id, session_id)
                        REFERENCES sessions(user_id, session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_token_usages_scope
                    ON token_usages(user_id, session_id, id);
                CREATE TABLE IF NOT EXISTS intent_classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    routed_intent TEXT NOT NULL,
                    data_action TEXT NULL,
                    confidence REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    fallback_reason TEXT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id, session_id)
                        REFERENCES sessions(user_id, session_id)
                );
                CREATE INDEX IF NOT EXISTS idx_intent_classifications_scope
                    ON intent_classifications(user_id, session_id, id);
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(token_usages)").fetchall()}
            if "requests" not in columns:
                connection.execute("ALTER TABLE token_usages ADD COLUMN requests INTEGER NOT NULL DEFAULT 0")
            intent_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(intent_classifications)").fetchall()
            }
            if "routed_intent" not in intent_columns:
                connection.execute(
                    "ALTER TABLE intent_classifications ADD COLUMN routed_intent TEXT NOT NULL DEFAULT 'chat'"
                )
                connection.execute(
                    """
                    UPDATE intent_classifications
                    SET routed_intent = intent
                    WHERE routed_intent = 'chat' AND fallback_reason IS NULL
                    """
                )

    def create_session(self, user_id: str) -> str:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = uuid.uuid4().hex
        self.ensure_session(user_id, session_id)
        return session_id

    def ensure_session(self, user_id: str, session_id: str) -> None:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = _validate_identifier(session_id, "session_id")
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO sessions(user_id, session_id, created_at) VALUES (?, ?, datetime('now'))",
                (user_id, session_id),
            )

    def append_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> int:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = _validate_identifier(session_id, "session_id")
        message = StoredMessage(role=role, content=content, metadata=metadata or {})
        self.ensure_session(user_id, session_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages(user_id, session_id, role, content, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    message.role,
                    message.content,
                    message.created_at.isoformat(),
                    json.dumps(message.metadata, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def get_messages(self, user_id: str, session_id: str, limit: int = 20) -> list[StoredMessage]:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = _validate_identifier(session_id, "session_id")
        if limit < 1:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, created_at, metadata_json
                FROM messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, session_id, limit),
            ).fetchall()
        return [self._stored_message(row) for row in reversed(rows)]

    def get_messages_after_id(self, user_id: str, session_id: str, message_id: int) -> list[StoredMessage]:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = _validate_identifier(session_id, "session_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, created_at, metadata_json
                FROM messages
                WHERE user_id = ? AND session_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (user_id, session_id, message_id),
            ).fetchall()
        return [self._stored_message(row) for row in rows]

    def get_all_messages(self, user_id: str, session_id: str) -> list[StoredMessage]:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = _validate_identifier(session_id, "session_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, created_at, metadata_json
                FROM messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY id ASC
                """,
                (user_id, session_id),
            ).fetchall()
        return [self._stored_message(row) for row in rows]

    @staticmethod
    def _stored_message(row: sqlite3.Row) -> StoredMessage:
        return StoredMessage(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    def save_summary(self, summary: SummaryRecord) -> None:
        user_id = _validate_identifier(summary.user_id, "user_id")
        session_id = _validate_identifier(summary.session_id, "session_id")
        self.ensure_session(user_id, session_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_summaries(
                    user_id, session_id, summary, summarized_through_message_id, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                    summary = excluded.summary,
                    summarized_through_message_id = excluded.summarized_through_message_id,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    session_id,
                    summary.summary,
                    summary.summarized_through_message_id,
                    summary.updated_at.isoformat(),
                ),
            )

    def get_summary(self, user_id: str, session_id: str) -> SummaryRecord | None:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = _validate_identifier(session_id, "session_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, session_id, summary, summarized_through_message_id, updated_at
                FROM session_summaries
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        if row is None:
            return None
        return SummaryRecord(
            user_id=row["user_id"],
            session_id=row["session_id"],
            summary=row["summary"],
            summarized_through_message_id=row["summarized_through_message_id"],
            updated_at=row["updated_at"],
        )

    def record_usage(
        self,
        user_id: str,
        session_id: str,
        operation: str,
        model: str,
        usage: TokenUsage,
    ) -> None:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = _validate_identifier(session_id, "session_id")
        if operation not in {"chat", "summary", "memory"}:
            raise ValueError("invalid usage operation")
        self.ensure_session(user_id, session_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO token_usages(
                    user_id, session_id, operation, model, requests, input_tokens, output_tokens,
                    total_tokens, estimated_context_tokens, context_window, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    user_id,
                    session_id,
                    operation,
                    model,
                    usage.requests,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    usage.estimated_context_tokens,
                    usage.context_window,
                ),
            )

    def record_intent_classification(
        self,
        user_id: str,
        session_id: str,
        decision: IntentDecision,
        routed_intent: IntentName,
        fallback_reason: str | None = None,
    ) -> None:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = _validate_identifier(session_id, "session_id")
        self.ensure_session(user_id, session_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO intent_classifications(
                    user_id, session_id, provider, model, intent, routed_intent, data_action,
                    confidence, latency_ms, input_tokens, output_tokens, total_tokens,
                    fallback_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    user_id,
                    session_id,
                    decision.provider,
                    decision.model,
                    decision.intent.value,
                    routed_intent.value,
                    decision.data_action.value if decision.data_action is not None else None,
                    decision.confidence,
                    decision.latency_ms,
                    decision.usage.input_tokens,
                    decision.usage.output_tokens,
                    decision.usage.total_tokens,
                    fallback_reason,
                ),
            )

    def get_intent_metrics(self, user_id: str, provider: str | None = None) -> dict:
        user_id = _validate_identifier(user_id, "user_id")
        query = """
            SELECT intent, routed_intent, latency_ms, input_tokens, output_tokens
            FROM intent_classifications
            WHERE user_id = ?
        """
        params: list[str] = [user_id]
        if provider is not None:
            query += " AND provider = ?"
            params.append(provider)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        latencies = sorted(float(row["latency_ms"]) for row in rows)
        classified_intent_counts: dict[str, int] = {}
        routed_intent_counts: dict[str, int] = {}
        for row in rows:
            classified_intent_counts[row["intent"]] = classified_intent_counts.get(row["intent"], 0) + 1
            routed_intent_counts[row["routed_intent"]] = routed_intent_counts.get(row["routed_intent"], 0) + 1
        return {
            "provider": provider,
            "request_count": len(rows),
            "classified_intent_counts": classified_intent_counts,
            "routed_intent_counts": routed_intent_counts,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50_latency_ms": self._percentile(latencies, 0.50),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "total_input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "total_output_tokens": sum(int(row["output_tokens"]) for row in rows),
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, math.ceil(len(values) * percentile) - 1))
        return values[index]

    def get_session_usage(self, user_id: str, session_id: str, current_turn: TokenUsage) -> SessionUsage:
        user_id = _validate_identifier(user_id, "user_id")
        session_id = _validate_identifier(session_id, "session_id")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COALESCE(SUM(requests), 0) AS requests,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(estimated_context_tokens), 0) AS estimated_context_tokens,
                    COALESCE(MAX(context_window), 0) AS context_window
                FROM token_usages
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        return SessionUsage(
            current_turn=current_turn,
            session_total=TokenUsage(
                requests=row["requests"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                total_tokens=row["total_tokens"],
                estimated_context_tokens=row["estimated_context_tokens"],
                context_window=row["context_window"],
            ),
        )
