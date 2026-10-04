import pytest
from pydantic import ValidationError

from app.nl2sql.models import NL2SQLState, ReflectionDecision, SQLDraft


def test_sql_draft_requires_structured_status_and_safe_fields():
    draft = SQLDraft(
        status="ok",
        sql="SELECT output_quantity FROM production_output LIMIT 100",
        tables=["production_output"],
        parameters={},
        explanation="查询产量",
    )

    assert draft.status == "ok"
    assert draft.sql is not None


def test_clarify_draft_does_not_require_sql():
    draft = SQLDraft(
        status="clarify",
        sql=None,
        tables=[],
        parameters={},
        explanation="请补充时间范围",
    )

    assert draft.sql is None


def test_ok_draft_without_sql_is_rejected():
    with pytest.raises(ValidationError):
        SQLDraft(status="ok", sql=None, tables=[], parameters={}, explanation="无 SQL")


def test_graph_state_has_bounded_attempt_fields():
    state = NL2SQLState.model_validate(
        {
            "user_id": "u1",
            "session_id": "s1",
            "user_message": "查询产量",
            "attempt": 0,
            "reflection_count": 0,
            "max_attempts": 3,
            "max_reflections": 2,
        }
    )

    assert state.attempt == 0
    assert state.reflection_count == 0


def test_reflection_decision_is_limited_to_known_values():
    assert ReflectionDecision(decision="pass", reason="结果符合问题").decision == "pass"
    with pytest.raises(ValidationError):
        ReflectionDecision(decision="retry_forever", reason="无")
