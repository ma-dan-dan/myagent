import pytest
from pydantic import ValidationError

from app.intent.models import DataAction, IntentDecision, IntentName


def test_non_data_intent_clears_data_action():
    result = IntentDecision(
        intent=IntentName.CHAT,
        data_action=DataAction.DELETE,
        confidence=0.9,
        provider="fake",
        model="fake-intent-v1",
        latency_ms=0,
    )

    assert result.data_action is None


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        IntentDecision(
            intent=IntentName.CHAT,
            confidence=1.1,
            provider="fake",
            model="fake-intent-v1",
            latency_ms=0,
        )
