import pytest

from app.intent.classifier import IntentClassificationError
from app.intent.fake_classifier import FakeIntentClassifier
from app.intent.models import IntentDecision, IntentName
from app.memory.models import TokenUsage
from app.intent.router import IntentRouter


def test_router_uses_classifier_result_above_threshold():
    classifier = FakeIntentClassifier.for_result("查今天的产量", "data_operation", "read", 0.95)
    result = IntentRouter(classifier, min_confidence=0.70).route("查今天的产量")

    assert result.decision.intent is IntentName.DATA_OPERATION
    assert result.decision.data_action.value == "read"
    assert result.routed_intent is IntentName.DATA_OPERATION
    assert result.fallback_reason is None


def test_router_falls_back_to_chat_for_low_confidence():
    classifier = FakeIntentClassifier.for_result("帮我看看", "nl2sql", None, 0.30)
    result = IntentRouter(classifier, min_confidence=0.70).route("帮我看看")

    assert result.decision.intent is IntentName.NL2SQL
    assert result.decision.provider == "fake"
    assert result.decision.model == "fake-intent-v1"
    assert result.routed_intent is IntentName.CHAT
    assert result.fallback_reason == "low_confidence"


def test_low_confidence_preserves_original_latency_and_usage():
    decision = IntentDecision(
        intent=IntentName.NL2SQL,
        confidence=0.2,
        provider="jev",
        model="typesafe/jev-1.13",
        latency_ms=85.4,
        usage=TokenUsage(requests=1, input_tokens=10, output_tokens=2, total_tokens=12),
    )
    classifier = FakeIntentClassifier({"ambiguous": decision})

    result = IntentRouter(classifier).route("ambiguous")

    assert result.decision.latency_ms == 85.4
    assert result.decision.usage.total_tokens == 12


def test_router_falls_back_to_chat_for_invalid_classifier_result():
    classifier = FakeIntentClassifier.for_invalid_result("无法解析")

    result = IntentRouter(classifier).route("无法解析")

    assert result.decision.intent is IntentName.CHAT
    assert result.decision.provider == "fallback"
    assert result.routed_intent is IntentName.CHAT
    assert result.fallback_reason == "invalid_result"


def test_router_does_not_hide_configuration_or_service_errors():
    classifier = FakeIntentClassifier.for_error(IntentClassificationError("配置缺失"))

    with pytest.raises(IntentClassificationError, match="配置缺失"):
        IntentRouter(classifier).route("你好")
