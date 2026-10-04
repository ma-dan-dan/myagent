import pytest
from pydantic import ValidationError

from app.agent.adapter import LLMResponse
from app.intent.classifier import IntentClassificationError
from app.intent.factory import IntentClassifierFactory
from app.intent import jev_classifier
from app.intent.llm_classifier import LLMIntentClassifier
from app.intent.models import DataAction, IntentDecision, IntentName
from app.config import IntentRuntimeConfig


class StubLLM:
    provider = "qwen"
    model = "test-qwen"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        return self.response


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


def test_llm_classifier_parses_valid_json_without_tools():
    llm = StubLLM(
        LLMResponse.message(
            '{"intent":"nl2sql","data_action":null,"confidence":0.91}'
        )
    )

    result = LLMIntentClassifier(llm).classify("帮我生成查询产量的 SQL")

    assert result.intent is IntentName.NL2SQL
    assert result.provider == "llm"
    assert result.model == "test-qwen"
    assert llm.calls[0][1] == []


def test_llm_classifier_rejects_invalid_json():
    llm = StubLLM(LLMResponse.message("不是 JSON"))

    with pytest.raises(IntentClassificationError):
        LLMIntentClassifier(llm).classify("你好")


def test_jev_factory_requires_typesafe_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("INTENT_PROVIDER", "jev")

    with pytest.raises(IntentClassificationError, match="TYPESAFE_API_KEY"):
        IntentClassifierFactory.from_env(StubLLM(LLMResponse.message("{}")))


def test_jev_classifier_maps_mocked_official_response(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def system_one(self, **kwargs):
            calls.append(("system_one", kwargs))
            return type(
                "Response",
                (),
                {
                    "answers": {
                        "intent": type("Answer", (), {"choice": "data_operation", "confidence": 0.92})(),
                        "data_action": type("Answer", (), {"choice": "read", "confidence": 0.88})(),
                    }
                },
            )()

    monkeypatch.setattr(jev_classifier, "TypeSafeClient", FakeClient)
    classifier = jev_classifier.JevIntentClassifier("test-key")

    result = classifier.classify("查今天产量")

    assert result.intent is IntentName.DATA_OPERATION
    assert result.data_action.value == "read"
    assert result.provider == "jev"
    assert result.model == "typesafe/jev-1.13"
    assert result.usage.total_tokens == 0
    assert calls[1][1]["model"] == "typesafe/jev-1.13"
    assert set(calls[1][1]["questions"]) == {"intent", "data_action"}


def test_intent_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("INTENT_PROVIDER", "unknown")

    with pytest.raises(IntentClassificationError, match="fake, jev, llm"):
        IntentClassifierFactory.from_env(StubLLM(LLMResponse.message("{}")))


def test_jev_factory_uses_centralized_intent_runtime_config(monkeypatch):
    monkeypatch.setattr(
        "app.intent.factory.get_intent_runtime_config",
        lambda: IntentRuntimeConfig("jev", "jev-key"),
    )

    classifier = IntentClassifierFactory.from_env(StubLLM(LLMResponse.message("{}")))

    assert isinstance(classifier, jev_classifier.JevIntentClassifier)
    assert classifier.api_key == "jev-key"
