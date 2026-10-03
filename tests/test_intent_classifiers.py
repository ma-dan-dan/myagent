import pytest
from pydantic import ValidationError

from app.agent.adapter import LLMResponse
from app.intent.classifier import IntentClassificationError
from app.intent.llm_classifier import LLMIntentClassifier
from app.intent.models import DataAction, IntentDecision, IntentName


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
