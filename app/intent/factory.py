from __future__ import annotations

from app.agent.adapter import LLMAdapter
from app.config import TYPESAFE_API_KEY_ENV, get_intent_runtime_config
from app.intent.classifier import IntentClassificationError, IntentClassifier
from app.intent.fake_classifier import FakeIntentClassifier
from app.intent.jev_classifier import JevIntentClassifier
from app.intent.llm_classifier import LLMIntentClassifier


class UnavailableIntentClassifier:
    def __init__(self, message: str) -> None:
        self.message = message

    def classify(self, message: str):
        raise IntentClassificationError(self.message)


class IntentClassifierFactory:
    ALLOWED_PROVIDERS = ("fake", "jev", "llm")

    @classmethod
    def from_env(cls, llm: LLMAdapter) -> IntentClassifier:
        runtime_config = get_intent_runtime_config()
        provider = runtime_config.provider
        if provider == "fake":
            return FakeIntentClassifier()
        if provider == "llm":
            return LLMIntentClassifier(llm)
        if provider == "jev":
            api_key = runtime_config.typesafe_api_key
            if not api_key or not api_key.strip():
                raise IntentClassificationError(f"缺少 Jev 配置：{TYPESAFE_API_KEY_ENV}。")
            return JevIntentClassifier(api_key)
        allowed = ", ".join(cls.ALLOWED_PROVIDERS)
        raise IntentClassificationError(f"不支持的 INTENT_PROVIDER: {provider!r}。允许值：{allowed}。")
