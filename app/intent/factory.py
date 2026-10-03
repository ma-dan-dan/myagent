from __future__ import annotations

import os

from app.agent.adapter import LLMAdapter
from app.config import DEFAULT_INTENT_PROVIDER, INTENT_PROVIDER_ENV
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
        provider = (os.getenv(INTENT_PROVIDER_ENV) or DEFAULT_INTENT_PROVIDER).strip().lower()
        if provider == "fake":
            return FakeIntentClassifier()
        if provider == "llm":
            return LLMIntentClassifier(llm)
        if provider == "jev":
            api_key = os.getenv("TYPESAFE_API_KEY")
            if not api_key or not api_key.strip():
                raise IntentClassificationError("缺少 Jev 配置：TYPESAFE_API_KEY。")
            return JevIntentClassifier(api_key)
        allowed = ", ".join(cls.ALLOWED_PROVIDERS)
        raise IntentClassificationError(f"不支持的 INTENT_PROVIDER: {provider!r}。允许值：{allowed}。")
