from __future__ import annotations

from collections.abc import Mapping

from app.intent.classifier import IntentClassificationError, InvalidIntentDecision
from app.intent.models import DataAction, IntentDecision, IntentName


class FakeIntentClassifier:
    provider = "fake"
    model = "fake-intent-v1"

    def __init__(
        self,
        results: Mapping[str, IntentDecision] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._results = dict(results or {})
        self._error = error

    @classmethod
    def for_result(
        cls,
        message: str,
        intent: str,
        data_action: str | None,
        confidence: float,
    ) -> "FakeIntentClassifier":
        return cls(
            {
                message: IntentDecision(
                    intent=intent,
                    data_action=data_action,
                    confidence=confidence,
                    provider=cls.provider,
                    model=cls.model,
                    latency_ms=0,
                )
            }
        )

    @classmethod
    def for_invalid_result(cls, message: str) -> "FakeIntentClassifier":
        return cls(error=InvalidIntentDecision("Fake 分类结果无效。"))

    @classmethod
    def for_error(cls, error: Exception) -> "FakeIntentClassifier":
        return cls(error=error)

    def classify(self, message: str) -> IntentDecision:
        if self._error is not None:
            raise self._error
        return self._results.get(
            message,
            IntentDecision(
                intent=IntentName.CHAT,
                confidence=1.0,
                provider=self.provider,
                model=self.model,
                latency_ms=0,
            ),
        )
