from __future__ import annotations

from app.intent.classifier import IntentClassifier, InvalidIntentDecision
from app.intent.models import IntentDecision, IntentName


class IntentRouter:
    def __init__(self, classifier: IntentClassifier, min_confidence: float = 0.70) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.classifier = classifier
        self.min_confidence = min_confidence

    def route(self, message: str) -> IntentDecision:
        try:
            decision = self.classifier.classify(message)
        except InvalidIntentDecision:
            return self._fallback("invalid_result")

        if decision.confidence < self.min_confidence:
            return decision.model_copy(
                update={
                    "intent": IntentName.CHAT,
                    "data_action": None,
                    "provider": "fallback",
                    "fallback_reason": "low_confidence",
                }
            )
        return decision

    @staticmethod
    def _fallback(reason: str) -> IntentDecision:
        return IntentDecision(
            intent=IntentName.CHAT,
            confidence=0,
            provider="fallback",
            model="intent-router",
            latency_ms=0,
            fallback_reason=reason,
        )
