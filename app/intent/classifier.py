from __future__ import annotations

from typing import Protocol

from app.intent.models import IntentDecision


class IntentClassifier(Protocol):
    def classify(self, message: str) -> IntentDecision:
        """Classify one user message without mutating session state."""


class IntentClassificationError(RuntimeError):
    """Raised when intent classification cannot produce a usable result."""


class InvalidIntentDecision(IntentClassificationError):
    """Raised when a classifier response cannot be validated as an intent result."""
