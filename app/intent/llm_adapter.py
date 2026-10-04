from __future__ import annotations

from typing import Any

from app.agent.adapter import LLMAdapter, LLMResponse, ToolSpec


class IntentLLMAdapter:
    """Thin adapter that identifies an independently configured intent model."""

    def __init__(self, adapter: LLMAdapter) -> None:
        self._adapter = adapter

    @property
    def provider(self) -> str:
        return str(getattr(self._adapter, "provider", "unknown") or "unknown")

    @property
    def model(self) -> str:
        return str(getattr(self._adapter, "model", "unknown") or "unknown")

    @property
    def base_url(self) -> str | None:
        return getattr(self._adapter, "base_url", None)

    @property
    def litellm_model_name(self) -> str:
        return str(getattr(self._adapter, "litellm_model_name", self.model) or self.model)

    def complete(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse:
        return self._adapter.complete(messages, tools)
