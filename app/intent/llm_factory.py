from __future__ import annotations

from app.agent.adapter import LLMAdapter
from app.agent.llm_factory import LLMAdapterConfig, LLMAdapterFactory
from app.config import get_intent_llm_runtime_config
from app.intent.llm_adapter import IntentLLMAdapter


class IntentLLMAdapterFactory:
    @classmethod
    def from_env(cls) -> LLMAdapter:
        try:
            runtime_config = get_intent_llm_runtime_config()
        except ValueError as exc:
            LLMAdapterFactory.create(LLMAdapterConfig(str(exc), None, None))
            raise
        adapter = LLMAdapterFactory.create(
            LLMAdapterConfig(
                provider=runtime_config.provider,
                api_key=runtime_config.api_key,
                model=runtime_config.model,
                base_url=runtime_config.base_url,
            )
        )
        return IntentLLMAdapter(adapter)
