from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Type

from app.agent.adapter import LLMAdapter, LLMConfigurationError, LiteLLMAdapter
from app.agent.llm_providers import DeepSeekLLMAdapter, OpenAILLMAdapter, QwenLLMAdapter


@dataclass(frozen=True)
class LLMAdapterConfig:
    provider: str
    api_key: str | None
    model: str | None
    base_url: str | None = None


class LLMAdapterFactory:
    _PROVIDERS: dict[str, Type[LiteLLMAdapter]] = {
        "openai": OpenAILLMAdapter,
        "deepseek": DeepSeekLLMAdapter,
        "qwen": QwenLLMAdapter,
    }

    @classmethod
    def _provider_class(cls, provider: str) -> Type[LiteLLMAdapter]:
        normalized = provider.strip().lower()
        adapter_class = cls._PROVIDERS.get(normalized)
        if adapter_class is None:
            raise LLMConfigurationError(
                f"不支持的 LLM_PROVIDER: {provider!r}。允许值：openai, deepseek, qwen。"
            )
        return adapter_class

    @classmethod
    def create(cls, config: LLMAdapterConfig) -> LLMAdapter:
        adapter_class = cls._provider_class(config.provider)
        return adapter_class(config.api_key, config.model, config.base_url)

    @classmethod
    def from_env(cls) -> LLMAdapter:
        provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        adapter_class = cls._provider_class(provider)
        config = LLMAdapterConfig(
            provider=provider,
            api_key=os.getenv(adapter_class.API_KEY_ENV),
            model=adapter_class.MODEL_ENV,
            base_url=os.getenv(adapter_class.BASE_URL_ENV),
        )
        return cls.create(config)
