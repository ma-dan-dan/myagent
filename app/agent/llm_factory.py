from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from app.agent.adapter import LLMAdapter, LLMConfigurationError, LiteLLMAdapter
from app.agent.llm_providers import DeepSeekLLMAdapter, OpenAILLMAdapter, QwenLLMAdapter
from app.config import get_llm_runtime_config


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
        try:
            runtime_config = get_llm_runtime_config()
        except ValueError as exc:
            cls._provider_class(str(exc))
            raise
        adapter_class = cls._provider_class(runtime_config.provider)
        config = LLMAdapterConfig(
            provider=runtime_config.provider,
            api_key=runtime_config.api_key,
            model=runtime_config.model,
            base_url=runtime_config.base_url,
        )
        return cls.create(config)
