import pytest

from app.agent.adapter import LLMConfigurationError
from app.agent.llm_factory import LLMAdapterFactory
from app.config import (
    get_intent_llm_runtime_config,
    get_intent_runtime_config,
    get_llm_runtime_config,
)


def test_get_llm_runtime_config_reads_selected_qwen_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "test-key")

    config = get_llm_runtime_config()

    assert config.provider == "qwen"
    assert config.api_key == "test-key"
    assert config.model == "deepseek-v4.1-flash"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_llm_base_url_uses_code_default_instead_of_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://override.example/v1")

    config = get_llm_runtime_config()

    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_get_intent_runtime_config_reads_jev_key(monkeypatch):
    monkeypatch.setenv("INTENT_PROVIDER", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-jev-key")

    config = get_intent_runtime_config()

    assert config.provider == "jev"
    assert config.typesafe_api_key == "test-jev-key"


def test_intent_llm_runtime_config_uses_selected_provider(monkeypatch):
    monkeypatch.setenv("INTENT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "intent-key")

    config = get_intent_llm_runtime_config()

    assert config.provider == "deepseek"
    assert config.api_key == "intent-key"
    assert config.model == "deepseek-flash"
    assert config.base_url == "https://api.deepseek.com"


def test_intent_llm_runtime_config_defaults_to_chat_provider(monkeypatch):
    monkeypatch.delenv("INTENT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "intent-key")

    config = get_intent_llm_runtime_config()

    assert config.provider == "qwen"
    assert config.model == "deepseek-v4.1-flash"


def test_runtime_config_uses_existing_default_providers(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("INTENT_PROVIDER", raising=False)

    assert get_llm_runtime_config().provider == "qwen"
    assert get_llm_runtime_config().model == "deepseek-v4.1-flash"
    assert get_intent_runtime_config().provider == "llm"


def test_missing_provider_key_keeps_existing_error_message(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)

    adapter = LLMAdapterFactory.from_env()

    with pytest.raises(LLMConfigurationError, match="QWEN_API_KEY"):
        adapter.complete([], [])
