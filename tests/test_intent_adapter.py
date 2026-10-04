import litellm

from app.agent.llm_factory import LLMAdapterFactory, LLMAdapterConfig
from app.config import INTENT_LLM_MODEL_CONFIGS
from app.intent.llm_adapter import IntentLLMAdapter
from app.intent.llm_factory import IntentLLMAdapterFactory


def test_intent_adapter_factory_creates_independent_provider_adapter(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "chat-key")
    monkeypatch.setenv("INTENT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "intent-key")

    chat_adapter = LLMAdapterFactory.from_env()
    intent_adapter = IntentLLMAdapterFactory.from_env()

    assert isinstance(intent_adapter, IntentLLMAdapter)
    assert intent_adapter is not chat_adapter
    assert intent_adapter.provider == "deepseek"
    assert intent_adapter.model == "deepseek-flash"
    assert intent_adapter.base_url == "https://api.deepseek.com"


def test_intent_adapter_reuses_provider_litellm_call(monkeypatch):
    calls = []
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: calls.append(kwargs)
        or {
            "choices": [{"message": {"content": "{\"intent\":\"chat\"}"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    )
    monkeypatch.setenv("INTENT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "intent-key")

    adapter = IntentLLMAdapterFactory.from_env()
    response = adapter.complete([{"role": "user", "content": "你好"}], [])

    assert response.content == '{"intent":"chat"}'
    assert calls == [
        {
            "model": "deepseek/deepseek-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "api_key": "intent-key",
            "api_base": "https://api.deepseek.com",
        }
    ]


def test_intent_adapter_uses_provider_specific_intent_model(monkeypatch):
    calls = []
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: calls.append(kwargs)
        or {"choices": [{"message": {"content": "分类结果"}}]},
    )
    monkeypatch.setenv("INTENT_LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "intent-key")
    monkeypatch.setitem(INTENT_LLM_MODEL_CONFIGS, "qwen", "qwen-intent-small")

    adapter = IntentLLMAdapterFactory.from_env()
    adapter.complete([{"role": "user", "content": "你好"}], [])

    assert adapter.model == "qwen-intent-small"
    assert calls[0]["model"] == "dashscope/qwen-intent-small"
