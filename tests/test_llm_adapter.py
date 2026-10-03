from types import SimpleNamespace

import litellm
import pytest

from app.agent.adapter import LLMConfigurationError, LLMResponse, ToolSpec
from app.agent.llm_factory import LLMAdapterConfig, LLMAdapterFactory
from app.agent.llm_providers import DeepSeekLLMAdapter, OpenAILLMAdapter, QwenLLMAdapter


def make_response(content=None, tool_calls=None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=usage,
    )


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "expected_model", "expected_base_url"),
    [
        ("openai", "gpt-4.1-mini", None, "gpt-4.1-mini", None),
        ("deepseek", "deepseek-chat", None, "deepseek/deepseek-chat", "https://api.deepseek.com"),
        (
            "qwen",
            "qwen-plus",
            None,
            "dashscope/qwen-plus",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    ],
)
def test_factory_maps_provider_model_and_default_base_url(
    provider, model, base_url, expected_model, expected_base_url, monkeypatch
):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return make_response(content="模型回复")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    adapter = LLMAdapterFactory.create(LLMAdapterConfig(provider, "key", model, base_url))

    result = adapter.complete([{"role": "user", "content": "你好"}], [])

    assert result == LLMResponse.message("模型回复")
    assert calls[0]["model"] == expected_model
    assert calls[0]["api_key"] == "key"
    assert calls[0].get("api_base") == expected_base_url
    if expected_base_url is None:
        assert "api_base" not in calls[0]


def test_openai_custom_base_url_uses_openai_prefix(monkeypatch):
    calls = []
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: calls.append(kwargs) or make_response(content="代理回复"),
    )
    adapter = LLMAdapterFactory.create(
        LLMAdapterConfig("openai", "key", "gpt-4.1-mini", "https://proxy.example/v1")
    )

    adapter.complete([{"role": "user", "content": "你好"}], [])

    assert calls[0]["model"] == "openai/gpt-4.1-mini"
    assert calls[0]["api_base"] == "https://proxy.example/v1"


def test_factory_creates_three_thin_provider_adapters():
    assert isinstance(LLMAdapterFactory.create(LLMAdapterConfig("openai", "k", "m")), OpenAILLMAdapter)
    assert isinstance(LLMAdapterFactory.create(LLMAdapterConfig("deepseek", "k", "m")), DeepSeekLLMAdapter)
    assert isinstance(LLMAdapterFactory.create(LLMAdapterConfig("qwen", "k", "m")), QwenLLMAdapter)


def test_factory_rejects_unknown_provider():
    with pytest.raises(LLMConfigurationError, match="openai, deepseek, qwen"):
        LLMAdapterFactory.create(LLMAdapterConfig("unknown", "k", "m"))


def test_factory_from_env_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")

    with pytest.raises(LLMConfigurationError, match="openai, deepseek, qwen"):
        LLMAdapterFactory.from_env()


def test_factory_reads_provider_specific_missing_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    adapter = LLMAdapterFactory.from_env()

    with pytest.raises(LLMConfigurationError, match="DEEPSEEK_API_KEY"):
        adapter.complete([], [])


def test_factory_uses_provider_model_constant_without_model_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "key")
    monkeypatch.delenv(QwenLLMAdapter.MODEL_ENV, raising=False)

    adapter = LLMAdapterFactory.from_env()

    assert adapter.model == QwenLLMAdapter.MODEL_ENV


def test_litellm_response_maps_function_call_and_tool_schema(monkeypatch):
    function_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="search_schema", arguments='{"query":"产量","limit":5}'),
    )
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return make_response(tool_calls=[function_call])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    adapter = LLMAdapterFactory.create(LLMAdapterConfig("openai", "key", "gpt-4.1-mini"))
    tools = [ToolSpec(name="search_schema", description="search", input_schema={"type": "object"})]

    result = adapter.complete([{"role": "user", "content": "找产量表"}], tools)

    assert result.kind == "tool_call"
    assert result.call_id == "call-1"
    assert result.tool_input == {"query": "产量", "limit": 5}
    assert calls[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search_schema",
                "description": "search",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert calls[0]["tool_choice"] == "auto"


def test_litellm_response_rejects_empty_message(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: make_response(content=None))
    adapter = LLMAdapterFactory.create(LLMAdapterConfig("openai", "key", "gpt-4.1-mini"))

    with pytest.raises(Exception, match="空"):
        adapter.complete([], [])


def test_litellm_response_extracts_provider_usage(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: make_response(
            content="模型回复",
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17),
        ),
    )
    adapter = LLMAdapterFactory.create(LLMAdapterConfig("openai", "key", "gpt-4.1-mini"))

    result = adapter.complete([], [])

    assert result.usage is not None
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 17
