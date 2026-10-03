from __future__ import annotations

import json
from typing import Any, Protocol, Literal
from urllib.parse import urlparse

import litellm
from pydantic import BaseModel, ConfigDict

from app.memory.models import TokenUsage


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any]


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["message", "tool_call"]
    content: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    call_id: str | None = None
    usage: TokenUsage | None = None

    @classmethod
    def message(cls, content: str, usage: TokenUsage | None = None) -> "LLMResponse":
        return cls(kind="message", content=content, usage=usage)

    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        tool_input: dict[str, Any],
        call_id: str | None = None,
        usage: TokenUsage | None = None,
    ) -> "LLMResponse":
        return cls(kind="tool_call", tool_name=tool_name, tool_input=tool_input, call_id=call_id, usage=usage)


class LLMAdapter(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse:
        """Return either a final message or one validated tool request."""


class LLMServiceUnavailable(RuntimeError):
    """Raised when the configured model service cannot serve a request."""


class LLMConfigurationError(LLMServiceUnavailable):
    """Raised when required model configuration is missing or invalid."""


class LiteLLMAdapter:
    """Shared non-streaming LiteLLM adapter for supported providers."""

    PROVIDER = ""
    MODEL_PREFIX = ""
    DEFAULT_BASE_URL: str | None = None
    API_KEY_ENV = ""
    MODEL_ENV = ""
    BASE_URL_ENV = ""

    def __init__(self, api_key: str | None, model: str | None, base_url: str | None = None) -> None:
        self._provider = self.PROVIDER
        self._api_key = (api_key or "").strip()
        self._model = (model or "").strip()
        configured_base_url = (base_url or "").strip()
        self._base_url = configured_base_url or self.DEFAULT_BASE_URL

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def litellm_model_name(self) -> str:
        if not self.model:
            return ""
        if "/" in self.model:
            return self.model
        if self.provider == "openai" and self.base_url:
            hostname = (urlparse(self.base_url).hostname or "").lower()
            if hostname != "api.openai.com":
                return f"openai/{self.model}"
        return f"{self.MODEL_PREFIX}{self.model}"

    def _ensure_configured(self) -> None:
        missing = []
        if not self.api_key:
            missing.append(self.API_KEY_ENV)
        if not self.model:
            missing.append(self.MODEL_ENV)
        if missing:
            raise LLMConfigurationError(f"缺少真实模型配置：{', '.join(missing)}。")

    @staticmethod
    def _tool_payload(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def _request_kwargs(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> dict[str, Any]:
        self._ensure_configured()
        request: dict[str, Any] = {
            "model": self.litellm_model_name,
            "messages": messages,
            "api_key": self.api_key,
        }
        if self.base_url:
            request["api_base"] = self.base_url
        if tools:
            request["tools"] = self._tool_payload(tools)
            request["tool_choice"] = "auto"
        return request

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _to_llm_response(self, response: Any) -> LLMResponse:
        usage = self._usage_from_response(response)
        choices = self._field(response, "choices", []) or []
        message = self._field(choices[0], "message") if choices else None
        if message is None:
            raise LLMServiceUnavailable("真实模型返回为空。")

        tool_calls = self._field(message, "tool_calls", []) or []
        if tool_calls:
            call = tool_calls[0]
            function = self._field(call, "function")
            name = self._field(function, "name") if function is not None else None
            raw_arguments = self._field(function, "arguments", "") if function is not None else ""
            if not name:
                raise LLMServiceUnavailable("真实模型返回了无效的 Tool 名称。")
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise LLMServiceUnavailable("真实模型返回了无效的 Tool 参数。") from exc
            if not isinstance(arguments, dict):
                raise LLMServiceUnavailable("真实模型 Tool 参数必须是对象。")
            return LLMResponse.tool_call(name, arguments, call_id=self._field(call, "id"), usage=usage)

        content = self._field(message, "content")
        if not content or not str(content).strip():
            raise LLMServiceUnavailable("真实模型返回了空消息。")
        return LLMResponse.message(str(content), usage=usage)

    def _usage_from_response(self, response: Any) -> TokenUsage | None:
        raw_usage = self._field(response, "usage")
        if raw_usage is None:
            return None
        input_tokens = self._field(raw_usage, "prompt_tokens")
        if input_tokens is None:
            input_tokens = self._field(raw_usage, "input_tokens", 0)
        output_tokens = self._field(raw_usage, "completion_tokens")
        if output_tokens is None:
            output_tokens = self._field(raw_usage, "output_tokens", 0)
        total_tokens = self._field(raw_usage, "total_tokens")
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        total_tokens = int(total_tokens) if total_tokens is not None else input_tokens + output_tokens
        return TokenUsage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse:
        try:
            response = litellm.completion(**self._request_kwargs(messages, tools))
            return self._to_llm_response(response)
        except LLMServiceUnavailable:
            raise
        except Exception as exc:
            raise LLMServiceUnavailable("真实模型服务调用失败。") from exc
