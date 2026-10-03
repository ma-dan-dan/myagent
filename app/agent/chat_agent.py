from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.adapter import LLMAdapter, LLMResponse, LLMServiceUnavailable
from app.agent.tool_registry import ToolRegistry
from app.memory.models import TokenUsage
from app.schemas.chat import SearchSchemaInput, ToolEvent


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ChatAgent:
    def __init__(
        self,
        llm: LLMAdapter,
        tool_registry: ToolRegistry,
        max_model_calls: int = 2,
        max_tool_calls: int = 1,
    ) -> None:
        if max_model_calls < 1 or max_tool_calls < 1:
            raise ValueError("agent budgets must be positive")
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls

    def run(self, messages: list[dict[str, Any]]) -> AgentResult:
        working_messages = [dict(message) for message in messages]
        events: list[ToolEvent] = []
        usage = TokenUsage()

        for _ in range(self.max_model_calls):
            try:
                response = self.llm.complete(working_messages, self.tool_registry.specs())
                response = LLMResponse.model_validate(response)
                if response.usage is not None:
                    usage = usage.add(response.usage)
            except LLMServiceUnavailable:
                raise
            except Exception:
                return AgentResult(message="模型暂时不可用，请稍后重试。", tool_events=events)

            if response.kind == "message":
                if response.content:
                    return AgentResult(message=response.content, tool_events=events, usage=usage)
                return AgentResult(message="模型返回了空回复，请换一种问法。", tool_events=events, usage=usage)

            if len(events) >= self.max_tool_calls:
                return AgentResult(message="本轮工具调用已达到上限，请稍后换一种问法。", tool_events=events, usage=usage)
            if response.tool_name != "search_schema" or response.tool_input is None:
                return AgentResult(message="模型请求了不可用的工具，本轮无法继续。", tool_events=events, usage=usage)

            try:
                tool_input = SearchSchemaInput.model_validate(response.tool_input)
                tool_output = self.tool_registry.execute(response.tool_name, tool_input)
                event = ToolEvent(
                    tool_name="search_schema",
                    input=tool_input,
                    status=tool_output.status,
                    summary=tool_output.message,
                )
            except Exception:
                return AgentResult(message="Schema 工具暂时不可用，请稍后重试。", tool_events=events, usage=usage)

            events.append(event)
            call_id = response.call_id or f"local-call-{len(events)}"
            working_messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": response.content or None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": response.tool_name,
                                    "arguments": json.dumps(tool_input.model_dump(), ensure_ascii=False),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": response.tool_name,
                        "tool_call_id": call_id,
                        "content": json.dumps(tool_output.model_dump(), ensure_ascii=False),
                    },
                ]
            )

        return AgentResult(message="本轮模型调用已达到上限，请稍后换一种问法。", tool_events=events, usage=usage)
