import json

import pytest

from app.agent.adapter import LLMResponse
from app.agent.chat_agent import ChatAgent
from app.agent.tool_registry import ToolRegistry
from app.memory.models import TokenUsage
from app.storage.schema_catalog import SchemaCatalog
from app.tools.schema_search import SchemaSearchTool


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0)


def build_agent(catalog_path, fake, **limits):
    registry = ToolRegistry([SchemaSearchTool(SchemaCatalog(catalog_path))])
    return ChatAgent(fake, registry, **limits)


def test_agent_returns_direct_model_reply_without_tools(catalog_path):
    fake = FakeLLM([LLMResponse.message("你好，我可以帮你了解可用的元数据。")])
    agent = build_agent(catalog_path, fake)

    result = agent.run([{"role": "user", "content": "你好"}])

    assert result.message == "你好，我可以帮你了解可用的元数据。"
    assert result.tool_events == []
    assert len(fake.calls) == 1


def test_agent_executes_schema_tool_then_uses_result_for_final_reply(catalog_path):
    fake = FakeLLM(
        [
            LLMResponse.tool_call("search_schema", {"query": "产量", "limit": 5}),
            LLMResponse.message("找到 production_output，它包含日期、产线和产量字段。"),
        ]
    )
    agent = build_agent(catalog_path, fake)

    result = agent.run([{"role": "user", "content": "产量相关的表有哪些？"}])

    assert result.message.startswith("找到 production_output")
    assert len(result.tool_events) == 1
    assert result.tool_events[0].tool_name == "search_schema"
    assert result.tool_events[0].status == "ok"
    assert "production_output" in json.dumps(fake.calls[1]["messages"], ensure_ascii=False)


def test_agent_exposes_empty_schema_result_to_model(catalog_path):
    fake = FakeLLM(
        [
            LLMResponse.tool_call("search_schema", {"query": "不存在的业务", "limit": 5}),
            LLMResponse.message("当前元数据中没有匹配的表，请换一种业务名称。"),
        ]
    )
    result = build_agent(catalog_path, fake).run([{"role": "user", "content": "找不存在的业务"}])

    assert result.tool_events[0].status == "empty"
    assert "没有匹配" in result.message


def test_agent_can_finish_with_a_normal_follow_up_question(catalog_path):
    fake = FakeLLM([LLMResponse.message("请补充时间范围和对象，我再帮你判断。")])

    result = build_agent(catalog_path, fake).run([{"role": "user", "content": "查产量"}])

    assert result.message == "请补充时间范围和对象，我再帮你判断。"
    assert result.tool_events == []


def test_agent_stops_after_tool_budget(catalog_path):
    fake = FakeLLM(
        [
            LLMResponse.tool_call("search_schema", {"query": "产量", "limit": 5}),
            LLMResponse.tool_call("search_schema", {"query": "产量", "limit": 5}),
            LLMResponse.message("这条回复不应被调用。"),
        ]
    )

    result = build_agent(catalog_path, fake, max_model_calls=2, max_tool_calls=1).run(
        [{"role": "user", "content": "找产量表"}]
    )

    assert len(fake.calls) == 2
    assert len(result.tool_events) == 1
    assert "上限" in result.message


def test_agent_default_budget_keeps_a_model_call_for_final_answer(catalog_path):
    fake = FakeLLM(
        [
            LLMResponse.tool_call("search_schema", {"query": "产量", "limit": 5}),
            LLMResponse.message("根据 Schema，production_output 是相关表。"),
        ]
    )

    result = build_agent(catalog_path, fake).run([{"role": "user", "content": "找产量表"}])

    assert result.message == "根据 Schema，production_output 是相关表。"
    assert len(result.tool_events) == 1
    assert len(fake.calls) == 2


def test_agent_does_not_execute_a_second_tool_after_tool_budget(catalog_path):
    fake = FakeLLM(
        [
            LLMResponse.tool_call("search_schema", {"query": "产量", "limit": 5}),
            LLMResponse.tool_call("search_schema", {"query": "销售", "limit": 5}),
        ]
    )

    result = build_agent(catalog_path, fake).run([{"role": "user", "content": "找相关表"}])

    assert len(fake.calls) == 2
    assert len(result.tool_events) == 1
    assert "上限" in result.message


def test_agent_aggregates_actual_usage_for_each_model_call(catalog_path):
    fake = FakeLLM(
        [
            LLMResponse.tool_call(
                "search_schema",
                {"query": "产量", "limit": 5},
                usage=TokenUsage(requests=1, input_tokens=20, output_tokens=4, total_tokens=24),
            ),
            LLMResponse.message(
                "根据 Schema，production_output 是相关表。",
                usage=TokenUsage(requests=1, input_tokens=30, output_tokens=8, total_tokens=38),
            ),
        ]
    )

    result = build_agent(catalog_path, fake).run([{"role": "user", "content": "找产量表"}])

    assert result.usage.requests == 2
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 12
    assert result.usage.total_tokens == 62
