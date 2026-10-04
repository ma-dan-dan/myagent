from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from app.agent.adapter import LLMAdapter, LLMConfigurationError, LLMResponse, LLMServiceUnavailable
from app.intent.classifier import IntentClassificationError, InvalidIntentDecision
from app.intent.models import IntentDecision
from app.memory.models import TokenUsage


_SYSTEM_PROMPT = """你是意图分类器，只根据当前用户消息进行分类。
一级意图必须是：
- chat：普通问答、项目讨论、解释或闲聊。
- data_operation：用户希望查询或操作业务数据；使用 data_action 表示 read、create、update、delete 或 unknown。
- nl2sql：用户明确要求生成、改写、解释或展示 SQL。
只输出一个 JSON 对象，不要 Markdown，不要解释，格式必须为：
{"intent":"chat|data_operation|nl2sql","data_action":"read|create|update|delete|unknown|null","confidence":0.0}
当 intent 不是 data_operation 时，data_action 必须为 null。"""


class LLMIntentClassifier:
    def __init__(self, llm: LLMAdapter) -> None:
        self.llm = llm

    def classify(self, message: str) -> IntentDecision:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        started = time.perf_counter()
        try:
            response = LLMResponse.model_validate(self.llm.complete(messages, tools=[]))
        except LLMConfigurationError as exc:
            raise IntentClassificationError(str(exc)) from exc
        except LLMServiceUnavailable as exc:
            raise IntentClassificationError("普通 LLM 意图分类服务不可用。") from exc
        except Exception as exc:
            raise InvalidIntentDecision("普通 LLM 返回了无效的意图分类结果。") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if response.kind != "message" or not response.content:
            raise InvalidIntentDecision("普通 LLM 返回了无效的意图分类结果。")
        try:
            payload = json.loads(response.content)
            decision = IntentDecision(
                **payload,
                provider=str(getattr(self.llm, "provider", "llm") or "llm"),
                model=str(getattr(self.llm, "model", "unknown") or "unknown"),
                latency_ms=latency_ms,
                usage=response.usage or TokenUsage(),
            )
        except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            raise InvalidIntentDecision("普通 LLM 返回了无效的意图分类结果。") from exc
        return decision
