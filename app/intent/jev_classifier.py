from __future__ import annotations

import time
from typing import Any

from typesafe_sdk import Choice, TypeSafeClient

from app.intent.classifier import IntentClassificationError, InvalidIntentDecision
from app.intent.models import DataAction, IntentDecision, IntentName
from app.memory.models import TokenUsage


class JevIntentClassifier:
    PROVIDER = "jev"
    MODEL = "typesafe/jev-1.13"

    def __init__(self, api_key: str | None) -> None:
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise IntentClassificationError("缺少 Jev 配置：TYPESAFE_API_KEY。")

    def classify(self, message: str) -> IntentDecision:
        started = time.perf_counter()
        client = TypeSafeClient(api_key=self.api_key)
        try:
            response = client.system_one(
                state={"message": message},
                model=self.MODEL,
                questions={
                    "intent": Choice(
                        instructions="判断用户消息的一级意图。",
                        criteria={
                            "chat": "普通问答、项目讨论、解释或闲聊。",
                            "data_operation": "希望查询或操作业务数据。",
                            "nl2sql": "明确要求生成、改写、解释或展示 SQL。",
                        },
                    ),
                    "data_action": Choice(
                        instructions="如果涉及数据操作，判断用户希望执行的二级操作。",
                        criteria={
                            "read": "查询或读取数据。",
                            "create": "创建数据。",
                            "update": "修改数据。",
                            "delete": "删除数据。",
                            "unknown": "无法确定具体操作。",
                        },
                    ),
                },
            )
        except Exception as exc:
            raise IntentClassificationError("Jev 意图分类服务不可用。") from exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        try:
            intent_answer = response.answers["intent"]
            intent = IntentName(intent_answer.choice)
            confidence = float(intent_answer.confidence)
            data_action: DataAction | None = None
            if intent is IntentName.DATA_OPERATION:
                data_action = DataAction(response.answers["data_action"].choice)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise InvalidIntentDecision("Jev 返回了无效的意图分类结果。") from exc

        return IntentDecision(
            intent=intent,
            data_action=data_action,
            confidence=confidence,
            provider=self.PROVIDER,
            model=self.MODEL,
            latency_ms=(time.perf_counter() - started) * 1000,
            usage=TokenUsage(),
        )
