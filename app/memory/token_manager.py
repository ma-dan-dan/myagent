from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import litellm

from app.memory.models import ContextPolicy


logger = logging.getLogger(__name__)


class TokenManager:
    def __init__(
        self,
        provider: str,
        model: str,
        policy: ContextPolicy | None = None,
        litellm_model_name: str | None = None,
        specs_path: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.policy = policy or ContextPolicy()
        self.litellm_model_name = litellm_model_name or model
        path = Path(specs_path) if specs_path else Path(__file__).resolve().parents[2] / "data" / "model_context_specs.json"
        self._specs = self._load_specs(path)
        spec = self._specs.get(provider, {}).get(model, {})
        self.context_window = int(spec.get("context_window", self.policy.default_context_window))
        self.output_reserve = int(spec.get("output_reserve", self.policy.default_output_reserve))

    @staticmethod
    def _load_specs(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def estimate(self, messages: list[dict[str, Any]]) -> int:
        try:
            value = litellm.token_counter(model=self.litellm_model_name, messages=messages)
            return max(0, int(value))
        except Exception as exc:
            logger.warning("LiteLLM token estimate failed; using UTF-8 fallback: %s", exc)
            byte_count = sum(len(str(message.get("content", "")).encode("utf-8")) for message in messages)
            return max(1, math.ceil(byte_count / 4))

    @property
    def hard_input_budget(self) -> int:
        return max(0, self.context_window - self.output_reserve)

    @property
    def soft_input_budget(self) -> int:
        return math.floor(self.hard_input_budget * self.policy.compact_ratio)

    @property
    def fixed_context_budget(self) -> int:
        return math.floor(self.soft_input_budget * self.policy.fixed_context_ratio)
