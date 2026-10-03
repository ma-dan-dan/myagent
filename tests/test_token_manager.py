import litellm
import pytest

from app.memory.models import ContextPolicy
from app.memory.token_manager import TokenManager


def test_token_manager_uses_litellm_token_counter(monkeypatch):
    calls = []

    def fake_counter(**kwargs):
        calls.append(kwargs)
        return 37

    monkeypatch.setattr(litellm, "token_counter", fake_counter)
    manager = TokenManager("openai", "gpt-4.1-mini")
    messages = [{"role": "user", "content": "hello"}]

    assert manager.estimate(messages) == 37
    assert calls == [{"model": "gpt-4.1-mini", "messages": messages}]
    assert manager.context_window == 128000
    assert manager.output_reserve == 4096


def test_token_manager_falls_back_to_utf8_estimate(monkeypatch):
    def fail_counter(**kwargs):
        raise RuntimeError("counter unavailable")

    monkeypatch.setattr(litellm, "token_counter", fail_counter)
    manager = TokenManager("unknown", "unknown", ContextPolicy(default_context_window=8192))

    assert manager.estimate([{"role": "user", "content": "你好"}]) == 2
    assert manager.context_window == 8192
    assert manager.output_reserve == 2048


def test_token_manager_exposes_complete_prompt_budget_formula(tmp_path):
    specs = tmp_path / "specs.json"
    specs.write_text(
        '{"temporary": {"large-model": {"context_window": 128000, "output_reserve": 4096}, '
        '"medium-model": {"context_window": 65536, "output_reserve": 4096}}}',
        encoding="utf-8",
    )
    manager = TokenManager(
        "temporary",
        "large-model",
        ContextPolicy(fixed_context_ratio=0.20),
        specs_path=specs,
    )
    medium = TokenManager("temporary", "medium-model", specs_path=specs)

    assert manager.hard_input_budget == 123904
    assert manager.soft_input_budget == 99123
    assert manager.fixed_context_budget == 19824
    assert medium.soft_input_budget == 49152
    assert medium.fixed_context_budget == 9830


def test_context_policy_rejects_invalid_budget_configuration():
    with pytest.raises(ValueError):
        ContextPolicy(fixed_context_ratio=0)
    with pytest.raises(ValueError):
        ContextPolicy(compact_ratio=0.8, fixed_context_ratio=0.8)
    with pytest.raises(ValueError):
        ContextPolicy(minimum_summary_tokens=127)
    with pytest.raises(ValueError):
        ContextPolicy(max_summary_calls_per_request=3)
    with pytest.raises(ValueError):
        ContextPolicy.model_validate({"long_term_" + "memory_limit": 3})
