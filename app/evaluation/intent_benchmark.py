from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from app.agent.llm_factory import LLMAdapterFactory
from app.intent.factory import IntentClassifierFactory
from app.intent.fake_classifier import FakeIntentClassifier
from app.intent.models import IntentDecision


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(result["latency_ms"]) for result in results]
    correct = sum(result["expected"] == result["actual"] for result in results)
    return {
        "accuracy": correct / len(results) if results else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "total_input_tokens": sum(int(result["input_tokens"]) for result in results),
        "total_output_tokens": sum(int(result["output_tokens"]) for result in results),
    }


def run_benchmark(classifier, cases: list[dict[str, Any]]) -> dict[str, Any]:
    fixture_mode = isinstance(classifier, FakeIntentClassifier)
    results: list[dict[str, Any]] = []
    provider = "unknown"
    model = "unknown"
    for case in cases:
        if fixture_mode:
            decision = IntentDecision(
                intent=case["expected_intent"],
                confidence=1.0,
                provider=FakeIntentClassifier.provider,
                model=FakeIntentClassifier.model,
                latency_ms=0,
            )
            elapsed_ms = 0.0
        else:
            started = time.perf_counter()
            decision = classifier.classify(case["message"])
            elapsed_ms = (time.perf_counter() - started) * 1000
        provider = decision.provider
        model = decision.model
        results.append(
            {
                "id": case["id"],
                "expected": case["expected_intent"],
                "actual": decision.intent.value,
                "latency_ms": elapsed_ms,
                "input_tokens": decision.usage.input_tokens,
                "output_tokens": decision.usage.output_tokens,
            }
        )
    return {
        "provider": provider,
        "model": model,
        "sample_count": len(results),
        "benchmark_mode": "fixture" if fixture_mode else "model",
        "comparable_to_real_models": not fixture_mode,
        **summarize_results(results),
        "results": results,
    }


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    cases_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "data" / "intent_eval_cases.json"
    return json.loads(cases_path.read_text(encoding="utf-8"))


def main() -> None:
    llm = LLMAdapterFactory.from_env()
    classifier = IntentClassifierFactory.from_env(llm)
    report = run_benchmark(classifier, load_cases())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
