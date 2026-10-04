from app.evaluation.intent_benchmark import load_cases, run_benchmark, summarize_results
from app.intent.fake_classifier import FakeIntentClassifier
from app.intent.models import IntentDecision, IntentName


def test_benchmark_calculates_accuracy_and_latency_percentiles():
    report = summarize_results(
        [
            {"expected": "chat", "actual": "chat", "latency_ms": 10, "input_tokens": 1, "output_tokens": 0},
            {"expected": "nl2sql", "actual": "chat", "latency_ms": 30, "input_tokens": 2, "output_tokens": 0},
        ]
    )

    assert report["accuracy"] == 0.5
    assert report["avg_latency_ms"] == 20.0
    assert report["p95_latency_ms"] == 30.0
    assert report["total_input_tokens"] == 3


def test_fake_benchmark_is_marked_as_fixture_not_model_comparison():
    report = run_benchmark(FakeIntentClassifier(), load_cases())

    assert report["sample_count"] == 15
    assert report["benchmark_mode"] == "fixture"
    assert report["comparable_to_real_models"] is False
    assert report["accuracy"] == 1.0


def test_model_benchmark_is_marked_comparable():
    class StubClassifier:
        def classify(self, message):
            return IntentDecision(
                intent=IntentName.CHAT,
                confidence=0.9,
                provider="llm",
                model="test-model",
                latency_ms=1,
            )

    report = run_benchmark(
        StubClassifier(),
        [{"id": "chat-1", "message": "你好", "expected_intent": "chat"}],
    )

    assert report["benchmark_mode"] == "model"
    assert report["comparable_to_real_models"] is True
