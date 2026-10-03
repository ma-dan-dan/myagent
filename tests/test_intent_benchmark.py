from app.evaluation.intent_benchmark import summarize_results


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


def test_fake_benchmark_returns_one_result_per_case(monkeypatch):
    from app.evaluation.intent_benchmark import run_benchmark
    from app.intent.fake_classifier import FakeIntentClassifier

    monkeypatch.setenv("INTENT_PROVIDER", "fake")
    report = run_benchmark(
        FakeIntentClassifier(),
        [
            {"id": "chat-1", "message": "你好", "expected_intent": "chat"},
            {"id": "sql-1", "message": "生成 SQL", "expected_intent": "nl2sql"},
        ],
    )

    assert report["sample_count"] == 2
    assert len(report["results"]) == 2
    assert report["results"][0]["actual"] == "chat"
