"""Eval harness tests — TDD: these must fail before implementation, pass after."""

from server.ml.eval.prompt_bench import BENCH, evaluate_prompt_bench
from server.ml.eval.metrics import genre_model_metrics


def test_bench_has_coverage():
    # At least 30 prompt paraphrases covering all intents
    assert len(BENCH) >= 30
    intents = {item["intent"] for item in BENCH}
    assert "trim" in intents
    assert "remove" in intents
    assert "isolate" in intents
    assert "mood" in intents
    assert "speed" in intents


def test_evaluate_prompt_bench_reports_accuracy():
    result = evaluate_prompt_bench()
    assert "accuracy" in result
    assert "total" in result
    assert result["total"] == len(BENCH)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert "per_intent" in result
    assert "failures" in result


def test_genre_model_metrics_shape():
    metrics = genre_model_metrics()
    # either modelUnavailable or real metrics
    assert "available" in metrics
    if metrics["available"]:
        assert "classes" in metrics
        assert "feature_size" in metrics
        assert "cv_note" in metrics
