from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.financial import build_financial_metrics, label_to_signal, summarize_financial
from src.evaluation.semantic import build_semantic_metrics, plot_semantic_confusion_matrix, summarize_semantic
from src.evaluation.textual import evaluate_textual


ROOT = Path(__file__).resolve().parents[1]


def test_textual_metrics_are_in_range() -> None:
    predictions_path = ROOT / "outputs" / "reports" / "test_predictions_textual.jsonl"
    output_path = ROOT / "outputs" / "reports" / "test_textual_metrics.csv"
    rows = [
        {
            "date_t": "2024-01-01",
            "date_t1": "2024-01-02",
            "prompt": "prompt",
            "real_news": "SPY rallies as investors buy broad market ETFs",
            "generated_news": "SPY rallies as investors buy broad market ETFs",
        }
    ]
    predictions_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    metrics = evaluate_textual(predictions_path, output_path, compute_bertscore=False)

    assert output_path.exists()
    assert metrics.loc[0, "rouge1"] == 1.0
    assert metrics.loc[0, "rougeL"] == 1.0
    assert metrics[["rouge1", "rouge2", "rougeL"]].ge(0).all().all()
    assert metrics[["rouge1", "rouge2", "rougeL"]].le(1).all().all()

    predictions_path.unlink()
    output_path.unlink()


def test_semantic_metrics_from_probabilities() -> None:
    predictions = [
        {
            "date_t": "2024-01-01",
            "date_t1": "2024-01-02",
            "real_news": "SPY rises after strong earnings",
            "generated_news": "SPY climbs as investors turn optimistic",
        },
        {
            "date_t": "2024-01-02",
            "date_t1": "2024-01-03",
            "real_news": "SPY falls as risk appetite weakens",
            "generated_news": "SPY remains steady in mixed trading",
        },
    ]
    real_probs = [
        {"negative": 0.10, "neutral": 0.20, "positive": 0.70},
        {"negative": 0.80, "neutral": 0.10, "positive": 0.10},
    ]
    generated_probs = [
        {"negative": 0.15, "neutral": 0.15, "positive": 0.70},
        {"negative": 0.20, "neutral": 0.60, "positive": 0.20},
    ]

    metrics = build_semantic_metrics(predictions, real_probs, generated_probs)
    summary = summarize_semantic(metrics)

    assert len(metrics) == 2
    assert metrics.loc[0, "real_label"] == "positive"
    assert metrics.loc[1, "generated_label"] == "neutral"
    assert metrics["kl_divergence"].ge(0).all()
    assert 0 <= summary["sentiment_match_accuracy"] <= 1
    assert summary["sentiment_match_accuracy"] == 0.5
    assert -1 <= summary["net_sentiment_pearson"] <= 1


def test_semantic_confusion_matrix_plot_is_written() -> None:
    output_path = ROOT / "outputs" / "reports" / "test_semantic_confusion_matrix.png"
    metrics = pd.DataFrame(
        [
            {"real_label": "positive", "generated_label": "neutral"},
            {"real_label": "neutral", "generated_label": "neutral"},
            {"real_label": "negative", "generated_label": "positive"},
        ]
    )

    plot_semantic_confusion_matrix(metrics, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    output_path.unlink()


def test_financial_metrics_join_sentiment_with_market_direction() -> None:
    semantic = pd.DataFrame(
        [
            {
                "date_t": "2024-01-01",
                "date_t1": "2024-01-02",
                "real_label": "positive",
                "generated_label": "positive",
            },
            {
                "date_t": "2024-01-02",
                "date_t1": "2024-01-06",
                "real_label": "negative",
                "generated_label": "neutral",
            },
            {
                "date_t": "2024-01-03",
                "date_t1": "2024-01-04",
                "real_label": "negative",
                "generated_label": "negative",
            },
        ]
    )
    market = pd.DataFrame(
        [
            {"Date": "2024-01-02", "direction": 1, "return_1d": 0.01},
            {"Date": "2024-01-08", "direction": 0, "return_1d": -0.02},
            {"Date": "2024-01-04", "direction": 0, "return_1d": -0.03},
        ]
    )

    metrics = build_financial_metrics(semantic, market)
    summary = summarize_financial(metrics)

    assert len(metrics) == 3
    assert label_to_signal("positive") == 1
    assert label_to_signal("negative") == -1
    assert label_to_signal("neutral") == 0
    assert metrics.loc[0, "actual_direction"] == 1
    assert metrics.loc[1, "actual_direction"] == -1
    assert metrics.loc[1, "target_market_date"] == "2024-01-08"
    assert summary["generated_signal_coverage"] == 2 / 3
    assert summary["generated_directional_accuracy"] == 1.0
    assert 0 <= summary["real_news_directional_accuracy"] <= 1
