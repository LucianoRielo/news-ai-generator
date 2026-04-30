from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.reporting.report import generate_report


ROOT = Path(__file__).resolve().parents[1]


def test_generate_report_writes_markdown() -> None:
    base_dir = ROOT / "outputs" / "reports" / "test_report_artifacts"
    base_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = base_dir / "predictions.jsonl"
    textual_path = base_dir / "textual.csv"
    semantic_path = base_dir / "semantic.csv"
    financial_path = base_dir / "financial.csv"
    report_path = base_dir / "REPORT.md"

    prediction = {
        "date_t": "2024-01-01",
        "date_t1": "2024-01-02",
        "prompt": "prompt",
        "real_news": "- SPY rises after strong market breadth",
        "generated_news": "- SPY rises as investors buy broad market ETFs",
    }
    predictions_path.write_text(json.dumps(prediction), encoding="utf-8")
    pd.DataFrame(
        [{"date_t": "2024-01-01", "date_t1": "2024-01-02", "rouge1": 0.5, "rouge2": 0.1, "rougeL": 0.4}]
    ).to_csv(textual_path, index=False)
    pd.DataFrame(
        [
            {
                "date_t": "2024-01-01",
                "date_t1": "2024-01-02",
                "real_label": "positive",
                "generated_label": "positive",
                "sentiment_match": True,
                "real_net_sentiment": 0.8,
                "generated_net_sentiment": 0.7,
                "kl_divergence": 0.05,
            }
        ]
    ).to_csv(semantic_path, index=False)
    pd.DataFrame(
        [
            {
                "date_t": "2024-01-01",
                "date_t1": "2024-01-02",
                "real_label": "positive",
                "generated_label": "positive",
                "real_signal": 1,
                "generated_signal": 1,
                "actual_direction": 1,
                "actual_return_1d": 0.01,
                "generated_direction_correct": True,
                "real_direction_correct": True,
                "abs_return_1d": 0.01,
                "volatility_bucket": "high",
            }
        ]
    ).to_csv(financial_path, index=False)

    with (ROOT / "config" / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["data"]["processed_dir"] = str(base_dir)
    config["evaluation"]["predictions_path"] = str(predictions_path)
    config["evaluation"]["textual_metrics_path"] = str(textual_path)
    config["evaluation"]["semantic_metrics_path"] = str(semantic_path)
    config["evaluation"]["financial_metrics_path"] = str(financial_path)
    config["evaluation"]["report_path"] = str(report_path)
    config["evaluation"]["reports_dir"] = str(base_dir)

    generate_report(config, output_path=report_path)

    report = report_path.read_text(encoding="utf-8")
    assert "# Reporte del experimento" in report
    assert "Resultados Textuales" in report
    assert "Resultados Semanticos" in report
    assert "Resultados Financieros" in report
    assert "SPY" in report

    for path in [predictions_path, textual_path, semantic_path, financial_path, report_path]:
        path.unlink()
    base_dir.rmdir()
