from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from src.data.build_dataset import REQUIRED_EXAMPLE_FIELDS, build_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_build_dataset_writes_temporal_splits() -> None:
    output_dir = ROOT / "outputs" / "reports" / "test_dataset_tmp"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    news = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "SPY", "headline": "Markets start year higher", "body": "", "source": "x"},
            {"date": "2024-01-02", "ticker": "SPY", "headline": "ETF inflows continue", "body": "Broad indexes rise.", "source": "x"},
            {"date": "2024-01-03", "ticker": "SPY", "headline": "Stocks pause", "body": "", "source": "x"},
            {"date": "2024-01-04", "ticker": "SPY", "headline": "Risk appetite improves", "body": "", "source": "x"},
            {"date": "2024-01-05", "ticker": "SPY", "headline": "Investors await payrolls", "body": "", "source": "x"},
            {"date": "2024-01-06", "ticker": "SPY", "headline": "Market narrative extends", "body": "", "source": "x"},
        ]
    )
    market = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "Open": [100, 101, 102, 103, 104, 105],
            "High": [101, 102, 103, 104, 105, 106],
            "Low": [99, 100, 101, 102, 103, 104],
            "Close": [100, 101, 102, 103, 104, 105],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500],
            "return_1d": [0.0, 0.01, 0.01, 0.01, 0.01, 0.01],
            "return_5d": [None, None, None, None, None, 0.05],
            "volume_ratio": [None, 1.1, 1.2, 1.3, 1.4, 1.5],
            "direction": [0, 1, 1, 1, 1, 1],
            "RSI": [None, 55, 56, 57, 58, 59],
            "MACD": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "SMA20": [None] * 6,
            "SMA50": [None] * 6,
        }
    )

    splits = build_dataset(
        news_df=news,
        market_df=market,
        ticker="SPY",
        k=3,
        split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
        output_dir=output_dir,
        max_news_per_day=2,
        max_text_chars=80,
        max_completion_news=2,
        include_body=False,
    )

    assert {split: len(rows) for split, rows in splits.items()} == {"train": 3, "val": 1, "test": 1}
    for split_name in ["train", "val", "test"]:
        path = output_dir / f"{split_name}.jsonl"
        assert path.exists()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows
        assert set(REQUIRED_EXAMPLE_FIELDS).issubset(rows[0])

    first = splits["train"][0]
    assert "[TICKER: SPY]" in first["prompt"]
    assert "[PRICE_CHANGE:" in first["prompt"]
    assert "[PREVIOUS NEWS]" in first["prompt"]
    assert "[NEXT DAY OUTLOOK]" in first["prompt"]
    assert first["prompt"].endswith("Sentiment:")
    assert first["completion"].startswith(" ")
    assert "Direction:" in first["completion"]
    assert "News:" in first["completion"]
    assert first["target_sentiment_label"] in {"negative", "neutral", "positive"}
    assert first["target_direction_label"] in {"down", "flat", "up"}
    assert first["completion"]
    assert max(len(row["prompt"] + row["completion"]) for rows in splits.values() for row in rows) < 4096
    assert max(row["date_t1"] for row in splits["train"]) < min(row["date_t1"] for row in splits["val"])
    assert max(row["date_t1"] for row in splits["val"]) < min(row["date_t1"] for row in splits["test"])

    shutil.rmtree(output_dir)


def test_build_dataset_keeps_tickers_separate() -> None:
    output_dir = ROOT / "outputs" / "reports" / "test_dataset_tmp_multi"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    rows = []
    market_rows = []
    for ticker in ["SPY", "QQQ"]:
        for day in range(1, 5):
            date = f"2024-01-0{day}"
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "headline": f"{ticker} headline {day}",
                    "body": "",
                    "source": "x",
                }
            )
            market_rows.append(
                {
                    "Date": date,
                    "ticker": ticker,
                    "Open": 100,
                    "High": 101,
                    "Low": 99,
                    "Close": 100 + day,
                    "Volume": 1000,
                    "return_1d": 0.01,
                    "return_5d": 0.02,
                    "volume_ratio": 1.0,
                    "direction": 1,
                    "RSI": 55,
                    "MACD": 0.1,
                    "SMA20": 100,
                    "SMA50": 100,
                }
            )

    splits = build_dataset(
        news_df=pd.DataFrame(rows),
        market_df=pd.DataFrame(market_rows),
        ticker=["SPY", "QQQ"],
        k=2,
        split_ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        output_dir=output_dir,
        max_news_per_day=1,
        max_text_chars=80,
        max_completion_news=1,
        include_body=False,
    )

    examples = [example for split in splits.values() for example in split]
    assert {example["ticker"] for example in examples} == {"SPY", "QQQ"}
    assert all(f"[TICKER: {example['ticker']}]" in example["prompt"] for example in examples)
    assert all(example["target_direction_label"] == "up" for example in examples)
    assert not any("QQQ headline" in example["prompt"] for example in examples if example["ticker"] == "SPY")
    assert not any("SPY headline" in example["prompt"] for example in examples if example["ticker"] == "QQQ")

    shutil.rmtree(output_dir)
