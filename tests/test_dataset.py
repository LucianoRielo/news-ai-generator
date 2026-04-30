from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from src.data.build_dataset import REQUIRED_EXAMPLE_FIELDS, build_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_build_dataset_writes_temporal_splits() -> None:
    output_dir = ROOT / "data" / "processed" / "test_tmp"
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
    assert first["completion"]
    assert max(len(row["prompt"] + row["completion"]) for rows in splits.values() for row in rows) < 4096
    assert max(row["date_t1"] for row in splits["train"]) < min(row["date_t1"] for row in splits["val"])
    assert max(row["date_t1"] for row in splits["val"]) < min(row["date_t1"] for row in splits["test"])

    shutil.rmtree(output_dir)
