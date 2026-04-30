from __future__ import annotations

import pandas as pd

from src.data.download_news import REQUIRED_COLUMNS, normalize_news
from src.data.download_market import add_market_features


def test_news_normalization_filters_and_orders_ticker() -> None:
    raw = pd.DataFrame(
        {
            "published_at": ["2024-01-03 09:00:00", "2024-01-01", "2024-01-02"],
            "symbol": ["MSFT", "AAPL", "AAPL,MSFT"],
            "title": ["Microsoft headline", "Apple headline", "Apple partner headline"],
            "content": ["body 1", "body 2", None],
            "publisher": ["wire", "blog", "wire"],
        }
    )

    normalized = normalize_news(raw, ticker="AAPL")

    assert list(normalized.columns) == REQUIRED_COLUMNS
    assert len(normalized) == 2
    assert normalized["date"].tolist() == ["2024-01-01", "2024-01-02"]
    assert normalized["ticker"].str.contains("AAPL").all()
    assert normalized["headline"].str.len().gt(0).all()


def test_fnspid_schema_and_date_range_are_supported() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2017-12-31 12:00:00 UTC", "2018-01-02 09:30:00 UTC", "2024-01-01 00:00:00 UTC"],
            "Article_title": ["Too old", "SPY rallies after Fed minutes", "Too new"],
            "Stock_symbol": ["SPY", "SPY", "SPY"],
            "Publisher": ["Benzinga", "Reuters", "Benzinga"],
            "Article": ["old body", "body", "new body"],
        }
    )

    normalized = normalize_news(raw, ticker="SPY", start_date="2018-01-01", end_date="2023-12-31")

    assert len(normalized) == 1
    assert normalized.loc[0, "date"] == "2018-01-02"
    assert normalized.loc[0, "ticker"] == "SPY"
    assert normalized.loc[0, "headline"] == "SPY rallies after Fed minutes"
    assert normalized.loc[0, "source"] == "Reuters"


def test_news_normalization_deduplicates_parquet_overlap() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-01 00:00:00 UTC", "2020-01-01 00:00:00 UTC"],
            "Article_title": ["Same headline", "Same headline"],
            "Stock_symbol": ["SPY", "SPY"],
            "Publisher": ["Reuters", "Reuters"],
            "Article": ["same body", "same body"],
        }
    )

    normalized = normalize_news(raw, ticker="SPY")

    assert len(normalized) == 1


def test_market_features_are_added() -> None:
    dates = pd.date_range("2024-01-01", periods=70, freq="B")
    raw = pd.DataFrame(
        {
            "Date": dates,
            "Open": range(100, 170),
            "High": range(101, 171),
            "Low": range(99, 169),
            "Close": range(100, 170),
            "Volume": range(1_000_000, 1_000_070),
        }
    )

    featured = add_market_features(raw)

    assert {"return_1d", "return_5d", "volume_ratio", "direction", "RSI", "MACD", "SMA20", "SMA50"}.issubset(
        featured.columns
    )
    assert featured["direction"].isin([0, 1]).all()
    assert featured.loc[50:, ["return_1d", "return_5d", "volume_ratio", "RSI", "MACD", "SMA20", "SMA50"]].notna().all().all()
