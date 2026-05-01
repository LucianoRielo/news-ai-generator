from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


MARKET_COLUMNS = [
    "Date",
    "ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "return_1d",
    "return_5d",
    "volume_ratio",
    "direction",
    "RSI",
    "MACD",
    "SMA20",
    "SMA50",
]


def download_market(
    ticker: str,
    start_date: str,
    end_date: str,
    output_path: str | Path,
) -> pd.DataFrame:
    """Download market data and add financial features."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    market = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if market.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    market = _flatten_yfinance_columns(market).reset_index()

    featured = add_market_features(market, start_date=start_date, end_date=end_date)
    featured["ticker"] = ticker.upper()
    featured = featured[MARKET_COLUMNS]
    featured.to_csv(output, index=False)
    return featured


def download_market_for_tickers(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_path: str | Path,
) -> pd.DataFrame:
    frames = [
        download_market_frame(ticker=ticker, start_date=start_date, end_date=end_date)
        for ticker in tickers
    ]
    market = pd.concat(frames, ignore_index=True).sort_values(["ticker", "Date"]).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    market.to_csv(output, index=False)
    return market


def download_market_frame(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    market = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if market.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    market = _flatten_yfinance_columns(market).reset_index()
    featured = add_market_features(market, start_date=start_date, end_date=end_date)
    featured["ticker"] = ticker.upper()
    return featured[MARKET_COLUMNS]


def add_market_features(
    market: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Add returns, volume ratio, direction, RSI, MACD, and moving averages."""
    df = market.copy()
    df = _flatten_yfinance_columns(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if start_date is not None:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["Date"] <= pd.to_datetime(end_date)]

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in df.columns:
            raise ValueError(f"Missing market column: {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20, min_periods=20).mean()
    df["direction"] = (df["return_1d"] > 0).astype(int)
    df["RSI"] = _rsi(df["Close"], window=14)
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["SMA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["SMA50"] = df["Close"].rolling(50, min_periods=50).mean()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    columns = [column for column in MARKET_COLUMNS if column in df.columns]
    return df[columns].reset_index(drop=True)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        flattened = df.copy()
        flattened.columns = [column[0] for column in flattened.columns]
        return flattened
    return df
