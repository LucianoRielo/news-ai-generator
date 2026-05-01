from __future__ import annotations

import re
from typing import Any


def get_config_tickers(data_config: dict[str, Any]) -> list[str]:
    """Return configured tickers, preserving compatibility with single-ticker configs."""
    values = data_config.get("tickers", [data_config["ticker"]])
    if isinstance(values, str):
        values = [values]
    tickers = [str(ticker).strip().upper() for ticker in values if str(ticker).strip()]
    if not tickers:
        raise ValueError("At least one ticker must be configured")
    return tickers


def prediction_ticker(prediction: dict[str, Any]) -> str:
    """Read ticker from prediction metadata or infer it from the prompt header."""
    ticker = str(prediction.get("ticker", "")).strip().upper()
    if ticker:
        return ticker

    match = re.search(r"\[TICKER:\s*([A-Za-z0-9._-]+)\]", str(prediction.get("prompt", "")))
    return match.group(1).upper() if match else ""
