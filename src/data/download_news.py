from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import list_repo_files


logger = logging.getLogger("news_ai_generator")

REQUIRED_COLUMNS = ["date", "ticker", "headline", "body", "source"]
FNSPID_COLUMNS = ["Date", "Article_title", "Stock_symbol", "Publisher", "Article"]

COLUMN_ALIASES = {
    "date": ["date", "datetime", "published_at", "publishedat", "time", "timestamp"],
    "ticker": ["ticker", "symbol", "stock", "stock_symbol", "stock_symbols", "symbols"],
    "headline": ["headline", "title", "news_title", "article_title", "summary", "news"],
    "body": ["body", "content", "article", "text", "description"],
    "source": ["source", "publisher", "provider", "site"],
}


def download_news(
    dataset_name: str,
    output_path: str | Path,
    ticker: str,
    dataset_source: str = "huggingface",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download FNSPID news for one ticker and save the normalized CSV."""
    if dataset_source not in {"huggingface", "hf"} or not dataset_name.lower().endswith("fnspid"):
        raise ValueError("This pipeline currently supports the FNSPID Hugging Face dataset only.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_df = _load_fnspid_news(dataset_name, ticker=ticker, start_date=start_date, end_date=end_date)
    normalized = normalize_news(raw_df, ticker=ticker, start_date=start_date, end_date=end_date)
    normalized.to_csv(output, index=False)
    return normalized


def download_news_for_tickers(
    dataset_name: str,
    output_path: str | Path,
    tickers: list[str],
    dataset_source: str = "huggingface",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download FNSPID news for multiple tickers and save one normalized CSV."""
    if dataset_source not in {"huggingface", "hf"} or not dataset_name.lower().endswith("fnspid"):
        raise ValueError("This pipeline currently supports the FNSPID Hugging Face dataset only.")

    frames = []
    for ticker in tickers:
        raw_df = _load_fnspid_news(dataset_name, ticker=ticker, start_date=start_date, end_date=end_date)
        frames.append(normalize_news(raw_df, ticker=ticker, start_date=start_date, end_date=end_date))

    normalized = pd.concat(frames, ignore_index=True)
    normalized = normalized.drop_duplicates(subset=["date", "ticker", "headline", "body", "source"])
    normalized = normalized.sort_values(["ticker", "date"]).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    return normalized


def normalize_news(
    df: pd.DataFrame,
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Normalize FNSPID-style financial news columns into the project schema."""
    source = df.copy()
    source.columns = [_clean_column_name(column) for column in source.columns]

    normalized = pd.DataFrame()
    for target_column, aliases in COLUMN_ALIASES.items():
        match = _first_existing_column(source.columns, aliases)
        if match is not None:
            normalized[target_column] = source[match]
        elif target_column in {"body", "source"}:
            normalized[target_column] = ""
        else:
            raise ValueError(f"Could not infer required column: {target_column}")

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce", utc=True, format="mixed")
    normalized["ticker"] = normalized["ticker"].fillna("").astype(str).str.upper()
    normalized["headline"] = normalized["headline"].fillna("").astype(str).str.strip()
    normalized["body"] = normalized["body"].fillna("").astype(str).str.strip()
    normalized["source"] = normalized["source"].fillna("").astype(str).str.strip()

    normalized = normalized.dropna(subset=["date"])
    normalized = _filter_news(normalized, ticker=ticker, start_date=start_date, end_date=end_date)
    normalized = normalized[normalized["headline"] != ""]
    normalized = normalized.drop_duplicates(subset=["date", "ticker", "headline", "body", "source"])
    normalized = normalized.sort_values("date").reset_index(drop=True)
    normalized["date"] = normalized["date"].dt.strftime("%Y-%m-%d")

    return normalized[REQUIRED_COLUMNS]


def _load_fnspid_news(
    dataset_name: str,
    ticker: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    parquet_files = _list_fnspid_stock_news_files(dataset_name)
    filters = _fnspid_filters(ticker=ticker, start_date=start_date, end_date=end_date)

    logger.info(
        "Loading FNSPID news for ticker=%s date_range=%s..%s from %s parquet files",
        ticker,
        start_date,
        end_date,
        len(parquet_files),
    )

    frames = []
    for file_path in parquet_files:
        parquet_url = f"hf://datasets/{dataset_name}/{file_path}"
        logger.info("Reading %s", parquet_url)
        frame = pd.read_parquet(parquet_url, columns=FNSPID_COLUMNS, filters=filters)
        logger.info("Loaded %s rows from %s", len(frame), file_path)
        if not frame.empty:
            frames.append(frame)

    if frames:
        return pd.concat(frames, ignore_index=True)

    available_range = _available_date_range(dataset_name, parquet_files, ticker)
    detail = (
        f" Available {ticker.upper()} range appears to be {available_range[0]}..{available_range[1]}."
        if available_range is not None
        else ""
    )
    raise ValueError(f"No FNSPID rows matched ticker/date filters.{detail}")


def _list_fnspid_stock_news_files(dataset_name: str) -> list[str]:
    parquet_files = [
        file_path
        for file_path in list_repo_files(dataset_name, repo_type="dataset")
        if file_path.startswith("Stock_news/") and file_path.endswith(".parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError("No Stock_news parquet files found in FNSPID.")
    return sorted(parquet_files)


def _fnspid_filters(
    ticker: str,
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, str, object]]:
    filters: list[tuple[str, str, object]] = [("Stock_symbol", "==", ticker.upper())]
    if start_date is not None:
        filters.append(("Date", ">=", f"{start_date} 00:00:00 UTC"))
    if end_date is not None:
        end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        filters.append(("Date", "<", f"{end_exclusive.strftime('%Y-%m-%d')} 00:00:00 UTC"))
    return filters


def _available_date_range(
    dataset_name: str,
    parquet_files: list[str],
    ticker: str,
) -> tuple[str, str] | None:
    dates = []
    for file_path in parquet_files:
        parquet_url = f"hf://datasets/{dataset_name}/{file_path}"
        frame = pd.read_parquet(
            parquet_url,
            columns=["Date", "Stock_symbol"],
            filters=[("Stock_symbol", "==", ticker.upper())],
        )
        if not frame.empty:
            dates.append(frame["Date"])

    if not dates:
        return None

    all_dates = pd.concat(dates, ignore_index=True)
    return str(all_dates.min()), str(all_dates.max())


def _filter_news(
    df: pd.DataFrame,
    ticker: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    filtered = df[df["ticker"].apply(lambda value: _matches_ticker(value, ticker.upper()))]
    if start_date is not None:
        filtered = filtered[filtered["date"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date is not None:
        end_timestamp = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        filtered = filtered[filtered["date"] < end_timestamp]
    return filtered


def _clean_column_name(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")


def _first_existing_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    available = set(columns)
    return next((alias for alias in aliases if alias in available), None)


def _matches_ticker(value: str, ticker: str) -> bool:
    tickers = {part.strip().upper() for part in str(value).replace(";", ",").replace("|", ",").split(",")}
    return ticker in tickers
