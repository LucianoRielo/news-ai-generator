from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_EXAMPLE_FIELDS = ["prompt", "completion", "date_t", "date_t1"]
MARKET_FEATURES = ["return_1d", "volume_ratio", "RSI"]


def build_dataset(
    news_df: pd.DataFrame,
    market_df: pd.DataFrame,
    ticker: str,
    k: int,
    split_ratios: dict[str, float],
    output_dir: str | Path,
    max_news_per_day: int = 5,
    max_text_chars: int = 360,
    max_completion_news: int = 5,
    include_body: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """Build temporal prompt/completion splits for financial narrative modeling."""
    news_by_day = _group_news_by_day(
        news_df,
        max_text_chars=max_text_chars,
        include_body=include_body,
    )
    market_by_day = _prepare_market(market_df)

    examples = []
    available_dates = sorted(news_by_day)
    for date_t in available_dates:
        date_t1 = _next_calendar_day(date_t)
        if date_t1 not in news_by_day or date_t not in market_by_day:
            continue

        context_dates = _context_dates(date_t, news_by_day.keys(), k)
        if not context_dates:
            continue

        prompt = _format_prompt(
            ticker=ticker,
            date_t=date_t,
            market_row=market_by_day[date_t],
            context_dates=context_dates,
            news_by_day=news_by_day,
            max_news_per_day=max_news_per_day,
        )
        completion = _format_completion(news_by_day[date_t1], max_completion_news=max_completion_news)
        if not completion:
            continue

        examples.append(
            {
                "prompt": prompt,
                "completion": completion,
                "date_t": date_t,
                "date_t1": date_t1,
            }
        )

    splits = _temporal_split(examples, split_ratios)
    _write_splits(splits, Path(output_dir))
    return splits


def load_raw_data(news_path: str | Path, market_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(news_path), pd.read_csv(market_path)


def _group_news_by_day(
    news_df: pd.DataFrame,
    max_text_chars: int,
    include_body: bool,
) -> dict[str, list[str]]:
    df = news_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])

    grouped: dict[str, list[str]] = {}
    for row in df.sort_values("date").itertuples(index=False):
        title = _clean_text(getattr(row, "headline", ""))
        body = _clean_text(getattr(row, "body", ""))
        text = title
        if include_body and body and body.lower() != "nan":
            text = f"{title} - {body}"
        text = _truncate(text, max_text_chars)
        if text:
            grouped.setdefault(row.date, []).append(text)
    return grouped


def _prepare_market(market_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    df = market_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["Date"])
    return {row["Date"]: row.to_dict() for _, row in df.iterrows()}


def _format_prompt(
    ticker: str,
    date_t: str,
    market_row: dict[str, Any],
    context_dates: list[str],
    news_by_day: dict[str, list[str]],
    max_news_per_day: int,
) -> str:
    price_change = _format_percent(market_row.get("return_1d"))
    volume_ratio = _format_float(market_row.get("volume_ratio"))
    rsi = _format_float(market_row.get("RSI"))

    lines = [
        f"[TICKER: {ticker}]",
        f"[DATE: {date_t}]",
        f"[PRICE_CHANGE: {price_change}]",
        f"[VOLUME_RATIO: {volume_ratio}]",
        f"[RSI: {rsi}]",
        "",
        "[PREVIOUS NEWS]",
    ]

    for context_date in context_dates:
        for item in news_by_day[context_date][:max_news_per_day]:
            lines.append(f"- {context_date}: {item}")

    lines.extend(["", "[NEXT DAY NEWS]"])
    return "\n".join(lines)


def _format_completion(news_items: list[str], max_completion_news: int) -> str:
    return "\n".join(f"- {item}" for item in news_items[:max_completion_news]).strip()


def _context_dates(date_t: str, dates: Any, k: int) -> list[str]:
    available = set(dates)
    current = pd.Timestamp(date_t)
    selected = []
    for offset in range(k - 1, -1, -1):
        candidate = (current - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        if candidate in available:
            selected.append(candidate)
    return selected


def _next_calendar_day(date_t: str) -> str:
    return (pd.Timestamp(date_t) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _temporal_split(
    examples: list[dict[str, str]],
    split_ratios: dict[str, float],
) -> dict[str, list[dict[str, str]]]:
    if not examples:
        return {"train": [], "val": [], "test": []}

    ordered = sorted(examples, key=lambda example: example["date_t1"])
    total = len(ordered)
    train_end = int(total * split_ratios["train"])
    val_end = train_end + int(total * split_ratios["val"])

    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def _write_splits(splits: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, examples in splits.items():
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as file:
            for example in examples:
                file.write(json.dumps(example, ensure_ascii=False) + "\n")


def _clean_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = text.replace("\n", " ")
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_percent(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "NA"
    return f"{number * 100:.2f}%"


def _format_float(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "NA"
    return f"{number:.2f}"
