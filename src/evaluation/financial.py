from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.structured_output import direction_to_signal


FINANCIAL_COLUMNS = [
    "ticker",
    "date_t",
    "date_t1",
    "target_market_date",
    "real_label",
    "generated_label",
    "target_direction_label",
    "generated_direction_label",
    "real_signal",
    "generated_signal",
    "actual_direction",
    "actual_return_1d",
    "generated_direction_correct",
    "real_direction_correct",
    "abs_return_1d",
    "volatility_bucket",
]


def evaluate_financial(
    semantic_metrics_path: str | Path,
    market_path: str | Path,
    output_path: str | Path,
    confusion_matrix_path: str | Path | None = None,
) -> pd.DataFrame:
    semantic = pd.read_csv(semantic_metrics_path)
    market = pd.read_csv(market_path)
    metrics = build_financial_metrics(semantic, market)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)

    if confusion_matrix_path is not None:
        plot_confusion_matrix(metrics, confusion_matrix_path)

    return metrics


def build_financial_metrics(semantic: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    required_semantic = {"date_t", "date_t1", "real_label", "generated_label"}
    missing_semantic = required_semantic.difference(semantic.columns)
    if missing_semantic:
        raise ValueError(f"Missing semantic columns: {sorted(missing_semantic)}")

    required_market = {"Date", "direction", "return_1d"}
    missing_market = required_market.difference(market.columns)
    if missing_market:
        raise ValueError(f"Missing market columns: {sorted(missing_market)}")

    semantic_df = semantic.copy()
    market_df = market.copy()
    semantic_df["prediction_order"] = range(len(semantic_df))
    semantic_df["date_t1"] = pd.to_datetime(semantic_df["date_t1"]).dt.strftime("%Y-%m-%d")
    market_df["Date"] = pd.to_datetime(market_df["Date"]).dt.strftime("%Y-%m-%d")
    if "ticker" not in semantic_df.columns:
        semantic_df["ticker"] = ""
    if "ticker" not in market_df.columns:
        market_df["ticker"] = semantic_df["ticker"].iloc[0] if not semantic_df.empty else ""
    semantic_df["ticker"] = semantic_df["ticker"].fillna("").astype(str).str.upper()
    market_df["ticker"] = market_df["ticker"].fillna("").astype(str).str.upper()
    market_df["return_1d"] = pd.to_numeric(market_df["return_1d"], errors="coerce")
    market_df["direction"] = pd.to_numeric(market_df["direction"], errors="coerce")

    semantic_df["target_timestamp"] = pd.to_datetime(semantic_df["date_t1"])
    market_df["market_timestamp"] = pd.to_datetime(market_df["Date"])

    merged = _merge_next_market_day(semantic_df, market_df)
    merged = merged.dropna(subset=["direction", "return_1d"])
    merged = merged.sort_values("prediction_order").reset_index(drop=True)
    merged["target_market_date"] = merged["Date"]

    if "target_direction_label" not in merged.columns:
        merged["target_direction_label"] = ""
    if "generated_direction_label" not in merged.columns:
        merged["generated_direction_label"] = ""

    merged["real_signal"] = merged["real_label"].map(label_to_signal)
    generated_direction_signal = merged["generated_direction_label"].map(direction_to_signal)
    merged["generated_signal"] = generated_direction_signal.where(
        merged["generated_direction_label"].fillna("").astype(str).str.len() > 0,
        merged["generated_label"].map(label_to_signal),
    )
    merged["actual_direction"] = merged["direction"].map(lambda value: 1 if int(value) == 1 else -1)
    merged["actual_return_1d"] = merged["return_1d"]
    merged["generated_direction_correct"] = merged["generated_signal"] == merged["actual_direction"]
    merged["real_direction_correct"] = merged["real_signal"] == merged["actual_direction"]
    merged["abs_return_1d"] = merged["actual_return_1d"].abs()
    median_abs_return = merged["abs_return_1d"].median()
    merged["volatility_bucket"] = merged["abs_return_1d"].map(
        lambda value: "high" if value >= median_abs_return else "low"
    )

    return merged[FINANCIAL_COLUMNS]


def _merge_next_market_day(semantic_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    merged_frames = []
    market_columns = ["market_timestamp", "Date", "direction", "return_1d"]
    for ticker, ticker_semantic in semantic_df.groupby("ticker", sort=False):
        ticker_market = market_df[market_df["ticker"] == ticker]
        if ticker_market.empty:
            continue
        merged_frames.append(
            pd.merge_asof(
                ticker_semantic.sort_values("target_timestamp"),
                ticker_market[market_columns].sort_values("market_timestamp"),
                left_on="target_timestamp",
                right_on="market_timestamp",
                direction="forward",
                tolerance=pd.Timedelta(days=3),
            )
        )

    if not merged_frames:
        return pd.DataFrame(columns=[*semantic_df.columns, "Date", "direction", "return_1d"])
    return pd.concat(merged_frames, ignore_index=True)


def summarize_financial(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty:
        return {
            "joined_rows": 0.0,
            "generated_directional_accuracy": 0.0,
            "real_news_directional_accuracy": 0.0,
            "generated_signal_coverage": 0.0,
            "real_signal_coverage": 0.0,
            "high_volatility_accuracy": 0.0,
            "low_volatility_accuracy": 0.0,
        }

    generated_active = metrics[metrics["generated_signal"] != 0]
    real_active = metrics[metrics["real_signal"] != 0]
    high_volatility = generated_active[generated_active["volatility_bucket"] == "high"]
    low_volatility = generated_active[generated_active["volatility_bucket"] == "low"]

    return {
        "joined_rows": float(len(metrics)),
        "generated_directional_accuracy": _accuracy(generated_active, "generated_direction_correct"),
        "real_news_directional_accuracy": _accuracy(real_active, "real_direction_correct"),
        "generated_signal_coverage": float(len(generated_active) / len(metrics)),
        "real_signal_coverage": float(len(real_active) / len(metrics)),
        "high_volatility_accuracy": _accuracy(high_volatility, "generated_direction_correct"),
        "low_volatility_accuracy": _accuracy(low_volatility, "generated_direction_correct"),
    }


def plot_confusion_matrix(metrics: pd.DataFrame, output_path: str | Path) -> None:
    active = metrics[metrics["generated_signal"] != 0]
    labels = [-1, 1]
    confusion = pd.crosstab(
        active["actual_direction"],
        active["generated_signal"],
        rownames=["actual"],
        colnames=["generated"],
    ).reindex(index=labels, columns=labels, fill_value=0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(confusion.values, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=["short", "long"])
    ax.set_yticks(range(len(labels)), labels=["down", "up"])
    ax.set_xlabel("Generated signal")
    ax.set_ylabel("Actual direction")

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            ax.text(
                column_index,
                row_index,
                int(confusion.values[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def label_to_signal(label: str) -> int:
    mapping = {"positive": 1, "negative": -1, "neutral": 0}
    return mapping.get(str(label).lower(), 0)


def _accuracy(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 0.0
    return float(frame[column].mean())
