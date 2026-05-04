from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.tickers import prediction_ticker


SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SEMANTIC_COLUMNS = [
    "ticker",
    "date_t",
    "date_t1",
    "real_negative",
    "real_neutral",
    "real_positive",
    "generated_negative",
    "generated_neutral",
    "generated_positive",
    "real_label",
    "generated_label",
    "sentiment_match",
    "target_sentiment_label",
    "structured_generated_label",
    "structured_sentiment_match",
    "target_direction_label",
    "generated_direction_label",
    "real_net_sentiment",
    "generated_net_sentiment",
    "kl_divergence",
]


def evaluate_semantic(
    predictions_path: str | Path,
    output_path: str | Path,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 8,
    confusion_matrix_path: str | Path | None = None,
) -> pd.DataFrame:
    predictions = _load_predictions(predictions_path)
    references = [row["real_news"] for row in predictions]
    candidates = [row["generated_news"] for row in predictions]

    all_probabilities = predict_sentiment_probabilities(
        references + candidates,
        model_name=model_name,
        batch_size=batch_size,
    )
    real_probs = all_probabilities[: len(references)]
    generated_probs = all_probabilities[len(references) :]

    metrics = build_semantic_metrics(predictions, real_probs, generated_probs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    if confusion_matrix_path is not None:
        plot_semantic_confusion_matrix(metrics, confusion_matrix_path)
    return metrics


def predict_sentiment_probabilities(
    texts: list[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 8,
) -> list[dict[str, float]]:
    if not texts:
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id_to_label = {
        int(index): str(label).lower()
        for index, label in getattr(model.config, "id2label", {}).items()
    }
    probabilities: list[dict[str, float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for row in batch_probs:
            probabilities.append(_normalize_model_probabilities(row, id_to_label))

    return probabilities


def build_semantic_metrics(
    predictions: list[dict[str, Any]],
    real_probabilities: list[dict[str, float]],
    generated_probabilities: list[dict[str, float]],
) -> pd.DataFrame:
    if not (len(predictions) == len(real_probabilities) == len(generated_probabilities)):
        raise ValueError("Predictions and sentiment probability lists must have the same length")

    rows = []
    for prediction, real_probs, generated_probs in zip(
        predictions,
        real_probabilities,
        generated_probabilities,
    ):
        real_label = max(SENTIMENT_LABELS, key=lambda label: real_probs[label])
        generated_label = max(SENTIMENT_LABELS, key=lambda label: generated_probs[label])
        real_net = real_probs["positive"] - real_probs["negative"]
        generated_net = generated_probs["positive"] - generated_probs["negative"]

        rows.append(
            {
                "date_t": prediction["date_t"],
                "ticker": prediction_ticker(prediction),
                "date_t1": prediction["date_t1"],
                "real_negative": real_probs["negative"],
                "real_neutral": real_probs["neutral"],
                "real_positive": real_probs["positive"],
                "generated_negative": generated_probs["negative"],
                "generated_neutral": generated_probs["neutral"],
                "generated_positive": generated_probs["positive"],
                "real_label": real_label,
                "generated_label": generated_label,
                "sentiment_match": real_label == generated_label,
                "target_sentiment_label": prediction.get("real_sentiment_label", ""),
                "structured_generated_label": prediction.get("generated_sentiment_label", ""),
                "structured_sentiment_match": (
                    prediction.get("real_sentiment_label", "")
                    == prediction.get("generated_sentiment_label", "")
                    and bool(prediction.get("generated_sentiment_label", ""))
                ),
                "target_direction_label": prediction.get("real_direction_label", ""),
                "generated_direction_label": prediction.get("generated_direction_label", ""),
                "real_net_sentiment": real_net,
                "generated_net_sentiment": generated_net,
                "kl_divergence": _kl_divergence(real_probs, generated_probs),
            }
        )

    return pd.DataFrame(rows, columns=SEMANTIC_COLUMNS)


def summarize_semantic(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty:
        return {
            "sentiment_match_accuracy": 0.0,
            "mean_kl_divergence": 0.0,
            "net_sentiment_pearson": 0.0,
            "neutral_baseline_accuracy": 0.0,
        }

    neutral_baseline = (metrics["real_label"] == "neutral").mean()
    pearson = metrics["real_net_sentiment"].corr(metrics["generated_net_sentiment"])
    if pd.isna(pearson):
        pearson = 0.0

    return {
        "sentiment_match_accuracy": float(metrics["sentiment_match"].mean()),
        "mean_kl_divergence": float(metrics["kl_divergence"].mean()),
        "net_sentiment_pearson": float(pearson),
        "neutral_baseline_accuracy": float(neutral_baseline),
        "structured_sentiment_match_accuracy": _optional_mean(metrics, "structured_sentiment_match"),
    }


def plot_semantic_confusion_matrix(metrics: pd.DataFrame, output_path: str | Path) -> None:
    confusion = pd.crosstab(
        metrics["real_label"],
        metrics["generated_label"],
        rownames=["real"],
        colnames=["generated"],
    ).reindex(index=SENTIMENT_LABELS, columns=SENTIMENT_LABELS, fill_value=0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion.values, cmap="Blues")
    ax.set_xticks(range(len(SENTIMENT_LABELS)), labels=SENTIMENT_LABELS)
    ax.set_yticks(range(len(SENTIMENT_LABELS)), labels=SENTIMENT_LABELS)
    ax.set_xlabel("Generated sentiment")
    ax.set_ylabel("Real sentiment")

    for row_index in range(len(SENTIMENT_LABELS)):
        for column_index in range(len(SENTIMENT_LABELS)):
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


def _normalize_model_probabilities(
    probabilities: np.ndarray,
    id_to_label: dict[int, str],
) -> dict[str, float]:
    mapped = {label: 0.0 for label in SENTIMENT_LABELS}
    for index, probability in enumerate(probabilities):
        label = id_to_label.get(index, SENTIMENT_LABELS[index] if index < len(SENTIMENT_LABELS) else "")
        label = label.lower()
        if label in mapped:
            mapped[label] = float(probability)

    total = sum(mapped.values())
    if total <= 0:
        raise ValueError("Could not map model outputs to negative/neutral/positive labels")
    return {label: value / total for label, value in mapped.items()}


def _kl_divergence(
    real_probs: dict[str, float],
    generated_probs: dict[str, float],
    epsilon: float = 1e-12,
) -> float:
    return float(
        sum(
            real_probs[label]
            * math.log((real_probs[label] + epsilon) / (generated_probs[label] + epsilon))
            for label in SENTIMENT_LABELS
        )
    )


def _load_predictions(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _optional_mean(metrics: pd.DataFrame, column: str) -> float:
    if column not in metrics:
        return 0.0
    active = metrics[metrics[column].notna()]
    if active.empty:
        return 0.0
    return float(active[column].mean())
