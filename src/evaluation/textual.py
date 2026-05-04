from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from src.utils.tickers import prediction_ticker


logger = logging.getLogger("news_ai_generator")

TEXTUAL_COLUMNS = [
    "ticker",
    "date_t",
    "date_t1",
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
]


def evaluate_textual(
    predictions_path: str | Path,
    output_path: str | Path,
    bertscore_model: str = "distilbert-base-uncased",
    compute_bertscore: bool = True,
) -> pd.DataFrame:
    predictions = _load_predictions(predictions_path)
    rows = []
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    references = [row["real_news"] for row in predictions]
    candidates = [row["generated_news"] for row in predictions]
    rouge_values = [scorer.score(reference, candidate) for reference, candidate in zip(references, candidates)]

    bert_precision = bert_recall = bert_f1 = [None] * len(predictions)
    if compute_bertscore and predictions:
        try:
            precision, recall, f1 = bert_score(
                candidates,
                references,
                model_type=bertscore_model,
                lang="en",
                verbose=False,
            )
            bert_precision = precision.tolist()
            bert_recall = recall.tolist()
            bert_f1 = f1.tolist()
        except Exception as exc:
            logger.warning("BERTScore failed; continuing with ROUGE-only textual metrics: %r", exc)

    for prediction, rouge, bp, br, bf in zip(predictions, rouge_values, bert_precision, bert_recall, bert_f1):
        rows.append(
            {
                "date_t": prediction["date_t"],
                "ticker": prediction_ticker(prediction),
                "date_t1": prediction["date_t1"],
                "rouge1": rouge["rouge1"].fmeasure,
                "rouge2": rouge["rouge2"].fmeasure,
                "rougeL": rouge["rougeL"].fmeasure,
                "bertscore_precision": bp,
                "bertscore_recall": br,
                "bertscore_f1": bf,
            }
        )

    metrics = pd.DataFrame(rows, columns=TEXTUAL_COLUMNS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    return metrics


def summarize_textual(metrics: pd.DataFrame) -> dict[str, float]:
    numeric = metrics.select_dtypes(include="number")
    return {column: float(numeric[column].mean()) for column in numeric.columns}


def _load_predictions(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
