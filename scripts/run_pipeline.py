from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from src.data.build_dataset import build_dataset, load_raw_data
from src.data.download_market import download_market
from src.data.download_news import download_news
from src.evaluation.financial import evaluate_financial, summarize_financial
from src.evaluation.semantic import evaluate_semantic, summarize_semantic
from src.evaluation.textual import evaluate_textual, summarize_textual
from src.model.generate import generate_predictions
from src.model.train import train_model
from src.reporting.report import generate_report
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.runs import create_run_config


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    run_id, run_root, config = create_run_config(
        base_config=base_config,
        runs_dir=args.runs_dir,
        run_name=args.name,
    )
    run_root.mkdir(parents=True, exist_ok=False)
    save_effective_config(config, run_root / "config.yaml")

    logger = setup_logger(
        log_dir=config["logging"]["dir"],
        log_file=config["logging"]["file"],
        level=config["logging"]["level"],
    )
    logger.info("Starting pipeline run %s", run_id)
    logger.info("Run directory: %s", run_root)

    started_at = datetime.now()
    stage_durations: dict[str, float] = {}
    status = "completed"
    error: str | None = None

    try:
        stage_durations["download_data"] = timed(lambda: run_download_data(config, logger))
        stage_durations["build_dataset"] = timed(lambda: run_build_dataset(config, logger))
        stage_durations["train_model"] = timed(lambda: run_train_model(config, logger))
        stage_durations["generate_predictions"] = timed(lambda: run_generate_predictions(config, logger))
        summaries = run_evaluation(config, logger)
        stage_durations["evaluate"] = summaries.pop("_duration")
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        logger.exception("Pipeline run failed")
        summaries = {}
        raise
    finally:
        completed_at = datetime.now()
        summary = build_run_summary(
            run_id=run_id,
            run_root=run_root,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            config=config,
            stage_durations=stage_durations,
            summaries=summaries,
            error=error,
        )
        write_json(run_root / "run_summary.json", summary)
        logger.info("Saved run summary to %s", run_root / "run_summary.json")
        if status == "completed":
            stage_durations["generate_report"] = timed(lambda: run_report(config, run_root, logger))
            completed_at = datetime.now()
            summary = build_run_summary(
                run_id=run_id,
                run_root=run_root,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                config=config,
                stage_durations=stage_durations,
                summaries=summaries,
                error=error,
            )
            write_json(run_root / "run_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full financial narrative pipeline into a timestamped run.")
    parser.add_argument("--config", default="config/config.yaml", help="Base config path.")
    parser.add_argument("--runs-dir", default="runs", help="Directory where timestamped runs are stored.")
    parser.add_argument("--name", default=None, help="Optional suffix for the run directory.")
    return parser.parse_args()


def run_download_data(config: dict[str, Any], logger: Any) -> None:
    data_config = config["data"]
    news = download_news(
        dataset_name=data_config["dataset_name"],
        dataset_source=data_config["dataset_source"],
        ticker=data_config["ticker"],
        output_path=data_config["raw_news_path"],
        start_date=data_config["start_date"],
        end_date=data_config["end_date"],
    )
    logger.info("Saved %s normalized news rows to %s", len(news), data_config["raw_news_path"])

    start_date = news["date"].min()
    end_date = news["date"].max()
    market = download_market(
        ticker=data_config["ticker"],
        start_date=start_date,
        end_date=end_date,
        output_path=data_config["raw_market_path"],
    )
    logger.info("Saved %s market rows to %s", len(market), data_config["raw_market_path"])


def run_build_dataset(config: dict[str, Any], logger: Any) -> None:
    data_config = config["data"]
    news_df, market_df = load_raw_data(data_config["raw_news_path"], data_config["raw_market_path"])
    splits = build_dataset(
        news_df=news_df,
        market_df=market_df,
        ticker=data_config["ticker"],
        k=data_config["context_window_days"],
        split_ratios=data_config["split_ratios"],
        output_dir=data_config["processed_dir"],
        max_news_per_day=data_config["max_news_per_day"],
        max_text_chars=data_config["max_text_chars"],
        max_completion_news=data_config["max_completion_news"],
        include_body=data_config.get("include_body", False),
    )
    logger.info(
        "Saved dataset splits to %s: train=%s val=%s test=%s",
        data_config["processed_dir"],
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )


def run_train_model(config: dict[str, Any], logger: Any) -> None:
    data_config = config["data"]
    model_config = config["model"]
    processed_dir = Path(data_config["processed_dir"])
    logger.info("Starting fine-tuning for %s", model_config["base_model"])
    train_model(
        train_path=processed_dir / "train.jsonl",
        val_path=processed_dir / "val.jsonl",
        model_name=model_config["base_model"],
        output_dir=model_config["output_dir"],
        max_length=model_config["max_length"],
        train_config=model_config["train"],
    )
    logger.info("Saved fine-tuned model to %s", model_config["output_dir"])


def run_generate_predictions(config: dict[str, Any], logger: Any) -> None:
    data_config = config["data"]
    model_config = config["model"]
    evaluation_config = config["evaluation"]
    predictions = generate_predictions(
        model_path=model_config["output_dir"],
        test_path=Path(data_config["processed_dir"]) / "test.jsonl",
        output_path=evaluation_config["predictions_path"],
        generation_config=model_config["generation"],
    )
    logger.info("Saved %s predictions to %s", len(predictions), evaluation_config["predictions_path"])


def run_evaluation(config: dict[str, Any], logger: Any) -> dict[str, Any]:
    start = perf_counter()
    evaluation_config = config["evaluation"]

    textual_metrics = evaluate_textual(
        predictions_path=evaluation_config["predictions_path"],
        output_path=evaluation_config["textual_metrics_path"],
        bertscore_model=evaluation_config["bertscore_model"],
        compute_bertscore=evaluation_config.get("compute_bertscore", True),
    )
    textual_summary = summarize_textual(textual_metrics)
    logger.info("Textual metrics mean: %s", textual_summary)

    semantic_metrics = evaluate_semantic(
        predictions_path=evaluation_config["predictions_path"],
        output_path=evaluation_config["semantic_metrics_path"],
        model_name=evaluation_config["finbert_model"],
        batch_size=evaluation_config.get("sentiment_batch_size", 8),
        confusion_matrix_path=evaluation_config["semantic_confusion_matrix_path"],
    )
    semantic_summary = summarize_semantic(semantic_metrics)
    logger.info("Semantic metrics summary: %s", semantic_summary)

    financial_metrics = evaluate_financial(
        semantic_metrics_path=evaluation_config["semantic_metrics_path"],
        market_path=config["data"]["raw_market_path"],
        output_path=evaluation_config["financial_metrics_path"],
        confusion_matrix_path=evaluation_config["financial_confusion_matrix_path"],
    )
    financial_summary = summarize_financial(financial_metrics)
    logger.info("Financial metrics summary: %s", financial_summary)

    return {
        "textual": textual_summary,
        "semantic": semantic_summary,
        "financial": financial_summary,
        "_duration": perf_counter() - start,
    }


def run_report(config: dict[str, Any], run_root: Path, logger: Any) -> None:
    report_path = config["evaluation"]["report_path"]
    generate_report(config=config, output_path=report_path, run_summary_path=run_root / "run_summary.json")
    logger.info("Saved report to %s", report_path)


def build_run_summary(
    run_id: str,
    run_root: Path,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    config: dict[str, Any],
    stage_durations: dict[str, float],
    summaries: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_dir": str(run_root),
        "status": status,
        "started_at": started_at.isoformat(timespec="seconds"),
        "completed_at": completed_at.isoformat(timespec="seconds"),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "stage_durations_seconds": stage_durations,
        "ticker": config["data"]["ticker"],
        "base_model": config["model"]["base_model"],
        "model_output_dir": config["model"]["output_dir"],
        "processed_examples": count_processed_examples(config["data"]["processed_dir"]),
        "metrics": summaries,
        "error": error,
    }


def count_processed_examples(processed_dir: str | Path) -> dict[str, int]:
    counts = {}
    for split in ["train", "val", "test"]:
        path = Path(processed_dir) / f"{split}.jsonl"
        if path.exists():
            counts[split] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return counts


def save_effective_config(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=False)


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def timed(callback: Any) -> float:
    start = perf_counter()
    callback()
    return perf_counter() - start


if __name__ == "__main__":
    main()
