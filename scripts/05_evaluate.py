from __future__ import annotations

from src.evaluation.financial import evaluate_financial, summarize_financial
from src.evaluation.semantic import evaluate_semantic, summarize_semantic
from src.evaluation.textual import evaluate_textual, summarize_textual
from src.utils.config import load_config
from src.utils.logging import setup_logger


def main() -> None:
    config = load_config()
    logger = setup_logger(
        log_dir=config["logging"]["dir"],
        log_file=config["logging"]["file"],
        level=config["logging"]["level"],
    )

    evaluation_config = config["evaluation"]
    textual_metrics = evaluate_textual(
        predictions_path=evaluation_config["predictions_path"],
        output_path=evaluation_config["textual_metrics_path"],
        bertscore_model=evaluation_config["bertscore_model"],
        compute_bertscore=evaluation_config.get("compute_bertscore", True),
    )
    summary = summarize_textual(textual_metrics)
    logger.info("Saved textual metrics to %s", evaluation_config["textual_metrics_path"])
    logger.info("Textual metrics mean: %s", summary)

    semantic_metrics = evaluate_semantic(
        predictions_path=evaluation_config["predictions_path"],
        output_path=evaluation_config["semantic_metrics_path"],
        model_name=evaluation_config["finbert_model"],
        batch_size=evaluation_config.get("sentiment_batch_size", 8),
        confusion_matrix_path=evaluation_config["semantic_confusion_matrix_path"],
    )
    semantic_summary = summarize_semantic(semantic_metrics)
    logger.info("Saved semantic metrics to %s", evaluation_config["semantic_metrics_path"])
    logger.info("Semantic metrics summary: %s", semantic_summary)

    financial_metrics = evaluate_financial(
        semantic_metrics_path=evaluation_config["semantic_metrics_path"],
        market_path=config["data"]["raw_market_path"],
        output_path=evaluation_config["financial_metrics_path"],
        confusion_matrix_path=evaluation_config["financial_confusion_matrix_path"],
    )
    financial_summary = summarize_financial(financial_metrics)
    logger.info("Saved financial metrics to %s", evaluation_config["financial_metrics_path"])
    logger.info("Financial metrics summary: %s", financial_summary)


if __name__ == "__main__":
    main()
