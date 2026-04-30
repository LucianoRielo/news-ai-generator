from __future__ import annotations

from src.data.build_dataset import build_dataset, load_raw_data
from src.utils.config import load_config
from src.utils.logging import setup_logger


def main() -> None:
    config = load_config()
    logger = setup_logger(
        log_dir=config["logging"]["dir"],
        log_file=config["logging"]["file"],
        level=config["logging"]["level"],
    )

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


if __name__ == "__main__":
    main()
