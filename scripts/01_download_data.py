from __future__ import annotations

from src.data.download_news import download_news
from src.data.download_market import download_market
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
    logger.info(
        "Saved %s market rows for %s between %s and %s to %s",
        len(market),
        data_config["ticker"],
        start_date,
        end_date,
        data_config["raw_market_path"],
    )


if __name__ == "__main__":
    main()
