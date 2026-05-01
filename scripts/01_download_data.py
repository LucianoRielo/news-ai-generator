from __future__ import annotations

from src.data.download_news import download_news_for_tickers
from src.data.download_market import download_market_for_tickers
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.tickers import get_config_tickers


def main() -> None:
    config = load_config()
    logger = setup_logger(
        log_dir=config["logging"]["dir"],
        log_file=config["logging"]["file"],
        level=config["logging"]["level"],
    )

    data_config = config["data"]
    tickers = get_config_tickers(data_config)
    news = download_news_for_tickers(
        dataset_name=data_config["dataset_name"],
        dataset_source=data_config["dataset_source"],
        tickers=tickers,
        output_path=data_config["raw_news_path"],
        start_date=data_config["start_date"],
        end_date=data_config["end_date"],
    )
    logger.info("Saved %s normalized news rows to %s", len(news), data_config["raw_news_path"])

    start_date = news["date"].min()
    end_date = news["date"].max()
    market = download_market_for_tickers(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        output_path=data_config["raw_market_path"],
    )
    logger.info(
        "Saved %s market rows for %s between %s and %s to %s",
        len(market),
        ",".join(tickers),
        start_date,
        end_date,
        data_config["raw_market_path"],
    )


if __name__ == "__main__":
    main()
