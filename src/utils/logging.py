from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(
    name: str = "news_ai_generator",
    log_dir: str | Path = "outputs/logs",
    log_file: str = "pipeline.log",
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Create a project logger that writes to console and outputs/logs."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_path = log_path / log_file
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == file_path.resolve()
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    ):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
