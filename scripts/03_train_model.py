from __future__ import annotations

from pathlib import Path

from src.model.train import train_model
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


if __name__ == "__main__":
    main()
