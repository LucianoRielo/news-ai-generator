from __future__ import annotations

from pathlib import Path

from src.model.generate import generate_predictions
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
    evaluation_config = config["evaluation"]
    test_path = Path(data_config["processed_dir"]) / "test.jsonl"

    predictions = generate_predictions(
        model_path=model_config["output_dir"],
        test_path=test_path,
        output_path=evaluation_config["predictions_path"],
        generation_config=model_config["generation"],
    )
    logger.info("Saved %s predictions to %s", len(predictions), evaluation_config["predictions_path"])


if __name__ == "__main__":
    main()
