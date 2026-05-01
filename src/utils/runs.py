from __future__ import annotations

import copy
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.tickers import get_config_tickers


def create_run_config(
    base_config: dict[str, Any],
    runs_dir: str | Path = "runs",
    run_name: str | None = None,
    now: datetime | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    tickers = get_config_tickers(base_config["data"])
    tickers_slug = "-".join(_slugify(str(ticker)) for ticker in tickers)
    name_parts = [
        timestamp,
        tickers_slug,
        _slugify(base_config["model"]["base_model"]),
    ]
    if run_name:
        name_parts.append(_slugify(run_name))

    run_id = "_".join(part for part in name_parts if part)
    run_root = Path(runs_dir) / run_id
    config = copy.deepcopy(base_config)

    config["data"]["raw_news_path"] = str(run_root / "data" / "raw" / "news.csv")
    config["data"]["raw_market_path"] = str(run_root / "data" / "raw" / "market.csv")
    config["data"]["processed_dir"] = str(run_root / "data" / "processed")

    model_dir_name = Path(base_config["model"]["output_dir"]).name or "model"
    config["model"]["output_dir"] = str(run_root / "models" / model_dir_name)

    config["evaluation"]["predictions_path"] = str(run_root / "generations" / "predictions.jsonl")
    config["evaluation"]["textual_metrics_path"] = str(run_root / "reports" / "textual_metrics.csv")
    config["evaluation"]["semantic_metrics_path"] = str(run_root / "reports" / "semantic_metrics.csv")
    config["evaluation"]["semantic_confusion_matrix_path"] = str(
        run_root / "reports" / "semantic_confusion_matrix.png"
    )
    config["evaluation"]["financial_metrics_path"] = str(run_root / "reports" / "financial_metrics.csv")
    config["evaluation"]["financial_confusion_matrix_path"] = str(
        run_root / "reports" / "financial_confusion_matrix.png"
    )
    config["evaluation"]["report_path"] = str(run_root / "reports" / "REPORT.md")

    config["logging"]["dir"] = str(run_root / "logs")
    config["logging"]["file"] = "pipeline.log"

    return run_id, run_root, config


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"
