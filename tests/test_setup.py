from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from src.utils.logging import setup_logger
from src.utils.runs import create_run_config
from src.utils.tickers import get_config_tickers


ROOT = Path(__file__).resolve().parents[1]


def test_expected_directories_exist() -> None:
    expected_dirs = [
        "config",
        "data/raw",
        "data/processed",
        "src/data",
        "src/model",
        "src/evaluation",
        "src/utils",
        "scripts",
        "tests",
        "notebooks",
        "outputs/models",
        "outputs/generations",
        "outputs/reports",
        "outputs/logs",
    ]

    missing = [path for path in expected_dirs if not (ROOT / path).is_dir()]

    assert not missing, f"Missing expected directories: {missing}"


def test_config_yaml_has_expected_keys() -> None:
    config_path = ROOT / "config" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["data"]["ticker"]
    assert get_config_tickers(config["data"]) == ["SPY", "QQQ"]
    assert config["data"]["context_window_days"] >= 1
    assert config["model"]["base_model"] in {"gpt2", "gpt2-medium"}
    assert config["model"]["max_length"] <= 1024
    assert config["logging"]["dir"] == "outputs/logs"

    split_total = sum(config["data"]["split_ratios"].values())
    assert split_total == pytest.approx(1.0)


def test_logger_initializes_and_writes_file() -> None:
    log_dir = ROOT / "outputs" / "logs"
    logger = setup_logger(name="setup_test_logger", log_dir=log_dir, log_file="test.log")

    logger.info("logger smoke test")

    log_file = log_dir / "test.log"
    assert log_file.exists()
    assert "logger smoke test" in log_file.read_text(encoding="utf-8")


def test_run_config_redirects_outputs_to_run_dir() -> None:
    config_path = ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    run_id, run_root, run_config = create_run_config(
        config,
        runs_dir="runs",
        run_name="smoke",
    )

    assert "spy" in run_id
    assert "smoke" in run_id
    assert run_root.parts[0] == "runs"
    assert run_config["data"]["raw_news_path"].startswith(str(run_root))
    assert run_config["data"]["processed_dir"].startswith(str(run_root))
    assert run_config["model"]["output_dir"].startswith(str(run_root))
    assert run_config["evaluation"]["financial_metrics_path"].startswith(str(run_root))
    assert run_config["evaluation"]["report_path"].startswith(str(run_root))
    assert run_config["logging"]["dir"].startswith(str(run_root))
