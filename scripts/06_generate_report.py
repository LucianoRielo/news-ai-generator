from __future__ import annotations

import argparse
from pathlib import Path

from src.reporting.report import generate_report
from src.utils.config import load_config
from src.utils.logging import setup_logger


def main() -> None:
    args = parse_args()
    config_path = Path(args.run_dir) / "config.yaml" if args.run_dir else Path(args.config)
    config = load_config(config_path)
    if args.run_dir:
        config["evaluation"]["report_path"] = str(Path(args.run_dir) / "reports" / "REPORT.md")

    logger = setup_logger(
        log_dir=config["logging"]["dir"],
        log_file=config["logging"]["file"],
        level=config["logging"]["level"],
    )
    run_summary_path = Path(args.run_dir) / "run_summary.json" if args.run_dir else None
    output_path = args.output or config["evaluation"].get("report_path")

    generate_report(config=config, output_path=output_path, run_summary_path=run_summary_path)
    logger.info("Saved report to %s", output_path or config["evaluation"].get("report_path"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a markdown report from pipeline outputs.")
    parser.add_argument("--config", default="config/config.yaml", help="Config path for fixed outputs.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing config.yaml and artifacts.")
    parser.add_argument("--output", default=None, help="Optional report output path.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
