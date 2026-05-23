from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.financial import evaluate_financial, summarize_financial
from src.evaluation.semantic import evaluate_semantic, summarize_semantic
from src.evaluation.textual import evaluate_textual, summarize_textual
from src.model.generate import generate_one, score_best_label
from src.model.train import load_jsonl
from src.reporting.report import generate_report
from src.utils.structured_output import DIRECTION_LABELS, SENTIMENT_LABELS, parse_outlook
from src.utils.tickers import prediction_ticker

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 without fine-tuning on an existing run test split.")
    parser.add_argument(
        "--source-run",
        default="runs/2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2",
        help="Run directory whose data split and evaluation config will be reused.",
    )
    parser.add_argument(
        "--output-run",
        default="runs/baseline-gpt2-no-finetune_nvda-amd",
        help="Run directory where baseline predictions and reports will be saved.",
    )
    parser.add_argument("--model-name", default="gpt2", help="Base Hugging Face causal LM to evaluate.")
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore if a faster smoke run is needed.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of test examples to generate.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate predictions even if an existing baseline predictions file is present.",
    )
    args = parser.parse_args()

    source_run = Path(args.source_run)
    output_run = Path(args.output_run)
    output_run.mkdir(parents=True, exist_ok=True)
    (output_run / "generations").mkdir(exist_ok=True)
    (output_run / "reports").mkdir(exist_ok=True)
    (output_run / "logs").mkdir(exist_ok=True)

    started_at = datetime.now()
    stage_durations: dict[str, float] = {}
    summaries: dict[str, Any] = {}
    status = "completed"
    error = None

    try:
        config = build_baseline_config(source_run, output_run, args.model_name, compute_bertscore=not args.skip_bertscore)
        config_path = output_run / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        start = perf_counter()
        predictions = generate_baseline_predictions(
            model_path=args.model_name,
            test_path=Path(config["data"]["processed_dir"]) / "test.jsonl",
            output_path=config["evaluation"]["predictions_path"],
            generation_config=config["model"]["generation"],
            limit=args.limit,
            force=args.force,
        )
        stage_durations["generate_predictions"] = perf_counter() - start

        start = perf_counter()
        textual_metrics = evaluate_textual(
            predictions_path=config["evaluation"]["predictions_path"],
            output_path=config["evaluation"]["textual_metrics_path"],
            bertscore_model=config["evaluation"]["bertscore_model"],
            compute_bertscore=config["evaluation"].get("compute_bertscore", True),
        )
        semantic_metrics = evaluate_semantic(
            predictions_path=config["evaluation"]["predictions_path"],
            output_path=config["evaluation"]["semantic_metrics_path"],
            model_name=config["evaluation"]["finbert_model"],
            batch_size=config["evaluation"].get("sentiment_batch_size", 8),
            confusion_matrix_path=config["evaluation"]["semantic_confusion_matrix_path"],
        )
        financial_metrics = evaluate_financial(
            semantic_metrics_path=config["evaluation"]["semantic_metrics_path"],
            market_path=config["data"]["raw_market_path"],
            output_path=config["evaluation"]["financial_metrics_path"],
            confusion_matrix_path=config["evaluation"]["financial_confusion_matrix_path"],
        )
        stage_durations["evaluate"] = perf_counter() - start

        summaries = {
            "textual": summarize_textual(textual_metrics),
            "semantic": summarize_semantic(semantic_metrics),
            "financial": summarize_financial(financial_metrics),
        }
        completed_at = datetime.now()
        summary = build_run_summary(
            run_id=output_run.name,
            run_root=output_run,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            config=config,
            predictions_count=len(predictions),
            stage_durations=stage_durations,
            summaries=summaries,
            error=error,
        )
        summary_path = output_run / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        generate_report(config=config, output_path=config["evaluation"]["report_path"], run_summary_path=summary_path)
    except Exception as exc:
        status = "failed"
        error = str(exc)
        completed_at = datetime.now()
        summary = build_run_summary(
            run_id=output_run.name,
            run_root=output_run,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            config=build_baseline_config(source_run, output_run, args.model_name, compute_bertscore=not args.skip_bertscore),
            predictions_count=0,
            stage_durations=stage_durations,
            summaries=summaries,
            error=error,
        )
        (output_run / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise


def build_baseline_config(
    source_run: Path,
    output_run: Path,
    model_name: str,
    compute_bertscore: bool,
) -> dict[str, Any]:
    source_config = yaml.safe_load((source_run / "config.yaml").read_text(encoding="utf-8"))
    config = dict(source_config)
    config["data"] = dict(source_config["data"])
    config["model"] = dict(source_config["model"])
    config["evaluation"] = dict(source_config["evaluation"])
    config["logging"] = dict(source_config.get("logging", {}))

    config["model"]["base_model"] = model_name
    config["model"]["output_dir"] = model_name
    config["model"]["train"] = dict(source_config["model"].get("train", {}))
    config["model"]["train"]["num_train_epochs"] = 0

    config["evaluation"]["predictions_path"] = str(output_run / "generations" / "predictions.jsonl")
    config["evaluation"]["textual_metrics_path"] = str(output_run / "reports" / "textual_metrics.csv")
    config["evaluation"]["semantic_metrics_path"] = str(output_run / "reports" / "semantic_metrics.csv")
    config["evaluation"]["semantic_confusion_matrix_path"] = str(
        output_run / "reports" / "semantic_confusion_matrix.png"
    )
    config["evaluation"]["financial_metrics_path"] = str(output_run / "reports" / "financial_metrics.csv")
    config["evaluation"]["financial_confusion_matrix_path"] = str(
        output_run / "reports" / "financial_confusion_matrix.png"
    )
    config["evaluation"]["report_path"] = str(output_run / "reports" / "REPORT.md")
    config["evaluation"]["reports_dir"] = str(output_run / "reports")
    config["evaluation"]["compute_bertscore"] = compute_bertscore

    config["logging"]["dir"] = str(output_run / "logs")
    config["logging"]["file"] = "baseline.log"
    return config


def generate_baseline_predictions(
    model_path: str,
    test_path: str | Path,
    output_path: str | Path,
    generation_config: dict[str, Any],
    limit: int | None = None,
    force: bool = False,
) -> list[dict[str, str]]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if force and output.exists():
        output.unlink()

    examples = load_jsonl(test_path)
    if limit is not None:
        examples = examples[:limit]

    existing = _load_jsonl_if_exists(output)
    completed_keys = {_prediction_key(prediction) for prediction in existing}
    if len(existing) >= len(examples):
        return existing[: len(examples)]

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions = list(existing)
    with output.open("a", encoding="utf-8") as file:
        for index, example in enumerate(examples, start=1):
            key = _example_key(example)
            if key in completed_keys:
                continue

            sentiment = score_best_label(
                model=model,
                tokenizer=tokenizer,
                prompt=example["prompt"],
                labels=sorted(SENTIMENT_LABELS),
                device=device,
            )
            direction_prompt = f"{example['prompt']} {sentiment}\nDirection:"
            direction = score_best_label(
                model=model,
                tokenizer=tokenizer,
                prompt=direction_prompt,
                labels=sorted(DIRECTION_LABELS),
                device=device,
            )
            narrative_prompt = f"{direction_prompt} {direction}\nNews:\n"
            generated = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=narrative_prompt,
                generation_config=generation_config,
                device=device,
            )
            real_text = f"Sentiment:{example['completion']}"
            generated_text = f"Sentiment: {sentiment}\nDirection: {direction}\nNews:\n{generated}"
            real_outlook = parse_outlook(real_text)
            generated_outlook = parse_outlook(generated_text)
            prediction = {
                "ticker": prediction_ticker(example),
                "date_t": example["date_t"],
                "date_t1": example["date_t1"],
                "prompt": example["prompt"],
                "real_news": real_outlook["narrative"],
                "generated_news": generated_outlook["narrative"],
                "real_outlook": real_text,
                "generated_outlook": generated_text,
                "real_sentiment_label": example.get("target_sentiment_label", real_outlook["sentiment"]),
                "generated_sentiment_label": sentiment,
                "real_direction_label": example.get("target_direction_label", real_outlook["direction"]),
                "generated_direction_label": direction,
            }
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            file.flush()
            predictions.append(prediction)
            if index % 10 == 0:
                print(f"Generated {len(predictions)}/{len(examples)} baseline predictions")

    return predictions


def build_run_summary(
    run_id: str,
    run_root: Path,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    config: dict[str, Any],
    predictions_count: int,
    stage_durations: dict[str, float],
    summaries: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_dir": str(run_root),
        "status": status,
        "started_at": started_at.isoformat(timespec="seconds"),
        "completed_at": completed_at.isoformat(timespec="seconds"),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "stage_durations_seconds": stage_durations,
        "tickers": config["data"].get("tickers", [config["data"].get("ticker", "")]),
        "ticker": config["data"].get("ticker"),
        "base_model": config["model"]["base_model"],
        "model_output_dir": config["model"]["output_dir"],
        "processed_examples": processed_counts(config["data"]["processed_dir"]),
        "predictions_count": predictions_count,
        "metrics": summaries,
        "error": error,
    }


def processed_counts(processed_dir: str | Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        path = Path(processed_dir) / f"{split}.jsonl"
        counts[split] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return counts


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _example_key(example: dict[str, Any]) -> tuple[str, str, str]:
    return prediction_ticker(example), str(example.get("date_t", "")), str(example.get("date_t1", ""))


def _prediction_key(prediction: dict[str, Any]) -> tuple[str, str, str]:
    return str(prediction.get("ticker", "")), str(prediction.get("date_t", "")), str(prediction.get("date_t1", ""))


if __name__ == "__main__":
    main()
