from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Select qualitative examples for the NLP presentation.")
    parser.add_argument(
        "--baseline-run",
        default="runs/baseline-gpt2-no-finetune_nvda-amd",
        help="Run directory for GPT-2 without fine-tuning.",
    )
    parser.add_argument(
        "--finetuned-run",
        default="runs/2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2",
        help="Comparable fine-tuned run directory.",
    )
    parser.add_argument(
        "--direction-run",
        default="runs/2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features",
        help="Run directory for the best directional-accuracy model.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/model-comparison/reports",
        help="Directory where qualitative example artifacts will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_run_frame(Path(args.baseline_run), "baseline")
    finetuned = load_run_frame(Path(args.finetuned_run), "finetuned")
    direction = load_run_frame(Path(args.direction_run), "direction")

    paired = baseline.merge(finetuned, on=["ticker", "date_t", "date_t1"], suffixes=("_baseline", "_finetuned"))
    examples = select_paired_examples(paired)
    direction_examples = select_single_run_examples(direction)

    examples_path = output_dir / "qualitative_examples.csv"
    pd.DataFrame(examples + direction_examples).to_csv(examples_path, index=False)

    report_path = output_dir / "qualitative_examples.md"
    report_path.write_text(build_markdown(examples, direction_examples), encoding="utf-8")

    print(f"Saved qualitative examples table to {examples_path}")
    print(f"Saved qualitative examples report to {report_path}")


def load_run_frame(run_dir: Path, label: str) -> pd.DataFrame:
    predictions = pd.DataFrame(load_jsonl(run_dir / "generations" / "predictions.jsonl"))
    textual = read_csv(run_dir / "reports" / "textual_metrics.csv")
    semantic = read_csv(run_dir / "reports" / "semantic_metrics.csv")
    financial = read_csv(run_dir / "reports" / "financial_metrics.csv")

    frame = predictions.copy()
    for metrics in [textual, semantic, financial]:
        if metrics.empty:
            continue
        keys = ["date_t", "date_t1"]
        if "ticker" in frame.columns and "ticker" in metrics.columns:
            keys = ["ticker", *keys]
        duplicate_columns = set(metrics.columns).intersection(frame.columns).difference(keys)
        frame = frame.merge(metrics.drop(columns=sorted(duplicate_columns)), on=keys, how="left")

    frame["run_label"] = label
    return frame


def select_paired_examples(paired: pd.DataFrame) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    used: set[tuple[str, str, str]] = set()

    specs = [
        (
            "Fine-tuning improves semantic alignment",
            paired[
                (paired.get("sentiment_match_finetuned") == True)
                & (paired.get("sentiment_match_baseline") == False)
                & paired.get("real_news_finetuned", "").fillna("").astype(str).str.strip().ne("")
            ].sort_values("rougeL_finetuned", ascending=False),
            "El fine-tuned logra match semantico donde GPT-2 base no, y ademas produce una narrativa mas cercana al dominio.",
        ),
        (
            "Financial metric can be misleading",
            paired[
                (paired.get("generated_direction_correct_baseline") == True)
                & (paired.get("sentiment_match_baseline") == False)
                & paired.get("real_news_baseline", "").fillna("").astype(str).str.strip().ne("")
            ].sort_values("rougeL_baseline", ascending=True),
            "GPT-2 base acierta la direccion pero con una narrativa semanticamente pobre; esto muestra por que directional accuracy sola no alcanza.",
        ),
        (
            "Best fine-tuned textual example",
            paired.sort_values("rougeL_finetuned", ascending=False),
            "Caso con mayor ROUGE-L del run fine-tuneado NVDA+AMD; util para mostrar que el modelo aprende formato y vocabulario financiero.",
        ),
        (
            "Fine-tuned failure case",
            paired[
                (paired.get("generated_direction_correct_finetuned") == False)
                & (paired.get("sentiment_match_finetuned") == False)
            ].sort_values("rougeL_finetuned", ascending=True),
            "Fallo claro del modelo fine-tuneado; conviene mostrarlo para que la defensa sea critica y no triunfalista.",
        ),
    ]

    for title, candidates, reading in specs:
        row = first_unused(candidates, used)
        if row is None:
            continue
        used.add(key(row))
        examples.append(format_paired_example(title, row, reading))

    return examples


def select_single_run_examples(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    candidates = frame[
        (frame.get("generated_direction_correct") == True)
        & (frame.get("generated_signal") != 0)
    ].sort_values("rougeL", ascending=False)
    if candidates.empty:
        return []
    row = candidates.iloc[0]
    return [
        {
            "section": "Best directional-features example",
            "ticker": row.get("ticker"),
            "date_t": row.get("date_t"),
            "date_t1": row.get("date_t1"),
            "reading": "Ejemplo del run con features direccionales: senal correcta con cobertura total, util para discutir la variante mas orientada a la metrica financiera.",
            "real_news": row.get("real_news"),
            "baseline_generated_news": "",
            "finetuned_generated_news": row.get("generated_news"),
            "baseline_sentiment": "",
            "finetuned_sentiment": row.get("generated_label", row.get("generated_sentiment_label")),
            "baseline_direction_correct": "",
            "finetuned_direction_correct": row.get("generated_direction_correct"),
            "baseline_rougeL": "",
            "finetuned_rougeL": row.get("rougeL"),
        }
    ]


def first_unused(candidates: pd.DataFrame, used: set[tuple[str, str, str]]) -> pd.Series | None:
    for _, row in candidates.iterrows():
        if key(row) not in used:
            return row
    return None


def format_paired_example(title: str, row: pd.Series, reading: str) -> dict[str, Any]:
    return {
        "section": title,
        "ticker": row.get("ticker"),
        "date_t": row.get("date_t"),
        "date_t1": row.get("date_t1"),
        "reading": reading,
        "real_news": row.get("real_news_baseline", row.get("real_news_finetuned")),
        "baseline_generated_news": row.get("generated_news_baseline"),
        "finetuned_generated_news": row.get("generated_news_finetuned"),
        "baseline_sentiment": row.get("generated_label_baseline", row.get("generated_sentiment_label_baseline")),
        "finetuned_sentiment": row.get("generated_label_finetuned", row.get("generated_sentiment_label_finetuned")),
        "baseline_direction_correct": row.get("generated_direction_correct_baseline"),
        "finetuned_direction_correct": row.get("generated_direction_correct_finetuned"),
        "baseline_rougeL": row.get("rougeL_baseline"),
        "finetuned_rougeL": row.get("rougeL_finetuned"),
    }


def build_markdown(examples: list[dict[str, Any]], direction_examples: list[dict[str, Any]]) -> str:
    lines = [
        "# Ejemplos cualitativos para la presentacion",
        "",
        "Estos ejemplos estan pensados para acompanar la tabla de metricas. La idea es mostrar que las metricas agregadas necesitan lectura cualitativa: un modelo puede acertar direccion y aun asi generar una narrativa pobre.",
        "",
    ]

    for example in examples:
        lines.extend(format_example_markdown(example, paired=True))

    if direction_examples:
        lines.extend(["## Variante con features direccionales", ""])
        for example in direction_examples:
            lines.extend(format_example_markdown(example, paired=False))

    return "\n".join(lines)


def format_example_markdown(example: dict[str, Any], paired: bool) -> list[str]:
    lines = [
        f"## {example['section']}",
        "",
        f"- Ticker: `{example.get('ticker', '')}`",
        f"- Fecha target: `{example.get('date_t1', '')}`",
        f"- Lectura: {example.get('reading', '')}",
        "",
        "| Metrica | GPT-2 base | Fine-tuned / variante |",
        "|---|---:|---:|",
        f"| ROUGE-L | {fmt(example.get('baseline_rougeL'))} | {fmt(example.get('finetuned_rougeL'))} |",
        f"| Sentimiento generado | {fmt(example.get('baseline_sentiment'))} | {fmt(example.get('finetuned_sentiment'))} |",
        f"| Direccion correcta | {fmt(example.get('baseline_direction_correct'))} | {fmt(example.get('finetuned_direction_correct'))} |",
        "",
        "**Noticia real:**",
        "",
        quote(example.get("real_news")),
        "",
    ]
    if paired:
        lines.extend(
            [
                "**GPT-2 base:**",
                "",
                quote(example.get("baseline_generated_news")),
                "",
                "**Fine-tuned:**",
                "",
                quote(example.get("finetuned_generated_news")),
                "",
            ]
        )
    else:
        lines.extend(["**Variante:**", "", quote(example.get("finetuned_generated_news")), ""])
    return lines


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def key(row: pd.Series) -> tuple[str, str, str]:
    return str(row.get("ticker", "")), str(row.get("date_t", "")), str(row.get("date_t1", ""))


def quote(value: Any, max_chars: int = 650) -> str:
    text = str(value or "").replace("\r", "").strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return "\n".join(f"> {line}" for line in text.splitlines())


def fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "si" if value else "no"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.4f}"


if __name__ == "__main__":
    main()
