from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


SLIDE_RUNS = [
    {
        "run_id": "2026-04-30_15-49-24_spy-qqq_gpt2_spy-qqq-baseline",
        "run": "SPY/QQQ baseline",
        "role": "Primer pipeline completo",
    },
    {
        "run_id": "baseline-gpt2-no-finetune_nvda-amd",
        "run": "GPT-2 no fine-tune",
        "role": "Baseline de modelo",
    },
    {
        "run_id": "2026-05-02_13-05-30_nvda_gpt2_nvda-single",
        "run": "NVDA single",
        "role": "Mejor BERTScore",
    },
    {
        "run_id": "2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2",
        "run": "NVDA + AMD",
        "role": "Mejor equilibrio",
    },
    {
        "run_id": "2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features",
        "run": "NVDA direction features",
        "role": "Mejor metrica financiera",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build slide-ready model comparison tables.")
    parser.add_argument(
        "--reports-dir",
        default="runs/model-comparison/reports",
        help="Directory containing comparison CSVs.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    comparison = pd.read_csv(reports_dir / "model_comparison.csv")
    significance = pd.read_csv(reports_dir / "directional_significance.csv")
    perplexity = pd.read_csv(reports_dir / "perplexity_summary.csv")

    table = build_slide_table(comparison, significance, perplexity)
    compact = build_compact_slide_table(table)
    csv_path = reports_dir / "slides_model_table.csv"
    md_path = reports_dir / "slides_model_table.md"
    compact_csv_path = reports_dir / "slides_model_table_compact.csv"
    compact_md_path = reports_dir / "slides_model_table_compact.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(table), encoding="utf-8")
    compact.to_csv(compact_csv_path, index=False)
    compact_md_path.write_text(to_markdown(compact, compact=True), encoding="utf-8")

    print(f"Saved slide CSV to {csv_path}")
    print(f"Saved slide Markdown to {md_path}")
    print(f"Saved compact slide CSV to {compact_csv_path}")
    print(f"Saved compact slide Markdown to {compact_md_path}")


def build_slide_table(
    comparison: pd.DataFrame,
    significance: pd.DataFrame,
    perplexity: pd.DataFrame,
) -> pd.DataFrame:
    comparison = comparison.set_index("run_id")
    significance = significance.set_index("run_id")
    perplexity = perplexity.set_index("run_id") if not perplexity.empty else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for spec in SLIDE_RUNS:
        run_id = spec["run_id"]
        metrics = comparison.loc[run_id] if run_id in comparison.index else pd.Series(dtype=object)
        sig = significance.loc[run_id] if run_id in significance.index else pd.Series(dtype=object)
        ppl = perplexity.loc[run_id] if run_id in perplexity.index else pd.Series(dtype=object)
        rows.append(
            {
                "Run": spec["run"],
                "Rol": spec["role"],
                "Tickers": metrics.get("tickers", ""),
                "Test": _int(metrics.get("test_examples")),
                "ROUGE-L": _num(metrics.get("rougeL")),
                "BERTScore F1": _num(metrics.get("bertscore_f1")),
                "Sent. match": _num(metrics.get("sentiment_match_accuracy")),
                "Neutral base": _num(metrics.get("neutral_baseline_accuracy")),
                "Dir. acc.": _num(metrics.get("generated_directional_accuracy")),
                "Coverage": _num(metrics.get("generated_signal_coverage")),
                "p vs 0.5": _num(sig.get("binomial_p_value_vs_0_5")),
                "Best ppl": _num(ppl.get("best_eval_perplexity")),
                "Lectura": reading_for(run_id),
            }
        )
    return pd.DataFrame(rows)


def build_compact_slide_table(table: pd.DataFrame) -> pd.DataFrame:
    return table[
        [
            "Run",
            "Rol",
            "ROUGE-L",
            "BERTScore F1",
            "Sent. match",
            "Dir. acc.",
            "Coverage",
            "p vs 0.5",
        ]
    ].copy()


def reading_for(run_id: str) -> str:
    readings = {
        "2026-04-30_15-49-24_spy-qqq_gpt2_spy-qqq-baseline": "Punto de partida razonable con ETFs amplios.",
        "baseline-gpt2-no-finetune_nvda-amd": "Acierta direccion pese a baja calidad semantica; alerta sobre la metrica financiera.",
        "2026-05-02_13-05-30_nvda_gpt2_nvda-single": "Mayor similitud semantica, pero peor direccion.",
        "2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2": "Mejor balance textual/semantico; modelo general mas defendible.",
        "2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features": "Mayor directional accuracy y coverage, sin significancia fuerte.",
    }
    return readings[run_id]


def to_markdown(frame: pd.DataFrame, compact: bool = False) -> str:
    lines = [
        "# Tabla final para slides" if not compact else "# Tabla compacta para slides",
        "",
        "Esta tabla esta reducida para presentacion. No incluye todas las metricas, solo las necesarias para sostener la narrativa.",
        "",
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    lines.extend(
        [
            "",
            "Lectura general:",
            "",
            "- El fine-tuning mejora adaptacion textual/semantica frente a GPT-2 base.",
            "- La mejor directional accuracy no prueba predictividad robusta.",
            "- El baseline sin fine-tuning muestra que la metrica financiera necesita controles adicionales.",
        ]
    )
    return "\n".join(lines)


def _num(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def _int(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(int(float(value)))


if __name__ == "__main__":
    main()
