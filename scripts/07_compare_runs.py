from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-run comparison reports.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run folders.")
    parser.add_argument(
        "--output-dir",
        default="runs/model-comparison/reports",
        help="Directory where comparison CSV and plots will be written.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_run_comparison(runs_dir)
    comparison_path = output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    loss_history = build_loss_history(runs_dir)
    loss_history_path = output_dir / "loss_history.csv"
    loss_history.to_csv(loss_history_path, index=False)

    perplexity = build_perplexity_summary(loss_history)
    perplexity_path = output_dir / "perplexity_summary.csv"
    perplexity.to_csv(perplexity_path, index=False)

    significance = build_directional_significance(runs_dir)
    significance_path = output_dir / "directional_significance.csv"
    significance.to_csv(significance_path, index=False)

    if not comparison.empty:
        plot_metric_comparison(comparison, output_dir / "model_comparison.png")
    if not loss_history.empty:
        plot_loss_curves(loss_history, output_dir / "training_loss_curves.png", "loss", "Training loss")
        plot_loss_curves(loss_history, output_dir / "eval_loss_curves.png", "eval_loss", "Validation loss")

    report_path = output_dir / "REPORT.md"
    write_global_report(comparison, perplexity, significance, report_path)

    print(f"Saved comparison table to {comparison_path}")
    print(f"Saved loss history to {loss_history_path}")
    print(f"Saved perplexity summary to {perplexity_path}")
    print(f"Saved directional significance to {significance_path}")
    print(f"Saved global report to {report_path}")


def build_run_comparison(runs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if "smoke" in run_dir.name:
            continue
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", {})
        examples = summary.get("processed_examples", {})
        durations = summary.get("stage_durations_seconds", {})

        rows.append(
            {
                "run_id": summary.get("run_id", run_dir.name),
                "tickers": ",".join(summary.get("tickers", []) or []),
                "base_model": summary.get("base_model"),
                "train_examples": examples.get("train"),
                "val_examples": examples.get("val"),
                "test_examples": examples.get("test"),
                "duration_hours": _seconds_to_hours(summary.get("duration_seconds")),
                "train_duration_hours": _seconds_to_hours(durations.get("train_model")),
                "rouge1": _metric(metrics, "textual", "rouge1"),
                "rouge2": _metric(metrics, "textual", "rouge2"),
                "rougeL": _metric(metrics, "textual", "rougeL"),
                "bertscore_f1": _metric(metrics, "textual", "bertscore_f1"),
                "sentiment_match_accuracy": _metric(metrics, "semantic", "sentiment_match_accuracy"),
                "neutral_baseline_accuracy": _metric(metrics, "semantic", "neutral_baseline_accuracy"),
                "net_sentiment_pearson": _metric(metrics, "semantic", "net_sentiment_pearson"),
                "generated_directional_accuracy": _metric(metrics, "financial", "generated_directional_accuracy"),
                "generated_signal_coverage": _metric(metrics, "financial", "generated_signal_coverage"),
                "real_news_directional_accuracy": _metric(metrics, "financial", "real_news_directional_accuracy"),
                "joined_rows": _metric(metrics, "financial", "joined_rows"),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        by=["generated_directional_accuracy", "bertscore_f1"],
        ascending=[False, False],
        na_position="last",
    )


def build_loss_history(runs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if "smoke" in run_dir.name:
            continue
        trainer_state = find_best_trainer_state(run_dir)
        if trainer_state is None:
            continue

        state = json.loads(trainer_state.read_text(encoding="utf-8"))
        for entry in state.get("log_history", []):
            if "loss" not in entry and "eval_loss" not in entry:
                continue
            rows.append(
                {
                    "run_id": run_dir.name,
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "loss": entry.get("loss"),
                    "eval_loss": entry.get("eval_loss"),
                    "learning_rate": entry.get("learning_rate"),
                    "trainer_state_path": str(trainer_state),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["run_id", "step", "epoch"])


def build_perplexity_summary(loss_history: pd.DataFrame) -> pd.DataFrame:
    if loss_history.empty or "eval_loss" not in loss_history:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    eval_rows = loss_history.dropna(subset=["eval_loss"]).sort_values(["run_id", "step"])
    for run_id, group in eval_rows.groupby("run_id", sort=True):
        best = group.loc[group["eval_loss"].idxmin()]
        final = group.iloc[-1]
        rows.append(
            {
                "run_id": run_id,
                "best_eval_step": best["step"],
                "best_eval_loss": best["eval_loss"],
                "best_eval_perplexity": _safe_perplexity(best["eval_loss"]),
                "final_eval_step": final["step"],
                "final_eval_loss": final["eval_loss"],
                "final_eval_perplexity": _safe_perplexity(final["eval_loss"]),
            }
        )

    return pd.DataFrame(rows).sort_values("best_eval_loss", na_position="last")


def build_directional_significance(runs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.glob("*/reports/financial_metrics.csv")):
        if "smoke" in metrics_path.parts[1]:
            continue
        metrics = pd.read_csv(metrics_path)
        required = {"generated_signal", "generated_direction_correct"}
        if not required.issubset(metrics.columns):
            continue

        active = metrics[metrics["generated_signal"].astype(bool)]
        active_n = len(active)
        correct = int(active["generated_direction_correct"].sum()) if active_n else 0
        accuracy = correct / active_n if active_n else None
        ci_low, ci_high = _wilson_interval(correct, active_n)

        rows.append(
            {
                "run_id": metrics_path.parts[1],
                "joined_rows": len(metrics),
                "active_n": active_n,
                "correct": correct,
                "directional_accuracy": accuracy,
                "signal_coverage": active_n / len(metrics) if len(metrics) else None,
                "binomial_p_value_vs_0_5": _binomial_two_sided(correct, active_n),
                "wilson_ci_low_95": ci_low,
                "wilson_ci_high_95": ci_high,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("directional_accuracy", ascending=False, na_position="last")


def find_best_trainer_state(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("models/**/trainer_state.json"))
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, float]:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return (-1, -1.0)
        return (int(state.get("global_step") or -1), float(state.get("epoch") or -1.0))

    return max(candidates, key=score)


def plot_metric_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("bertscore_f1", "BERTScore F1"),
        ("sentiment_match_accuracy", "Sentiment match"),
        ("generated_directional_accuracy", "Directional accuracy"),
        ("generated_signal_coverage", "Signal coverage"),
    ]
    labels = [_short_label(run_id) for run_id in comparison["run_id"]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for ax, (column, title) in zip(axes.ravel(), metrics):
        values = comparison[column]
        ax.bar(labels, values, color="#2f6f7e")
        ax.set_title(title)
        ax.set_ylim(0, max(1.0, float(values.max(skipna=True) or 1.0)))
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Cross-run model comparison", fontsize=16)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_loss_curves(loss_history: pd.DataFrame, output_path: Path, column: str, title: str) -> None:
    values = loss_history.dropna(subset=[column])
    if values.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for run_id, group in values.groupby("run_id", sort=True):
        group = group.sort_values("step")
        ax.plot(group["step"], group[column], marker="o", linewidth=1.5, markersize=3, label=_short_label(run_id))

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel(column)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_global_report(
    comparison: pd.DataFrame,
    perplexity: pd.DataFrame,
    significance: pd.DataFrame,
    output_path: Path,
) -> None:
    selected_runs = [
        "2026-04-30_15-49-24_spy-qqq_gpt2_spy-qqq-baseline",
        "baseline-gpt2-no-finetune_nvda-amd",
        "2026-05-02_13-05-30_nvda_gpt2_nvda-single",
        "2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2",
        "2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features",
    ]

    selected = comparison[comparison["run_id"].isin(selected_runs)].copy()
    selected["run_label"] = selected["run_id"].map(_presentation_label)
    selected = selected.set_index("run_id").reindex(selected_runs).reset_index()

    selected_columns = [
        "run_label",
        "tickers",
        "train_examples",
        "test_examples",
        "rougeL",
        "bertscore_f1",
        "sentiment_match_accuracy",
        "neutral_baseline_accuracy",
        "generated_directional_accuracy",
        "generated_signal_coverage",
    ]

    perplexity_selected = perplexity[perplexity["run_id"].isin(selected_runs)].copy()
    perplexity_selected["run_label"] = perplexity_selected["run_id"].map(_presentation_label)
    perplexity_selected = perplexity_selected.set_index("run_id").reindex(selected_runs).reset_index()

    significance_selected = significance[significance["run_id"].isin(selected_runs)].copy()
    significance_selected["run_label"] = significance_selected["run_id"].map(_presentation_label)
    significance_selected = significance_selected.set_index("run_id").reindex(selected_runs).reset_index()

    best_textual = _best_row(comparison, "bertscore_f1")
    best_general = comparison.loc[
        comparison["sentiment_match_accuracy"].fillna(-1).idxmax()
    ] if not comparison.empty else None
    best_financial = _best_row(comparison, "generated_directional_accuracy")
    best_perplexity = _best_row(perplexity, "best_eval_perplexity", ascending=True)

    content = [
        "# Reporte global de comparacion",
        "",
        "## Objetivo",
        "",
        (
            "Este reporte compara los principales experimentos de fine-tuning de GPT-2 para generacion "
            "de narrativas financieras. La lectura central es que las metricas textuales, semanticas y "
            "financieras capturan propiedades distintas del sistema, por lo que no conviene resumir el "
            "proyecto con una unica metrica."
        ),
        "",
        "## Runs recomendados para la presentacion",
        "",
        _markdown_table(
            selected[selected_columns],
            {
                "run_label": "Run",
                "tickers": "Tickers",
                "train_examples": "Train",
                "test_examples": "Test",
                "rougeL": "ROUGE-L",
                "bertscore_f1": "BERTScore F1",
                "sentiment_match_accuracy": "Sent. match",
                "neutral_baseline_accuracy": "Neutral base",
                "generated_directional_accuracy": "Dir. acc.",
                "generated_signal_coverage": "Coverage",
            },
        ),
        "",
        "Lectura:",
        "",
        "- `SPY/QQQ baseline` sirve como punto de partida del pipeline completo.",
        "- `GPT-2 no fine-tune` aisla cuanto aporta el entrenamiento especifico frente al modelo base.",
        "- `NVDA single` muestra el mejor BERTScore F1, pero no el mejor resultado financiero.",
        "- `NVDA + AMD` es el run mas equilibrado para defender como modelo general.",
        "- `NVDA direction features` es el mejor en directional accuracy, aunque no prueba predictividad robusta.",
        "",
        "## Graficos",
        "",
        "![Comparacion de modelos](model_comparison.png)",
        "",
        "![Training loss](training_loss_curves.png)",
        "",
        "![Validation loss](eval_loss_curves.png)",
        "",
        "## Perplexity",
        "",
        (
            "La perplexity se calcula como `exp(eval_loss)`. Sirve para interpretar la perdida de "
            "language modeling, pero no reemplaza las metricas downstream."
        ),
        "",
        _markdown_table(
            perplexity_selected[
                [
                    "run_label",
                    "best_eval_step",
                    "best_eval_loss",
                    "best_eval_perplexity",
                    "final_eval_loss",
                    "final_eval_perplexity",
                ]
            ],
            {
                "run_label": "Run",
                "best_eval_step": "Best step",
                "best_eval_loss": "Best eval loss",
                "best_eval_perplexity": "Best ppl",
                "final_eval_loss": "Final eval loss",
                "final_eval_perplexity": "Final ppl",
            },
        ),
        "",
        "Lectura:",
        "",
        (
            "- El mejor valor de perplexity no coincide necesariamente con el mejor resultado financiero. "
            "Esto es una conclusion importante: optimizar la perdida de lenguaje no garantiza optimizar "
            "la utilidad de la narrativa como senal."
        ),
        "",
        "## Significancia de directional accuracy",
        "",
        (
            "Para la directional accuracy se calcula un test binomial bilateral contra azar (`p = 0.5`) "
            "usando solo los casos donde el modelo emite una senal activa. Tambien se reporta un intervalo "
            "de confianza de Wilson al 95%."
        ),
        "",
        _markdown_table(
            significance_selected[
                [
                    "run_label",
                    "active_n",
                    "correct",
                    "directional_accuracy",
                    "signal_coverage",
                    "binomial_p_value_vs_0_5",
                    "wilson_ci_low_95",
                    "wilson_ci_high_95",
                ]
            ],
            {
                "run_label": "Run",
                "active_n": "N activo",
                "correct": "Correctas",
                "directional_accuracy": "Dir. acc.",
                "signal_coverage": "Coverage",
                "binomial_p_value_vs_0_5": "p vs 0.5",
                "wilson_ci_low_95": "CI 95 low",
                "wilson_ci_high_95": "CI 95 high",
            },
        ),
        "",
        "Lectura:",
        "",
        (
            "- El baseline sin fine-tuning tiene ROUGE-L bajo y sentiment match pobre, pero obtiene una "
            "directional accuracy alta. Esto sugiere que la metrica financiera puede capturar sesgos de la "
            "regla de etiquetado o de la distribucion del test, no solo calidad generativa."
        ),
        (
            "- Ninguno de los runs principales alcanza evidencia estadistica fuerte contra azar al nivel "
            "`p < 0.05`. Por eso la conclusion correcta es que hay senales exploratorias, no predictividad "
            "robusta demostrada."
        ),
        "- Coverage y accuracy deben leerse juntos: una senal selectiva no equivale a una senal emitida siempre.",
        "",
        "## Hallazgos principales",
        "",
        f"- Mejor BERTScore F1: `{_format_best(best_textual, 'bertscore_f1')}`.",
        f"- Mejor sentiment match: `{_format_best(best_general, 'sentiment_match_accuracy')}`.",
        f"- Mejor directional accuracy: `{_format_best(best_financial, 'generated_directional_accuracy')}`.",
        f"- Mejor perplexity: `{_format_best(best_perplexity, 'best_eval_perplexity')}`.",
        "",
        "## Conclusiones para defender",
        "",
        (
            "1. El modelo aprende rasgos del dominio financiero: vocabulario, estilo de titulares y patrones "
            "generales de narrativa."
        ),
        (
            "2. La similitud textual o semantica no alcanza para afirmar utilidad financiera. El run con mejor "
            "BERTScore no es el mejor en directional accuracy."
        ),
        (
            "3. El baseline sin fine-tuning obliga a ser prudentes: una directional accuracy alta puede aparecer "
            "aun con baja calidad semantica, por lo que la metrica financiera debe interpretarse con cuidado."
        ),
        (
            "4. La evaluacion semantica con FinBERT muestra una limitacion clara: varios modelos no superan "
            "el baseline neutral."
        ),
        (
            "5. La directional accuracy tiene algunos resultados por encima de 0.5, pero sin significancia "
            "estadistica fuerte. Debe presentarse como analisis exploratorio."
        ),
        (
            "6. El aporte mas solido del proyecto es metodologico: un pipeline reproducible de NLP generativo "
            "con evaluacion textual, semantica y downstream."
        ),
        "",
        "## Proximos pasos antes de slides",
        "",
        "- Revisar `qualitative_examples.md` para elegir los ejemplos finales de slides.",
        "- Elegir una matriz de confusion semantica y una financiera para la presentacion.",
        "- Convertir este reporte en 8-10 slides.",
        "",
        "## Reportes auxiliares",
        "",
        "- [Ejemplos cualitativos](qualitative_examples.md)",
        "- [Tabla compacta para slides](slides_model_table_compact.md)",
        "- [Tabla completa para slides](slides_model_table.md)",
        "- [Tabla completa de comparacion](model_comparison.csv)",
        "- [Significancia direccional](directional_significance.csv)",
        "- [Perplexity](perplexity_summary.csv)",
        "- [Matriz semantica GPT-2 no fine-tune](selected_confusion_matrices/semantic_baseline_gpt2_no_finetune.png)",
        "- [Matriz semantica fine-tuned NVDA + AMD](selected_confusion_matrices/semantic_finetuned_nvda_amd.png)",
        "- [Matriz financiera direction features](selected_confusion_matrices/financial_direction_features.png)",
        "",
    ]

    output_path.write_text("\n".join(content), encoding="utf-8")


def _metric(metrics: dict[str, Any], group: str, name: str) -> Any:
    return metrics.get(group, {}).get(name)


def _seconds_to_hours(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / 3600


def _safe_perplexity(loss: Any) -> float | None:
    if loss is None or pd.isna(loss):
        return None
    loss = float(loss)
    if loss > 50:
        return None
    return math.exp(loss)


def _binomial_two_sided(successes: int, trials: int, p: float = 0.5) -> float | None:
    if trials <= 0:
        return None

    probabilities = [
        math.comb(trials, value) * (p**value) * ((1 - p) ** (trials - value))
        for value in range(trials + 1)
    ]
    observed_probability = probabilities[successes]
    return min(1.0, sum(prob for prob in probabilities if prob <= observed_probability + 1e-15))


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None

    phat = successes / trials
    denominator = 1 + z**2 / trials
    center = (phat + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * trials)) / trials) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _markdown_table(frame: pd.DataFrame, headers: dict[str, str]) -> str:
    if frame.empty:
        return "No hay datos disponibles."

    columns = list(headers)
    lines = [
        "| " + " | ".join(headers[column] for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_format_cell(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer() and abs(numeric) >= 10:
        return str(int(numeric))
    return f"{numeric:.4f}"


def _best_row(frame: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return None
    index = values.idxmin() if ascending else values.idxmax()
    return frame.loc[index]


def _format_best(row: pd.Series | None, column: str) -> str:
    if row is None:
        return "no disponible"
    return f"{_presentation_label(str(row['run_id']))} = {_format_cell(row[column])}"


def _presentation_label(run_id: str) -> str:
    labels = {
        "2026-04-30_15-49-24_spy-qqq_gpt2_spy-qqq-baseline": "SPY/QQQ baseline",
        "2026-05-02_13-05-30_nvda_gpt2_nvda-single": "NVDA single",
        "2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2": "NVDA + AMD",
        "2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features": "NVDA direction features",
        "2026-05-03_12-41-00_nvda_gpt2_nvda-structured-prompt": "NVDA structured prompt",
        "2026-05-04_00-31-31_nvda_gpt2_nvda-tagged-structured-prompt": "NVDA tagged structured",
        "2026-05-04_09-09-04_nvda_gpt2_nvda-label-scoring": "NVDA label scoring",
        "2026-05-05_10-41-34_nvda-amd-avgo_gpt2_nvda-amd-avgo": "NVDA + AMD + AVGO",
        "2026-05-12_09-20-01_nvda-amd_gpt2_nvda-amd-label-scoring": "NVDA + AMD label scoring",
        "gpt2-spy-20260430-150000": "SPY legacy run",
        "baseline-gpt2-no-finetune_nvda-amd": "GPT-2 no fine-tune",
        "baseline-gpt2-no-finetune_smoke": "GPT-2 smoke baseline",
    }
    return labels.get(run_id, _short_label(run_id).replace("\n", " "))


def _short_label(run_id: str) -> str:
    label = run_id
    for prefix in [
        "2026-04-30_15-49-24_",
        "2026-05-02_13-05-30_",
        "2026-05-03_01-12-20_",
        "2026-05-03_12-41-00_",
        "2026-05-04_00-31-31_",
        "2026-05-04_09-09-04_",
        "2026-05-04_18-09-40_",
        "2026-05-05_10-41-34_",
        "2026-05-12_09-20-01_",
    ]:
        label = label.replace(prefix, "")
    return label.replace("gpt2_", "").replace("_", "\n")


if __name__ == "__main__":
    main()
