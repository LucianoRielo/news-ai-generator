from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def generate_report(
    config: dict[str, Any],
    output_path: str | Path | None = None,
    run_summary_path: str | Path | None = None,
) -> str:
    evaluation = config["evaluation"]
    data = config["data"]
    model = config["model"]

    report_path = Path(output_path or evaluation.get("report_path", Path(evaluation["reports_dir"]) / "REPORT.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = _load_jsonl(evaluation["predictions_path"])
    textual = _read_csv_if_exists(evaluation["textual_metrics_path"])
    semantic = _read_csv_if_exists(evaluation["semantic_metrics_path"])
    financial = _read_csv_if_exists(evaluation["financial_metrics_path"])
    run_summary = _load_json_if_exists(run_summary_path) if run_summary_path else None

    markdown = build_report_markdown(
        config=config,
        predictions=predictions,
        textual=textual,
        semantic=semantic,
        financial=financial,
        run_summary=run_summary,
        report_dir=report_path.parent,
    )
    report_path.write_text(markdown, encoding="utf-8")
    return markdown


def build_report_markdown(
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
    textual: pd.DataFrame,
    semantic: pd.DataFrame,
    financial: pd.DataFrame,
    run_summary: dict[str, Any] | None = None,
    report_dir: str | Path | None = None,
) -> str:
    data = config["data"]
    model = config["model"]
    evaluation = config["evaluation"]
    report_dir = Path(report_dir or evaluation["reports_dir"])
    processed_counts = _processed_counts(data["processed_dir"])
    textual_summary = _numeric_means(textual)
    semantic_summary = _semantic_summary(semantic)
    financial_summary = _financial_summary(financial)
    semantic_confusion = _semantic_confusion_table(semantic)
    examples = _select_examples(predictions, textual, semantic, financial)

    lines = [
        "# Reporte del experimento",
        "",
        "## Resumen",
        "",
        (
            f"Este experimento fine-tunea `{model['base_model']}` para generar narrativas financieras "
            f"de `{data['ticker']}` para el dia `t+1`, usando noticias y features de mercado de una "
            f"ventana previa de `{data['context_window_days']}` dias."
        ),
        "",
    ]
    if run_summary:
        lines.extend(
            [
                f"- Run: `{run_summary.get('run_id', 'unknown')}`",
                f"- Estado: `{run_summary.get('status', 'unknown')}`",
                f"- Duracion: `{run_summary.get('duration_seconds', 0):.1f}` segundos",
                "",
            ]
        )

    lines.extend(
        [
            "## Datos",
            "",
            f"- Dataset: `{data['dataset_name']}`",
            f"- Activo: `{data['ticker']}`",
            f"- Rango configurado: `{data['start_date']}` a `{data['end_date']}`",
            f"- Splits procesados: train={processed_counts.get('train', 0)}, val={processed_counts.get('val', 0)}, test={processed_counts.get('test', 0)}",
            f"- Predicciones evaluadas: {len(predictions)}",
            "",
            "## Metodo",
            "",
            "- Formato de entrada: ticker, fecha, retorno diario, volumen relativo, RSI y noticias previas.",
            "- Target: titulares/noticias del dia calendario siguiente.",
            "- Entrenamiento: causal language modeling con perdida enmascarada sobre el prompt.",
            f"- Modelo base: `{model['base_model']}`",
            f"- Epocas: `{model['train']['num_train_epochs']}`",
            f"- Learning rate: `{model['train']['learning_rate']}`",
            f"- Decoding: `do_sample={model['generation']['do_sample']}`, `max_new_tokens={model['generation']['max_new_tokens']}`",
            "",
            "## Resultados Textuales",
            "",
            _markdown_table(
                [
                    ("ROUGE-1", textual_summary.get("rouge1")),
                    ("ROUGE-2", textual_summary.get("rouge2")),
                    ("ROUGE-L", textual_summary.get("rougeL")),
                    ("BERTScore P", textual_summary.get("bertscore_precision")),
                    ("BERTScore R", textual_summary.get("bertscore_recall")),
                    ("BERTScore F1", textual_summary.get("bertscore_f1")),
                ],
                headers=("Metrica", "Media"),
            ),
            "",
            "Lectura: ROUGE bajo indica poca coincidencia literal con las noticias reales. BERTScore moderado sugiere cercania semantica general, probablemente influida por vocabulario financiero compartido.",
            "",
            "## Resultados Semanticos",
            "",
            _markdown_table(
                [
                    ("Sentiment match accuracy", semantic_summary.get("sentiment_match_accuracy")),
                    ("Neutral baseline accuracy", semantic_summary.get("neutral_baseline_accuracy")),
                    ("Mean KL divergence", semantic_summary.get("mean_kl_divergence")),
                    ("Net sentiment Pearson", semantic_summary.get("net_sentiment_pearson")),
                ],
                headers=("Metrica", "Valor"),
            ),
            "",
            "Matriz real vs generado segun FinBERT:",
            "",
            semantic_confusion,
            "",
            _image_link(report_dir, evaluation.get("semantic_confusion_matrix_path"), "Matriz de confusion semantica"),
            "",
            "Lectura: el modelo tiende a neutralizar el tono. La evaluacion semantica no supera el baseline trivial de predecir siempre neutral.",
            "",
            "## Resultados Financieros",
            "",
            _markdown_table(
                [
                    ("Filas joineadas", financial_summary.get("joined_rows")),
                    ("Directional accuracy generado", financial_summary.get("generated_directional_accuracy")),
                    ("Directional accuracy noticia real", financial_summary.get("real_news_directional_accuracy")),
                    ("Cobertura senal generada", financial_summary.get("generated_signal_coverage")),
                    ("Cobertura senal real", financial_summary.get("real_signal_coverage")),
                    ("Accuracy alta volatilidad", financial_summary.get("high_volatility_accuracy")),
                    ("Accuracy baja volatilidad", financial_summary.get("low_volatility_accuracy")),
                ],
                headers=("Metrica", "Valor"),
            ),
            "",
            _image_link(report_dir, evaluation.get("financial_confusion_matrix_path"), "Matriz de confusion financiera"),
            "",
            "Lectura: cuando el modelo genera una senal no neutral, la precision direccional es razonable, pero la cobertura es baja. Esto debe interpretarse como una senal selectiva, no como una estrategia completa.",
            "",
            "## Ejemplos Cualitativos",
            "",
            *examples,
            "## Limitaciones",
            "",
            "- GPT-2 small tiene capacidad limitada y tiende a generar titulares genericos.",
            "- El target usa titulares agregados por dia, no articulos completos curados.",
            "- FinBERT mide tono financiero, pero no garantiza causalidad ni prediccion de precio.",
            "- La metrica financiera usa una regla simple de sentimiento a long/short/hold.",
            "- La muestra de test es chica para concluir robustez estadistica.",
            "",
            "## Proximos Experimentos",
            "",
            "- Comparar contra mas epochs y contra GPT-2 medium si hay VRAM.",
            "- Probar prompts mas estructurados con menos titulares y features mas claras.",
            "- Ajustar decoding para reducir titulares clickbait/genericos.",
            "- Extender a varios ETFs/tickers cuando el pipeline este estable.",
            "- Agregar una tabla comparativa entre carpetas `runs/`.",
            "",
        ]
    )
    return "\n".join(lines)


def _select_examples(
    predictions: list[dict[str, Any]],
    textual: pd.DataFrame,
    semantic: pd.DataFrame,
    financial: pd.DataFrame,
) -> list[str]:
    if not predictions:
        return ["No hay predicciones disponibles.", ""]

    merged = pd.DataFrame(predictions)
    for frame in [textual, semantic, financial]:
        if not frame.empty:
            merged = merged.merge(frame, on=["date_t", "date_t1"], how="left", suffixes=("", "_metric"))

    examples = []
    picks = [
        ("Caso con buena similitud textual", _idxmax(merged, "rougeL")),
        ("Caso con sentimiento coincidente", _first_true(merged, "sentiment_match")),
        ("Caso con senal financiera correcta", _first_true(merged, "generated_direction_correct")),
    ]
    used: set[int] = set()
    for title, index in picks:
        if index is None or index in used:
            continue
        used.add(index)
        row = merged.iloc[index]
        examples.extend(_format_example(title, row))

    if not examples:
        examples.extend(_format_example("Ejemplo del test set", merged.iloc[0]))
    return examples


def _format_example(title: str, row: pd.Series) -> list[str]:
    return [
        f"### {title}",
        "",
        f"- Fecha target: `{row.get('date_t1', '')}`",
        f"- Sentimiento real/generado: `{row.get('real_label', 'n/a')}` / `{row.get('generated_label', 'n/a')}`",
        f"- Direccion real/senal generada: `{row.get('actual_direction', 'n/a')}` / `{row.get('generated_signal', 'n/a')}`",
        "",
        "**Noticia real:**",
        "",
        _quote_block(str(row.get("real_news", ""))),
        "",
        "**Noticia generada:**",
        "",
        _quote_block(str(row.get("generated_news", ""))),
        "",
    ]


def _quote_block(text: str, max_chars: int = 700) -> str:
    cleaned = text.replace("\r", "").strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    return "\n".join(f"> {line}" for line in cleaned.splitlines())


def _semantic_summary(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty:
        return {}
    pearson = 0.0
    if len(metrics) > 1:
        pearson = metrics["real_net_sentiment"].corr(metrics["generated_net_sentiment"])
    return {
        "sentiment_match_accuracy": float(metrics["sentiment_match"].mean()),
        "neutral_baseline_accuracy": float((metrics["real_label"] == "neutral").mean()),
        "mean_kl_divergence": float(metrics["kl_divergence"].mean()),
        "net_sentiment_pearson": 0.0 if pd.isna(pearson) else float(pearson),
    }


def _financial_summary(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty:
        return {}
    generated_active = metrics[metrics["generated_signal"] != 0]
    real_active = metrics[metrics["real_signal"] != 0]
    high = generated_active[generated_active["volatility_bucket"] == "high"]
    low = generated_active[generated_active["volatility_bucket"] == "low"]
    return {
        "joined_rows": float(len(metrics)),
        "generated_directional_accuracy": _mean_bool(generated_active, "generated_direction_correct"),
        "real_news_directional_accuracy": _mean_bool(real_active, "real_direction_correct"),
        "generated_signal_coverage": float(len(generated_active) / len(metrics)),
        "real_signal_coverage": float(len(real_active) / len(metrics)),
        "high_volatility_accuracy": _mean_bool(high, "generated_direction_correct"),
        "low_volatility_accuracy": _mean_bool(low, "generated_direction_correct"),
    }


def _semantic_confusion_table(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "No hay metricas semanticas disponibles."
    table = pd.crosstab(metrics["real_label"], metrics["generated_label"]).reindex(
        index=["negative", "neutral", "positive"],
        columns=["negative", "neutral", "positive"],
        fill_value=0,
    )
    rows = [("| Real \\ Generado | negative | neutral | positive |"), ("|---|---:|---:|---:|")]
    for label, values in table.iterrows():
        rows.append(f"| {label} | {values['negative']} | {values['neutral']} | {values['positive']} |")
    return "\n".join(rows)


def _markdown_table(rows: list[tuple[str, Any]], headers: tuple[str, str]) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---:|"]
    for name, value in rows:
        lines.append(f"| {name} | {_format_value(value)} |")
    return "\n".join(lines)


def _image_link(report_dir: Path, image_path: str | Path | None, alt: str) -> str:
    if not image_path:
        return ""
    image = Path(image_path)
    if image.exists():
        try:
            image = image.relative_to(report_dir)
        except ValueError:
            pass
    return f"![{alt}]({image.as_posix()})"


def _numeric_means(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    numeric = frame.select_dtypes(include="number")
    return {column: float(numeric[column].mean()) for column in numeric.columns}


def _processed_counts(processed_dir: str | Path) -> dict[str, int]:
    counts = {}
    for split in ["train", "val", "test"]:
        path = Path(processed_dir) / f"{split}.jsonl"
        if path.exists():
            counts[split] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return counts


def _read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    return [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_json_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _idxmax(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame or frame[column].dropna().empty:
        return None
    return int(frame[column].idxmax())


def _first_true(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame:
        return None
    matches = frame.index[frame[column] == True].tolist()
    return int(matches[0]) if matches else None


def _mean_bool(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(frame[column].mean())
