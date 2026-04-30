# Proyecto NLP — Generador de narrativas financieras futuras

## Contexto y objetivo

Este documento es la especificación completa para un proyecto de la materia de NLP. El objetivo es **fine-tunear GPT-2 (decoder-only)** localmente para generar la noticia financiera del día `t+1` dadas las noticias y datos de mercado de los días `t-k` hasta `t`. Después evaluamos si el modelo "entiende" la narrativa del mercado en tres niveles: textual, semántico y financiero.

**Hipótesis a validar:** un modelo de lenguaje fine-tuneado sobre narrativas financieras puede aprender patrones discursivos que correlacionen con movimientos de precio futuros.

**Restricciones técnicas:**

- Entrenamiento local (CPU o GPU modesta tipo RTX 3060 / 4060)
- GPT-2 small (124M params) o medium (355M) según VRAM disponible
- Pipeline reproducible end-to-end con un solo comando

---

## Stack técnico

- Python 3.10+
- PyTorch 2.x
- Hugging Face `transformers`, `datasets`, `accelerate`
- `yfinance` para datos de mercado
- `pandas`, `numpy` para manipulación
- `evaluate`, `rouge-score`, `bert-score` para métricas
- FinBERT (`ProsusAI/finbert`) para evaluación de sentimiento
- `pytest` para testing
- `python-dotenv` para configuración

---

## Estructura del proyecto

```
welcome-mr-nlp-forecasting/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_news.py
│   │   ├── download_market.py
│   │   └── build_dataset.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── generate.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── textual.py
│   │   ├── semantic.py
│   │   └── financial.py
│   └── utils/
│       ├── __init__.py
│       └── logging.py
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_build_dataset.py
│   ├── 03_train_model.py
│   ├── 04_generate_predictions.py
│   └── 05_evaluate.py
├── tests/
│   ├── test_data.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_evaluation.py
├── notebooks/
│   └── exploration.ipynb
└── outputs/
    ├── models/
    ├── generations/
    └── reports/
```

---

## Decisión de dataset

**Dataset principal: FNSPID (Financial News and Stock Price Integration Dataset)** desde Hugging Face (`beachside1234/FNSPID`).

**Activo elegido para el MVP: `SPY`.** El proyecto queda planteado como generación de narrativa de mercado asociada al S&P 500 ETF, no como noticia corporativa de una empresa puntual.

**Por qué este dataset/activo:**

- FNSPID tiene noticias con timestamp y ticker asociado.
- `SPY` tiene cobertura suficiente para un experimento end-to-end: 7196 noticias normalizadas entre `2017-08-18` y `2023-12-16`.
- El precio objetivo se obtiene con `yfinance` para el mismo activo (`SPY`), lo que mantiene alineadas noticias y mercado.
- Usar un solo activo mantiene el scope controlado y evita mezclar narrativas de empresas distintas.

**Decisión técnica:** para FNSPID no usamos streaming fila por fila, porque es lento para rangos chicos. El downloader lee los Parquet de `Stock_news/` con filtros por `Stock_symbol` y `Date`.

**Extensión futura:** cuando el pipeline funcione completo para `SPY`, se puede generalizar a varios tickers agregando una lista `tickers` en config y joineando mercado por `(ticker, date)`.

---

## Paso a paso

Cada paso tiene: **objetivo**, **acciones**, **testing automatizado** y **testing manual**. Solamente se avanza al siguiente paso cuando los tests del paso actual pasan.

---

### Paso 0 — Setup del proyecto

**Objetivo:** dejar listo el esqueleto del repo con dependencias instaladas y configuración base.

**Estado actual:** implementado y ejecutado.

**Acciones realizadas:**

1. Crear el directorio del proyecto y el árbol de carpetas según la estructura de arriba
2. Crear `requirements.txt` con todas las dependencias del stack
3. Crear `config/config.yaml` con los hiperparámetros configurables (ticker, modelo base, learning rate, batch size, context window k, max_new_tokens, etc.)
4. Crear `.env.example` para variables sensibles (tokens de API si se usan)
5. Crear `src/utils/logging.py` con un logger configurado que escriba a `outputs/logs/`
6. Crear un `Makefile` o un `scripts/run_all.sh` que orqueste todo el pipeline

**Testing automatizado (`tests/test_setup.py`):**

- Verificar que todas las carpetas existen
- Verificar que `config.yaml` se parsea correctamente y tiene las claves esperadas
- Verificar que el logger se inicializa sin errores

**Testing manual:**

- Correr `python -c "import torch; print(torch.cuda.is_available())"` y confirmar disponibilidad de GPU (si aplica)
- Correr `pytest tests/test_setup.py -v` y ver todo en verde

---

### Paso 1 — Descarga de noticias financieras

**Objetivo:** descargar noticias de FNSPID para `SPY` y dejarlas en `data/raw/news.csv` con columnas estándar: `date`, `ticker`, `headline`, `body`, `source`.

**Estado actual:** implementado y validado para `SPY`.

**Acciones realizadas:**

1. Implementar `src/data/download_news.py` con una función `download_news(dataset_name, output_path, ticker, start_date, end_date)` que:
   - Soporte el dataset `beachside1234/FNSPID` desde Hugging Face.
   - Lea los archivos Parquet de `Stock_news/` con filtros por `Stock_symbol` y `Date`.
   - Normalice las columnas de FNSPID (`Date`, `Article_title`, `Stock_symbol`, `Publisher`, `Article`) al schema único: `date`, `ticker`, `headline`, `body`, `source`.
   - Filtre por el ticker y rango definidos en `config/config.yaml`.
   - Elimine duplicados entre Parquets.
   - Ordene por fecha ascendente.
   - Guarde como CSV en `data/raw/news.csv`.
2. `scripts/01_download_data.py` llama a esta función usando la configuración actual:
   - `ticker: SPY`
   - `start_date: "2017-08-18"`
   - `end_date: "2023-12-16"`

**Testing automatizado (`tests/test_data.py`):**

- Verificar normalización de columnas al schema esperado.
- Verificar filtrado por ticker.
- Verificar filtrado por rango de fechas.
- Verificar deduplicación de filas repetidas.

**Testing manual realizado:**

- Correr `python -m scripts.01_download_data`.
- Verificar `data/raw/news.csv`.
- Resultado actual: 7196 noticias normalizadas de `SPY`, rango `2017-08-18` a `2023-12-16`.

**Criterio de aceptación:** cumplido. Hay más de 1000 noticias para `SPY`, distribuidas en varios años.

---

### Paso 2 — Descarga de datos de mercado

**Objetivo:** descargar precio histórico de `SPY` desde `yfinance` y guardarlo en `data/raw/market.csv`.

**Estado actual:** implementado y validado.

**Acciones realizadas:**

1. Implementar `src/data/download_market.py` con una función `download_market(ticker, start_date, end_date, output_path)` que:
   - Use `yfinance.download(ticker, start, end)`.
   - Calcule `return_1d`, `return_5d`, `volume_ratio`, `direction`.
   - Calcule indicadores técnicos básicos: RSI(14), MACD, SMA(20), SMA(50).
   - Guarde como CSV en `data/raw/market.csv`.
2. `scripts/01_download_data.py` descarga mercado usando el rango efectivo de noticias normalizadas.

**Testing automatizado (`tests/test_data.py`):**

- Verificar que las features de mercado se calculan correctamente sobre una serie sintética.
- Verificar columnas: `Open`, `High`, `Low`, `Close`, `Volume`, `return_1d`, `return_5d`, `volume_ratio`, `direction`, `RSI`, `MACD`, `SMA20`, `SMA50`.
- Verificar que los indicadores dejan de ser `NaN` después de las ventanas necesarias.

**Testing manual realizado:**

- Correr `python -m scripts.01_download_data`.
- Verificar `data/raw/market.csv`.
- Resultado actual: 1593 filas de mercado para `SPY`, rango `2017-08-18` a `2023-12-15`.

**Nota:** el rango de mercado termina en `2023-12-15` porque `2023-12-16` no fue día hábil.

---

### Paso 3 — Construcción del dataset de entrenamiento

**Objetivo:** generar `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl` con pares `(prompt, completion)` listos para fine-tuning.

**Estado actual:** implementado y validado.

**Acciones realizadas:**

1. Implementar `src/data/build_dataset.py` con la función `build_dataset(news_df, market_df, ticker, k, split_ratios, output_dir, ...)` que:
   - Agrupa noticias por día.
   - Para cada día `t`, arma un prompt con features de mercado y noticias previas de la ventana `t-k..t`.
   - Usa como completion las noticias del día calendario `t+1`, concatenadas como lista.
   - Usa titulares limpios como texto principal (`include_body: false`) para reducir boilerplate y ruido de artículos largos.
   - Normaliza caracteres raros a ASCII legible.
   - Limita el tamaño del texto para que GPT-2 pueda entrenar dentro de 1024 tokens:
     - `max_news_per_day: 3`
     - `max_text_chars: 180`
     - `max_completion_news: 3`
     - `include_body: false`
   - Hace split temporal, nunca random: 70% train, 15% val, 15% test.
   - Guarda JSONL con campos `prompt`, `completion`, `date_t`, `date_t1`.
2. Implementar `scripts/02_build_dataset.py` para leer `data/raw/news.csv` y `data/raw/market.csv`, construir los splits y guardarlos en `data/processed/`.

**Formato del prompt:**

     ```
     [TICKER: SPY]
     [DATE: 2023-05-14]
     [PRICE_CHANGE: -1.2%]
     [VOLUME_RATIO: 1.4]
     [RSI: 38]

     [PREVIOUS NEWS]
     - 2023-05-12: S&P 500 ETF sees renewed inflows...
     - 2023-05-13: Investors rotate into S&P 500 ETFs...
     - 2023-05-14: Broad market sentiment improves after inflation data...

     [NEXT DAY NEWS]
     ```

**Testing automatizado (`tests/test_dataset.py`):**

- Verificar que los tres archivos existen
- Verificar que cada línea es JSON válido con los campos esperados
- Verificar que el split es temporal: `max(train.date_t1) < min(val.date_t1) < min(test.date_t1)`
- Verificar que el prompt contiene el header del ticker, las features de mercado, y al menos una noticia previa
- Verificar que `completion` no está vacío
- Verificar que la longitud de caracteres queda dentro de un presupuesto conservador para GPT-2.

**Testing manual realizado:**

- Correr `python -m scripts.02_build_dataset`.
- Revisar un ejemplo de prompt/completion generado.
- Tokenizar todos los ejemplos con el tokenizer de GPT-2.
- Resultado de splits:
  - `train`: 616 ejemplos, `2017-08-22` a `2022-09-07`
  - `val`: 132 ejemplos, `2022-09-08` a `2023-05-09`
  - `test`: 133 ejemplos, `2023-05-10` a `2023-12-16`
- Verificación GPT-2 con dataset limpio:
  - Total: 881 ejemplos
  - Máximo: 390 tokens
  - Promedio: 257 tokens
  - Ejemplos sobre 1024 tokens: 0

**Criterio de aceptación:** cumplido. Hay más de 500 ejemplos de train, split temporal correcto y todos los ejemplos entran en el contexto de GPT-2.

---

### Paso 4 — Fine-tuning de GPT-2

**Objetivo:** entrenar GPT-2 sobre el dataset y guardar el modelo en `outputs/models/gpt2-financial-narrative/`.

**Estado actual:** implementado, entrenado y validado con una generación manual.

**Acciones realizadas:**

1. Implementar `src/model/train.py` que:
   - Cargue `gpt2` (base) desde Hugging Face
   - Tokenice los datasets con el tokenizer de GPT-2 (agregar `pad_token = eos_token`)
   - Use el formato de causal LM con **pérdida enmascarada sobre el prompt** (`labels = -100` en posiciones del prompt y padding).
   - Use `transformers.Trainer` con `TrainingArguments`:
     - `num_train_epochs`: 3
     - `per_device_train_batch_size`: 2 (subir si hay VRAM)
     - `gradient_accumulation_steps`: 4
     - `learning_rate`: 5e-5
     - `warmup_steps`: 20
     - `eval_strategy`: "steps", `eval_steps`: 50
     - `save_strategy`: "steps", `save_steps`: 50
     - `fp16`: True si hay GPU compatible
     - `save_total_limit`: 2
   - Guarde el mejor checkpoint según loss de validación
2. Crear `scripts/03_train_model.py` que orqueste el entrenamiento

**Testing automatizado (`tests/test_model.py`):**

- Test de tokenización: verificar que el masking de labels está bien (las posiciones del prompt deben tener `-100`)
- Test de dataset: verificar que cada item devuelve `input_ids`, `attention_mask` y `labels` como tensores listos para el modelo.

**Testing manual realizado:**

- Correr `python -m scripts.03_train_model`.
- Entrenamiento local con CUDA disponible.
- Duración aproximada: 51 minutos.
- Primer entrenamiento guardado en `outputs/models/gpt2-financial-narrative/`.
- Segundo entrenamiento limpio guardado en `outputs/models/gpt2-financial-narrative-clean/`.
- Dataset limpio: titulares solamente, sin `body`, con normalización de caracteres.
- Loss del segundo entrenamiento:
  - train loss final: `2.914`
  - eval loss inicial: `3.345`
  - eval loss final: `3.029`
- Generación greedy manual desde un prompt de validación produjo un titular legible, aunque todavía genérico.

**Criterio de aceptación:** cumplido parcialmente. La limpieza mejoró la estabilidad textual, pero GPT-2 small sigue generando titulares genéricos/clickbait con sampling. Para evaluación usaremos decoding determinístico o conservador.

---

### Paso 5 — Generación de predicciones sobre el test set

**Objetivo:** generar la "noticia de t+1" para cada ejemplo del test set y guardarla en `outputs/generations/predictions.jsonl`.

**Estado actual:** implementado y ejecutado sobre el test set.

**Acciones realizadas:**

1. Implementar `src/model/generate.py` con una función `generate_predictions(model_path, test_path, output_path, generation_config)`:
   - Cargar el modelo fine-tuneado
   - Para cada ejemplo del test set:
     - Tokenizar el prompt
     - Generar con `model.generate()` usando:
        - `min_new_tokens`: 30
        - `max_new_tokens`: 120
        - `do_sample`: false
        - `repetition_penalty`: 1.25
        - `pad_token_id`: tokenizer.eos_token_id
     - Guardar el output decodificado, junto con `date_t`, `date_t1`, el prompt original, la completion real, y la generación
2. Crear `scripts/04_generate_predictions.py` que orqueste

**Testing automatizado (`tests/test_model.py`):**

- Verificar que `predictions.jsonl` existe y tiene tantas líneas como el test set
- Verificar que cada línea tiene los campos: `date_t`, `date_t1`, `prompt`, `real_news`, `generated_news`
- Verificar que `generated_news` no está vacío y no es idéntico al prompt

**Testing manual realizado:**

- Correr `python -m scripts.04_generate_predictions`.
- Resultado actual:
  - `outputs/generations/predictions.jsonl`
  - 133 predicciones, una por ejemplo de test.
  - 0 generaciones vacías.
  - 0 generaciones idénticas al prompt.
  - Longitud media: 26.5 palabras.

**Nota cualitativa:** las predicciones son legibles, pero todavía tienden a ser genéricas/clickbait. Esto se reportará en el análisis cualitativo y se medirá en los pasos de evaluación.

---

### Paso 6 — Evaluación textual (ROUGE + BERTScore)

**Objetivo:** medir similitud entre noticia generada y noticia real, a nivel de superficie y a nivel semántico denso.

**Estado actual:** implementado y ejecutado.

**Acciones realizadas:**

1. Implementar `src/evaluation/textual.py` con la función `evaluate_textual(predictions_path, output_path)` que:
   - Calcule ROUGE-1, ROUGE-2, ROUGE-L para cada par (real, generated)
   - Calcule BERTScore (precision, recall, F1) usando `distilbert-base-uncased`
   - Guarde un DataFrame con métricas por ejemplo en `outputs/reports/textual_metrics.csv`
   - Imprima medias agregadas en consola
2. `scripts/05_evaluate.py` ejecuta la evaluación textual usando `outputs/generations/predictions.jsonl`.

**Testing automatizado (`tests/test_evaluation.py`):**

- Verificar que el archivo de métricas se genera
- Verificar que todas las métricas están en `[0, 1]`
- Test sanity: pasar dos textos idénticos como input y verificar que ROUGE-1 = 1.0

**Testing manual realizado:**

- Correr `python -m scripts.05_evaluate`.
- Resultado actual: `outputs/reports/textual_metrics.csv` con 133 filas.
- Medias:
  - ROUGE-1: `0.0960`
  - ROUGE-2: `0.0053`
  - ROUGE-L: `0.0682`
  - BERTScore precision: `0.7340`
  - BERTScore recall: `0.7380`
  - BERTScore F1: `0.7358`

**Lectura inicial:** ROUGE bajo confirma poca coincidencia superficial con las noticias reales; BERTScore moderado indica cierta cercanía semántica general, aunque probablemente influida por vocabulario financiero compartido.

---

### Paso 7 — Evaluación semántica (sentimiento con FinBERT)

**Objetivo:** medir si el sentimiento del texto generado coincide con el sentimiento del texto real.

**Estado actual:** implementado y ejecutado.

**Acciones realizadas:**

1. Implementar `src/evaluation/semantic.py` con la función `evaluate_semantic(predictions_path, output_path, model_name, batch_size)` que:
   - Carga `ProsusAI/finbert` desde Hugging Face.
   - Para cada par `(real_news, generated_news)`, obtiene la distribución de sentimiento `{negative, neutral, positive}`.
   - Usa el orden real de labels declarado por el modelo para evitar asumir el orden de logits.
   - Calcula por ejemplo:
     - label real y generado por `argmax`
     - `sentiment_match`
     - score neto `positive - negative`
     - KL divergence entre distribuciones real/generada
   - Guarda resultados en `outputs/reports/semantic_metrics.csv`.
2. Extender `scripts/05_evaluate.py` para correr evaluación textual y semántica en el mismo comando.
3. Agregar configuración:
   - `semantic_metrics_path: outputs/reports/semantic_metrics.csv`
   - `sentiment_batch_size: 8`

**Testing automatizado (`tests/test_evaluation.py`):**

- Verificar que las métricas semánticas se construyen correctamente desde probabilidades simuladas.
- Verificar que `sentiment_match_accuracy` queda entre 0 y 1.
- Verificar que `kl_divergence` es no negativa.
- Verificar que la correlación de sentimiento neto queda en el rango esperado.

**Testing manual realizado:**

- Correr `python -m scripts.05_evaluate`.
- Resultado actual: `outputs/reports/semantic_metrics.csv` con 133 filas.
- Resumen:
  - `sentiment_match_accuracy`: `0.3985`
  - `neutral_baseline_accuracy`: `0.4812`
  - `mean_kl_divergence`: `1.3610`
  - `net_sentiment_pearson`: `-0.0569`

**Lectura inicial:** el modelo no supera el baseline trivial de predecir siempre `neutral`. Esto es una señal importante para el reporte: las generaciones son legibles, pero no conservan de forma confiable el tono financiero de la noticia real.

**Criterio de aceptación:** no cumplido. El resultado se mantiene como hallazgo experimental y motiva el Paso 8 para medir si, aun con sentimiento débil, existe alguna señal direccional financiera.

---

### Paso 8 — Evaluación financiera (señal direccional)

**Objetivo:** validar si la noticia generada tiene poder predictivo sobre la dirección real del precio en `t+1`.

**Estado actual:** implementado y ejecutado.

**Acciones realizadas:**

1. Implementar `src/evaluation/financial.py` con la función `evaluate_financial(semantic_metrics_path, market_path, output_path, confusion_matrix_path)` que:
   - Reutiliza `semantic_metrics.csv` para no volver a correr FinBERT.
   - Deriva una **señal** desde `generated_label`:
     - `positive` -> señal = +1 (long)
     - `negative` -> señal = -1 (short)
     - `neutral` -> señal = 0 (hold)
   - Deriva el baseline techo desde `real_label` con la misma regla.
   - Joinea contra `market.csv` usando `date_t1` y la próxima rueda disponible con tolerancia de 3 días, para cubrir fines de semana y feriados.
   - Calcula:
     - **Directional accuracy** (excluyendo neutrales)
     - **Confusion matrix** (señal vs dirección real)
     - **Hit rate por subgrupo:** días con alta volatilidad vs baja volatilidad
     - **Baseline techo:** señal derivada de la noticia REAL
   - Guarda:
     - `outputs/reports/financial_metrics.csv`
     - `outputs/reports/financial_confusion_matrix.png`
2. Extender `scripts/05_evaluate.py` para correr evaluación textual, semántica y financiera en el mismo comando.
3. Agregar configuración:
   - `financial_metrics_path: outputs/reports/financial_metrics.csv`
   - `financial_confusion_matrix_path: outputs/reports/financial_confusion_matrix.png`

**Testing automatizado (`tests/test_evaluation.py`):**

- Verificar que el sentimiento se mapea correctamente a señal financiera.
- Verificar que el join con mercado usa la próxima rueda disponible cuando `date_t1` cae fuera de mercado.
- Verificar que `directional_accuracy` está entre 0 y 1.
- Verificar que la cobertura de señal se calcula correctamente excluyendo neutrales.

**Testing manual realizado:**

- Correr `python -m scripts.05_evaluate`.
- Resultado actual: `outputs/reports/financial_metrics.csv` con 132 filas joineadas sobre 133 predicciones.
- Resultado adicional: `outputs/reports/financial_confusion_matrix.png`.
- Resumen:
  - `generated_directional_accuracy`: `0.6250`
  - `real_news_directional_accuracy`: `0.6232`
  - `generated_signal_coverage`: `0.3636`
  - `real_signal_coverage`: `0.5227`
  - `high_volatility_accuracy`: `0.6522`
  - `low_volatility_accuracy`: `0.6000`

**Lectura inicial:** cuando GPT-2 genera una señal no neutral, acierta la dirección de la próxima rueda en 62.5% de los casos activos. Pero la cobertura es baja: solo genera señal long/short en 36.4% de los ejemplos. Esto sugiere una señal selectiva, no una estrategia completa.

**Criterio de aceptación:** cumplido como evaluación experimental. El resultado debe reportarse junto con su baja cobertura y compararse contra baselines en el informe final.

---

### Paso 9 — Reporte final y notebook de presentación

**Objetivo:** consolidar resultados en un notebook reproducible y un informe markdown listos para presentar en la materia.

**Estado actual:** reporte markdown implementado; notebook pendiente.

**Acciones:**

1. Implementar `src/reporting/report.py` y `scripts/06_generate_report.py` para generar `REPORT.md` desde:
   - `predictions.jsonl`
   - `textual_metrics.csv`
   - `semantic_metrics.csv`
   - `financial_metrics.csv`
   - matrices de confusión semántica y financiera
2. Integrar el reporte en `scripts/run_pipeline.py`, de modo que cada carpeta `runs/<run_id>/` termine con `reports/REPORT.md`.
3. Crear `notebooks/exploration.ipynb` que:
   - Cargue todos los outputs
   - Tenga celdas con:
     - Distribución de fechas y tickers en el dataset
     - Curvas de loss del entrenamiento
     - Tabla resumen con todas las métricas (ROUGE, BERTScore, sentiment match, directional accuracy)
     - 3 ejemplos cualitativos: (a) un caso donde el modelo acertó tono y dirección, (b) un caso donde acertó tono pero falló dirección, (c) un caso donde falló todo
     - Confusion matrix de la señal vs dirección real
     - Discusión de limitaciones
4. Crear `outputs/reports/REPORT.md` con la estructura típica de paper corto:
   - Introducción (motivación + hipótesis)
   - Datos (qué dataset, qué ticker, qué rango)
   - Método (arquitectura, formato de input, objetivo de entrenamiento)
   - Experimentos (hiperparámetros, hardware, tiempo de entrenamiento)
   - Resultados (las tres tablas de métricas)
   - Análisis cualitativo (los 3 ejemplos)
   - Limitaciones y trabajo futuro
   - Referencias

**Testing manual:**

- Correr el notebook completo de cero y verificar que todas las celdas se ejecutan sin errores
- Leer el `REPORT.md` como si fuera el evaluador de la materia: ¿se entiende qué se hizo y qué se encontró?

---

## Comando único para correr todo

**Estado actual:** implementado con runs versionados.

El comando recomendado es:

```bash
make run
```

Internamente ejecuta:

```bash
python -m scripts.run_pipeline
```

Cada ejecución crea una carpeta nueva en `runs/` con fecha/hora, ticker y modelo:

```text
runs/YYYY-MM-DD_HH-MM-SS_spy_gpt2/
  config.yaml
  run_summary.json
  logs/pipeline.log
  data/raw/news.csv
  data/raw/market.csv
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl
  models/gpt2-financial-narrative-clean/
  generations/predictions.jsonl
  reports/textual_metrics.csv
  reports/semantic_metrics.csv
  reports/financial_metrics.csv
  reports/semantic_confusion_matrix.png
  reports/financial_confusion_matrix.png
```

Para nombrar una ejecución experimental:

```bash
python -m scripts.run_pipeline --name more-epochs
```

También queda disponible el comando legacy:

```bash
make run-all
```

Ese flujo legacy ejecuta los scripts individuales en orden y escribe en los paths fijos de `config/config.yaml`.

El comando de validación es:

```bash
make test
```

---

## Riesgos identificados y mitigaciones

1. **El modelo genera texto genérico** ("markets remained volatile"). Mitigación: condicionar fuertemente con features numéricas en el prompt y usar `repetition_penalty`. Si persiste, aumentar epochs o subir a GPT-2 medium.

2. **Dataset desbalanceado** (días sin noticias o con muchas). Mitigación: en `build_dataset.py`, si un día tiene múltiples noticias, concatenarlas con un separador. Si no tiene noticia, omitir ese día como target.

3. **Overfitting** con dataset chico. Mitigación: early stopping basado en val loss + dropout adicional si hace falta.

4. **FinBERT no captura el matiz** de noticias muy técnicas. Mitigación: reportar el techo del modelo (señal de noticias reales pasada por FinBERT) como referencia. El gap real-vs-generado es lo que importa.

5. **Leakage temporal.** Mitigación: split estrictamente temporal, nunca random. Documentar las fechas exactas de cada split.

---

## Entregable final esperado

- Repo público o zip con todo el código
- Modelo entrenado (puede ser link a HF Hub o checkpoint local)
- Notebook ejecutable end-to-end
- `REPORT.md` de 4-6 páginas equivalentes
- Presentación oral de 10-15 minutos con foco en: (1) por qué el approach es novedoso, (2) qué dijeron las métricas, (3) los 3 ejemplos cualitativos
