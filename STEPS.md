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

**Dataset principal: FNSPID (Financial News and Stock Price Integration Dataset)** o como alternativa más simple **Kaggle "Daily Financial News for 6000+ Stocks"** (también conocido como "US Financial News Articles").

**Por qué este:**

- Tiene noticias con timestamp exacto y ticker asociado
- Cobertura de varios años, lo que permite tener suficientes pares (input, target)
- Es el dataset de referencia más usado en papers recientes de NLP financiero

**Plan B si FNSPID es muy pesado:** usar `kaggle datasets download -d aaron7sun/stocknews` (Reddit/Reuters headlines + DJIA), que es más liviano (~30 MB) y suficiente para un proof-of-concept.

**Tickers a usar:** empezamos con un solo ticker (`AAPL` o `MSFT` — alta cobertura de noticias en inglés) para mantener el scope manejable. Si funciona, escalamos a un set de 5 tickers usando un token especial de condicionamiento.

---

## Paso a paso

Cada paso tiene: **objetivo**, **acciones**, **testing automatizado** y **testing manual**. Solamente se avanza al siguiente paso cuando los tests del paso actual pasan.

---

### Paso 0 — Setup del proyecto

**Objetivo:** dejar listo el esqueleto del repo con dependencias instaladas y configuración base.

**Acciones:**

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

**Objetivo:** descargar el dataset de noticias y dejarlo en `data/raw/news.csv` con columnas estándar: `date`, `ticker`, `headline`, `body` (opcional), `source`.

**Acciones:**

1. Implementar `src/data/download_news.py` con una función `download_news(dataset_name, output_path)` que:
   - Si el dataset es de Kaggle, use la API de Kaggle (`kaggle datasets download`)
   - Si es de Hugging Face, use `datasets.load_dataset()`
   - Normalice las columnas a un schema único: `date` (datetime), `ticker` (str), `headline` (str), `body` (str o vacío), `source` (str)
   - Filtre por el ticker definido en `config.yaml`
   - Ordene por fecha ascendente
   - Guarde como CSV en `data/raw/news.csv`
2. Crear `scripts/01_download_data.py` que llame a esta función

**Testing automatizado (`tests/test_data.py::test_news_download`):**

- Verificar que el CSV existe después de correr el script
- Verificar que tiene las columnas esperadas
- Verificar que `date` es parseable como datetime
- Verificar que no hay filas con `headline` vacío
- Verificar que el ticker filtrado coincide con `config.ticker`

**Testing manual:**

- Abrir `data/raw/news.csv` y leer 10 filas al azar — confirmar que las noticias son coherentes y están bien parseadas
- Verificar el rango de fechas: debería cubrir al menos 2-3 años
- Contar noticias por mes (`df.groupby(df.date.dt.to_period('M')).size()`) para detectar gaps temporales grandes

**Criterio de aceptación:** al menos 1000 noticias para el ticker elegido, distribuidas razonablemente en el tiempo.

---

### Paso 2 — Descarga de datos de mercado

**Objetivo:** descargar precio histórico del ticker desde `yfinance` y guardarlo en `data/raw/market.csv`.

**Acciones:**

1. Implementar `src/data/download_market.py` con una función `download_market(ticker, start_date, end_date, output_path)` que:
   - Use `yfinance.download(ticker, start, end)`
   - Calcule features adicionales: `return_1d` (retorno diario), `return_5d`, `volume_ratio` (volumen / promedio móvil 20d), `direction` (1 si return positivo, 0 si negativo)
   - Calcule indicadores técnicos básicos: RSI(14), MACD, SMA(20), SMA(50)
   - Guarde como CSV con index = fecha
2. Las fechas de inicio y fin deben coincidir con el rango cubierto por las noticias

**Testing automatizado (`tests/test_data.py::test_market_download`):**

- Verificar que el CSV existe
- Verificar columnas: `Open`, `High`, `Low`, `Close`, `Volume`, `return_1d`, `RSI`, `direction`
- Verificar que no hay NaN en columnas críticas (después de los primeros N días que necesitan los indicadores)
- Verificar que la cantidad de días hábiles es razonable (~252 por año)

**Testing manual:**

- Plotear `Close` con matplotlib y confirmar que se ve la serie temporal correcta
- Verificar a ojo un par de fechas específicas contra Yahoo Finance web

---

### Paso 3 — Construcción del dataset de entrenamiento

**Objetivo:** generar `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl` con pares `(prompt, completion)` listos para fine-tuning.

**Acciones:**

1. Implementar `src/data/build_dataset.py` con la función `build_dataset(news_df, market_df, k, split_ratios)` que:
   - Para cada día `t` con noticia disponible, arme un prompt con el formato:

     ```
     [TICKER: AAPL]
     [DATE: 2023-05-14]
     [PRICE_CHANGE: -1.2%]
     [VOLUME_RATIO: 1.4]
     [RSI: 38]

     [PREVIOUS NEWS]
     - 2023-05-12: Apple announces new chip...
     - 2023-05-13: Analysts upgrade AAPL...
     - 2023-05-14: Apple posts earnings...

     [NEXT DAY NEWS]
     ```

   - El completion sea la noticia (o noticias concatenadas) del día `t+1`
   - Usar ventana deslizante de `k` días de contexto (default `k=3`)
   - Hacer split temporal (no random): primeros 70% = train, siguientes 15% = val, últimos 15% = test
   - Guardar como JSONL con campos `prompt`, `completion`, `date_t`, `date_t1`

**Testing automatizado (`tests/test_dataset.py`):**

- Verificar que los tres archivos existen
- Verificar que cada línea es JSON válido con los campos esperados
- Verificar que el split es temporal: `max(train.date_t1) < min(val.date_t1) < min(test.date_t1)`
- Verificar que el prompt contiene el header del ticker, las features de mercado, y al menos una noticia previa
- Verificar que `completion` no está vacío
- Verificar que la longitud máxima de prompt + completion no excede el context window de GPT-2 (1024 tokens)

**Testing manual:**

- Imprimir 3 ejemplos al azar de `train.jsonl` y leerlos completos — confirmar que el formato es legible y tiene sentido
- Verificar el balance: ¿cuántos pares totales? ¿hay suficiente para entrenar? (mínimo deseable: 500 pares de train)
- Tokenizar 5 ejemplos con el tokenizer de GPT-2 y ver el conteo de tokens — la mayoría debe estar bajo 1024

---

### Paso 4 — Fine-tuning de GPT-2

**Objetivo:** entrenar GPT-2 sobre el dataset y guardar el modelo en `outputs/models/gpt2-financial-narrative/`.

**Acciones:**

1. Implementar `src/model/train.py` que:
   - Cargue `gpt2` (base) desde Hugging Face
   - Tokenice los datasets con el tokenizer de GPT-2 (agregar `pad_token = eos_token`)
   - Use el formato de causal LM: concatenar `prompt + completion` y entrenar a predecir el próximo token, pero **enmascarar la pérdida sobre el prompt** (loss solo sobre la completion). Esto se hace seteando `labels = -100` en las posiciones del prompt.
   - Use `transformers.Trainer` con `TrainingArguments`:
     - `num_train_epochs`: 3
     - `per_device_train_batch_size`: 2 (subir si hay VRAM)
     - `gradient_accumulation_steps`: 4
     - `learning_rate`: 5e-5
     - `warmup_steps`: 100
     - `evaluation_strategy`: "steps", `eval_steps`: 200
     - `save_strategy`: "steps", `save_steps`: 500
     - `fp16`: True si hay GPU compatible
     - `logging_dir`: `outputs/logs/`
   - Guarde el mejor checkpoint según loss de validación
2. Crear `scripts/03_train_model.py` que orqueste el entrenamiento

**Testing automatizado (`tests/test_model.py::test_training_smoke`):**

- Smoke test: correr 1 epoch sobre 10 ejemplos (subset) y verificar que:
  - El modelo se carga sin errores
  - La loss disminuye en los primeros pasos
  - Se generan checkpoints en disco
- Test de tokenización: verificar que el masking de labels está bien (las posiciones del prompt deben tener `-100`)

**Testing manual:**

- Mirar las curvas de train/val loss en TensorBoard o en los logs — debe haber descenso claro
- Verificar que la val loss no diverge de la train loss (señal de overfitting muy temprano)
- Después del entrenamiento, cargar el modelo y generar 3 ejemplos de inferencia greedy con prompts del val set — leer las salidas y confirmar que son coherentes (no random tokens)

**Criterio de aceptación:** val loss final < val loss inicial × 0.7, y las generaciones manuales son legibles como noticias financieras (aunque sean genéricas).

---

### Paso 5 — Generación de predicciones sobre el test set

**Objetivo:** generar la "noticia de t+1" para cada ejemplo del test set y guardarla en `outputs/generations/predictions.jsonl`.

**Acciones:**

1. Implementar `src/model/generate.py` con una función `generate_predictions(model_path, test_path, output_path, generation_config)`:
   - Cargar el modelo fine-tuneado
   - Para cada ejemplo del test set:
     - Tokenizar el prompt
     - Generar con `model.generate()` usando:
       - `max_new_tokens`: 150
       - `do_sample`: True
       - `temperature`: 0.8
       - `top_p`: 0.9
       - `repetition_penalty`: 1.2
       - `pad_token_id`: tokenizer.eos_token_id
     - Guardar el output decodificado, junto con `date_t`, `date_t1`, el prompt original, la completion real, y la generación
2. Crear `scripts/04_generate_predictions.py` que orqueste

**Testing automatizado (`tests/test_model.py::test_generation`):**

- Verificar que `predictions.jsonl` existe y tiene tantas líneas como el test set
- Verificar que cada línea tiene los campos: `date_t`, `date_t1`, `prompt`, `real_news`, `generated_news`
- Verificar que `generated_news` no está vacío y no es idéntico al prompt
- Verificar que la longitud media de `generated_news` está en un rango razonable (50-200 palabras)

**Testing manual:**

- Leer 5 generaciones al azar y compararlas lado a lado con la noticia real — anotar:
  - ¿El tono coincide (bullish vs bearish)?
  - ¿Menciona temas plausibles para ese ticker?
  - ¿Tiene errores obvios (repetición, incoherencia)?

---

### Paso 6 — Evaluación textual (ROUGE + BERTScore)

**Objetivo:** medir similitud entre noticia generada y noticia real, a nivel de superficie y a nivel semántico denso.

**Acciones:**

1. Implementar `src/evaluation/textual.py` con la función `evaluate_textual(predictions_path, output_path)` que:
   - Calcule ROUGE-1, ROUGE-2, ROUGE-L para cada par (real, generated)
   - Calcule BERTScore (precision, recall, F1) usando un modelo base como `roberta-large` o `distilbert-base-uncased`
   - Guarde un DataFrame con métricas por ejemplo en `outputs/reports/textual_metrics.csv`
   - Imprima medias agregadas en consola

**Testing automatizado (`tests/test_evaluation.py::test_textual`):**

- Verificar que el archivo de métricas se genera
- Verificar que todas las métricas están en `[0, 1]`
- Test sanity: pasar dos textos idénticos como input y verificar que ROUGE-1 = 1.0

**Testing manual:**

- Inspeccionar el ejemplo con ROUGE-L más alto y el más bajo — leer ambos pares (real vs generated) para entender qué está capturando la métrica
- Reportar la media y la desviación estándar de cada métrica en el informe final

---

### Paso 7 — Evaluación semántica (sentimiento con FinBERT)

**Objetivo:** medir si el sentimiento del texto generado coincide con el sentimiento del texto real.

**Acciones:**

1. Implementar `src/evaluation/semantic.py` con la función `evaluate_semantic(predictions_path, output_path)` que:
   - Cargue `ProsusAI/finbert` desde Hugging Face
   - Para cada par (real, generated), obtenga la distribución de sentimiento `{positive, negative, neutral}` para cada uno
   - Calcule:
     - **Sentiment match accuracy:** porcentaje de pares donde el argmax coincide
     - **KL divergence** entre distribuciones reales y generadas
     - **Pearson correlation** entre el "score net" (`positive - negative`) real y generado
   - Guarde resultados en `outputs/reports/semantic_metrics.csv`

**Testing automatizado (`tests/test_evaluation.py::test_semantic`):**

- Verificar que las distribuciones suman ~1.0
- Verificar que `sentiment_match_accuracy` está entre 0 y 1
- Test sanity: textos claramente positivos como "Stock surges 20% on record earnings" deben dar argmax = positive

**Testing manual:**

- Inspeccionar 5 casos donde el sentimiento NO coincide — entender por qué (¿el modelo es genérico? ¿hay ironía? ¿el tema cambió?)
- Comparar el accuracy contra un baseline trivial (siempre predecir "neutral"): si el modelo no supera al baseline, hay un problema

**Criterio de aceptación:** sentiment match accuracy > baseline neutral por al menos 5 puntos porcentuales.

---

### Paso 8 — Evaluación financiera (señal direccional)

**Objetivo:** validar si la noticia generada tiene poder predictivo sobre la dirección real del precio en `t+1`.

**Acciones:**

1. Implementar `src/evaluation/financial.py` con la función `evaluate_financial(predictions_path, market_path, output_path)` que:
   - Para cada predicción, derive una **señal** de la noticia generada:
     - Pasarla por FinBERT
     - Si argmax = positive → señal = +1 (long)
     - Si argmax = negative → señal = -1 (short)
     - Si neutral → señal = 0 (hold)
   - Joinearlo con el `direction` real del día `t+1` desde `market.csv`
   - Calcular:
     - **Directional accuracy** (excluyendo neutrales)
     - **Confusion matrix** (señal vs dirección real)
     - **Hit rate por subgrupo:** días con alta volatilidad vs baja volatilidad
     - **Comparación con dos baselines:**
       - Baseline 1: señal derivada de la noticia REAL (techo del modelo)
       - Baseline 2: señal random
   - Guardar en `outputs/reports/financial_metrics.csv` y un plot de confusion matrix

**Testing automatizado (`tests/test_evaluation.py::test_financial`):**

- Verificar que `directional_accuracy` está entre 0 y 1
- Verificar que el join entre predicciones y market data no pierde más del 5% de filas
- Test sanity: si pasamos las noticias REALES como "predicciones", la accuracy debe ser >= 50%

**Testing manual:**

- Leer la confusion matrix y entender los errores: ¿el modelo es muy bullish? ¿muy neutral?
- Comparar contra el baseline aleatorio (50%): si la accuracy es < 50%, podemos invertir la señal y reportar eso como hallazgo curioso
- Comparar contra el techo del modelo (señal de noticias reales): el gap nos dice cuánto se pierde por la generación vs cuánto por la limitación de FinBERT

---

### Paso 9 — Reporte final y notebook de presentación

**Objetivo:** consolidar resultados en un notebook reproducible y un informe markdown listos para presentar en la materia.

**Acciones:**

1. Crear `notebooks/exploration.ipynb` que:
   - Cargue todos los outputs
   - Tenga celdas con:
     - Distribución de fechas y tickers en el dataset
     - Curvas de loss del entrenamiento
     - Tabla resumen con todas las métricas (ROUGE, BERTScore, sentiment match, directional accuracy)
     - 3 ejemplos cualitativos: (a) un caso donde el modelo acertó tono y dirección, (b) un caso donde acertó tono pero falló dirección, (c) un caso donde falló todo
     - Confusion matrix de la señal vs dirección real
     - Discusión de limitaciones
2. Crear `outputs/reports/REPORT.md` con la estructura típica de paper corto:
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

Crear un script `scripts/run_all.sh` (o un `Makefile`) que ejecute en orden:

```bash
python scripts/01_download_data.py
python scripts/02_build_dataset.py
python scripts/03_train_model.py
python scripts/04_generate_predictions.py
python scripts/05_evaluate.py
```

Y un comando `make test` que corra `pytest tests/ -v` para validar todo.

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
