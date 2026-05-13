# news-ai-generator

Proyecto de NLP para fine-tunear GPT-2 en narrativas financieras de mercado. La
configuracion inicial usa noticias asociadas a `SPY` y `QQQ` en FNSPID y precios
de esos ETFs para generar un outlook estructurado del dia `t+1` usando una
ventana previa `t-k..t`.

El completion entrenado tiene este formato:

```text
 positive|neutral|negative
Direction: up|flat|down
News:
- ...
```

Durante inferencia, `Sentiment` y `Direction` se eligen por scoring entre
opciones cerradas; la narrativa se genera despues condicionada por esas labels.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

## Pipeline completo

```bash
make run
```

Ese comando crea una carpeta nueva en `runs/` con fecha, ticker y modelo. Por
ejemplo:

```text
runs/2026-04-30_15-30-12_spy-qqq_gpt2/
  config.yaml
  run_summary.json
  stage_timings.csv
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

`run_summary.json` guarda metricas agregadas y `stage_timings.csv` guarda el
tiempo de cada etapa del pipeline para comparar experimentos.

Para nombrar una ejecucion:

```bash
python -m scripts.run_pipeline --name more-epochs
```

Para correr el experimento de NVDA solamente:

```bash
python -m scripts.run_pipeline --config config/experiments/nvda.yaml --name nvda-single
```

Para correr el experimento de semiconductores `NVDA + AMD`:

```bash
python -m scripts.run_pipeline --config config/experiments/nvda-amd.yaml --name nvda-amd
```

Para correr el experimento de semiconductores `NVDA + AMD + AVGO`:

```bash
python -m scripts.run_pipeline --config config/experiments/nvda-amd-avgo.yaml --name nvda-amd-avgo
```

Para regenerar solo el reporte con los outputs actuales:

```bash
make report
```

El flujo legacy sigue disponible con `make run-all`, pero escribe en los paths
fijos definidos en `config/config.yaml`.
