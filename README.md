# news-ai-generator

Proyecto de NLP para fine-tunear GPT-2 en narrativas financieras de mercado. La
configuracion inicial usa noticias asociadas a `SPY` en FNSPID y precios de SPY
para generar la narrativa del dia `t+1` usando una ventana previa `t-k..t`.

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
runs/2026-04-30_15-30-12_spy_gpt2/
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

Para nombrar una ejecucion:

```bash
python -m scripts.run_pipeline --name more-epochs
```

Para regenerar solo el reporte con los outputs actuales:

```bash
make report
```

El flujo legacy sigue disponible con `make run-all`, pero escribe en los paths
fijos definidos en `config/config.yaml`.
