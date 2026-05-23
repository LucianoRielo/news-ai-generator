# Outline de slides - Defensa NLP

## Slide 1 - Titulo

**Titulo:** Fine-tuning de GPT-2 para generacion de narrativas financieras

**Mostrar:**

- Nombre del proyecto.
- Materia: NLP.
- Modelo base: GPT-2.
- Dominio: noticias financieras + datos de mercado.

**Decir:**

> El objetivo no es presentar un sistema de trading, sino estudiar un problema de NLP aplicado: dado contexto financiero previo, fine-tuneamos un modelo generativo para producir una narrativa del dia siguiente y evaluamos que tan util o coherente resulta.

## Slide 2 - Pregunta del proyecto

**Titulo:** Pregunta experimental

**Mostrar:**

- Puede GPT-2 generar narrativas financieras plausibles?
- El fine-tuning mejora frente a GPT-2 base?
- La narrativa generada tiene consistencia semantica?
- La senal generada se relaciona con la direccion real del activo?

**Decir:**

> La pregunta central tiene tres capas: calidad textual, alineacion semantica y utilidad downstream. La parte interesante es que esas tres capas no necesariamente coinciden.

## Slide 3 - Conexion con NLP

**Titulo:** Conceptos de NLP usados

**Mostrar:**

- Transfer learning.
- Language modeling causal.
- Tokenizacion subword.
- Embeddings contextuales.
- Fine-tuning.
- Evaluacion generativa.

**Decir:**

> Usamos GPT-2 como modelo preentrenado y lo adaptamos al dominio financiero. El entrenamiento sigue siendo causal language modeling: el modelo aprende a predecir los siguientes tokens, pero condicionado por prompts con ticker, fecha, features de mercado y noticias previas.

## Slide 4 - Pipeline

**Titulo:** Pipeline experimental

**Mostrar:**

1. Dataset FNSPID + mercado.
2. Construccion de prompts.
3. Split temporal train/val/test.
4. Fine-tuning GPT-2.
5. Generacion sobre test.
6. Evaluacion textual, semantica y financiera.

**Decir:**

> El split temporal es importante porque evita mezclar futuro con pasado. Para cada dia se arma una ventana de contexto y el target es la narrativa o outlook del dia siguiente.

## Slide 5 - Formato del problema

**Titulo:** Prompt y target

**Mostrar:**

Ejemplo abreviado de prompt:

```text
[TICKER: AMD]
[DATE: 2023-04-18]
[PRICE_CHANGE: -0.10%]
[VOLUME_RATIO: 0.80]
[RSI: 36.50]

[PREVIOUS NEWS]
- ...

[NEXT DAY NEWS]
```

**Decir:**

> La tarea se formula como prompt/completion. En algunos runs el target es solo texto de noticias; en otros agregamos estructura explicita de sentimiento y direccion. Eso permite comparar si la estructura ayuda o no.

## Slide 6 - Runs principales

**Titulo:** Experimentos comparados

**Mostrar:** tabla desde `runs/model-comparison/reports/REPORT.md`.

Runs:

- SPY/QQQ baseline.
- GPT-2 no fine-tune.
- NVDA single.
- NVDA + AMD.
- NVDA direction features.

**Decir:**

> No todos los runs responden la misma pregunta. El baseline sin fine-tuning mide cuanto aporta entrenar. NVDA + AMD es el run mas equilibrado. Direction features es el run mas orientado a maximizar la metrica financiera.

## Slide 7 - Metricas

**Titulo:** Evaluacion multicapa

**Mostrar:**

| Capa | Metrica | Que mide |
|---|---|---|
| Textual | ROUGE-L | Solapamiento secuencial |
| Semantica | BERTScore / FinBERT | Cercania semantica y sentimiento |
| Financiera | Directional accuracy + coverage | Relacion con direccion real |
| Estadistica | p-value binomial | Si supera azar con evidencia |

**Decir:**

> En generacion no alcanza una metrica. ROUGE puede ser bajo aunque el texto sea plausible. BERTScore captura mas semantica. FinBERT permite evaluar tono financiero. Y la metrica financiera es downstream, pero debe interpretarse con coverage y significancia.

## Slide 8 - Resultados agregados

**Titulo:** Las metricas no se mueven juntas

**Mostrar:**

- `runs/model-comparison/reports/model_comparison.png`
- `runs/model-comparison/reports/slides_model_table_compact.md`
- Tabla compacta:
  - GPT-2 no fine-tune: ROUGE-L `0.0402`, sentiment match `0.2730`, dir. acc. `0.5494`.
  - NVDA + AMD: ROUGE-L `0.0736`, sentiment match `0.4417`, dir. acc. `0.5299`.
  - Direction features: ROUGE-L `0.0634`, sentiment match `0.3459`, dir. acc. `0.5564`.

**Decir:**

> El fine-tuned NVDA + AMD mejora claramente en calidad textual y semantica frente a GPT-2 base. Pero GPT-2 base tiene una directional accuracy alta. Eso indica que la metrica financiera puede estar capturando sesgos del test o de la regla de etiquetado, no necesariamente comprension financiera.

## Slide 9 - Loss y perplexity

**Titulo:** Entrenamiento y language modeling

**Mostrar:**

- `runs/model-comparison/reports/training_loss_curves.png`
- `runs/model-comparison/reports/eval_loss_curves.png`
- `runs/model-comparison/reports/perplexity_summary.csv`

**Decir:**

> La loss permite verificar que el fine-tuning efectivamente ajusta el modelo al corpus. Pero el run con mejor perplexity no es necesariamente el mejor en directional accuracy. Esto separa optimizacion de lenguaje de utilidad downstream.

## Slide 10 - Confusion matrices semanticas

**Titulo:** GPT-2 base vs fine-tuned

**Mostrar:**

- `runs/model-comparison/reports/selected_confusion_matrices/semantic_baseline_gpt2_no_finetune.png`
- `runs/model-comparison/reports/selected_confusion_matrices/semantic_finetuned_nvda_amd.png`

**Decir:**

> FinBERT muestra que el baseline sin fine-tuning tiene bajo match de sentimiento. El fine-tuned mejora, pero aun no supera el baseline neutral. Esto es una limitacion importante: el modelo genera lenguaje financiero, pero no controla bien el tono respecto de la noticia real.

## Slide 11 - Matriz financiera

**Titulo:** Senal direccional

**Mostrar:**

- `runs/model-comparison/reports/selected_confusion_matrices/financial_direction_features.png`
- Fragmento de `directional_significance.csv`.

**Decir:**

> El run con direction features tiene la mejor directional accuracy entre los runs principales y coverage total. Sin embargo, el test binomial no da evidencia fuerte contra azar. Por eso lo presentamos como senal exploratoria, no como capacidad predictiva demostrada.

## Slide 12 - Ejemplos cualitativos

**Titulo:** Mirar ejemplos cambia la lectura

**Mostrar:**

- `runs/model-comparison/reports/qualitative_examples.md`
- Elegir dos:
  - "Fine-tuning improves semantic alignment".
  - "Financial metric can be misleading".

**Decir:**

> En un ejemplo, el fine-tuning mejora ROUGE-L y sentimiento. En otro, GPT-2 base acierta direccion pero genera una narrativa pobre. Eso justifica por que necesitamos evaluacion multicapa.

## Slide 13 - Limitaciones

**Titulo:** Limitaciones

**Mostrar:**

- Dataset chico.
- Titulares agregados, no articulos completos.
- GPT-2 small.
- FinBERT como evaluador externo imperfecto.
- Directional accuracy sensible a distribucion de clases.
- No hay significancia estadistica fuerte.

**Decir:**

> La conclusion no es que el modelo predice precios. La conclusion es que el pipeline permite medir donde el modelo mejora y donde no. La parte financiera queda como analisis exploratorio.

## Slide 14 - Conclusiones

**Titulo:** Conclusiones

**Mostrar:**

1. GPT-2 fine-tuneado aprende vocabulario y formato financiero.
2. El fine-tuning mejora calidad textual/semantica frente a GPT-2 base.
3. Las metricas textuales, semanticas y financieras no son equivalentes.
4. La directional accuracy no es concluyente sin significancia y coverage.
5. El aporte principal es el pipeline reproducible y la evaluacion critica.

**Decir:**

> El aprendizaje principal del proyecto es metodologico. En NLP aplicado, especialmente con generacion, una sola metrica puede ser enganosa. La evaluacion tiene que combinar calidad de texto, semantica y comportamiento downstream.

## Slide 15 - Cierre

**Titulo:** Proximos pasos

**Mostrar:**

- Mejor baseline estadistico de direccion.
- Mas datos y otros tickers.
- GPT-2 medium o modelo financiero especializado.
- Mejor control de estructura de salida.
- Evaluacion humana de plausibilidad.
- Analisis de atencion como interpretabilidad exploratoria.

**Decir:**

> El siguiente paso seria separar mejor calidad generativa de senal financiera: agregar baselines estadisticos, evaluar robustez por ticker y mejorar la estructura del target.
