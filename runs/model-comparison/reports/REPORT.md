# Reporte global de comparacion

## Objetivo

Este reporte compara los principales experimentos de fine-tuning de GPT-2 para generacion de narrativas financieras. La lectura central es que las metricas textuales, semanticas y financieras capturan propiedades distintas del sistema, por lo que no conviene resumir el proyecto con una unica metrica.

## Runs recomendados para la presentacion

| Run | Tickers | Train | Test | ROUGE-L | BERTScore F1 | Sent. match | Neutral base | Dir. acc. | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY/QQQ baseline | SPY,QQQ | 1410 | 303 | 0.0615 | 0.7318 | 0.3861 | 0.4950 | 0.5282 | 0.4718 |
| GPT-2 no fine-tune | NVDA,AMD | 1517 | 326 | 0.0402 |  | 0.2730 | 0.5828 | 0.5494 | 1.0000 |
| NVDA single | NVDA | 618 | 133 | 0.0697 | 0.7438 | 0.3459 | 0.4211 | 0.4615 | 0.5909 |
| NVDA + AMD | NVDA,AMD | 1517 | 326 | 0.0736 | 0.7383 | 0.4417 | 0.5368 | 0.5299 | 0.3611 |
| NVDA direction features | NVDA | 617 | 133 | 0.0634 | 0.7337 | 0.3459 | 0.4211 | 0.5564 | 1.0000 |

Lectura:

- `SPY/QQQ baseline` sirve como punto de partida del pipeline completo.
- `GPT-2 no fine-tune` aisla cuanto aporta el entrenamiento especifico frente al modelo base.
- `NVDA single` muestra el mejor BERTScore F1, pero no el mejor resultado financiero.
- `NVDA + AMD` es el run mas equilibrado para defender como modelo general.
- `NVDA direction features` es el mejor en directional accuracy, aunque no prueba predictividad robusta.

## Graficos

![Comparacion de modelos](model_comparison.png)

![Training loss](training_loss_curves.png)

![Validation loss](eval_loss_curves.png)

## Perplexity

La perplexity se calcula como `exp(eval_loss)`. Sirve para interpretar la perdida de language modeling, pero no reemplaza las metricas downstream.

| Run | Best step | Best eval loss | Best ppl | Final eval loss | Final ppl |
| --- | --- | --- | --- | --- | --- |
| SPY/QQQ baseline | 531.0000 | 2.6850 | 14.6586 | 2.6850 | 14.6586 |
|  |  |  |  |  |  |
| NVDA single | 234.0000 | 3.3894 | 29.6482 | 3.3894 | 29.6482 |
| NVDA + AMD | 570.0000 | 2.9597 | 19.2916 | 2.9597 | 19.2916 |
| NVDA direction features | 234.0000 | 2.8033 | 16.4998 | 2.8033 | 16.4998 |

Lectura:

- El mejor valor de perplexity no coincide necesariamente con el mejor resultado financiero. Esto es una conclusion importante: optimizar la perdida de lenguaje no garantiza optimizar la utilidad de la narrativa como senal.

## Significancia de directional accuracy

Para la directional accuracy se calcula un test binomial bilateral contra azar (`p = 0.5`) usando solo los casos donde el modelo emite una senal activa. Tambien se reporta un intervalo de confianza de Wilson al 95%.

| Run | N activo | Correctas | Dir. acc. | Coverage | p vs 0.5 | CI 95 low | CI 95 high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPY/QQQ baseline | 142 | 75 | 0.5282 | 0.4718 | 0.5571 | 0.4464 | 0.6084 |
| GPT-2 no fine-tune | 324 | 178 | 0.5494 | 1.0000 | 0.0849 | 0.4949 | 0.6027 |
| NVDA single | 78 | 36 | 0.4615 | 0.5909 | 0.5716 | 0.3553 | 0.5714 |
| NVDA + AMD | 117 | 62 | 0.5299 | 0.3611 | 0.5793 | 0.4400 | 0.6180 |
| NVDA direction features | 133 | 74 | 0.5564 | 1.0000 | 0.2246 | 0.4716 | 0.6381 |

Lectura:

- El baseline sin fine-tuning tiene ROUGE-L bajo y sentiment match pobre, pero obtiene una directional accuracy alta. Esto sugiere que la metrica financiera puede capturar sesgos de la regla de etiquetado o de la distribucion del test, no solo calidad generativa.
- Ninguno de los runs principales alcanza evidencia estadistica fuerte contra azar al nivel `p < 0.05`. Por eso la conclusion correcta es que hay senales exploratorias, no predictividad robusta demostrada.
- Coverage y accuracy deben leerse juntos: una senal selectiva no equivale a una senal emitida siempre.

## Hallazgos principales

- Mejor BERTScore F1: `NVDA single = 0.7438`.
- Mejor sentiment match: `NVDA + AMD = 0.4417`.
- Mejor directional accuracy: `NVDA direction features = 0.5564`.
- Mejor perplexity: `NVDA tagged structured = 9.7271`.

## Conclusiones para defender

1. El modelo aprende rasgos del dominio financiero: vocabulario, estilo de titulares y patrones generales de narrativa.
2. La similitud textual o semantica no alcanza para afirmar utilidad financiera. El run con mejor BERTScore no es el mejor en directional accuracy.
3. El baseline sin fine-tuning obliga a ser prudentes: una directional accuracy alta puede aparecer aun con baja calidad semantica, por lo que la metrica financiera debe interpretarse con cuidado.
4. La evaluacion semantica con FinBERT muestra una limitacion clara: varios modelos no superan el baseline neutral.
5. La directional accuracy tiene algunos resultados por encima de 0.5, pero sin significancia estadistica fuerte. Debe presentarse como analisis exploratorio.
6. El aporte mas solido del proyecto es metodologico: un pipeline reproducible de NLP generativo con evaluacion textual, semantica y downstream.

## Proximos pasos antes de slides

- Revisar `qualitative_examples.md` para elegir los ejemplos finales de slides.
- Elegir una matriz de confusion semantica y una financiera para la presentacion.
- Convertir este reporte en 8-10 slides.

## Reportes auxiliares

- [Ejemplos cualitativos](qualitative_examples.md)
- [Tabla compacta para slides](slides_model_table_compact.md)
- [Tabla completa para slides](slides_model_table.md)
- [Tabla completa de comparacion](model_comparison.csv)
- [Significancia direccional](directional_significance.csv)
- [Perplexity](perplexity_summary.csv)
- [Matriz semantica GPT-2 no fine-tune](selected_confusion_matrices/semantic_baseline_gpt2_no_finetune.png)
- [Matriz semantica fine-tuned NVDA + AMD](selected_confusion_matrices/semantic_finetuned_nvda_amd.png)
- [Matriz financiera direction features](selected_confusion_matrices/financial_direction_features.png)
