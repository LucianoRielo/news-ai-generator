# Analisis de runs sin usar el run binario

## Decision principal

Se descarta el run:

`2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features`

Motivo: aunque tiene la mejor directional accuracy entre los fine-tuned, introduce un cambio fuerte de formulacion: features binarias orientadas directamente a direccion. Para una defensa de NLP, puede contaminar la narrativa porque parece optimizar la metrica financiera mas que estudiar la generacion de texto financiero.

La comparacion queda mas limpia si usamos una progresion:

1. GPT-2 sin fine-tuning.
2. Fine-tuning en SPY/QQQ, textos mas neutrales y de mercado amplio.
3. Fine-tuning en NVDA single.
4. Mejor variante NVDA estructurada/label-aware.
5. Fine-tuning multi-ticker NVDA + AMD.

Esta seleccion permite discutir aprendizaje de dominio, especializacion por activo, estructura del output y generalizacion a mas de un ticker.

## Runs seleccionados

| Rol | Run |
|---|---|
| Modelo base | `baseline-gpt2-no-finetune_nvda-amd` |
| Dominio amplio/neutral | `2026-04-30_15-49-24_spy-qqq_gpt2_spy-qqq-baseline` |
| Especializacion en NVDA | `2026-05-02_13-05-30_nvda_gpt2_nvda-single` |
| Mejor NVDA estructurado/label-aware | `2026-05-04_09-09-04_nvda_gpt2_nvda-label-scoring` |
| Multi-ticker semiconductor | `2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2` |

## Tabla comparativa

| Run | Tickers | Train | Test | ROUGE-L | BERT F1 | Sent. match | Dir. acc. | Coverage | p-value | Best ppl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 no fine-tune | NVDA,AMD | 1517 | 326 | 0.040 | n/a | 0.273 | 0.549 | 1.000 | 0.085 | n/a |
| SPY/QQQ fine-tuned | SPY,QQQ | 1410 | 303 | 0.062 | 0.732 | 0.386 | 0.528 | 0.472 | 0.557 | 14.659 |
| NVDA single | NVDA | 618 | 133 | 0.070 | 0.744 | 0.346 | 0.462 | 0.591 | 0.572 | 29.648 |
| NVDA structured/label scoring | NVDA | 617 | 133 | 0.066 | 0.740 | 0.376 | 0.496 | 1.000 | 1.000 | 16.567 |
| NVDA + AMD | NVDA,AMD | 1517 | 326 | 0.074 | 0.738 | 0.442 | 0.530 | 0.361 | 0.579 | 19.292 |

## Es correcta la comparacion propuesta?

Si, es una comparacion mas defendible que la anterior, con una salvedad:

> No conviene venderla como ranking de modelos, sino como progresion experimental.

La progresion cuenta una historia clara:

- GPT-2 base genera texto financiero generico, pero con baja calidad semantica.
- SPY/QQQ muestra que el fine-tuning en un dominio mas neutral mejora la adaptacion textual/semantica.
- NVDA single mejora BERTScore, pero no la metrica financiera.
- NVDA label scoring muestra que estructurar/scoring ayuda a coverage y mantiene buena calidad textual.
- NVDA + AMD es el mejor balance general de calidad textual/semantica.

## Por que elegir `NVDA label scoring` como mejor variante estructurada

Hay tres candidatos relacionados con estructura:

| Run | ROUGE-L | BERT F1 | Sent. match | Dir. acc. | Coverage | Best ppl |
|---|---:|---:|---:|---:|---:|---:|
| NVDA structured prompt | 0.035 | n/a | 0.308 | 0.400 | 0.602 | 13.811 |
| NVDA tagged structured prompt | 0.042 | 0.687 | 0.383 | 0.483 | 0.436 | 9.727 |
| NVDA label scoring | 0.066 | 0.740 | 0.376 | 0.496 | 1.000 | 16.567 |

`Tagged structured prompt` tiene la mejor perplexity (`9.727`) y levemente mejor sentiment match (`0.383`) que label scoring (`0.376`), pero cae mucho en ROUGE-L y BERTScore.

Desde criterio de ML aplicado, `NVDA label scoring` es mejor para presentar porque:

- mantiene ROUGE-L alto para NVDA (`0.066`);
- mantiene BERTScore alto (`0.740`);
- mejora coverage a `1.000`;
- mejora directional accuracy frente a NVDA single (`0.496` vs `0.462`);
- evita el cambio mas agresivo del run binario.

Conclusion:

> Si el objetivo es mostrar una variante estructurada razonable sin usar el binario, `NVDA label scoring` es el mejor candidato.

## Lectura por eje

### 1. Fine-tuning vs modelo base

Comparacion:

- GPT-2 no fine-tune: ROUGE-L `0.040`, sentiment match `0.273`.
- NVDA + AMD: ROUGE-L `0.074`, sentiment match `0.442`.

Lectura:

> El fine-tuning mejora calidad textual y semantica.

Pero:

- GPT-2 base tiene directional accuracy `0.549`.
- NVDA + AMD tiene directional accuracy `0.530`.

Lectura:

> Directional accuracy puede estar capturando sesgos de distribucion o de la regla de etiquetado, no comprension financiera.

### 2. SPY/QQQ como dominio mas neutral

SPY/QQQ:

- ROUGE-L `0.062`.
- BERTScore F1 `0.732`.
- Sentiment match `0.386`.
- Directional accuracy `0.528`.
- Coverage `0.472`.

Lectura:

> Los ETFs amplios producen un baseline fine-tuned razonable y mas neutral, pero no dominan ninguna metrica.

Esto sirve para mostrar que el pipeline funciona fuera de NVDA/AMD.

### 3. NVDA single

NVDA single:

- BERTScore F1 `0.744`, el mas alto de la seleccion.
- ROUGE-L `0.070`.
- Sentiment match `0.346`.
- Directional accuracy `0.462`.

Lectura:

> Especializar en NVDA mejora similitud semantica superficial, pero no alcanza para mejorar sentimiento ni direccion.

### 4. NVDA label scoring

NVDA label scoring:

- ROUGE-L `0.066`.
- BERTScore F1 `0.740`.
- Sentiment match `0.376`.
- Directional accuracy `0.496`.
- Coverage `1.000`.

Lectura:

> La estructura/scoring mejora cobertura y mantiene calidad textual, pero no genera una senal financiera robusta.

### 5. NVDA + AMD

NVDA + AMD:

- ROUGE-L `0.074`, el mejor de la seleccion.
- Sentiment match `0.442`, el mejor de la seleccion.
- Directional accuracy `0.530`.
- Coverage `0.361`.

Lectura:

> Es el mejor balance general. Agregar AMD ayuda a calidad textual/semantica, aunque la cobertura financiera baja.

## Sobre loss y perplexity

La loss baja dentro de un run indica aprendizaje del objetivo de ese run.

Pero no hay que comparar loss como si todos los targets fueran iguales.

Ejemplo:

- `NVDA tagged structured prompt` tiene la mejor perplexity (`9.727`).
- Pero no es el mejor en ROUGE-L, BERTScore ni directional accuracy.

Esto sugiere que su target era mas facil o mas regular, no que sea el mejor modelo global.

Frase recomendada:

> La loss confirma aprendizaje dentro de cada configuracion; no decide por si sola el mejor run entre targets distintos.

## Recomendacion final para la presentacion

Usar los cinco runs seleccionados:

1. GPT-2 no fine-tune.
2. SPY/QQQ fine-tuned.
3. NVDA single.
4. NVDA label scoring.
5. NVDA + AMD.

No usar el run binario en la narrativa principal.

Si alguien pregunta por que:

> Lo descartamos porque cambia demasiado la formulacion hacia una tarea direccional. Para defender NLP generativo, preferimos comparar variantes de fine-tuning y estructura sin introducir features binarias orientadas directamente a la metrica financiera.

