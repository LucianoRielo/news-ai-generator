# Revision de runs sin incluir el run binario

## Criterio de seleccion

Se descarta el run:

`2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features`

Motivo: el usuario pidio excluirlo. Ademas, ese run cambia el problema al incorporar features/labels mas directamente orientadas a direccion, por lo que no es la comparacion mas limpia para discutir fine-tuning generativo de narrativas.

La seleccion se hace con criterio de machine learning:

1. Definir objetivo primario.
2. Comparar contra baseline limpio.
3. Separar ablations por familia.
4. No elegir por una sola metrica.
5. Interpretar loss/perplexity solo dentro del contexto del target.

## Objetivo primario recomendado

El objetivo central de la presentacion deberia ser:

> Evaluar si el fine-tuning de GPT-2 mejora la generacion de narrativas financieras frente a GPT-2 base, y analizar si las metricas textuales/semanticas se relacionan o no con una señal financiera exploratoria.

Con este objetivo, el mejor run principal es:

`2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2`

Nombre para slides:

`NVDA + AMD fine-tuned`

## Runs centrales recomendados

### 1. GPT-2 no fine-tune

Run:

`baseline-gpt2-no-finetune_nvda-amd`

Rol:

Baseline limpio de modelo.

Metricas:

| Metrica | Valor |
|---|---:|
| ROUGE-L | 0.040 |
| Sentiment match | 0.273 |
| Neutral baseline | 0.583 |
| Directional accuracy | 0.549 |
| Coverage | 1.000 |
| p-value | 0.085 |

Lectura:

GPT-2 base genera peor texto y peor alineacion semantica, pero tiene directional accuracy relativamente alta. Esto indica que la metrica financiera puede estar capturando sesgos de clase, distribucion del test o de la regla de conversion a señal.

Conclusion:

> Directional accuracy no puede interpretarse como comprension financiera si no esta acompañada por calidad semantica, coverage, p-value y baselines.

### 2. NVDA + AMD fine-tuned

Run:

`2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2`

Rol:

Modelo principal defendible.

Metricas:

| Metrica | Valor |
|---|---:|
| Train examples | 1517 |
| Test examples | 326 |
| ROUGE-L | 0.074 |
| BERTScore F1 | 0.738 |
| Sentiment match | 0.442 |
| Neutral baseline | 0.537 |
| Directional accuracy | 0.530 |
| Coverage | 0.361 |
| p-value | 0.579 |
| Best perplexity | 19.29 |

Lectura:

Es el run con mejor balance general:

- mejora ROUGE-L frente a GPT-2 base;
- mejora sentiment match frente a GPT-2 base;
- usa el mismo universo NVDA+AMD que el baseline;
- tiene dataset mas grande que NVDA single;
- no depende del run binario.

Conclusion:

> Es el mejor modelo para defender calidad de NLP, aunque no sea el mejor en directional accuracy.

### 3. NVDA single

Run:

`2026-05-02_13-05-30_nvda_gpt2_nvda-single`

Rol:

Baseline fine-tuned de un solo ticker.

Metricas:

| Metrica | Valor |
|---|---:|
| Train examples | 618 |
| Test examples | 133 |
| ROUGE-L | 0.070 |
| BERTScore F1 | 0.744 |
| Sentiment match | 0.346 |
| Directional accuracy | 0.462 |
| Coverage | 0.591 |
| Best perplexity | 29.65 |

Lectura:

Tiene el mejor BERTScore F1, pero bajo sentiment match y peor directional accuracy. Sirve para mostrar que una metrica semantica general no alcanza para validar el comportamiento financiero.

Conclusion:

> Buen parecido semantico general no implica mejor señal financiera ni mejor control de sentimiento.

## Runs secundarios para ablation

### 4. NVDA tagged structured

Run:

`2026-05-04_00-31-31_nvda_gpt2_nvda-tagged-structured-prompt`

Rol:

Ablation de target/prompt estructurado.

Metricas:

| Metrica | Valor |
|---|---:|
| ROUGE-L | 0.042 |
| BERTScore F1 | 0.687 |
| Sentiment match | 0.383 |
| Directional accuracy | 0.483 |
| Coverage | 0.436 |
| Best perplexity | 9.73 |

Lectura:

Tiene la mejor perplexity, pero no el mejor resultado externo. Esto es muy importante para explicar ML:

> Un target mas regular puede bajar la loss sin mejorar las metricas que realmente importan.

Conclusion:

> Loss/perplexity no son suficientes para elegir el mejor run cuando cambian prompt y target.

### 5. NVDA label scoring

Run:

`2026-05-04_09-09-04_nvda_gpt2_nvda-label-scoring`

Rol:

Ablation de scoring estructurado de labels.

Metricas:

| Metrica | Valor |
|---|---:|
| ROUGE-L | 0.066 |
| BERTScore F1 | 0.740 |
| Sentiment match | 0.376 |
| Directional accuracy | 0.496 |
| Coverage | 1.000 |
| Best perplexity | 16.57 |

Lectura:

Mejora coverage a 1.0, pero no mejora directional accuracy. Esto separa cobertura de calidad.

Conclusion:

> Emitir una señal siempre no significa emitir una mejor señal.

### 6. NVDA + AMD + AVGO

Run:

`2026-05-05_10-41-34_nvda-amd-avgo_gpt2_nvda-amd-avgo`

Rol:

Ablation de escalado de tickers/datos.

Metricas:

| Metrica | Valor |
|---|---:|
| Train examples | 1869 |
| Test examples | 402 |
| ROUGE-L | 0.066 |
| BERTScore F1 | 0.733 |
| Sentiment match | 0.323 |
| Directional accuracy | 0.490 |
| Coverage | 1.000 |
| Best perplexity | 11.13 |

Lectura:

Agregar mas tickers y mas datos no mejora automaticamente. Puede introducir heterogeneidad.

Conclusion:

> Mas datos no siempre gana si cambia la distribucion.

## Runs que dejaria fuera de la tabla principal

### SPY/QQQ baseline

Run:

`2026-04-30_15-49-24_spy-qqq_gpt2_spy-qqq-baseline`

Por que no usarlo como central:

- usa otro universo de activos;
- no comparte test con GPT-2 no fine-tune;
- sirve como antecedente historico, no como comparacion limpia.

Puede mencionarse en una nota o appendix.

### NVDA structured prompt

Run:

`2026-05-03_12-41-00_nvda_gpt2_nvda-structured-prompt`

Por que no usarlo como central:

- es un caso negativo claro;
- puede servir como ejemplo de que estructurar prompt no alcanza;
- pero no hace falta en la narrativa principal si hay poco espacio.

## Seleccion final recomendada para slides

### Tabla principal

Usar 4 runs:

| Run | Rol |
|---|---|
| GPT-2 no fine-tune | baseline limpio |
| NVDA single | fine-tuning de un ticker |
| NVDA + AMD | mejor modelo principal |
| NVDA tagged structured | mejor perplexity, mal ejemplo para elegir por loss |

### Tabla secundaria / ablation

Usar 3 comparaciones:

#### Fine-tuning

| Run | ROUGE-L | Sent. match | Dir. acc. |
|---|---:|---:|---:|
| GPT-2 no fine-tune | 0.040 | 0.273 | 0.549 |
| NVDA + AMD | 0.074 | 0.442 | 0.530 |

#### Escalado de datos

| Run | Train | ROUGE-L | Sent. match | Dir. acc. |
|---|---:|---:|---:|---:|
| NVDA single | 618 | 0.070 | 0.346 | 0.462 |
| NVDA + AMD | 1517 | 0.074 | 0.442 | 0.530 |
| NVDA + AMD + AVGO | 1869 | 0.066 | 0.323 | 0.490 |

#### Loss/perplexity

| Run | Best ppl | Lectura |
|---|---:|---|
| NVDA tagged structured | 9.73 | mejor perplexity, no mejor resultado externo |
| NVDA + AMD | 19.29 | mejor balance NLP |
| NVDA single | 29.65 | mejor BERTScore, peor financiero |

## Conclusion experta de ML

La seleccion correcta no es "el run con mayor directional accuracy", sino el run que mejor responde al objetivo principal bajo una comparacion justa.

Como el objetivo es NLP generativo aplicado a finanzas, el run mas defendible es:

`NVDA + AMD fine-tuned`

Motivo:

- compara limpio contra GPT-2 base;
- mejora ROUGE-L y sentiment match;
- usa mas datos que NVDA single;
- evita la trampa de elegir por loss/perplexity;
- evita el run binario pedido como excluido;
- permite explicar que la metrica financiera no esta validada.

Frase final:

> Desde una mirada de machine learning, el mejor run no es el que maximiza una metrica aislada, sino el que mejora el objetivo primario bajo una comparacion controlada. En este caso, NVDA + AMD es el mejor candidato para defender adaptacion NLP; la señal financiera queda como analisis exploratorio.

