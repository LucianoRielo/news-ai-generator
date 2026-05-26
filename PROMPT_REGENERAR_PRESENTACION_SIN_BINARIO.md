# Prompt para regenerar presentacion sin run binario

Quiero que modifiques la presentacion PowerPoint existente:

`presentacion_nlp_finanzas.pptx`

Debes actualizar principalmente las slides de resultados, entrenamiento, matrices/ejemplos y conclusiones. El objetivo es **eliminar completamente el run binario** de la narrativa y reemplazarlo por una comparacion mas clara desde machine learning.

## Regla principal

No usar ni mencionar como run principal:

`2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features`

Tampoco usar frases como:

- "direction features es el mejor";
- "run binario";
- "features binarias";
- "binary direction features".

Si aparece en una imagen vieja, no usar esa imagen.

## Nueva seleccion de runs

Usar estos cinco runs:

| Rol | Run legible | Run ID |
|---|---|---|
| Modelo base | GPT-2 no fine-tune | `baseline-gpt2-no-finetune_nvda-amd` |
| Dominio amplio/neutral | SPY/QQQ fine-tuned | `2026-04-30_15-49-24_spy-qqq_gpt2_spy-qqq-baseline` |
| Especializacion NVDA | NVDA single | `2026-05-02_13-05-30_nvda_gpt2_nvda-single` |
| Variante estructurada | NVDA label scoring | `2026-05-04_09-09-04_nvda_gpt2_nvda-label-scoring` |
| Multi-ticker | NVDA + AMD | `2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2` |

## Nueva historia de la presentacion

La comparacion ya no debe ser un ranking general de modelos. Debe presentarse como una **progresion experimental**:

1. Que hace GPT-2 sin fine-tuning?
2. Que aporta fine-tuning en un dominio amplio/neutral como SPY/QQQ?
3. Que pasa al especializar en NVDA?
4. Que aporta una salida estructurada/label-aware sin usar el run binario?
5. Que pasa al sumar AMD y pasar a multi-ticker?

Idea central:

> El fine-tuning mejora adaptacion textual y semantica al dominio financiero. Sin embargo, directional accuracy sigue siendo exploratoria y no prueba prediccion de mercado. La loss/perplexity confirma aprendizaje dentro de cada configuracion, pero no decide por si sola el mejor run entre targets distintos.

## Datos clave a usar

Usar `selected_runs_no_binary.csv` y `selected_runs_no_binary.md` como fuente principal de la tabla.

Tabla principal:

| Run | Tickers | Train | Test | ROUGE-L | BERT F1 | Sent. match | Dir. acc. | Coverage | p-value | Best ppl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 no fine-tune | NVDA,AMD | 1517 | 326 | 0.040 | n/a | 0.273 | 0.549 | 1.000 | 0.085 | n/a |
| SPY/QQQ fine-tuned | SPY,QQQ | 1410 | 303 | 0.062 | 0.732 | 0.386 | 0.528 | 0.472 | 0.557 | 14.659 |
| NVDA single | NVDA | 618 | 133 | 0.070 | 0.744 | 0.346 | 0.462 | 0.591 | 0.572 | 29.648 |
| NVDA label scoring | NVDA | 617 | 133 | 0.066 | 0.740 | 0.376 | 0.496 | 1.000 | 1.000 | 16.567 |
| NVDA + AMD | NVDA,AMD | 1517 | 326 | 0.074 | 0.738 | 0.442 | 0.530 | 0.361 | 0.579 | 19.292 |

## Slides a modificar

### Slide 8 - Resultados

**Titulo sugerido:**

`Resultados: progresion experimental sin run binario`

**Contenido:**

No mostrar una tabla gigante de ranking. Mostrar una progresion en 5 pasos:

1. `GPT-2 no fine-tune`: baseline de modelo.
2. `SPY/QQQ fine-tuned`: dominio amplio/neutral.
3. `NVDA single`: especializacion en un activo.
4. `NVDA label scoring`: salida estructurada/label-aware.
5. `NVDA + AMD`: mejor balance multi-ticker.

**Visual recomendado:**

Usar la imagen:

`selected_runs_no_binary_metrics.png`

Y/o una tabla compacta basada en `selected_runs_no_binary.md`.

**Callouts obligatorios:**

- `El fine-tuning mejora ROUGE-L y sentiment match frente a GPT-2 base.`
- `La directional accuracy no sigue necesariamente a las metricas de NLP.`
- `NVDA + AMD es el mejor balance textual/semantico.`

**Notas del presentador:**

> Esta slide no rankea modelos. Muestra una progresion experimental. GPT-2 base sirve para medir el aporte del fine-tuning. SPY/QQQ muestra un dominio amplio y mas neutral. NVDA single evalua especializacion. NVDA label scoring prueba estructura sin usar el run binario. NVDA + AMD es el mejor balance general.

### Slide 9 - Loss / Perplexity

**Titulo sugerido:**

`Loss y perplexity: aprendizaje, no ranking final`

**Imagenes obligatorias:**

Usar:

- `selected_training_loss_no_binary.png`
- `selected_eval_loss_no_binary.png`

Opcional:

- `selected_runs_no_binary_perplexity.png`

**Mensaje principal:**

> La loss baja dentro de cada run muestra aprendizaje del objetivo. Pero no se debe usar como ranking absoluto cuando cambian prompts y targets.

**Callout obligatorio:**

`Menor perplexity no implica mejor resultado semantico o financiero.`

**Ejemplo obligatorio:**

Mencionar que `NVDA tagged structured prompt` tenia la mejor perplexity aproximada (`9.73`), pero no fue seleccionado como mejor estructurado porque era peor en ROUGE-L/BERTScore que `NVDA label scoring`.

**Notas del presentador:**

> En ML, la loss es valida para monitorear entrenamiento, pero hay que saber que objetivo esta optimizando. Si el target es mas regular, la loss puede bajar mas sin que eso signifique mejor generacion o mejor metrica financiera.

### Slide 10 - Semantica

**Titulo sugerido:**

`Semantica: del modelo base al fine-tuning`

**Imagenes recomendadas:**

Usar dos matrices lado a lado:

- `semantic_baseline_gpt2_no_finetune.png`
- `semantic_nvda_amd.png`

Si se quiere mostrar el paso intermedio, usar tambien:

- `semantic_spy_qqq.png`

**Captions:**

- GPT-2 no fine-tune: `sentiment match = 0.273`.
- SPY/QQQ: `sentiment match = 0.386`.
- NVDA + AMD: `sentiment match = 0.442`.

**Callout obligatorio:**

`El fine-tuning mejora alineacion semantica, pero el baseline neutral sigue siendo competitivo.`

**Notas del presentador:**

> Esta es la evidencia mas clara a favor del fine-tuning: sentiment match sube de 0.273 a 0.442. Aun asi, no se debe exagerar porque el baseline neutral es alto.

### Slide 11 - Metrica financiera

**Titulo sugerido:**

`Directional accuracy: resultado exploratorio`

**Importante:**

No usar `financial_direction_features.png` porque corresponde al run binario descartado.

**Imagen recomendada:**

Usar:

- `financial_nvda_amd.png`

Opcionalmente, comparar con:

- `financial_no_finetune.png`
- `financial_spy_qqq.png`

**Tabla obligatoria:**

| Run | Dir. acc. | Coverage | p-value |
|---|---:|---:|---:|
| GPT-2 no fine-tune | 0.549 | 1.000 | 0.085 |
| SPY/QQQ fine-tuned | 0.528 | 0.472 | 0.557 |
| NVDA single | 0.462 | 0.591 | 0.572 |
| NVDA label scoring | 0.496 | 1.000 | 1.000 |
| NVDA + AMD | 0.530 | 0.361 | 0.579 |

**Callout obligatorio:**

`Ningun run seleccionado demuestra predictividad robusta: todos quedan sin significancia fuerte.`

**Notas del presentador:**

> GPT-2 base tiene la mayor directional accuracy de esta seleccion, pero mala semantica. Eso confirma que la directional accuracy puede capturar sesgos del dataset o de la regla de etiquetado. Por eso la tratamos como exploratoria.

### Slide 12 - Cierre

**Titulo sugerido:**

`Conclusiones`

**Bullets recomendados:**

1. `GPT-2 base genera texto financiero generico, pero con baja calidad semantica.`
2. `El fine-tuning mejora ROUGE-L y sentiment match.`
3. `NVDA label scoring es la mejor variante estructurada sin usar el run binario.`
4. `NVDA + AMD es el mejor balance general textual/semantico.`
5. `Directional accuracy queda como senal exploratoria, no como evidencia predictiva.`

**Cierre abajo:**

`El aporte principal es metodologico: pipeline reproducible + evaluacion multicapa.`

## Modificaciones menores en otras slides

### Slide 2

Corregir el titulo si dice `Varías`. Debe decir:

`Varias capas de analisis`

o:

`Tres capas de analisis`

### Slide 6

Corregir:

`prompt/competition`

por:

`prompt/completion`

### Slide 7

Revisar numeracion si aparece repetida como `06`.

## Imagenes que NO usar

No usar:

- `financial_direction_features.png`
- cualquier figura o texto que destaque el run binario.

## Imagenes que SI usar

- `selected_runs_no_binary_metrics.png`
- `selected_training_loss_no_binary.png`
- `selected_eval_loss_no_binary.png`
- `selected_runs_no_binary_perplexity.png`
- `semantic_baseline_gpt2_no_finetune.png`
- `semantic_spy_qqq.png`
- `semantic_nvda_amd.png`
- `financial_nvda_amd.png`
- opcional: `financial_no_finetune.png`, `financial_spy_qqq.png`

## Estilo

Mantener el estilo visual del PowerPoint actual.

Priorizar:

- pocas tablas;
- numeros con 3 decimales;
- titulos claros;
- callouts cortos;
- lectura critica de ML.

No saturar con todos los runs disponibles. Solo usar la seleccion indicada.

