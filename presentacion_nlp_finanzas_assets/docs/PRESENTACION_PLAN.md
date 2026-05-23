# Plan de presentacion - Proyecto NLP financiero

## Objetivo de la defensa

Presentar el proyecto como un pipeline completo de NLP aplicado a finanzas: fine-tuning de un modelo de lenguaje causal para generar narrativas financieras, evaluado desde tres niveles complementarios.

La tesis principal no debe ser "el modelo predice el mercado". La tesis defendible es:

> El fine-tuning adapta GPT-2 al estilo y vocabulario financiero, pero la evaluacion muestra que similitud textual, consistencia semantica y utilidad financiera no necesariamente se mueven juntas. El aporte principal del trabajo es construir y analizar criticamente un pipeline generativo con evaluacion multicapa.

## Preguntas guia

- Puede GPT-2 fine-tuneado generar narrativas financieras plausibles a partir de contexto historico?
- El modelo captura tono financiero comparable al de noticias reales?
- Las senales positivas/negativas generadas tienen relacion con la direccion real del activo?
- Que diferencia hay entre parecerse semanticamente a una noticia y ser util para una metrica downstream financiera?

## Conceptos de NLP que hay que conectar con la materia

- Transfer learning: GPT-2 parte de preentrenamiento general y se adapta al dominio financiero.
- Modelo de lenguaje causal: predice el siguiente token condicionado por tokens anteriores.
- Tokenizacion subword: GPT-2 procesa unidades menores que palabras completas.
- Embeddings contextuales: una misma palabra puede representarse distinto segun el contexto.
- Fine-tuning: ajuste de pesos sobre una tarea especifica.
- Prompt/completion: formulacion generativa supervisada.
- Evaluacion generativa: ROUGE, BERTScore y evaluacion semantica externa.
- Clasificacion semantica auxiliar: FinBERT como evaluador de sentimiento financiero.

## Runs principales a mostrar

### 1. Baseline experimental SPY/QQQ

- Run: `2026-04-30_15-49-24_spy-qqq_gpt2_spy-qqq-baseline`
- Por que importa:
  - Primer experimento robusto con ETFs amplios.
  - Sirve como punto de partida frente a runs posteriores.
  - Muestra resultados moderados con cobertura parcial.
- Lectura:
  - Directional accuracy aproximada: `0.528`.
  - Signal coverage aproximada: `0.472`.
  - Buen run para introducir el pipeline completo.

### 2. Baseline NVDA single

- Run: `2026-05-02_13-05-30_nvda_gpt2_nvda-single`
- Por que importa:
  - Mejor BERTScore F1 entre los runs principales.
  - Muestra que buen parecido semantico/textual no garantiza utilidad financiera.
- Lectura:
  - BERTScore F1 aproximado: `0.744`.
  - Directional accuracy aproximada: `0.462`.

### 3. NVDA + AMD

- Run: `2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2`
- Por que importa:
  - Mejor equilibrio general.
  - Mejor ROUGE-L y mejor sentiment match entre los runs seleccionados.
  - Dataset mas grande que NVDA single.
- Lectura:
  - ROUGE-L aproximado: `0.0736`.
  - Sentiment match aproximado: `0.442`.
  - Directional accuracy aproximada: `0.530`.
  - Coverage aproximada: `0.361`.
- Este es probablemente el run mas defendible como "mejor modelo general".

### 4. GPT-2 sin fine-tuning

- Run: `baseline-gpt2-no-finetune_nvda-amd`
- Por que importa:
  - Es el baseline mas importante para aislar el efecto del fine-tuning.
  - Usa el mismo test set que el run NVDA + AMD.
  - Muestra que la directional accuracy puede ser relativamente alta incluso con baja calidad textual/semantica.
- Lectura:
  - ROUGE-L aproximado: `0.0402`.
  - Sentiment match aproximado: `0.2730`.
  - Directional accuracy aproximada: `0.5494`.
  - Coverage: `1.0`.
- Mensaje clave:
  - El fine-tuning mejora la adaptacion NLP al dominio, pero la metrica financiera simple puede estar capturando sesgos de etiquetado o distribucion.

### 5. NVDA con binary direction features

- Run: `2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features`
- Por que importa:
  - Mejor resultado financiero.
  - Sirve para mostrar que agregar features explicitas cambia el comportamiento.
- Lectura:
  - Directional accuracy aproximada: `0.556`.
  - Coverage: `1.0`.
  - Sentiment match bajo frente al baseline neutral.
- Mensaje clave:
  - Es interesante como senal, pero no alcanza para afirmar predictividad robusta.

### Run negativo o de ablacion

- Run: `2026-05-03_12-41-00_nvda_gpt2_nvda-structured-prompt`
- Por que mencionarlo:
  - El prompt estructurado por si solo no mejoro.
  - Sirve para mostrar aprendizaje experimental: no toda estructura agrega valor.

## Metricas relevantes

### 1. Loss, eval loss y perplexity

Uso en la presentacion:

- Demostrar que hubo fine-tuning real.
- Mostrar dinamica de entrenamiento.
- Detectar overfitting si training loss baja pero eval loss empeora.

Accion:

- Extraer `loss` y `eval_loss` de `trainer_state.json`.
- Calcular `perplexity = exp(eval_loss)` cuando la loss sea razonable.
- Agregar curvas al reporte global.

### 2. ROUGE-L y BERTScore F1

Uso en la presentacion:

- ROUGE-L mide coincidencia lexica/secuencial.
- BERTScore mide similitud semantica usando embeddings.
- ROUGE bajo es esperable en generacion de titulares, porque puede haber muchas formas validas de narrar el mismo contexto.

Mensaje:

- No alcanza con metricas textuales para validar utilidad financiera.

### 3. Sentiment match vs neutral baseline

Uso en la presentacion:

- Evaluar si el tono financiero generado coincide con el tono de las noticias reales.
- Comparar contra un baseline trivial de predecir siempre neutral.

Mensaje:

- Varios runs no superan el baseline neutral.
- Esto revela una limitacion semantica importante.

### 4. Directional accuracy + signal coverage

Uso en la presentacion:

- Medir si la senal generada coincide con la direccion real del activo.
- Mostrar siempre junto con coverage.

Mensaje:

- Accuracy sin coverage puede ser enganosa.
- Una senal selectiva y una senal que predice siempre tienen interpretaciones distintas.

### 5. Significancia estadistica

Uso en la presentacion:

- Comparar directional accuracy contra azar (`p = 0.5`) usando test binomial.
- Agregar intervalo de confianza.

Mensaje esperado:

- Los resultados actuales parecen exploratorios.
- No hay evidencia suficiente para afirmar predictividad estadisticamente significativa.

## Artefactos a generar

- `runs/model-comparison/reports/model_comparison.csv`
- `runs/model-comparison/reports/loss_history.csv`
- `runs/model-comparison/reports/model_comparison.png`
- `runs/model-comparison/reports/training_loss_curves.png`
- `runs/model-comparison/reports/eval_loss_curves.png`
- `runs/model-comparison/reports/directional_significance.csv`
- `runs/model-comparison/reports/perplexity_summary.csv`
- `runs/model-comparison/reports/REPORT.md`
- `runs/model-comparison/reports/qualitative_examples.md`
- `runs/model-comparison/reports/qualitative_examples.csv`
- `runs/model-comparison/reports/slides_model_table.md`
- `runs/model-comparison/reports/slides_model_table.csv`
- `runs/model-comparison/reports/slides_model_table_compact.md`
- `runs/model-comparison/reports/slides_model_table_compact.csv`
- `runs/model-comparison/reports/selected_confusion_matrices/semantic_baseline_gpt2_no_finetune.png`
- `runs/model-comparison/reports/selected_confusion_matrices/semantic_finetuned_nvda_amd.png`
- `runs/model-comparison/reports/selected_confusion_matrices/financial_direction_features.png`
- `SLIDES_OUTLINE.md`
- `PRESENTACION_GUION.md`
- `presentacion_nlp_finanzas.pptx`
- `runs/baseline-gpt2-no-finetune_nvda-amd/reports/REPORT.md`

## Trabajo tecnico pendiente

### Prioridad alta

- [x] Crear script de comparacion entre runs.
- [x] Generar curvas iniciales de loss y eval loss.
- [x] Agregar perplexity final por run.
- [x] Agregar test binomial e intervalos de confianza para directional accuracy.
- [x] Crear reporte global `runs/model-comparison/reports/REPORT.md`.
- [x] Generar baseline GPT-2 sin fine-tuning usando el mismo test set y las mismas metricas.
- [x] Recalcular tabla comparativa incluyendo el baseline no fine-tuneado.

### Prioridad media

- [x] Seleccionar ejemplos cualitativos de los runs principales:
  - buen caso textual;
  - buen caso financiero;
  - fallo semantico;
  - generacion generica o clickbait.
- [x] Agregar confusion matrices seleccionadas al reporte global.
- [x] Preparar outline de slides con contenido y dialogo.
- [x] Preparar tabla final para slides con runs principales y metricas clave.

### Prioridad baja / opcional

- [ ] Extraer attention weights de un ejemplo.
- [ ] Visualizar atencion sobre tokens del prompt.
- [ ] Presentarlo solo como analisis interpretativo, no como prueba causal.

## Baseline GPT-2 sin fine-tuning

Motivo:

- Es necesario para aislar el aporte del fine-tuning.
- Permite responder: el modelo aprendio algo del dominio o GPT-2 base ya produce lenguaje financiero generico?

Plan:

1. Elegir un test set representativo, preferentemente el de NVDA + AMD o NVDA single.
2. Generar predicciones con `gpt2` base usando exactamente los mismos prompts.
3. Guardar resultados como un run separado.
4. Ejecutar las mismas metricas:
   - textual;
   - semantica;
   - financiera.
5. Incorporar el baseline a `model_comparison.csv`.

Estado:

- Completado en `runs/baseline-gpt2-no-finetune_nvda-amd`.
- BERTScore no quedo disponible para este baseline por un problema local del tokenizer, pero ROUGE, FinBERT y metricas financieras se calcularon correctamente.
- Resultado interpretativo: GPT-2 base obtiene directional accuracy alta, pero con baja calidad textual/semantica. Esto obliga a presentar la metrica financiera como exploratoria.

## Estructura sugerida de slides

### Slide 1 - Titulo

Fine-tuning de GPT-2 para generacion de narrativas financieras.

### Slide 2 - Pregunta del proyecto

Podemos generar narrativas financieras plausibles y extraer de ellas senales utiles?

### Slide 3 - Conexion con NLP

Transfer learning, GPT-2, language modeling causal, tokenizacion, embeddings contextuales y fine-tuning.

### Slide 4 - Dataset y pipeline

FNSPID + datos de mercado, ventana temporal, prompt, completion, split temporal, generacion y evaluacion.

### Slide 5 - Modelo

GPT-2 fine-tuneado con perdida enmascarada sobre el prompt.

### Slide 6 - Evaluacion multicapa

Textual: ROUGE/BERTScore.
Semantica: FinBERT.
Financiera: directional accuracy + coverage.

### Slide 7 - Comparacion de runs

Tabla compacta con los 4 runs principales.

### Slide 8 - Curvas de entrenamiento

Training loss, eval loss y perplexity.

### Slide 9 - Confusion matrices y ejemplos

Mostrar una matriz semantica y una matriz financiera. Acompanarlo con 1 o 2 generaciones.

### Slide 10 - Conclusiones

- El modelo aprende estilo y vocabulario financiero.
- La similitud textual no implica utilidad financiera.
- El sentimiento generado no supera consistentemente el baseline neutral.
- La directional accuracy es interesante pero no estadisticamente concluyente.
- El aporte fuerte es el pipeline reproducible y la evaluacion critica.

## Dialogo base

Inicio:

> Este trabajo no busca vender un sistema de trading, sino estudiar un problema de NLP aplicado: dado contexto financiero previo, entrenamos un modelo generativo para producir una narrativa del dia siguiente.

Modelo:

> Usamos GPT-2 porque permite mostrar transfer learning: partimos de un modelo preentrenado en lenguaje general y lo fine-tuneamos sobre un dominio especifico, usando prompts con informacion de mercado y noticias previas.

Evaluacion:

> Como la tarea es generativa, una unica metrica no alcanza. Por eso evaluamos en tres niveles: parecido textual, consistencia semantica y una metrica downstream financiera.

Resultado:

> El punto mas interesante es que las metricas no se mueven juntas. El modelo que mejor se parece semanticamente a las noticias reales no es necesariamente el que obtiene mejor directional accuracy.

Conclusion:

> Los resultados sugieren adaptacion al dominio, pero no evidencia suficiente de predictividad robusta. La conclusion principal es metodologica: el proyecto muestra como construir, evaluar y discutir criticamente un sistema generativo aplicado a finanzas.

## Criterio de exito de la presentacion

La defensa deberia dejar claro que:

- el proyecto usa conceptos centrales de NLP;
- el pipeline es reproducible;
- las metricas fueron elegidas con criterio;
- las conclusiones son prudentes;
- el trabajo identifica limitaciones y proximos pasos razonables.
