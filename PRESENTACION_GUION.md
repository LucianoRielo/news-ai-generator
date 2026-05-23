# Guion de presentacion - Proyecto NLP financiero

## Idea central

Este trabajo no debe presentarse como un predictor de mercado. La idea central es:

> Construimos un pipeline de NLP generativo aplicado a finanzas, fine-tuneando GPT-2 para generar narrativas financieras y evaluandolo con metricas textuales, semanticas y financieras. La conclusion principal es que el fine-tuning mejora la adaptacion al dominio, pero la utilidad financiera no queda demostrada de forma robusta.

## Version corta de 5 minutos

> Este proyecto toma GPT-2, un modelo de lenguaje causal preentrenado, y lo fine-tunea para generar narrativas financieras del dia siguiente usando como contexto ticker, fecha, features de mercado y noticias previas.
>
> La idea no es afirmar que el modelo predice el mercado, sino analizar una tarea de NLP aplicada. Por eso evaluamos en tres capas: calidad textual con ROUGE y BERTScore, consistencia semantica con FinBERT, y una metrica downstream financiera basada en directional accuracy y coverage.
>
> El resultado principal es que las metricas no se mueven juntas. GPT-2 sin fine-tuning tiene mala calidad textual y semantica, con ROUGE-L `0.040` y sentiment match `0.273`, pero aun asi logra directional accuracy `0.549`. Eso muestra que la metrica financiera puede capturar sesgos y no necesariamente comprension del dominio.
>
> El run fine-tuneado `NVDA + AMD` es el mas equilibrado: mejora ROUGE-L a `0.074` y sentiment match a `0.442`, aunque su directional accuracy queda en `0.530`. El run con features direccionales logra la mejor directional accuracy, `0.556`, pero sin significancia estadistica fuerte.
>
> La conclusion es prudente: el fine-tuning adapta GPT-2 al lenguaje financiero, pero no demuestra predictividad robusta. El aporte fuerte del trabajo es el pipeline reproducible y la evaluacion critica multicapa, que muestra por que en NLP generativo no alcanza una sola metrica.

## Slide 1 - Titulo

**Decir:**

> El proyecto se llama fine-tuning de GPT-2 para generacion de narrativas financieras. Lo planteo como un problema de NLP aplicado: dado un contexto financiero previo, queremos generar una narrativa del dia siguiente y despues evaluar si esa narrativa se parece a noticias reales, si mantiene un tono financiero coherente y si de alguna forma se relaciona con la direccion del activo.

**Enfasis:**

- No decir "prediccion bursatil" como promesa.
- Decir "narrativas financieras" y "evaluacion downstream".

## Slide 2 - Pregunta experimental

**Decir:**

> La pregunta no es una sola. Primero: puede GPT-2 generar texto financiero plausible? Segundo: el fine-tuning mejora frente a GPT-2 base? Tercero: el sentimiento o la direccion que aparece en la narrativa tiene relacion con el comportamiento real del mercado?
>
> Esta separacion es importante porque una respuesta puede ser positiva en una capa y negativa en otra. Un texto puede sonar financiero, pero no tener buen sentimiento. O puede acertar una direccion por sesgo de datos, pero generar una narrativa pobre.

**Enfasis:**

- La defensa gira alrededor de evaluacion multicapa.
- Evitar una unica metrica ganadora.

## Slide 3 - Conexion con NLP

**Decir:**

> Desde NLP, el proyecto usa transfer learning. Partimos de GPT-2, que ya aprendio patrones generales del lenguaje en preentrenamiento, y hacemos fine-tuning sobre un corpus financiero.
>
> GPT-2 es un modelo de lenguaje causal: predice el siguiente token condicionado por los tokens anteriores. Trabaja con tokenizacion subword, no necesariamente palabras completas. Durante el fine-tuning, el modelo ajusta sus pesos para que, dado un prompt con ticker, fecha, features de mercado y noticias previas, genere una continuacion parecida al target financiero.

**Conectar con materia:**

- Transfer learning.
- Tokenizacion.
- Embeddings contextuales.
- Language modeling.
- Fine-tuning.

## Slide 4 - Pipeline

**Decir:**

> El pipeline empieza con datos de noticias financieras y datos de mercado. Para cada ticker y fecha se construye una ventana de contexto con noticias previas y features como retorno, volumen relativo o RSI, segun el experimento.
>
> Despues hacemos un split temporal en train, validation y test. Esto es importante porque en finanzas no queremos entrenar con informacion futura. Luego fine-tuneamos GPT-2, generamos predicciones sobre test y evaluamos.

**Enfasis:**

- Split temporal.
- Pipeline reproducible.
- Misma evaluacion para todos los runs.

## Slide 5 - Prompt y target

**Decir:**

> La formulacion es prompt/completion. El prompt contiene el contexto disponible hasta el dia t: ticker, fecha, features numericas y noticias previas. El target es la narrativa o outlook del dia t+1.
>
> En algunos runs el target era mas libre, como titulares o noticias. En otros agregamos estructura explicita, por ejemplo sentimiento y direccion. Eso permite comparar si estructurar la salida ayuda al modelo.

**Frase util:**

> Esta decision convierte el problema en una tarea generativa supervisada.

## Slide 6 - Runs principales

**Decir:**

> No todos los runs cumplen la misma funcion. SPY/QQQ es el primer baseline experimental completo. GPT-2 no fine-tuneado es el baseline mas importante para medir si el entrenamiento aporta algo. NVDA single sirve para ver comportamiento en un solo activo. NVDA + AMD es el run mas equilibrado en calidad textual y semantica. Y NVDA direction features es la variante mas orientada a la metrica financiera.

**Lectura rapida:**

- `SPY/QQQ baseline`: punto de partida.
- `GPT-2 no fine-tune`: baseline de modelo.
- `NVDA single`: mejor BERTScore.
- `NVDA + AMD`: mejor equilibrio general.
- `NVDA direction features`: mejor directional accuracy.

## Slide 7 - Evaluacion multicapa

**Decir:**

> Para evaluar generacion no alcanza una metrica. Por eso usamos tres capas.
>
> La primera es textual: ROUGE-L y BERTScore. ROUGE mira solapamiento mas literal, mientras que BERTScore captura similitud semantica usando embeddings.
>
> La segunda es semantica financiera: usamos FinBERT para clasificar sentimiento de la noticia real y de la generada, y medimos si coinciden.
>
> La tercera es downstream financiera: convertimos la salida generada en una senal direccional y medimos si coincide con la direccion real del activo. Pero siempre junto con coverage y test de significancia.

**Enfasis:**

- Directional accuracy sola puede enganar.
- Coverage y p-value son obligatorios.

## Slide 8 - Resultados agregados

**Decir:**

> Esta tabla resume el punto mas importante del proyecto: las metricas no se mueven juntas.
>
> GPT-2 sin fine-tuning tiene ROUGE-L bajo, `0.040`, y sentiment match bajo, `0.273`. Sin embargo, tiene directional accuracy de `0.549` con coverage `1.0`. Esto es una alerta: una metrica financiera alta puede aparecer incluso cuando la calidad semantica es pobre.
>
> El run NVDA + AMD tiene mejor calidad textual y semantica: ROUGE-L `0.074` y sentiment match `0.442`. Pero su directional accuracy es `0.530`, menor que el baseline sin fine-tuning.
>
> El run con direction features tiene la mejor directional accuracy, `0.556`, pero tampoco alcanza significancia fuerte.

**Conclusion de la slide:**

> El fine-tuning mejora adaptacion al dominio, pero no demuestra predictividad financiera robusta.

## Slide 9 - Loss y perplexity

**Decir:**

> Las curvas de loss muestran que hubo fine-tuning real: el modelo ajusta la tarea de language modeling sobre el corpus financiero.
>
> Tambien calculamos perplexity como exponencial de la eval loss. Es una metrica clasica para modelos de lenguaje: menor perplexity significa que el modelo asigna mas probabilidad al texto de validacion.
>
> Pero hay un hallazgo importante: el mejor valor de perplexity no coincide necesariamente con el mejor resultado financiero. Eso separa dos cosas: aprender a modelar texto y producir una senal financiera util.

**Enfasis:**

- Perplexity mide lenguaje, no trading.
- Buen language model no implica buena senal downstream.

## Slide 10 - Matrices semanticas

**Decir:**

> Estas matrices comparan sentimiento real contra sentimiento generado, usando FinBERT como evaluador externo.
>
> En GPT-2 base, el match semantico es bajo. El modelo puede generar texto que parece financiero, pero no necesariamente con el tono correcto.
>
> Con fine-tuning, especialmente en NVDA + AMD, mejora la alineacion, pero todavia no supera claramente el baseline neutral. Esto muestra una limitacion fuerte: el modelo tiende a producir salidas genericas o neutrales y le cuesta controlar el sentimiento.

**Frase clave:**

> El fine-tuning mejora, pero no resuelve completamente la semantica financiera.

## Slide 11 - Matriz financiera

**Decir:**

> La matriz financiera mira otra cosa: si la direccion generada coincide con la direccion real del activo.
>
> El run con features direccionales obtiene la mejor directional accuracy entre los runs principales, alrededor de `0.556`, con coverage total. Pero el p-value contra azar es aproximadamente `0.225`, asi que no hay evidencia estadistica fuerte.
>
> Por eso esta parte la presento como una senal exploratoria. Es interesante, pero no suficiente para afirmar que el modelo predice el mercado.

**Enfasis:**

- No sobreactuar `0.556`.
- Decir "exploratorio".

## Slide 12 - Ejemplos cualitativos

**Decir:**

> Los ejemplos ayudan a entender por que las metricas agregadas no alcanzan.
>
> En un caso, el fine-tuning mejora la alineacion semantica y ROUGE-L frente a GPT-2 base. Esto muestra que el entrenamiento si adapta el modelo al dominio.
>
> Pero en otro caso, GPT-2 base acierta la direccion mientras genera una narrativa pobre o generica. Ese ejemplo es central: muestra que directional accuracy puede verse bien aunque el texto no sea bueno.

**Conclusion de la slide:**

> La evaluacion cuantitativa necesita lectura cualitativa.

## Slide 13 - Limitaciones

**Decir:**

> Las limitaciones son varias. El dataset de test es chico para concluir robustez. El target son titulares agregados por dia, no articulos completos curados. GPT-2 small tiene capacidad limitada y tiende a generar frases genericas o clickbait.
>
> Ademas, FinBERT es un evaluador automatico, no una verdad absoluta. Y la directional accuracy depende mucho de como convertimos sentimiento o direccion estructurada en senal financiera.
>
> La mas importante: no hay significancia estadistica fuerte en directional accuracy. Por eso la conclusion financiera tiene que ser prudente.

## Slide 14 - Conclusiones

**Decir:**

> La primera conclusion es que GPT-2 fine-tuneado aprende vocabulario y formato financiero.
>
> La segunda es que el fine-tuning mejora la calidad textual y semantica frente a GPT-2 base, especialmente en el run NVDA + AMD.
>
> La tercera es que las metricas no son equivalentes: BERTScore, sentiment match y directional accuracy miden cosas distintas.
>
> La cuarta es que la directional accuracy no alcanza para afirmar predictividad. Hace falta coverage, significancia y mejores baselines estadisticos.
>
> Entonces, el aporte principal del proyecto es metodologico: construir un pipeline reproducible de NLP generativo y evaluarlo criticamente en varias capas.

## Slide 15 - Proximos pasos

**Decir:**

> Como trabajo futuro, agregaria baselines estadisticos mas fuertes para la direccion, por ejemplo siempre predecir la clase mayoritaria o usar una regla simple basada en retorno previo.
>
> Tambien probaria modelos mas adecuados al dominio, como un GPT mas grande o modelos financieros. Mejoraria la estructura de salida para controlar sentimiento y direccion. Y sumaria evaluacion humana de plausibilidad.
>
> Como analisis interpretativo, se puede explorar atencion sobre tokens del prompt, pero lo presentaria con cuidado porque atencion no equivale directamente a explicacion causal.

## Cierre de 30 segundos

> En resumen, el proyecto muestra que fine-tunear GPT-2 permite adaptar un modelo generativo al dominio financiero, pero tambien muestra por que evaluar generacion es dificil. Una narrativa puede parecer financiera sin ser semanticamente correcta, y una senal puede acertar direccion sin demostrar capacidad predictiva. Por eso la contribucion principal es el pipeline de evaluacion multicapa y la lectura critica de sus resultados.

## Preguntas posibles

### Por que GPT-2 y no BERT?

Porque la tarea principal es generativa. BERT es encoder y esta pensado para comprension o clasificacion; GPT-2 es causal y genera texto autoregresivamente.

### Por que ROUGE es bajo?

Porque se comparan titulares generados contra titulares reales. Hay muchas formas distintas de narrar el mismo contexto, asi que la coincidencia literal suele ser baja. Por eso tambien usamos BERTScore y FinBERT.

### Si GPT-2 base tiene buena directional accuracy, para que sirve el fine-tuning?

Sirve para mejorar adaptacion textual y semantica al dominio. El resultado del baseline muestra que directional accuracy sola no es suficiente y puede capturar sesgos. Esa es justamente una conclusion importante del trabajo.

### El modelo predice el mercado?

No hay evidencia suficiente para decir eso. Algunos runs superan `0.5` en directional accuracy, pero sin significancia estadistica fuerte. Lo correcto es hablar de senales exploratorias.

### Que run defenderias como mejor?

Para calidad general, `NVDA + AMD`, porque tiene mejor equilibrio textual/semantico. Para metrica financiera, `NVDA direction features`, pero con la aclaracion de que no es estadisticamente concluyente.

### Que agregarias si tuvieras mas tiempo?

Un baseline estadistico financiero mas fuerte, evaluacion humana, mas datos, modelos mas grandes o especializados, y mejor control estructural de la salida.
