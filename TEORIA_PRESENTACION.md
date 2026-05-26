# Teoria para entender la presentacion

Este apunte es para leer antes de practicar la presentacion. La idea es que puedas explicar el proyecto con seguridad sin perderte en detalles tecnicos.

## 1. Que problema resolvemos

El proyecto usa NLP para generar **narrativas financieras**.

Dado un contexto como:

```text
Ticker: AMD
Fecha: 2023-04-18
Retorno diario: -0.10%
RSI: 36.50
Noticias previas:
- AMD Stock Sinks As Market Gains...
```

el modelo intenta generar una narrativa del dia siguiente:

```text
AMD shares moved higher after renewed optimism around AI chips...
```

Importante: **no estamos demostrando que el modelo predice el mercado**. Estamos evaluando si un modelo generativo puede aprender estilo financiero y si sus salidas tienen alguna relacion exploratoria con señales de mercado.

## 2. GPT-2 en pocas palabras

GPT-2 es un **modelo de lenguaje causal**.

Esto significa que predice el siguiente token usando los tokens anteriores:

```text
The stock moved ___
```

El modelo estima que palabras como `higher`, `lower`, `after`, etc. pueden venir despues.

Formalmente, modela una secuencia asi:

```text
P(w1, w2, ..., wn) = P(w1) * P(w2|w1) * ... * P(wn|w1...wn-1)
```

En simple: GPT-2 aprende a continuar texto.

## 3. Transfer learning y fine-tuning

**Transfer learning** es reutilizar un modelo ya entrenado en una tarea general y adaptarlo a una tarea especifica.

En este proyecto:

1. GPT-2 ya sabe lenguaje general.
2. Lo entrenamos un poco mas con textos financieros.
3. Ese proceso se llama **fine-tuning**.

Ejemplo:

- GPT-2 base puede generar texto financiero generico.
- GPT-2 fine-tuneado deberia usar mejor vocabulario del dominio: `earnings`, `shares`, `analysts`, `AI demand`, `semiconductors`.

## 4. Tokenizacion y embeddings

Los modelos no leen palabras como humanos. Leen **tokens**.

Un token puede ser:

- una palabra completa: `stock`
- parte de una palabra: `invest`, `ing`
- un simbolo: `%`, `:`, `[`

GPT-2 usa tokenizacion subword. Esto ayuda con palabras raras o tecnicas.

Cada token se transforma en un vector llamado **embedding**. Ese vector representa informacion aprendida sobre el token y su contexto.

Ejemplo intuitivo:

```text
Nvidia reported strong earnings
Nvidia shares fell after earnings
```

La palabra `earnings` aparece en ambos casos, pero el contexto cambia su interpretacion.

## 5. Prompt / completion

El proyecto formula la tarea como:

```text
prompt -> completion
```

El **prompt** contiene el contexto disponible:

- ticker;
- fecha;
- retornos;
- volumen;
- RSI;
- noticias previas.

La **completion** es lo que queremos que el modelo genere:

- noticia del dia siguiente;
- outlook;
- sentimiento;
- direccion esperada, segun el run.

Esto convierte el problema en una tarea generativa supervisada.

## 6. Que son los runs

Un **run** es un experimento completo con una configuracion concreta.

Ejemplos:

- `GPT-2 no fine-tune`: GPT-2 base, sin entrenamiento adicional.
- `NVDA single`: fine-tuning solo con NVDA.
- `NVDA + AMD`: fine-tuning con dos tickers.
- `NVDA direction features`: run con features orientadas a direccion.

Cada run tiene:

- datos;
- modelo;
- predicciones;
- metricas;
- reportes.

## 7. Por que usamos baselines

Un **baseline** es un punto de comparacion simple.

Sin baseline, no sabemos si el modelo realmente mejora algo.

En este proyecto usamos especialmente:

### GPT-2 sin fine-tuning

Sirve para preguntar:

> El fine-tuning aporta algo o GPT-2 base ya genera algo parecido?

Resultado interesante:

- GPT-2 base tiene mala calidad textual/semantica.
- Pero obtiene directional accuracy relativamente alta.

Eso nos enseña que directional accuracy puede ser engañosa.

### Neutral baseline

En sentimiento financiero muchas noticias son neutrales. Entonces un modelo tonto que siempre dice `neutral` puede tener buena accuracy.

Por eso comparamos sentiment match contra:

```text
predecir siempre neutral
```

Si nuestro modelo no supera eso, la metrica semantica no es fuerte.

## 8. Metricas textuales

### ROUGE-L

ROUGE-L mide solapamiento entre texto generado y texto real, mirando subsecuencias comunes.

Ejemplo:

Texto real:

```text
Nvidia shares rise after strong AI demand
```

Texto generado:

```text
Nvidia stock rises on AI demand
```

Hay palabras e ideas compartidas, entonces ROUGE puede subir.

Limitacion: si el modelo dice algo semanticamente parecido pero con otras palabras, ROUGE puede ser bajo.

Por eso en generacion financiera esperamos ROUGE bajo. No hay una unica forma correcta de escribir una noticia.

### BERTScore

BERTScore compara textos usando embeddings, no solo palabras exactas.

Intenta capturar similitud semantica.

Ejemplo:

```text
shares rise
stock climbs
```

ROUGE puede ver pocas palabras iguales. BERTScore puede entender que son parecidas.

## 9. Metricas semanticas con FinBERT

FinBERT es un modelo entrenado para sentimiento financiero.

Clasifica texto como:

- `positive`
- `neutral`
- `negative`

En el proyecto usamos FinBERT para comparar:

```text
sentimiento de noticia real vs sentimiento de noticia generada
```

Ejemplo:

Noticia real:

```text
AMD shares fall after weak guidance
```

Sentimiento real: `negative`

Si el modelo genera:

```text
AMD shares rally after strong demand
```

Sentimiento generado: `positive`

Entonces falla el match semantico.

## 10. Metricas financieras

La metrica financiera principal es **directional accuracy**.

Pregunta:

> Cuando el modelo genera una señal positiva o negativa, coincide con la direccion real del activo?

Ejemplo:

- El modelo genera señal `up`.
- El activo sube al dia siguiente.
- Eso cuenta como correcto.

Pero hay que tener cuidado.

### Coverage

Coverage mide cuantas veces el modelo emite una señal activa.

Ejemplo:

Modelo A:

- acierta 60%;
- pero solo emite señal en 10 casos.

Modelo B:

- acierta 55%;
- emite señal en 100 casos.

No son comparables sin mirar coverage.

### p-value

El p-value compara el resultado contra azar.

En directional accuracy usamos azar como:

```text
p = 0.5
```

Si un modelo acierta 56%, puede sonar bien. Pero si hay pocos ejemplos, puede ser suerte.

En este proyecto, los p-values no dan evidencia fuerte al nivel clasico `p < 0.05`.

Por eso decimos:

> señales exploratorias, no predictividad robusta.

## 11. Perplexity y loss

Durante el entrenamiento, el modelo minimiza **loss**.

Loss baja significa:

> el modelo asigna mas probabilidad al texto correcto.

La **perplexity** es:

```text
perplexity = exp(eval_loss)
```

Menor perplexity suele significar mejor modelo de lenguaje.

Pero atencion:

> Mejor perplexity no garantiza mejor directional accuracy.

Porque una cosa es generar texto probable y otra es generar una señal financiera util.

## 12. Teoria financiera minima

### Ticker

Es el simbolo de un activo.

Ejemplos:

- `NVDA`: Nvidia.
- `AMD`: Advanced Micro Devices.
- `SPY`: ETF del S&P 500.
- `QQQ`: ETF del Nasdaq 100.

### Retorno

Mide cuanto subio o bajo el precio.

Ejemplo:

```text
Precio ayer: 100
Precio hoy: 103
Retorno: +3%
```

### Volumen

Cantidad negociada de un activo.

Volumen alto puede indicar mucho interes del mercado.

### RSI

RSI significa Relative Strength Index.

Es un indicador tecnico entre 0 y 100.

Lectura tipica:

- RSI bajo: posible sobreventa.
- RSI alto: posible sobrecompra.

No es una verdad absoluta; es solo una feature.

### Direccion

En el proyecto simplificamos la direccion del activo:

- sube: `up` o `1`
- baja: `down` o `-1`
- sin señal: `neutral` o `0`

Esta simplificacion permite medir directional accuracy.

## 13. Lectura correcta de los resultados

Resultados clave:

- GPT-2 sin fine-tuning:

  - ROUGE-L `0.040`
  - sentiment match `0.273`
  - directional accuracy `0.549`
- NVDA + AMD:

  - ROUGE-L `0.074`
  - BERTScore F1 `0.738`
  - sentiment match `0.442`
  - directional accuracy `0.530`
- NVDA direction features:

  - directional accuracy `0.556`
  - coverage `1.0`
  - p-value aprox. `0.225`

Interpretacion:

1. El fine-tuning mejora la calidad textual/semantica.
2. Directional accuracy no sigue exactamente a las metricas de NLP.
3. GPT-2 base puede acertar direccion aun generando texto pobre.
4. No hay evidencia estadistica fuerte de prediccion financiera.
5. El valor del proyecto esta en el pipeline y la evaluacion critica.

## 14. Frases que conviene usar

Usar:

- "adaptacion al dominio financiero"
- "evaluacion multicapa"
- "señal exploratoria"
- "sin evidencia estadistica fuerte"
- "pipeline reproducible"
- "metricas complementarias"

Evitar:

- "el modelo predice el mercado"
- "el modelo entiende finanzas"
- "la estrategia es rentable"
- "la accuracy demuestra capacidad predictiva"

## 15. Si te preguntan algo dificil

### El modelo predice el mercado?

No. Algunos runs superan `0.5` en directional accuracy, pero sin significancia estadistica fuerte. Lo correcto es hablar de señales exploratorias.

### Entonces para que sirve?

Sirve como pipeline de NLP aplicado: permite entrenar, generar y evaluar narrativas financieras desde distintas perspectivas.

### Por que GPT-2?

Porque es un modelo generativo causal. BERT seria mas natural para clasificacion, pero GPT-2 es adecuado para generar texto.

### Por que ROUGE es tan bajo?

Porque generar noticias no tiene una unica respuesta correcta. Dos textos pueden hablar de lo mismo con palabras distintas.

### Que run defenderias?

Para calidad general: `NVDA + AMD`.

Para metrica financiera: `NVDA direction features`, pero aclarando que no es concluyente.

### Cual es la conclusion final?

El fine-tuning ayuda a adaptar GPT-2 al lenguaje financiero, pero la utilidad financiera no queda demostrada de forma robusta. El aporte fuerte es la evaluacion multicapa.
