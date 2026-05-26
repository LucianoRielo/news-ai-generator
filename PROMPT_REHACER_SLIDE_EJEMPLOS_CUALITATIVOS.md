# Prompt para rehacer slide de ejemplos cualitativos

Quiero que reemplaces la diapositiva actual de **"Ejemplos cualitativos"** por una slide más clara, basada en ejemplos reales generados por dos runs:

- `GPT-2 no fine-tune`: `baseline-gpt2-no-finetune_nvda-amd`
- `NVDA + AMD fine-tuned`: `2026-05-03_01-12-20_nvda-amd_gpt2_nvda-ambd-gpt2`

## Objetivo de la slide

La slide debe mostrar texto real generado por ambos modelos, no solo descripciones. La idea es demostrar cualitativamente dos cosas:

1. El fine-tuning mejora el formato/vocabulario financiero y la alineación semántica.
2. Una métrica financiera puede ser engañosa: GPT-2 base puede acertar dirección aunque genere una narrativa pobre.

## Título sugerido

`Ejemplos cualitativos: qué cambia con fine-tuning`

## Subtítulo sugerido

`Mismo tipo de prompt, dos comportamientos distintos: GPT-2 base vs. GPT-2 fine-tuneado.`

## Diseño recomendado

Usar 2 bloques o cards horizontales:

- Card izquierda: ejemplo donde el fine-tuning mejora.
- Card derecha: ejemplo donde la métrica financiera puede engañar.

Cada card debe tener:

- ticker y fecha;
- noticia real abreviada;
- salida GPT-2 base;
- salida fine-tuned NVDA + AMD;
- lectura breve.

No usar párrafos largos. Recortar los textos generados con puntos suspensivos si hace falta.

## Card 1 - Fine-tuning mejora alineación semántica

**Header:**

`Caso A · mejora semántica`

**Ticker y fecha:**

`AMD · target 2023-06-21`

**Métricas:**

`ROUGE-L: 0.031 → 0.179`

`Sentimiento: GPT-2 base falla · fine-tuned acierta`

**Noticia real:**

```text
- Noteworthy Wednesday Option Activity: SBGI, AMD, HUM
- Unusual Options Activity and Flow in Advanced Micro Devices (AMD)
```

**GPT-2 base generado:**

```text
AMD has been in talks with Intel about their next generation of CPUs and GPUs...
```

**Fine-tuned generado:**

```text
...And More on Intel and Advanced Micro Devices (AMD) in Q1 Earnings Conference Call Transcripts...
```

**Lectura breve:**

`El fine-tuned no es perfecto, pero se acerca más al formato de titular financiero y mejora el match semántico.`

## Card 2 - La métrica financiera puede engañar

**Header:**

`Caso B · directional accuracy no alcanza`

**Ticker y fecha:**

`AMD · target 2023-05-04`

**Métricas:**

`Dirección correcta: GPT-2 base sí · fine-tuned no`

`ROUGE-L: 0.000 → 0.035`

**Noticia real:**

```text
- Why Advanced Micro Devices Stock Zoomed Higher on AI News Today
- Qualcomm amplifies chip gloom with 'sobering report'
```

**GPT-2 base generado:**

```text
-- 10/20 at 11 AM EDT | CNBC Newsroom – "The Dow Jones Industrial Average has fallen..."
```

**Fine-tuned generado:**

```text
... Intel's New Chips Could Make It a Winner in 2020 and Beyond
3 Reasons To Invest $100 In NVIDIA Today!
```

**Lectura breve:**

`GPT-2 base acierta la dirección, pero el texto es genérico y poco relacionado. Esto muestra por qué directional accuracy sola no valida la calidad del modelo.`

## Mensaje final de la slide

Agregar un callout inferior:

```text
La inspección cualitativa confirma la lectura de métricas: fine-tuning mejora lenguaje financiero, pero una señal direccional correcta no implica una buena narrativa.
```

## Notas del presentador

Decir algo así:

> Esta slide aterriza las métricas en ejemplos concretos. En el primer caso, GPT-2 base genera una explicación genérica sobre CPUs, mientras que el modelo fine-tuneado produce algo más parecido a un titular financiero del dominio AMD. No es perfecto, pero mejora el formato y la alineación semántica.
>
> En el segundo caso pasa algo importante: GPT-2 base acierta la dirección, pero el texto generado es una nota genérica de CNBC sobre el Dow Jones, poco relacionada con AMD. Entonces una directional accuracy correcta no significa que el modelo haya entendido el contexto financiero.
>
> La conclusión es que necesitamos mirar métricas y ejemplos juntos.

## Estilo visual

- Mantener el estilo actual de la presentación.
- Usar dos columnas.
- Usar colores distintos para GPT-2 base y fine-tuned.
- No poner textos largos: máximo 2 líneas por generación.
- Resaltar los números clave: `0.031 → 0.179` y `dirección correcta pero texto pobre`.

