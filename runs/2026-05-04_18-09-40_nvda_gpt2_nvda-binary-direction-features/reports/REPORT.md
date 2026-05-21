# Reporte del experimento

## Resumen

Este experimento fine-tunea `gpt2` para generar narrativas financieras de `NVDA` para el dia `t+1`, usando noticias y features de mercado de una ventana previa de `3` dias.

- Run: `2026-05-04_18-09-40_nvda_gpt2_nvda-binary-direction-features`
- Estado: `completed`
- Duracion: `14166.4` segundos

Tiempos por etapa:

| Etapa | Segundos |
|---|---:|
| download_data | 27.0534 |
| build_dataset | 2.3531 |
| train_model | 13623.4420 |
| generate_predictions | 490.0709 |
| evaluate | 23.3520 |

## Datos

- Dataset: `beachside1234/FNSPID`
- Activos configurados: `NVDA`
- Activos presentes en predicciones: `NVDA`
- Rango configurado: `2017-08-18` a `2023-12-16`
- Splits procesados: train=617, val=132, test=133
- Predicciones evaluadas: 133

## Metodo

- Formato de entrada: ticker, fecha, retornos, volatilidad, volumen, tendencia, momentum y noticias previas.
- Target: outlook estructurado del dia siguiente con sentimiento, direccion y narrativa.
- Entrenamiento: causal language modeling con perdida enmascarada sobre el prompt.
- Modelo base: `gpt2`
- Epocas: `3`
- Learning rate: `5e-05`
- Decoding: `do_sample=False`, `max_new_tokens=120`

## Resultados Textuales

| Metrica | Media |
|---|---:|
| ROUGE-1 | 0.0908 |
| ROUGE-2 | 0.0025 |
| ROUGE-L | 0.0634 |
| BERTScore P | 0.7260 |
| BERTScore R | 0.7423 |
| BERTScore F1 | 0.7337 |

Lectura: ROUGE bajo indica poca coincidencia literal con las noticias reales. BERTScore moderado sugiere cercania semantica general, probablemente influida por vocabulario financiero compartido.

## Resultados Semanticos

| Metrica | Valor |
|---|---:|
| Sentiment match accuracy | 0.3459 |
| Structured sentiment match accuracy | 0.7594 |
| Neutral baseline accuracy | 0.4211 |
| Mean KL divergence | 1.7602 |
| Net sentiment Pearson | -0.1122 |

Matriz real vs generado segun FinBERT:

| Real \ Generado | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 4 | 4 | 16 |
| neutral | 13 | 18 | 25 |
| positive | 14 | 15 | 24 |

![Matriz de confusion semantica](semantic_confusion_matrix.png)

Lectura: el modelo tiende a neutralizar el tono. La evaluacion semantica no supera el baseline trivial de predecir siempre neutral.

## Resultados Financieros

| Metrica | Valor |
|---|---:|
| Filas joineadas | 133.0000 |
| Directional accuracy generado | 0.5564 |
| Directional accuracy noticia real | 0.6753 |
| Cobertura senal generada | 1.0000 |
| Cobertura senal real | 0.5789 |
| Accuracy alta volatilidad | 0.5373 |
| Accuracy baja volatilidad | 0.5758 |

![Matriz de confusion financiera](financial_confusion_matrix.png)

Lectura: cuando el modelo genera una senal no neutral, la precision direccional es razonable, pero la cobertura es baja. Esto debe interpretarse como una senal selectiva, no como una estrategia completa.

## Ejemplos Cualitativos

### Caso con buena similitud textual

- Fecha target: `2023-11-04`
- Sentimiento real/generado: `neutral` / `neutral`
- Direccion real/senal generada: `1` / `1`

**Noticia real:**

> - These 2 Artificial Intelligence (AI) Stocks Have a Lot to Prove Before I'm Willing to Buy
> - 1 Massive Reason to Buy Nvidia Stock Right Now, and It's Not Artificial Intelligence (AI)

**Noticia generada:**

> -- The Top 50 Most Valuable Stock Companies Of All Time -- Why You Should Invest $250 Today And Buy Now...and Keep It For a Long Term

### Caso con sentimiento coincidente

- Fecha target: `2023-06-08`
- Sentimiento real/generado: `neutral` / `neutral`
- Direccion real/senal generada: `1` / `1`

**Noticia real:**

> - Zacks Earnings Trends Highlights: Meta Platforms and Nvidia
> - AI Could Send These 2 Nasdaq Stocks to the Moon
> - Apple Stock is on Fire. Invest in it with These 3 ETFs

**Noticia generada:**

> -$1 Billion in Sales From Nvidia and AMD In FY14; Analysts Say More Than 50% Of These Are Unsold Shares... And That's Not Even Close To What It Looks Like

## Limitaciones

- GPT-2 small tiene capacidad limitada y tiende a generar titulares genericos.
- El target usa titulares agregados por dia, no articulos completos curados.
- FinBERT mide tono financiero, pero no garantiza causalidad ni prediccion de precio.
- La metrica financiera usa la direccion estructurada generada cuando esta disponible.
- La muestra de test es chica para concluir robustez estadistica.

## Proximos Experimentos

- Comparar contra mas epochs y contra GPT-2 medium si hay VRAM.
- Probar prompts mas estructurados con menos titulares y features mas claras.
- Ajustar decoding para reducir titulares clickbait/genericos.
- Extender a varios ETFs/tickers cuando el pipeline este estable.
- Agregar una tabla comparativa entre carpetas `runs/`.
