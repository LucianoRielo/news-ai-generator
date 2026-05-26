# Ejemplos cualitativos para la presentacion

Estos ejemplos estan pensados para acompanar la tabla de metricas. La idea es mostrar que las metricas agregadas necesitan lectura cualitativa: un modelo puede acertar direccion y aun asi generar una narrativa pobre.

## Fine-tuning improves semantic alignment

- Ticker: `AMD`
- Fecha target: `2023-06-21`
- Lectura: El fine-tuned logra match semantico donde GPT-2 base no, y ademas produce una narrativa mas cercana al dominio.

| Metrica | GPT-2 base | Fine-tuned / variante |
|---|---:|---:|
| ROUGE-L | 0.0305 | 0.1786 |
| Sentimiento generado | positive | neutral |
| Direccion correcta | no | no |

**Noticia real:**

> - Noteworthy Wednesday Option Activity: SBGI, AMD, HUM
> - Unusual Options Activity and Flow in Advanced Micro Devices (AMD)

**GPT-2 base:**

> AMD has been in talks with Intel about their next generation of CPUs and GPUs (the new chips are expected by early 2017). It is believed that they will be able offer more than just an "AI" chip but also include some other features such as intelligent computing capabilities like GPU acceleration which could allow them access into markets where there isn't much demand from traditional CPU vendors due mainly because it would require less power consumption compared against current processors on average at around 1GHz per core while still being capable enough under normal conditions without any additional cost associated costs related hardware...

**Fine-tuned:**

> ...And More on Intel and Advanced Micro Devices (AMD) in Q1 Earnings Conference Call Transcripts Posted Today by Micron Technology News Service On Mar 21st., 2015 at 10 AM ET Share [Read more...]

## Financial metric can be misleading

- Ticker: `AMD`
- Fecha target: `2023-05-04`
- Lectura: GPT-2 base acierta la direccion pero con una narrativa semanticamente pobre; esto muestra por que directional accuracy sola no alcanza.

| Metrica | GPT-2 base | Fine-tuned / variante |
|---|---:|---:|
| ROUGE-L | 0.0000 | 0.0351 |
| Sentimiento generado | negative | neutral |
| Direccion correcta | si | no |

**Noticia real:**

> - Why Advanced Micro Devices Stock Zoomed Higher on AI News Today
> - Qualcomm amplifies chip gloom with 'sobering report'

**GPT-2 base:**

> -- 10/20 at 11 AM EDT | CNBC Newsroom – "The Dow Jones Industrial Average has fallen by more than 1 percent since its peak in late September and now stands just shy a point above record highs."

**Fine-tuned:**

> ... Intel's New Chips Could Make It a Winner in 2020 and Beyond
> 3 Reasons To Invest $100 In NVIDIA Today! 1), 3 Ways You Can Play Crypto

## Best fine-tuned textual example

- Ticker: `NVDA`
- Fecha target: `2023-12-15`
- Lectura: Caso con mayor ROUGE-L del run fine-tuneado NVDA+AMD; util para mostrar que el modelo aprende formato y vocabulario financiero.

| Metrica | GPT-2 base | Fine-tuned / variante |
|---|---:|---:|
| ROUGE-L | 0.0625 | 0.2326 |
| Sentimiento generado | neutral | neutral |
| Direccion correcta | si | no |

**Noticia real:**

> - 3 Growth Stocks to Buy and Hold Forever
> - Why Intel Stock Popped Today

**GPT-2 base:**

> 1st day - 10/30 at 11am EST | 1pm PST / 2pm EDT
> 
> ...and more news from around the world here .

**Fine-tuned:**

> - Why AMD Is a Buy Now and Hold On Forever...and What It Means For NVIDIA Investors Today
> 
> /rblinkflux via CNBC / Bloomberg

## Fine-tuned failure case

- Ticker: `NVDA`
- Fecha target: `2023-08-11`
- Lectura: Fallo claro del modelo fine-tuneado; conviene mostrarlo para que la defensa sea critica y no triunfalista.

| Metrica | GPT-2 base | Fine-tuned / variante |
|---|---:|---:|
| ROUGE-L | 0.0698 | 0.0000 |
| Sentimiento generado | neutral | neutral |
| Direccion correcta | no | no |

**Noticia real:**

> - 3 Stocks That Could Make You a Millionaire by 2024
> - US STOCKS-S&P 500, Nasdaq fall as megacaps slide after producer prices data

**GPT-2 base:**

> 1) New York Times reports that Apple is considering buying back its stock at $3 per share (or about 2% above current market value). It's not clear if it will do so before next year but there are signs they may have already done just such an deal with Facebook last week .  I'm sure we'll see more details soon...

**Fine-tuned:**

> ... 2 Biggest Upside Charts For Wednesday Morning Market Share : AMD (NASDAQV), NVIDIA (NVDC) AND TSLA NKEYS

## Variante con features direccionales

## Best directional-features example

- Ticker: `NVDA`
- Fecha target: `2023-11-04`
- Lectura: Ejemplo del run con features direccionales: senal correcta con cobertura total, util para discutir la variante mas orientada a la metrica financiera.

| Metrica | GPT-2 base | Fine-tuned / variante |
|---|---:|---:|
| ROUGE-L |  | 0.1695 |
| Sentimiento generado |  | neutral |
| Direccion correcta |  | si |

**Noticia real:**

> - These 2 Artificial Intelligence (AI) Stocks Have a Lot to Prove Before I'm Willing to Buy
> - 1 Massive Reason to Buy Nvidia Stock Right Now, and It's Not Artificial Intelligence (AI)

**Variante:**

> -- The Top 50 Most Valuable Stock Companies Of All Time -- Why You Should Invest $250 Today And Buy Now...and Keep It For a Long Term
