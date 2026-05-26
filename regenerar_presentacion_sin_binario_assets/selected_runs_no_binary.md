# Runs seleccionados sin binario

| Run | Tickers | Train | Test | ROUGE-L | BERT F1 | Sent. match | Dir. acc. | Coverage | p-value | Best ppl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 no fine-tune | NVDA,AMD | 1517 | 326 | 0.040 |  | 0.273 | 0.549 | 1.000 | 0.085 |  |
| SPY/QQQ fine-tuned | SPY,QQQ | 1410 | 303 | 0.062 | 0.732 | 0.386 | 0.528 | 0.472 | 0.557 | 14.659 |
| NVDA single | NVDA | 618 | 133 | 0.070 | 0.744 | 0.346 | 0.462 | 0.591 | 0.572 | 29.648 |
| NVDA structured/label scoring | NVDA | 617 | 133 | 0.066 | 0.740 | 0.376 | 0.496 | 1.000 | 1.000 | 16.567 |
| NVDA+AMD fine-tuned | NVDA,AMD | 1517 | 326 | 0.074 | 0.738 | 0.442 | 0.530 | 0.361 | 0.579 | 19.292 |