from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.train import load_jsonl
from src.utils.structured_output import parse_outlook
from src.utils.tickers import prediction_ticker


PREDICTION_FIELDS = [
    "ticker",
    "date_t",
    "date_t1",
    "prompt",
    "real_news",
    "generated_news",
    "real_outlook",
    "generated_outlook",
    "real_sentiment_label",
    "generated_sentiment_label",
    "real_direction_label",
    "generated_direction_label",
]


def generate_predictions(
    model_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    generation_config: dict[str, Any],
) -> list[dict[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    examples = load_jsonl(test_path)
    predictions = []
    for example in examples:
        generated = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=example["prompt"],
            generation_config=generation_config,
            device=device,
        )
        real_outlook = parse_outlook(example["completion"])
        generated_outlook = parse_outlook(generated)
        predictions.append(
            {
                "ticker": prediction_ticker(example),
                "date_t": example["date_t"],
                "date_t1": example["date_t1"],
                "prompt": example["prompt"],
                "real_news": real_outlook["narrative"],
                "generated_news": generated_outlook["narrative"],
                "real_outlook": example["completion"],
                "generated_outlook": generated,
                "real_sentiment_label": example.get("target_sentiment_label", real_outlook["sentiment"]),
                "generated_sentiment_label": generated_outlook["sentiment"],
                "real_direction_label": example.get("target_direction_label", real_outlook["direction"]),
                "generated_direction_label": generated_outlook["direction"],
            }
        )

    write_predictions(predictions, output_path)
    return predictions


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    generation_config: dict[str, Any],
    device: str = "cpu",
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    do_sample = bool(generation_config.get("do_sample", False))
    generate_kwargs = {
        "max_new_tokens": generation_config.get("max_new_tokens", 120),
        "min_new_tokens": generation_config.get("min_new_tokens", 0),
        "do_sample": do_sample,
        "repetition_penalty": generation_config.get("repetition_penalty", 1.0),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = generation_config.get("temperature", 0.8)
        generate_kwargs["top_p"] = generation_config.get("top_p", 0.9)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def write_predictions(predictions: list[dict[str, str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
