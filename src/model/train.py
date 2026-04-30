from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class FinancialNarrativeDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, str]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = tokenize_prompt_completion(
            prompt=self.examples[index]["prompt"],
            completion=self.examples[index]["completion"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}


def train_model(
    train_path: str | Path,
    val_path: str | Path,
    model_name: str,
    output_dir: str | Path,
    max_length: int,
    train_config: dict[str, Any],
) -> Trainer:
    """Fine-tune a causal LM with loss masked over the prompt tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = FinancialNarrativeDataset(load_jsonl(train_path), tokenizer, max_length=max_length)
    eval_dataset = FinancialNarrativeDataset(load_jsonl(val_path), tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config.get(
            "per_device_eval_batch_size",
            train_config["per_device_train_batch_size"],
        ),
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        warmup_steps=train_config["warmup_steps"],
        eval_strategy="steps",
        eval_steps=train_config["eval_steps"],
        save_strategy="steps",
        save_steps=train_config["save_steps"],
        logging_dir="outputs/logs",
        logging_steps=train_config.get("logging_steps", 50),
        save_total_limit=train_config.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def tokenize_prompt_completion(
    prompt: str,
    completion: str,
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_text = completion + tokenizer.eos_token
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids.extend([tokenizer.pad_token_id] * padding_length)
        labels.extend([-100] * padding_length)
        attention_mask.extend([0] * padding_length)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_jsonl(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
