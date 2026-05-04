from __future__ import annotations

import json
from pathlib import Path

import torch

from src.model.generate import PREDICTION_FIELDS, generate_one, score_best_label, write_predictions
from src.model.train import FinancialNarrativeDataset, tokenize_prompt_completion
from src.utils.structured_output import parse_outlook


ROOT = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) % 50 + 2 for character in text]


def test_tokenization_masks_prompt_labels() -> None:
    tokenizer = TinyTokenizer()
    prompt = "[TICKER: SPY]\n[NEXT DAY OUTLOOK]\nSentiment:"
    completion = " positive\nDirection: up\nNews:\n- Market rallies"

    encoded = tokenize_prompt_completion(prompt, completion, tokenizer, max_length=160)
    prompt_length = len(tokenizer.encode(prompt))

    assert encoded["labels"][:prompt_length] == [-100] * prompt_length
    assert any(label != -100 for label in encoded["labels"][prompt_length:])
    assert len(encoded["input_ids"]) == 160
    assert len(encoded["attention_mask"]) == 160
    assert len(encoded["labels"]) == 160


def test_dataset_returns_model_ready_tensors() -> None:
    dataset = FinancialNarrativeDataset(
        examples=[{"prompt": "prompt", "completion": "completion"}],
        tokenizer=TinyTokenizer(),
        max_length=32,
    )

    item = dataset[0]

    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape[0] == 32
    assert item["labels"].shape[0] == 32


class TinyCallableTokenizer(TinyTokenizer):
    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        del return_tensors
        return {"input_ids": torch.tensor([self.encode(text)], dtype=torch.long)}

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "generated headline"


class TinyModel:
    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        generated = torch.tensor([[11, 12, 13]], dtype=torch.long)
        return torch.cat([input_ids.cpu(), generated], dim=1)


def test_generate_one_returns_only_generated_text() -> None:
    generated = generate_one(
        model=TinyModel(),
        tokenizer=TinyCallableTokenizer(),
        prompt="prompt",
        generation_config={"do_sample": False, "max_new_tokens": 5, "min_new_tokens": 1},
    )

    assert generated == "generated headline"


def test_parse_outlook_supports_tagged_and_legacy_formats() -> None:
    tagged = parse_outlook("[SENTIMENT=positive]\n[DIRECTION=up]\n[NARRATIVE]\n- Market rallies")
    legacy = parse_outlook("Sentiment: negative\nDirection: down\nNews:\n- Market falls")

    assert tagged == {
        "sentiment": "positive",
        "direction": "up",
        "narrative": "- Market rallies",
    }
    assert legacy == {
        "sentiment": "negative",
        "direction": "down",
        "narrative": "- Market falls",
    }


class ScoringTokenizer(TinyTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        mapping = {" good": [10], " bad": [20]}
        return mapping.get(text, [2, *mapping.get(text[-5:], [30])])


class ScoringModel:
    def __call__(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, 64), -10.0)
        logits[:, :, 10] = 5.0
        logits[:, :, 20] = 1.0
        return type("Output", (), {"logits": logits})


def test_score_best_label_uses_closed_options() -> None:
    label = score_best_label(
        model=ScoringModel(),
        tokenizer=ScoringTokenizer(),
        prompt="prompt",
        labels=["bad", "good"],
    )

    assert label == "good"


def test_write_predictions_jsonl() -> None:
    path = ROOT / "outputs" / "generations" / "test_predictions.jsonl"
    prediction = {
        "ticker": "SPY",
        "date_t": "2024-01-01",
        "date_t1": "2024-01-02",
        "prompt": "prompt",
        "real_news": "real",
        "generated_news": "generated",
        "real_outlook": "Sentiment: positive\nDirection: up\nNews:\nreal",
        "generated_outlook": "Sentiment: positive\nDirection: up\nNews:\ngenerated",
        "real_sentiment_label": "positive",
        "generated_sentiment_label": "positive",
        "real_direction_label": "up",
        "generated_direction_label": "up",
    }

    write_predictions([prediction], path)

    row = json.loads(path.read_text(encoding="utf-8"))
    assert list(row) == PREDICTION_FIELDS
    assert row["generated_news"]
    assert row["generated_news"] != row["prompt"]
    path.unlink()
