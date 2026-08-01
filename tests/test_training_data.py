from __future__ import annotations

import json
from dataclasses import replace

import pytest

from token_efficient_reasoning.config import ExperimentConfig
from token_efficient_reasoning.controlled.schema import CanonicalExample
from token_efficient_reasoning.training.config import M2Config
from token_efficient_reasoning.training.data import (
    IGNORE_INDEX,
    CausalDataCollator,
    audit_matched_examples,
    build_training_examples,
    summarize_token_exposure,
    tokenize_training_example,
)


class CharacterTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert kwargs == {"enable_thinking": True}
        return f"<user>{messages[0]['content']}</user><assistant>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def repository_contract():
    config = M2Config.from_json("configs/m2/smoke_protocol.json")
    m0 = ExperimentConfig.from_json(config.m0_config_path)
    checkpoint = next(item for item in m0.checkpoints if item.id == config.checkpoint_id)
    return config, checkpoint


def first_training_example() -> CanonicalExample:
    with open("data/m1/splits/train.jsonl", encoding="utf-8") as handle:
        return CanonicalExample.from_dict(json.loads(handle.readline()))


def test_conditions_share_questions_and_only_change_reasoning_targets() -> None:
    config, checkpoint = repository_contract()
    source = first_training_example()

    by_condition = build_training_examples([source], config, checkpoint)
    audit = audit_matched_examples(by_condition)

    assert audit["passed"]
    assert audit["item_count"] == 1
    assert by_condition["c2"][0].reasoning_target == source.renderings["r0"]
    assert by_condition["c3"][0].reasoning_target == source.renderings["r2"]
    assert by_condition["c4"][0].reasoning_target == source.renderings["r3"]
    assert {rows[0].user_prompt for rows in by_condition.values()} == {
        by_condition["c2"][0].user_prompt
    }
    assert {rows[0].answer_target for rows in by_condition.values()} == {source.answer}


def test_tokenization_masks_prompt_and_supervises_complete_target() -> None:
    config, checkpoint = repository_contract()
    row = build_training_examples([first_training_example()], config, checkpoint)["c4"][0]
    tokenizer = CharacterTokenizer()

    tokenized = tokenize_training_example(row, tokenizer, checkpoint)

    prompt_labels = tokenized.labels[: tokenized.prompt_token_count]
    target_labels = tokenized.labels[tokenized.prompt_token_count :]
    assert set(prompt_labels) == {IGNORE_INDEX}
    assert target_labels == tokenized.input_ids[tokenized.prompt_token_count :]
    assert tokenized.reasoning_token_count == len(row.reasoning_target)
    assert tokenized.target_token_count > tokenized.reasoning_token_count
    assert len(tokenized.input_ids) == len(tokenized.attention_mask) == len(tokenized.labels)


def test_tokenization_rejects_overflow_instead_of_truncating() -> None:
    config, checkpoint = repository_contract()
    row = build_training_examples([first_training_example()], config, checkpoint)["c4"][0]

    with pytest.raises(ValueError, match="exceeding max_sequence_length"):
        tokenize_training_example(
            row,
            CharacterTokenizer(),
            checkpoint,
            max_sequence_length=8,
        )


def test_condition_audit_reports_invariant_changes() -> None:
    config, checkpoint = repository_contract()
    rows = build_training_examples([first_training_example()], config, checkpoint)
    rows["c4"][0] = replace(rows["c4"][0], answer_target="tampered")

    audit = audit_matched_examples(rows)

    assert not audit["passed"]
    assert audit["differences"][0]["field"] == "answer_target"


def test_token_exposure_is_reported_per_condition() -> None:
    config, checkpoint = repository_contract()
    rows = build_training_examples([first_training_example()], config, checkpoint)
    tokenizer = CharacterTokenizer()
    tokenized = {
        condition: [
            tokenize_training_example(example, tokenizer, checkpoint) for example in examples
        ]
        for condition, examples in rows.items()
    }

    report = summarize_token_exposure(tokenized)

    assert set(report["conditions"]) == {"c2", "c3", "c4"}
    assert (
        report["conditions"]["c2"]["total_reasoning_tokens"]
        > report["conditions"]["c4"]["total_reasoning_tokens"]
    )


def test_causal_collator_masks_right_padding() -> None:
    collator = CausalDataCollator(pad_token_id=0, pad_to_multiple_of=4)

    batch = collator(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3, 0], [4, 5, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert batch["labels"].tolist() == [
        [-100, 2, 3, -100],
        [-100, 5, -100, -100],
    ]
