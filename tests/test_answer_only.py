from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from token_efficient_reasoning import answer_only
from token_efficient_reasoning.budget_analysis import verify_manifest
from token_efficient_reasoning.config import ExperimentConfig
from token_efficient_reasoning.controlled.families.arithmetic import ArithmeticFamily
from token_efficient_reasoning.controlled.rendering import render_example
from token_efficient_reasoning.primary.config import M3Config
from token_efficient_reasoning.training.data import (
    ANSWER_ONLY_CONDITION,
    CausalDataCollator,
    build_training_examples,
    summarize_token_exposure,
    tokenize_training_example,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return f"<user>{messages[0]['content']}</user><assistant>"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(map(ord, text))


def contract():
    primary = M3Config.from_json("configs/m3/protocol.json")
    protocol = answer_only.AnswerOnlyProtocol.from_primary(primary, "followup-test", "test-hash")
    checkpoint = next(
        c
        for c in ExperimentConfig.from_json("configs/m0.json").checkpoints
        if c.id == primary.checkpoint_id
    )
    return primary, protocol, checkpoint


def source(split="train"):
    return render_example(ArithmeticFamily().generate(seed=42, depth=1, split=split))


def test_answer_only_preserves_training_contract_and_matched_order():
    primary, protocol, checkpoint = contract()
    assert protocol.training_seeds == primary.training_seeds
    assert protocol.optimizer_updates == primary.optimizer_updates == 500
    assert protocol.inference_family != primary.analysis_version
    examples = [
        source(),
        render_example(ArithmeticFamily().generate(seed=43, depth=1, split="train")),
    ]
    rows = build_training_examples(examples, protocol, checkpoint)
    baseline = build_training_examples(examples, primary, checkpoint)
    for condition in baseline:
        assert rows[condition] == baseline[condition]
    answers = rows[ANSWER_ONLY_CONDITION]
    assert [row.item_id for row in answers] == [example.id for example in examples]
    for answer, verbose in zip(answers, rows["c2"], strict=True):
        assert answer.user_prompt == verbose.user_prompt
        assert answer.answer_target == verbose.answer_target
        assert answer.reasoning_target == ""
        assert answer.target_text == (
            checkpoint.reasoning_close
            + checkpoint.answer_open
            + answer.answer_target
            + checkpoint.answer_close
        )


def test_answer_only_masks_prompt_but_supervises_delimiters_and_answer():
    _, protocol, checkpoint = contract()
    row = build_training_examples([source()], protocol, checkpoint)[ANSWER_ONLY_CONDITION][0]
    tokenized = tokenize_training_example(row, CharacterTokenizer(), checkpoint)
    n = tokenized.prompt_token_count
    assert tokenized.reasoning_token_count == 0
    assert tokenized.labels[:n] == (-100,) * n
    assert tokenized.labels[n:] == tuple(map(ord, row.target_text))
    exposure = summarize_token_exposure({ANSWER_ONLY_CONDITION: [tokenized]})
    assert exposure["conditions"][ANSWER_ONLY_CONDITION]["total_reasoning_tokens"] == 0
    with pytest.raises(ValueError, match="exceeding"):
        tokenize_training_example(row, CharacterTokenizer(), checkpoint, max_sequence_length=8)
    batch = CausalDataCollator(0, 8)(
        [
            {
                "input_ids": list(tokenized.input_ids),
                "labels": list(tokenized.labels),
                "attention_mask": list(tokenized.attention_mask),
            }
        ]
    )
    length = len(tokenized.input_ids)
    assert all(value == -100 for value in batch["labels"][0, length:].tolist())
    assert not any(batch["attention_mask"][0, length:].tolist())


@pytest.mark.parametrize(
    "id,representation", [("c2", "answer_only"), (ANSWER_ONLY_CONDITION, "r0")]
)
def test_reserved_condition_cannot_alias_a_primary_condition(id, representation):
    _, protocol, checkpoint = contract()
    protocol = replace(protocol, conditions=(answer_only.FollowupCondition(id, representation),))
    with pytest.raises(ValueError, match="reserved"):
        build_training_examples([source()], protocol, checkpoint)


def test_primary_config_still_rejects_answer_only_condition():
    config = json.loads(Path("configs/m3/protocol.json").read_text())
    config["conditions"].append({"id": ANSWER_ONLY_CONDITION, "representation": "answer_only"})
    with pytest.raises(ValueError):
        M3Config.from_dict(config)


def test_cpu_preparation_manifest_roundtrip_and_no_overwrite(tmp_path, monkeypatch):
    primary, _, _ = contract()
    monkeypatch.setattr(
        answer_only, "verify_manifest", lambda path: primary.dataset_manifest_sha256
    )
    monkeypatch.setattr(answer_only, "load_examples", lambda path: [source(path.stem)])
    output = tmp_path / "prepared"
    report = answer_only.prepare(
        Path("configs/followup/answer_only.json"),
        Path.cwd(),
        output,
        tokenizer=CharacterTokenizer(),
    )
    assert report["training_performed"] is False
    assert report["splits"]["train"]["matching"]["passed"]
    verify_manifest(output)
    targets = output / "targets" / ANSWER_ONLY_CONDITION / "train.jsonl"
    assert json.loads(targets.read_text())["reasoning_target"] == ""
    with pytest.raises(FileExistsError):
        answer_only.prepare(
            Path("configs/followup/answer_only.json"),
            Path.cwd(),
            output,
            tokenizer=CharacterTokenizer(),
        )


def test_preparation_rejects_unpinned_inputs_before_writing(tmp_path):
    config = json.loads(Path("configs/followup/answer_only.json").read_text())
    config["source_protocol_sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hash mismatch"):
        answer_only.prepare(path, Path.cwd(), tmp_path / "bad", tokenizer=CharacterTokenizer())
    assert not (tmp_path / "bad").exists()
