from __future__ import annotations

import json
import os

import pytest

from token_efficient_reasoning.config import ExperimentConfig
from token_efficient_reasoning.controlled.schema import CanonicalExample
from token_efficient_reasoning.training.config import M2Config
from token_efficient_reasoning.training.data import (
    IGNORE_INDEX,
    build_training_examples,
    tokenize_training_example,
)

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="model tests disabled"),
]


@pytest.mark.parametrize("checkpoint_id", ["qwen3-0.6b", "deepseek-r1-distill-qwen-1.5b"])
def test_pinned_tokenizer_protocol_boundaries(checkpoint_id: str) -> None:
    from transformers import AutoTokenizer

    config = ExperimentConfig.from_json("configs/m0.json")
    checkpoint = next(item for item in config.checkpoints if item.id == checkpoint_id)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.model_id, revision=checkpoint.tokenizer_revision
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 1 + 1?"}],
        tokenize=False,
        add_generation_prompt=True,
        **checkpoint.template_kwargs,
    )
    if not rendered.endswith(checkpoint.reasoning_prefix):
        rendered += checkpoint.reasoning_prefix

    assert rendered.endswith("<think>\n")
    assert tokenizer.encode(checkpoint.reasoning_close, add_special_tokens=False)
    assert tokenizer.encode(checkpoint.answer_open, add_special_tokens=False)
    assert tokenizer.encode(checkpoint.answer_close, add_special_tokens=False)


def test_pinned_qwen_tokenizer_matches_m2_training_contract() -> None:
    from transformers import AutoTokenizer

    m2 = M2Config.from_json("configs/m2/smoke_protocol.json")
    m0 = ExperimentConfig.from_json(m2.m0_config_path)
    checkpoint = next(item for item in m0.checkpoints if item.id == m2.checkpoint_id)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.model_id,
        revision=checkpoint.tokenizer_revision,
    )
    with open("data/m1/splits/train.jsonl", encoding="utf-8") as handle:
        source = CanonicalExample.from_dict(json.loads(handle.readline()))

    rows = build_training_examples([source], m2, checkpoint)

    for condition_id, examples in rows.items():
        tokenized = tokenize_training_example(examples[0], tokenizer, checkpoint)
        representation = examples[0].representation
        assert (
            tokenized.reasoning_token_count
            == source.token_measurements[checkpoint.id][representation]
        )
        assert set(tokenized.labels[: tokenized.prompt_token_count]) == {IGNORE_INDEX}
        assert (
            tokenized.labels[tokenized.prompt_token_count :]
            == tokenized.input_ids[tokenized.prompt_token_count :]
        )
        assert tokenized.condition_id == condition_id
