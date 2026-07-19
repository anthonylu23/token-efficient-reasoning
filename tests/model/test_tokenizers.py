from __future__ import annotations

import os

import pytest

from token_efficient_reasoning.config import ExperimentConfig

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
