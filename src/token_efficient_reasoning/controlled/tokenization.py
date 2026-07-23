"""Pinned-tokenizer measurement and compression acceptance checks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from token_efficient_reasoning.config import ExperimentConfig

from .config import M1Config
from .schema import REPRESENTATIONS, CanonicalExample


def load_pinned_tokenizers(
    config: M1Config, *, m0_config_path: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from transformers import AutoTokenizer

    m0_config = ExperimentConfig.from_json(m0_config_path)
    selected = {
        checkpoint.id: checkpoint
        for checkpoint in m0_config.checkpoints
        if checkpoint.id in config.tokenizer_checkpoint_ids
    }
    missing = sorted(set(config.tokenizer_checkpoint_ids) - selected.keys())
    if missing:
        raise ValueError(f"M1 config references unknown M0 tokenizers: {missing}")
    tokenizers: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for checkpoint_id in config.tokenizer_checkpoint_ids:
        checkpoint = selected[checkpoint_id]
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.model_id,
            revision=checkpoint.tokenizer_revision,
            trust_remote_code=checkpoint.trust_remote_code,
        )
        tokenizers[checkpoint_id] = tokenizer
        provenance[checkpoint_id] = {
            "model_id": checkpoint.model_id,
            "tokenizer_revision": checkpoint.tokenizer_revision,
            "name_or_path": tokenizer.name_or_path,
            "vocab_size": len(tokenizer),
        }
    return tokenizers, provenance


def measure_example(example: CanonicalExample, tokenizers: dict[str, Any]) -> CanonicalExample:
    measurements: dict[str, dict[str, int]] = {}
    for checkpoint_id, tokenizer in tokenizers.items():
        measurements[checkpoint_id] = {
            representation: len(
                tokenizer.encode(example.renderings[representation], add_special_tokens=False)
            )
            for representation in REPRESENTATIONS
        }
    return replace(example, token_measurements=measurements)


def compression_rejection_reason(
    example: CanonicalExample, required_representations: tuple[str, ...]
) -> str | None:
    for checkpoint_id, measurements in example.token_measurements.items():
        r0_count = measurements["r0"]
        for representation in required_representations:
            if measurements[representation] >= r0_count:
                return (
                    f"{representation} is not shorter than r0 for {checkpoint_id}: "
                    f"{measurements[representation]} >= {r0_count}"
                )
    return None
