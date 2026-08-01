"""Matched M2 training examples, causal labels, and exposure audits."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from token_efficient_reasoning.config import CheckpointConfig, ConditionConfig
from token_efficient_reasoning.controlled.adapters import to_evaluation_item
from token_efficient_reasoning.controlled.schema import CanonicalExample, stable_hash
from token_efficient_reasoning.hf_backend import render_model_prompt
from token_efficient_reasoning.prompting import build_user_prompt

IGNORE_INDEX = -100


class SupervisedProtocol(Protocol):
    """Shared subset of the M2 and M3 supervised-training contracts."""

    checkpoint_id: str
    train_split: str
    validation_split: str
    conditions: Sequence[Any]


@dataclass(frozen=True)
class TrainingExample:
    """One condition-specific target derived from a canonical M1 item."""

    item_id: str
    condition_id: str
    representation: str
    split: str
    task_family: str
    depth: int
    source_instance_hash: str
    user_prompt: str
    reasoning_target: str
    answer_target: str
    target_text: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizedTrainingExample:
    """A causal-LM sequence with prompt tokens excluded from the loss."""

    item_id: str
    condition_id: str
    representation: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_token_count: int
    reasoning_token_count: int
    target_token_count: int
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _target_text(
    example: CanonicalExample,
    representation: str,
    checkpoint: CheckpointConfig,
) -> str:
    return (
        example.renderings[representation]
        + checkpoint.reasoning_close
        + checkpoint.answer_open
        + example.answer
        + checkpoint.answer_close
    )


def build_training_examples(
    examples: Iterable[CanonicalExample],
    config: SupervisedProtocol,
    checkpoint: CheckpointConfig,
) -> dict[str, list[TrainingExample]]:
    """Build matched C2-C4 targets in source-manifest order."""

    if checkpoint.id != config.checkpoint_id:
        raise ValueError(
            f"checkpoint {checkpoint.id!r} does not match protocol {config.checkpoint_id!r}"
        )
    source = list(examples)
    if not source:
        raise ValueError("cannot build training targets from an empty split")
    splits = {example.split for example in source}
    allowed_splits = {config.train_split, config.validation_split}
    if len(splits) != 1 or not splits <= allowed_splits:
        raise ValueError("examples must come from exactly one configured train or validation split")

    prompt_condition = ConditionConfig(id="m2-training", instruction="")
    prompts = {
        example.id: build_user_prompt(to_evaluation_item(example), prompt_condition)
        for example in source
    }
    result: dict[str, list[TrainingExample]] = {}
    for condition in config.conditions:
        rows: list[TrainingExample] = []
        for example in source:
            if condition.representation not in example.renderings:
                raise ValueError(f"item {example.id!r} lacks {condition.representation} rendering")
            rows.append(
                TrainingExample(
                    item_id=example.id,
                    condition_id=condition.id,
                    representation=condition.representation,
                    split=example.split,
                    task_family=example.family,
                    depth=example.depth,
                    source_instance_hash=example.instance_hash,
                    user_prompt=prompts[example.id],
                    reasoning_target=example.renderings[condition.representation],
                    answer_target=example.answer,
                    target_text=_target_text(
                        example,
                        condition.representation,
                        checkpoint,
                    ),
                )
            )
        result[condition.id] = rows
    audit = audit_matched_examples(result)
    if not audit["passed"]:
        raise ValueError(f"condition matching failed: {audit['differences']}")
    return result


def audit_matched_examples(
    by_condition: dict[str, list[TrainingExample]],
) -> dict[str, Any]:
    """Confirm conditions differ only in their declared representation targets."""

    if not by_condition:
        raise ValueError("cannot audit an empty condition mapping")
    baseline_id = next(iter(by_condition))
    baseline = by_condition[baseline_id]
    differences: list[dict[str, Any]] = []
    invariant_fields = (
        "item_id",
        "split",
        "task_family",
        "depth",
        "source_instance_hash",
        "user_prompt",
        "answer_target",
    )
    for condition_id, rows in by_condition.items():
        if len(rows) != len(baseline):
            differences.append(
                {
                    "condition_id": condition_id,
                    "field": "item_count",
                    "expected": len(baseline),
                    "actual": len(rows),
                }
            )
            continue
        for index, (expected, actual) in enumerate(zip(baseline, rows, strict=True)):
            for field in invariant_fields:
                if getattr(expected, field) != getattr(actual, field):
                    differences.append(
                        {
                            "condition_id": condition_id,
                            "index": index,
                            "item_id": actual.item_id,
                            "field": field,
                        }
                    )
    item_ids = [row.item_id for row in baseline]
    return {
        "schema_version": 1,
        "passed": not differences,
        "baseline_condition_id": baseline_id,
        "condition_ids": list(by_condition),
        "item_count": len(baseline),
        "item_ids_sha256": stable_hash(item_ids),
        "differences": differences,
    }


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(int(token) for token in tokenizer.encode(text, add_special_tokens=False))


def tokenize_training_example(
    example: TrainingExample,
    tokenizer: Any,
    checkpoint: CheckpointConfig,
    *,
    max_sequence_length: int | None = None,
) -> TokenizedTrainingExample:
    """Tokenize exactly as inference starts, masking every prompt token."""

    if max_sequence_length is not None and max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    _, prompt_ids = render_model_prompt(tokenizer, checkpoint, example.user_prompt)
    reasoning_ids = _encode(tokenizer, example.reasoning_target)
    suffix = (
        checkpoint.reasoning_close
        + checkpoint.answer_open
        + example.answer_target
        + checkpoint.answer_close
    )
    suffix_ids = _encode(tokenizer, suffix)
    target_ids = reasoning_ids + suffix_ids
    input_ids = prompt_ids + target_ids
    if max_sequence_length is not None and len(input_ids) > max_sequence_length:
        raise ValueError(
            f"item {example.item_id!r} condition {example.condition_id!r} requires "
            f"{len(input_ids)} tokens, exceeding max_sequence_length={max_sequence_length}"
        )
    labels = (IGNORE_INDEX,) * len(prompt_ids) + target_ids
    return TokenizedTrainingExample(
        item_id=example.item_id,
        condition_id=example.condition_id,
        representation=example.representation,
        input_ids=input_ids,
        attention_mask=(1,) * len(input_ids),
        labels=labels,
        prompt_token_count=len(prompt_ids),
        reasoning_token_count=len(reasoning_ids),
        target_token_count=len(target_ids),
    )


def summarize_token_exposure(
    by_condition: dict[str, list[TokenizedTrainingExample]],
) -> dict[str, Any]:
    """Report supervised-token differences without pretending they are matched."""

    rows: dict[str, Any] = {}
    for condition_id, examples in by_condition.items():
        if not examples:
            raise ValueError(f"condition {condition_id!r} has no tokenized examples")
        target_counts = [example.target_token_count for example in examples]
        reasoning_counts = [example.reasoning_token_count for example in examples]
        rows[condition_id] = {
            "item_count": len(examples),
            "total_target_tokens": sum(target_counts),
            "total_reasoning_tokens": sum(reasoning_counts),
            "total_sequence_tokens": sum(len(example.input_ids) for example in examples),
            "median_sequence_tokens": statistics.median(
                len(example.input_ids) for example in examples
            ),
            "p95_sequence_tokens": _percentile(
                [len(example.input_ids) for example in examples], 0.95
            ),
            "maximum_sequence_tokens": max(len(example.input_ids) for example in examples),
            "median_target_tokens": statistics.median(target_counts),
            "median_reasoning_tokens": statistics.median(reasoning_counts),
        }
    return {"schema_version": 1, "conditions": rows}


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile over an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class CausalTrainingDataset:
    """Torch-compatible immutable view over pre-tokenized causal-LM examples."""

    def __init__(self, examples: Sequence[TokenizedTrainingExample]) -> None:
        if not examples:
            raise ValueError("training dataset must not be empty")
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {
            "input_ids": list(example.input_ids),
            "attention_mask": list(example.attention_mask),
            "labels": list(example.labels),
        }


@dataclass(frozen=True)
class CausalDataCollator:
    """Right-pad inputs and masks without introducing supervised padding labels."""

    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __post_init__(self) -> None:
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        if self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("cannot collate an empty batch")
        maximum = max(len(feature["input_ids"]) for feature in features)
        padded_length = (
            (maximum + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
        ) * self.pad_to_multiple_of
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            length = len(feature["input_ids"])
            if len(feature["attention_mask"]) != length or len(feature["labels"]) != length:
                raise ValueError("input_ids, attention_mask, and labels must have equal lengths")
            padding = padded_length - length
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [IGNORE_INDEX] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
