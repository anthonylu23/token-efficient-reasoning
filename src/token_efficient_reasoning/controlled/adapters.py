"""Narrow adapters from controlled data into downstream project interfaces."""

from __future__ import annotations

from token_efficient_reasoning.data import EvaluationItem

from .schema import CanonicalExample


def to_evaluation_item(example: CanonicalExample) -> EvaluationItem:
    """Expose a controlled example to the fixed-budget M0 evaluator."""

    return EvaluationItem(
        id=example.id,
        prompt=example.prompt,
        reference_answer=example.answer,
        answer_type=example.answer_type,
        task_family=example.family,
    )
