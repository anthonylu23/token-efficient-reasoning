"""Conservative exact-answer normalization and scoring."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_INTEGER_PATTERN = re.compile(r"[-+]?\d[\d,]*")
_DECIMAL_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")


def normalize_answer(value: str, answer_type: str) -> str:
    stripped = value.strip()
    if answer_type == "exact":
        return " ".join(stripped.casefold().split())
    pattern = _INTEGER_PATTERN if answer_type == "integer" else _DECIMAL_PATTERN
    matches = pattern.findall(stripped)
    if not matches:
        return ""
    candidate = matches[-1].replace(",", "")
    if answer_type == "integer":
        try:
            return str(int(candidate))
        except ValueError:
            return ""
    try:
        normalized = Decimal(candidate).normalize()
    except InvalidOperation:
        return ""
    return format(normalized, "f")


def score_answer(prediction: str, reference: str, answer_type: str) -> tuple[bool, str, str]:
    normalized_prediction = normalize_answer(prediction, answer_type)
    normalized_reference = normalize_answer(reference, answer_type)
    return (
        bool(normalized_prediction) and normalized_prediction == normalized_reference,
        normalized_prediction,
        normalized_reference,
    )
