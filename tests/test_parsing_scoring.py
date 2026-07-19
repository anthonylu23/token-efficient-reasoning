from __future__ import annotations

import pytest

from token_efficient_reasoning.parsing import parse_regions
from token_efficient_reasoning.scoring import normalize_answer, score_answer


def test_parse_valid_regions() -> None:
    parsed = parse_regions("<think>1 + 1 = 2</think><answer>2</answer>")

    assert parsed.ok
    assert parsed.reasoning == "1 + 1 = 2"
    assert parsed.answer == "2"


@pytest.mark.parametrize(
    "text,error",
    [
        ("<think>x</think><answer>2", "missing:</answer>"),
        ("<think>x</think><answer>2</answer><answer>3</answer>", "duplicate:<answer>"),
        ("<answer>2</answer><think>x</think>", "misordered_delimiters"),
        ("<think></think><answer>2</answer>", "empty_reasoning"),
        ("<think>x</think><answer></answer>", "empty_answer"),
    ],
)
def test_parse_malformed_regions(text: str, error: str) -> None:
    parsed = parse_regions(text)

    assert not parsed.ok
    assert error in parsed.errors


def test_numeric_normalization_uses_last_number() -> None:
    assert normalize_answer("Therefore, 1 + 1 = 2", "integer") == "2"
    assert normalize_answer("1,250.00", "decimal") == "1250"


def test_score_answer_is_fail_closed_without_number() -> None:
    correct, prediction, reference = score_answer("unknown", "12", "integer")

    assert not correct
    assert prediction == ""
    assert reference == "12"
