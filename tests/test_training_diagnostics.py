from __future__ import annotations

import json

import pytest

from token_efficient_reasoning.controlled.schema import CanonicalExample
from token_efficient_reasoning.training.diagnostics import diagnose_trace


def first_example(family: str | None = None) -> CanonicalExample:
    with open("data/m1/splits/validation.jsonl", encoding="utf-8") as handle:
        for line in handle:
            example = CanonicalExample.from_dict(json.loads(line))
            if family is None or example.family == family:
                return example
    raise AssertionError(f"validation fixture lacks family {family!r}")


def test_canonical_trace_is_parseable_and_replayable() -> None:
    example = first_example()

    result = diagnose_trace(example, example.renderings["r3"], "r3")

    assert result["trace_parse_success"]
    assert result["verified_prefix_transitions"] == example.depth
    assert result["complete_trace"]
    assert result["replay_valid"]
    assert result["replayed_answer"] == example.answer


def test_truncated_trace_retains_verified_prefix_diagnostic() -> None:
    example = first_example()
    first_line = example.renderings["r3"].splitlines()[0]

    result = diagnose_trace(example, first_line, "r3")

    assert result["trace_parse_success"]
    assert result["verified_prefix_transitions"] == 1
    assert not result["complete_trace"]
    assert not result["replay_valid"]


@pytest.mark.parametrize(
    ("family", "reasoning_text"),
    [
        ("multi_operation_arithmetic", "s1=5>div(5)>1\ns2=1>"),
        ("directed_graph_traversal", "e1(N18,black,N12)\ne2(N12"),
    ],
)
def test_low_budget_incomplete_tail_preserves_canonical_prefix(
    family: str,
    reasoning_text: str,
) -> None:
    example = first_example(family)

    result = diagnose_trace(
        example,
        reasoning_text,
        "r3",
        allow_trailing_fragment=True,
    )

    assert not result["trace_parse_success"]
    assert result["trailing_fragment_recovered"]
    assert result["parsed_transition_count"] == 1
    assert result["verified_prefix_transitions"] == 1
    assert result["invalid_transition_count"] == 0
    assert result["invalid_transition_event"]
    assert not result["complete_trace"]
    assert result["trace_parse_error"]


def test_unforced_malformed_suffix_does_not_recover_prefix() -> None:
    example = first_example()
    reasoning_text = example.renderings["r3"].splitlines()[0] + "\ns2=1>"

    result = diagnose_trace(example, reasoning_text, "r3")

    assert not result["trace_parse_success"]
    assert not result["trailing_fragment_recovered"]
    assert result["parsed_transition_count"] == 0
    assert result["verified_prefix_transitions"] == 0
    assert result["invalid_transition_event"]


def test_forced_noncanonical_suffix_does_not_recover_prefix() -> None:
    example = first_example()
    first_line = example.renderings["r3"].splitlines()[0]

    result = diagnose_trace(
        example,
        first_line + "\nnot the next canonical line",
        "r3",
        allow_trailing_fragment=True,
    )

    assert result["parsed_transition_count"] == 0
    assert result["verified_prefix_transitions"] == 0
    assert result["invalid_transition_event"]


def test_malformed_trace_fails_closed_as_adverse_validity_event() -> None:
    result = diagnose_trace(first_example(), "not a trace", "r3")

    assert not result["trace_parse_success"]
    assert result["verified_prefix_transitions"] == 0
    assert result["invalid_transition_count"] == 0
    assert result["invalid_transition_event"]
    assert result["trace_parse_error"]
