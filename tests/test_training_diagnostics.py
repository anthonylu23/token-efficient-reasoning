from __future__ import annotations

import json

from token_efficient_reasoning.controlled.schema import CanonicalExample
from token_efficient_reasoning.training.diagnostics import diagnose_trace


def first_example() -> CanonicalExample:
    with open("data/m1/splits/validation.jsonl", encoding="utf-8") as handle:
        return CanonicalExample.from_dict(json.loads(handle.readline()))


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


def test_malformed_trace_fails_closed() -> None:
    result = diagnose_trace(first_example(), "not a trace", "r3")

    assert not result["trace_parse_success"]
    assert result["verified_prefix_transitions"] == 0
    assert result["trace_parse_error"]
