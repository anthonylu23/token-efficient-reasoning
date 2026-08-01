"""Executable diagnostics for model-generated controlled reasoning traces."""

from __future__ import annotations

from typing import Any

from token_efficient_reasoning.controlled.families import FAMILIES
from token_efficient_reasoning.controlled.schema import REPRESENTATIONS, CanonicalExample


def diagnose_trace(
    example: CanonicalExample,
    reasoning_text: str,
    expected_representation: str | None,
) -> dict[str, Any]:
    """Parse a trace and measure how much of the canonical transition path it verifies."""

    stripped = reasoning_text.strip()
    line_count = len(stripped.splitlines()) if stripped else 0
    lines = stripped.splitlines()
    repeated_line_count = len(lines) - len(set(lines))
    base = {
        "expected_representation": expected_representation,
        "detected_representation": None,
        "trace_parse_success": False,
        "trace_parse_error": None,
        "parsed_transition_count": 0,
        "verified_prefix_transitions": 0,
        "invalid_transition_count": 0,
        "complete_trace": False,
        "replay_valid": False,
        "replayed_answer": None,
        "reasoning_empty": not stripped,
        "reasoning_line_count": line_count,
        "repeated_line_count": repeated_line_count,
        "prompt_copied": bool(stripped and example.prompt in stripped),
    }
    if not stripped:
        base["trace_parse_error"] = "empty reasoning"
        return base
    family = FAMILIES[example.family]
    candidates = (
        (expected_representation,)
        if expected_representation is not None
        else tuple(REPRESENTATIONS)
    )
    best: dict[str, Any] | None = None
    errors: list[str] = []
    for representation in candidates:
        try:
            parsed = family.parse(stripped, representation)
        except ValueError as error:
            errors.append(f"{representation}:{error}")
            continue
        verified_prefix = 0
        for actual, canonical in zip(parsed, example.transitions, strict=False):
            if actual != canonical:
                break
            verified_prefix += 1
        complete = parsed == example.transitions
        replay_valid = False
        replayed_answer: str | None = None
        if complete:
            try:
                replayed_answer = family.replay(example, parsed)
                replay_valid = replayed_answer == example.answer
            except ValueError:
                replay_valid = False
        candidate = {
            **base,
            "detected_representation": representation,
            "trace_parse_success": True,
            "parsed_transition_count": len(parsed),
            "verified_prefix_transitions": verified_prefix,
            "invalid_transition_count": len(parsed) - verified_prefix,
            "complete_trace": complete,
            "replay_valid": replay_valid,
            "replayed_answer": replayed_answer,
        }
        if best is None or (
            candidate["verified_prefix_transitions"],
            candidate["complete_trace"],
        ) > (
            best["verified_prefix_transitions"],
            best["complete_trace"],
        ):
            best = candidate
    if best is not None:
        return best
    base["trace_parse_error"] = " | ".join(errors)
    return base
