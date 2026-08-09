"""Executable diagnostics for model-generated controlled reasoning traces."""

from __future__ import annotations

from typing import Any

from token_efficient_reasoning.controlled.families import FAMILIES
from token_efficient_reasoning.controlled.schema import REPRESENTATIONS, CanonicalExample


def diagnose_trace(
    example: CanonicalExample,
    reasoning_text: str,
    expected_representation: str | None,
    *,
    allow_trailing_fragment: bool = False,
) -> dict[str, Any]:
    """Parse a trace and measure how much of the canonical transition path it verifies.

    A token budget commonly cuts the final rendered line in half.  Such a line still makes
    the whole trace syntactically invalid, but complete lines before it remain useful evidence
    of verified progress and are diagnosed as a parsed prefix.
    """

    stripped = reasoning_text.strip()
    lines = stripped.splitlines()
    line_count = len(lines)
    repeated_line_count = len(lines) - len(set(lines))
    base = {
        "expected_representation": expected_representation,
        "detected_representation": None,
        "trace_parse_success": False,
        "trace_parse_error": None,
        "trailing_fragment_recovered": False,
        "parsed_transition_count": 0,
        "verified_prefix_transitions": 0,
        "invalid_transition_count": 0,
        # This is deliberately distinct from invalid_transition_count.  A syntax failure is
        # an adverse validity event even though it did not produce a countable transition.
        "invalid_transition_event": True,
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

    def make_candidate(
        parsed: tuple[Any, ...],
        representation: str,
        *,
        parse_error: str | None = None,
    ) -> dict[str, Any]:
        verified_prefix = 0
        for actual, canonical in zip(parsed, example.transitions, strict=False):
            if actual != canonical:
                break
            verified_prefix += 1
        invalid_count = len(parsed) - verified_prefix
        full_parse = parse_error is None
        complete = full_parse and parsed == example.transitions
        replay_valid = False
        replayed_answer: str | None = None
        if complete:
            try:
                replayed_answer = family.replay(example, parsed)
                replay_valid = replayed_answer == example.answer
            except ValueError:
                replay_valid = False
        return {
            **base,
            "detected_representation": representation,
            "trace_parse_success": full_parse,
            "trace_parse_error": parse_error,
            "trailing_fragment_recovered": not full_parse,
            "parsed_transition_count": len(parsed),
            "verified_prefix_transitions": verified_prefix,
            "invalid_transition_count": invalid_count,
            "invalid_transition_event": not full_parse or invalid_count > 0,
            "complete_trace": complete,
            "replay_valid": replay_valid,
            "replayed_answer": replayed_answer,
        }

    def consider(candidate: dict[str, Any]) -> None:
        nonlocal best
        rank = (
            candidate["verified_prefix_transitions"],
            candidate["complete_trace"],
            candidate["trace_parse_success"],
            candidate["parsed_transition_count"],
        )
        if best is None:
            best = candidate
            return
        best_rank = (
            best["verified_prefix_transitions"],
            best["complete_trace"],
            best["trace_parse_success"],
            best["parsed_transition_count"],
        )
        if rank > best_rank:
            best = candidate

    for representation in candidates:
        try:
            parsed = family.parse(stripped, representation)
        except ValueError as error:
            formatted_error = f"{representation}:{error}"
            errors.append(formatted_error)
            # Only the final physical line may be discarded.  This avoids scanning past a
            # malformed interior line while recovering the usual budget-truncated suffix.
            if allow_trailing_fragment and len(lines) > 1:
                try:
                    parsed_prefix = family.parse("\n".join(lines[:-1]), representation)
                except ValueError:
                    continue
                prefix_length = len(parsed_prefix)
                canonical_prefix = example.transitions[:prefix_length]
                if parsed_prefix != canonical_prefix or prefix_length >= len(example.transitions):
                    continue
                canonical_next_line = family.render(
                    (example.transitions[prefix_length],), representation
                )
                trailing_line = lines[-1]
                if not canonical_next_line.startswith(trailing_line) or (
                    trailing_line == canonical_next_line
                ):
                    continue
                consider(
                    make_candidate(
                        parsed_prefix,
                        representation,
                        parse_error=formatted_error,
                    )
                )
            continue
        consider(make_candidate(parsed, representation))
    if best is not None:
        return best
    base["trace_parse_error"] = " | ".join(errors)
    return base
