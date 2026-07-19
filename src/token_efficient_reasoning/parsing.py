"""Strict text parsing for canonical reasoning and answer regions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedRegions:
    reasoning: str
    answer: str
    status: str
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status == "valid"


def parse_regions(
    text: str,
    *,
    reasoning_open: str = "<think>",
    reasoning_close: str = "</think>",
    answer_open: str = "<answer>",
    answer_close: str = "</answer>",
) -> ParsedRegions:
    markers = (reasoning_open, reasoning_close, answer_open, answer_close)
    errors: list[str] = []
    for marker in markers:
        count = text.count(marker)
        if count == 0:
            errors.append(f"missing:{marker}")
        elif count > 1:
            errors.append(f"duplicate:{marker}")
    if errors:
        return ParsedRegions(reasoning="", answer="", status="malformed", errors=tuple(errors))

    positions = tuple(text.index(marker) for marker in markers)
    if positions != tuple(sorted(positions)):
        return ParsedRegions(
            reasoning="", answer="", status="malformed", errors=("misordered_delimiters",)
        )

    reasoning_start = positions[0] + len(reasoning_open)
    reasoning = text[reasoning_start : positions[1]].strip()
    answer_start = positions[2] + len(answer_open)
    answer = text[answer_start : positions[3]].strip()
    if not reasoning:
        errors.append("empty_reasoning")
    if not answer:
        errors.append("empty_answer")
    status = "valid" if not errors else "malformed"
    return ParsedRegions(reasoning=reasoning, answer=answer, status=status, errors=tuple(errors))
