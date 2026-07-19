"""Token-native two-phase generation with explicit reasoning and answer budgets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

StopCallback = Callable[[tuple[int, ...]], bool]


@dataclass(frozen=True)
class BackendOutput:
    """New tokens returned by a generation backend for one phase."""

    token_ids: tuple[int, ...]
    finish_reason: str
    latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.finish_reason not in {"stop", "eos", "length"}:
            raise ValueError(f"unsupported finish reason: {self.finish_reason}")


class GenerationBackend(Protocol):
    """Minimal interface needed by the budget protocol."""

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        should_stop: StopCallback,
    ) -> BackendOutput: ...


@dataclass(frozen=True)
class PhaseResult:
    raw_token_ids: tuple[int, ...]
    content_token_ids: tuple[int, ...]
    delimiter_token_ids: tuple[int, ...]
    overflow_token_ids: tuple[int, ...]
    termination: str
    finish_reason: str
    latency_seconds: float

    @property
    def delimiter_was_generated(self) -> bool:
        return bool(self.delimiter_token_ids)


@dataclass(frozen=True)
class BudgetedGeneration:
    prompt_token_ids: tuple[int, ...]
    reasoning: PhaseResult
    answer: PhaseResult
    reasoning_close_token_ids: tuple[int, ...]
    answer_open_token_ids: tuple[int, ...]
    answer_close_token_ids: tuple[int, ...]

    @property
    def forced_reasoning_transition(self) -> bool:
        return self.reasoning.termination != "natural_close"

    @property
    def effective_token_ids(self) -> tuple[int, ...]:
        return (
            self.prompt_token_ids
            + self.reasoning.content_token_ids
            + self.reasoning_close_token_ids
            + self.answer_open_token_ids
            + self.answer.content_token_ids
            + self.answer_close_token_ids
        )


def _longest_delimiter_prefix_suffix(generated: tuple[int, ...], delimiter: tuple[int, ...]) -> int:
    """Return the suffix length that could still complete ``delimiter``."""

    if not generated or len(delimiter) <= 1:
        return 0
    maximum = min(len(generated), len(delimiter) - 1)
    for length in range(maximum, 0, -1):
        if generated[-length:] == delimiter[:length]:
            return length
    return 0


def _matching_delimiter(
    generated: tuple[int, ...], delimiters: tuple[tuple[int, ...], ...]
) -> tuple[int, ...]:
    matches = [
        delimiter
        for delimiter in delimiters
        if len(generated) >= len(delimiter) and generated[-len(delimiter) :] == delimiter
    ]
    return max(matches, key=len, default=())


def phase_should_stop_any(
    generated: tuple[int, ...],
    delimiters: tuple[tuple[int, ...], ...],
    content_budget: int,
) -> bool:
    if not delimiters or any(not delimiter for delimiter in delimiters):
        raise ValueError("delimiter variants must be non-empty")
    if _matching_delimiter(generated, delimiters):
        return True
    pending = max(
        _longest_delimiter_prefix_suffix(generated, delimiter) for delimiter in delimiters
    )
    committed_content = len(generated) - pending
    return committed_content >= content_budget


def phase_should_stop(
    generated: tuple[int, ...], delimiter: tuple[int, ...], content_budget: int
) -> bool:
    """Stop after a full delimiter or ``content_budget`` committed content tokens.

    A possible partial delimiter may extend past the nominal boundary. If it does
    not complete, the overflow is discarded from the effective transcript.
    """

    return phase_should_stop_any(generated, (delimiter,), content_budget)


def finalize_phase(
    output: BackendOutput,
    *,
    delimiter: tuple[int, ...],
    delimiter_variants: tuple[tuple[int, ...], ...] | None = None,
    content_budget: int,
    phase_name: str,
) -> PhaseResult:
    if content_budget <= 0:
        raise ValueError("content_budget must be positive")
    if not delimiter:
        raise ValueError("delimiter must not be empty")

    raw = output.token_ids
    variants = delimiter_variants or (delimiter,)
    matched_delimiter = _matching_delimiter(raw, variants)
    if matched_delimiter:
        content = raw[: -len(matched_delimiter)]
        if len(content) <= content_budget:
            return PhaseResult(
                raw_token_ids=raw,
                content_token_ids=content,
                delimiter_token_ids=matched_delimiter,
                overflow_token_ids=(),
                termination="natural_close",
                finish_reason=output.finish_reason,
                latency_seconds=output.latency_seconds,
            )

    content = raw[:content_budget]
    overflow = raw[content_budget:]
    if len(content) >= content_budget:
        termination = "budget_forced" if phase_name == "reasoning" else "allowance_exhausted"
    elif output.finish_reason == "eos":
        termination = "eos"
    else:
        termination = "incomplete"
    return PhaseResult(
        raw_token_ids=raw,
        content_token_ids=content,
        delimiter_token_ids=(),
        overflow_token_ids=overflow,
        termination=termination,
        finish_reason=output.finish_reason,
        latency_seconds=output.latency_seconds,
    )


class BudgetedGenerator:
    """Execute reasoning and answer generation with independent token budgets."""

    def __init__(self, backend: GenerationBackend) -> None:
        self.backend = backend

    def generate(
        self,
        prompt_token_ids: tuple[int, ...],
        *,
        reasoning_budget: int,
        answer_allowance: int,
        reasoning_close_token_ids: tuple[int, ...],
        answer_open_token_ids: tuple[int, ...],
        answer_close_token_ids: tuple[int, ...],
        reasoning_close_variants: tuple[tuple[int, ...], ...] | None = None,
        answer_close_variants: tuple[tuple[int, ...], ...] | None = None,
    ) -> BudgetedGeneration:
        if reasoning_budget <= 0 or answer_allowance <= 0:
            raise ValueError("reasoning and answer budgets must be positive")
        if not reasoning_close_token_ids or not answer_open_token_ids or not answer_close_token_ids:
            raise ValueError("protocol delimiters must tokenize to non-empty sequences")

        reasoning_variants = reasoning_close_variants or (reasoning_close_token_ids,)
        answer_variants = answer_close_variants or (answer_close_token_ids,)
        reasoning_raw = self.backend.generate(
            prompt_token_ids,
            max_new_tokens=reasoning_budget + max(map(len, reasoning_variants)) - 1,
            should_stop=lambda ids: phase_should_stop_any(
                ids, reasoning_variants, reasoning_budget
            ),
        )
        reasoning = finalize_phase(
            reasoning_raw,
            delimiter=reasoning_close_token_ids,
            delimiter_variants=reasoning_variants,
            content_budget=reasoning_budget,
            phase_name="reasoning",
        )

        answer_prompt = (
            prompt_token_ids
            + reasoning.content_token_ids
            + reasoning_close_token_ids
            + answer_open_token_ids
        )
        answer_raw = self.backend.generate(
            answer_prompt,
            max_new_tokens=answer_allowance + max(map(len, answer_variants)) - 1,
            should_stop=lambda ids: phase_should_stop_any(ids, answer_variants, answer_allowance),
        )
        answer = finalize_phase(
            answer_raw,
            delimiter=answer_close_token_ids,
            delimiter_variants=answer_variants,
            content_budget=answer_allowance,
            phase_name="answer",
        )
        return BudgetedGeneration(
            prompt_token_ids=prompt_token_ids,
            reasoning=reasoning,
            answer=answer,
            reasoning_close_token_ids=reasoning_close_token_ids,
            answer_open_token_ids=answer_open_token_ids,
            answer_close_token_ids=answer_close_token_ids,
        )
