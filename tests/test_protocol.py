from __future__ import annotations

from dataclasses import dataclass

from token_efficient_reasoning.protocol import (
    BackendOutput,
    BatchBudgetedGenerator,
    BudgetedGenerator,
    phase_should_stop,
    phase_should_stop_any,
)


@dataclass
class Script:
    token_ids: tuple[int, ...]
    finish_reason: str = "length"


class ScriptedBackend:
    def __init__(self, *scripts: Script) -> None:
        self.scripts = list(scripts)
        self.inputs: list[tuple[int, ...]] = []

    def generate(self, input_ids, *, max_new_tokens, should_stop):
        self.inputs.append(input_ids)
        script = self.scripts.pop(0)
        emitted: list[int] = []
        for token in script.token_ids[:max_new_tokens]:
            emitted.append(token)
            if should_stop(tuple(emitted)):
                return BackendOutput(tuple(emitted), "stop", 0.01)
        return BackendOutput(tuple(emitted), script.finish_reason, 0.01)


class ScriptedBatchBackend:
    def __init__(self, *calls: tuple[Script, ...]) -> None:
        self.calls = list(calls)
        self.inputs: list[tuple[tuple[int, ...], ...]] = []
        self.states_seen: list[tuple[object, ...]] = []

    def create_sampling_states(self, seeds):
        return tuple({"seed": seed} for seed in seeds)

    def generate_batch(
        self, input_ids, *, max_new_tokens, should_stop, sampling_states
    ):
        self.inputs.append(input_ids)
        self.states_seen.append(sampling_states)
        scripts = self.calls.pop(0)
        results = []
        for script, callback in zip(scripts, should_stop, strict=True):
            emitted: list[int] = []
            reason = script.finish_reason
            for token in script.token_ids[:max_new_tokens]:
                emitted.append(token)
                if callback(tuple(emitted)):
                    reason = "stop"
                    break
            results.append(BackendOutput(tuple(emitted), reason, 0.02))
        return tuple(results)


def _generate(backend: ScriptedBackend, budget: int = 3):
    return BudgetedGenerator(backend).generate(
        (100,),
        reasoning_budget=budget,
        answer_allowance=2,
        reasoning_close_token_ids=(9, 10),
        answer_open_token_ids=(11,),
        answer_close_token_ids=(12,),
    )


def test_natural_delimiters_are_excluded_from_content_counts() -> None:
    backend = ScriptedBackend(Script((1, 2, 9, 10)), Script((42, 12)))

    result = _generate(backend)

    assert result.reasoning.content_token_ids == (1, 2)
    assert result.reasoning.termination == "natural_close"
    assert result.answer.content_token_ids == (42,)
    assert result.answer.termination == "natural_close"
    assert result.effective_token_ids == (100, 1, 2, 9, 10, 11, 42, 12)


def test_budget_forces_transition_without_charging_answer_tokens() -> None:
    backend = ScriptedBackend(Script((1, 2, 3, 4)), Script((42, 43, 44)))

    result = _generate(backend)

    assert result.reasoning.content_token_ids == (1, 2, 3)
    assert result.reasoning.termination == "budget_forced"
    assert result.reasoning.overflow_token_ids == ()
    assert result.answer.content_token_ids == (42, 43)
    assert result.answer.termination == "allowance_exhausted"
    assert result.forced_reasoning_transition
    assert backend.inputs[1] == (100, 1, 2, 3, 9, 10, 11)


def test_partial_multi_token_delimiter_can_complete_at_boundary() -> None:
    backend = ScriptedBackend(Script((1, 2, 9, 10)), Script((42, 12)))

    result = _generate(backend, budget=3)

    assert result.reasoning.content_token_ids == (1, 2)
    assert result.reasoning.delimiter_token_ids == (9, 10)


def test_divergent_partial_delimiter_is_trimmed_to_exact_budget() -> None:
    backend = ScriptedBackend(Script((1, 2, 9, 8)), Script((42, 12)))

    result = _generate(backend, budget=3)

    assert result.reasoning.content_token_ids == (1, 2, 9)
    assert result.reasoning.overflow_token_ids == (8,)
    assert len(result.reasoning.content_token_ids) == 3
    assert result.reasoning.termination == "budget_forced"


def test_eos_before_budget_is_explicit_and_answer_is_still_attempted() -> None:
    backend = ScriptedBackend(Script((1,), "eos"), Script((42,), "eos"))

    result = _generate(backend)

    assert result.reasoning.termination == "eos"
    assert result.answer.termination == "eos"
    assert result.effective_token_ids == (100, 1, 9, 10, 11, 42, 12)


def test_stopper_waits_for_possible_multi_token_delimiter() -> None:
    assert not phase_should_stop((1, 2, 9), (9, 10), 3)
    assert phase_should_stop((1, 2, 9, 10), (9, 10), 3)
    assert phase_should_stop((1, 2, 3), (9, 10), 3)


def test_contextual_delimiter_variant_is_excluded_from_content() -> None:
    backend = ScriptedBackend(Script((1, 2, 9, 13)), Script((42, 12)))

    result = BudgetedGenerator(backend).generate(
        (100,),
        reasoning_budget=5,
        answer_allowance=2,
        reasoning_close_token_ids=(9, 10),
        reasoning_close_variants=((9, 10), (9, 13)),
        answer_open_token_ids=(11,),
        answer_close_token_ids=(12,),
    )

    assert result.reasoning.content_token_ids == (1, 2)
    assert result.reasoning.delimiter_token_ids == (9, 13)
    assert result.effective_token_ids[:5] == (100, 1, 2, 9, 10)


def test_variant_stopper_accepts_fused_trailing_whitespace() -> None:
    variants = ((9, 10), (9, 13))

    assert phase_should_stop_any((1, 9, 10), variants, 4)
    assert phase_should_stop_any((1, 9, 13), variants, 4)


def test_batched_protocol_stops_rows_independently_and_reuses_rng_states() -> None:
    backend = ScriptedBatchBackend(
        (
            Script((1, 9, 10, 99)),
            Script((2, 3, 4, 5)),
        ),
        (
            Script((41, 12, 99)),
            Script((42, 43, 44)),
        ),
    )
    states = backend.create_sampling_states((7, 8))

    results = BatchBudgetedGenerator(backend).generate(
        ((100,), (200,)),
        sampling_states=states,
        reasoning_budget=3,
        answer_allowance=2,
        reasoning_close_token_ids=(9, 10),
        answer_open_token_ids=(11,),
        answer_close_token_ids=(12,),
    )

    assert results[0].reasoning.content_token_ids == (1,)
    assert results[0].reasoning.termination == "natural_close"
    assert results[1].reasoning.content_token_ids == (2, 3, 4)
    assert results[1].reasoning.termination == "budget_forced"
    assert results[0].answer.content_token_ids == (41,)
    assert results[1].answer.content_token_ids == (42, 43)
    assert backend.inputs[1] == (
        (100, 1, 9, 10, 11),
        (200, 2, 3, 4, 9, 10, 11),
    )
    assert backend.states_seen == [states, states]
