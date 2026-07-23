"""Fail-closed semantic and counterfactual verification."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from .families import FAMILIES
from .schema import REPRESENTATIONS, CanonicalExample


def verify_example(
    example: CanonicalExample,
    *,
    require_tokens: bool = False,
    required_compression: tuple[str, ...] = (),
) -> None:
    """Verify canonical semantics and every rendered trace, raising on any mismatch."""

    example.validate_basic()
    try:
        family = FAMILIES[example.family]
    except KeyError as error:
        raise ValueError(f"unknown controlled task family: {example.family!r}") from error
    solved_answer = family.solve(example)
    if solved_answer != example.answer:
        raise ValueError(f"independent solver answer mismatch for {example.id!r}")
    canonical_answer = family.replay(example, example.transitions)
    if canonical_answer != example.answer:
        raise ValueError(f"canonical answer mismatch for {example.id!r}")
    if set(example.renderings) != set(REPRESENTATIONS):
        raise ValueError(f"item {example.id!r} does not contain exactly R0-R3")
    for representation in REPRESENTATIONS:
        parsed = family.parse(example.renderings[representation], representation)
        if parsed != example.transitions:
            raise ValueError(f"{representation.upper()} transition mismatch for {example.id!r}")
        rendered_answer = family.replay(example, parsed)
        if rendered_answer != example.answer:
            raise ValueError(f"{representation.upper()} answer mismatch for {example.id!r}")
    if require_tokens and not example.token_measurements:
        raise ValueError(f"item {example.id!r} has no tokenizer measurements")
    for checkpoint_id, measurements in example.token_measurements.items():
        if set(measurements) != set(REPRESENTATIONS):
            raise ValueError(
                f"item {example.id!r} has incomplete measurements for {checkpoint_id!r}"
            )
        if any(not isinstance(count, int) or count <= 0 for count in measurements.values()):
            raise ValueError(f"item {example.id!r} has invalid tokenizer counts")
        for representation in required_compression:
            if measurements[representation] >= measurements["r0"]:
                raise ValueError(
                    f"item {example.id!r} has non-compressed {representation} for {checkpoint_id!r}"
                )


def verify_counterfactual_rejection(example: CanonicalExample) -> None:
    """Confirm a representative semantic mutation cannot replay as the accepted trace."""

    family = FAMILIES[example.family]
    first = example.transitions[0]
    if isinstance(first.after, int):
        changed_after: int | str = first.after + 1
    else:
        changed_after = f"{first.after}_changed"
    mutated_transition = replace(first, after=changed_after)
    mutated = (mutated_transition, *example.transitions[1:])
    try:
        family.replay(example, mutated)
    except ValueError:
        return
    raise ValueError(f"counterfactual transition was accepted for {example.id!r}")


def verify_problem_mutation_rejection(example: CanonicalExample) -> None:
    """Confirm changing a relevant operand or edge invalidates the original path."""

    family = FAMILIES[example.family]
    changed_graph = copy.deepcopy(example.solution_graph)
    if example.family == "multi_operation_arithmetic":
        changed_graph["operations"][0]["argument"] += 1
    elif example.family == "directed_graph_traversal":
        first = example.transitions[0]
        matching = next(
            edge
            for edge in changed_graph["edges"]
            if edge["from"] == first.before and edge["label"] == first.argument
        )
        matching["to"] = next(node for node in changed_graph["nodes"] if node != matching["to"])
    else:
        raise ValueError(f"no problem mutation is defined for {example.family!r}")
    mutated_example = replace(example, solution_graph=changed_graph)
    try:
        family.replay(mutated_example, example.transitions)
    except ValueError:
        return
    raise ValueError(f"counterfactual problem was accepted for {example.id!r}")


def validate_examples(
    examples: list[CanonicalExample],
    *,
    require_tokens: bool = False,
    required_compression: tuple[str, ...] = (),
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    counterfactual_failures: list[dict[str, str]] = []
    for example in examples:
        try:
            verify_example(
                example,
                require_tokens=require_tokens,
                required_compression=required_compression,
            )
        except ValueError as error:
            failures.append({"item_id": example.id, "error": str(error)})
            continue
        try:
            verify_counterfactual_rejection(example)
            verify_problem_mutation_rejection(example)
        except ValueError as error:
            counterfactual_failures.append({"item_id": example.id, "error": str(error)})
    return {
        "schema_version": 1,
        "item_count": len(examples),
        "verified_count": len(examples) - len(failures),
        "verification_rate": (len(examples) - len(failures)) / len(examples) if examples else 0,
        "failures": failures,
        "counterfactual_checks": len(examples) * 2,
        "counterfactual_failures": counterfactual_failures,
        "passed": not failures and not counterfactual_failures,
    }
