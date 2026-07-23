"""Controlled sequential-integer arithmetic tasks."""

from __future__ import annotations

import random
import re
from dataclasses import replace

from ..schema import CanonicalExample, Transition, stable_hash

_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_VERBS = {
    "add": "Adding",
    "sub": "Subtracting",
    "mul": "Multiplying by",
    "div": "Dividing by",
}
_VERB_ACTIONS = {value: key for key, value in _VERBS.items()}


def _apply(action: str, before: int, argument: int) -> int:
    if action == "add":
        return before + argument
    if action == "sub":
        return before - argument
    if action == "mul":
        return before * argument
    if action == "div":
        if argument == 0 or before % argument:
            raise ValueError("arithmetic division must be exact and nonzero")
        return before // argument
    raise ValueError(f"unknown arithmetic action: {action!r}")


class ArithmeticFamily:
    name = "multi_operation_arithmetic"

    def generate(self, *, seed: int, depth: int, split: str) -> CanonicalExample:
        if depth <= 0:
            raise ValueError("arithmetic depth must be positive")
        rng = random.Random(seed)
        initial = rng.randint(3, 24)
        state = initial
        transitions: list[Transition] = []
        operations: list[dict[str, int | str]] = []
        for index in range(1, depth + 1):
            candidates: list[tuple[str, int]] = [
                ("add", rng.randint(1, 12)),
            ]
            if state >= 2:
                candidates.append(("sub", rng.randint(1, min(12, state))))
            if 0 < state <= 2_000:
                candidates.append(("mul", rng.randint(2, 4)))
            divisors = [value for value in range(2, 10) if state and state % value == 0]
            if divisors:
                candidates.append(("div", rng.choice(divisors)))
            action, argument = rng.choice(candidates)
            after = _apply(action, state, argument)
            transition = Transition(
                index=index,
                kind="arithmetic",
                before=state,
                action=action,
                argument=argument,
                after=after,
            )
            transitions.append(transition)
            operations.append({"action": action, "argument": argument})
            state = after

        irrelevant = {
            "label": rng.choice(["blue", "copper", "striped", "wooden"]),
            "count": rng.randint(2, 30),
        }
        template_ids = (
            ("arith_inventory",)
            if split == "test_compositional"
            else ("arith_counter", "arith_machine")
        )
        template_id = rng.choice(template_ids)
        prompt = self._prompt(template_id, initial, operations, irrelevant)
        surface_form_id = f"{template_id}:ops-v1:distractor-v1"
        example = CanonicalExample(
            id="pending",
            family=self.name,
            split=split,
            seed=seed,
            depth=depth,
            template_id=template_id,
            surface_form_id=surface_form_id,
            prompt=prompt,
            initial_state={"value": initial, "irrelevant_fact": irrelevant},
            solution_graph={"operations": operations},
            transitions=tuple(transitions),
            answer=str(state),
            answer_type="integer",
            instance_hash="pending",
        )
        instance_hash = stable_hash(example.semantic_payload())
        return replace(
            example,
            id=f"arith-{split}-d{depth}-{seed:016x}-{instance_hash[:10]}",
            instance_hash=instance_hash,
        )

    def _prompt(
        self,
        template_id: str,
        initial: int,
        operations: list[dict[str, int | str]],
        irrelevant: dict[str, int | str],
    ) -> str:
        instructions = []
        for operation in operations:
            action = str(operation["action"])
            argument = operation["argument"]
            phrases = {
                "add": f"add {argument}",
                "sub": f"subtract {argument}",
                "mul": f"multiply by {argument}",
                "div": f"divide by {argument}",
            }
            instructions.append(phrases[action])
        sequence = ", then ".join(instructions)
        distractor = (
            f"A separate box contains {irrelevant['count']} {irrelevant['label']} markers; "
            "that fact does not affect the calculation."
        )
        if template_id == "arith_counter":
            return (
                f"A counter starts at {initial}. In order, {sequence}. {distractor} "
                "What is the counter's final value?"
            )
        if template_id == "arith_machine":
            return (
                f"A number machine receives {initial} and performs these commands in order: "
                f"{sequence}. {distractor} What number leaves the machine?"
            )
        if template_id == "arith_inventory":
            return (
                f"An inventory register initially reads {initial}. Process these updates in order: "
                f"{sequence}. {distractor} Report the final register value."
            )
        raise ValueError(f"unknown arithmetic template: {template_id!r}")

    def render(self, transitions: tuple[Transition, ...], representation: str) -> str:
        lines: list[str] = []
        for item in transitions:
            symbol = _SYMBOLS[item.action]
            if representation == "r0":
                lines.append(
                    f"Step {item.index}: The current value is {item.before}. "
                    f"{_VERBS[item.action]} {item.argument} gives {item.after}."
                )
            elif representation == "r1":
                lines.append(
                    f"Step {item.index}: {item.before} {symbol} {item.argument} = {item.after}."
                )
            elif representation == "r2":
                lines.append(f"{item.index}:{item.before}{symbol}{item.argument}={item.after}")
            elif representation == "r3":
                lines.append(
                    f"s{item.index}={item.before}>{item.action}({item.argument})>{item.after}"
                )
            else:
                raise ValueError(f"unknown representation: {representation!r}")
        return "\n".join(lines)

    def parse(self, text: str, representation: str) -> tuple[Transition, ...]:
        patterns = {
            "r0": re.compile(
                r"Step (?P<i>\d+): The current value is (?P<b>-?\d+)\. "
                r"(?P<a>Adding|Subtracting|Multiplying by|Dividing by) "
                r"(?P<x>-?\d+) gives (?P<z>-?\d+)\."
            ),
            "r1": re.compile(
                r"Step (?P<i>\d+): (?P<b>-?\d+) (?P<a>[+*/-]) "
                r"(?P<x>-?\d+) = (?P<z>-?\d+)\."
            ),
            "r2": re.compile(r"(?P<i>\d+):(?P<b>-?\d+)(?P<a>[+*/-])(?P<x>-?\d+)=(?P<z>-?\d+)"),
            "r3": re.compile(
                r"s(?P<i>\d+)=(?P<b>-?\d+)>(?P<a>add|sub|mul|div)"
                r"\((?P<x>-?\d+)\)>(?P<z>-?\d+)"
            ),
        }
        if representation not in patterns:
            raise ValueError(f"unknown representation: {representation!r}")
        parsed: list[Transition] = []
        for line in text.splitlines():
            match = patterns[representation].fullmatch(line)
            if match is None:
                raise ValueError(f"invalid {representation.upper()} arithmetic line: {line!r}")
            action_text = match.group("a")
            if representation == "r0":
                action = _VERB_ACTIONS[action_text]
            elif representation in {"r1", "r2"}:
                action = {value: key for key, value in _SYMBOLS.items()}[action_text]
            else:
                action = action_text
            parsed.append(
                Transition(
                    index=int(match.group("i")),
                    kind="arithmetic",
                    before=int(match.group("b")),
                    action=action,
                    argument=int(match.group("x")),
                    after=int(match.group("z")),
                )
            )
        if not parsed:
            raise ValueError("arithmetic trace must not be empty")
        return tuple(parsed)

    def replay(self, example: CanonicalExample, transitions: tuple[Transition, ...]) -> str:
        state = int(example.initial_state["value"])
        operations = example.solution_graph["operations"]
        if len(transitions) != len(operations):
            raise ValueError("arithmetic trace does not contain every required operation")
        for index, (transition, operation) in enumerate(
            zip(transitions, operations, strict=True), start=1
        ):
            expected_action = operation["action"]
            expected_argument = int(operation["argument"])
            if transition.index != index or transition.kind != "arithmetic":
                raise ValueError(f"invalid arithmetic transition metadata at step {index}")
            if transition.before != state:
                raise ValueError(f"arithmetic state mismatch at step {index}")
            if transition.action != expected_action or transition.argument != expected_argument:
                raise ValueError(f"arithmetic operation mismatch at step {index}")
            after = _apply(transition.action, state, int(transition.argument))
            if transition.after != after:
                raise ValueError(f"arithmetic result mismatch at step {index}")
            state = after
        return str(state)

    def solve(self, example: CanonicalExample) -> str:
        state = int(example.initial_state["value"])
        for operation in example.solution_graph["operations"]:
            state = _apply(str(operation["action"]), state, int(operation["argument"]))
        return str(state)
