"""Controlled labeled directed-graph traversal tasks."""

from __future__ import annotations

import random
import re
from dataclasses import replace

from ..schema import CanonicalExample, Transition, stable_hash

_LABELS = ("red", "blue", "green", "gold", "white", "black")


class GraphTraversalFamily:
    name = "directed_graph_traversal"

    def generate(self, *, seed: int, depth: int, split: str) -> CanonicalExample:
        if depth <= 0:
            raise ValueError("graph depth must be positive")
        rng = random.Random(seed)
        node_count = depth * 2 + 3
        nodes = [f"N{index:02d}" for index in range(node_count)]
        rng.shuffle(nodes)
        path = nodes[: depth + 1]
        branch_nodes = nodes[depth + 1 : depth * 2 + 1]
        route_labels: list[str] = []
        edges: list[dict[str, str]] = []
        transitions: list[Transition] = []
        for index in range(depth):
            route_label = rng.choice(_LABELS)
            route_labels.append(route_label)
            edges.append({"from": path[index], "label": route_label, "to": path[index + 1]})
            transitions.append(
                Transition(
                    index=index + 1,
                    kind="graph_edge",
                    before=path[index],
                    action="follow",
                    argument=route_label,
                    after=path[index + 1],
                )
            )
            distractor_label = rng.choice([label for label in _LABELS if label != route_label])
            edges.append(
                {"from": path[index], "label": distractor_label, "to": branch_nodes[index]}
            )
        edges.append(
            {
                "from": nodes[-2],
                "label": rng.choice(_LABELS),
                "to": nodes[-1],
            }
        )
        edges.sort(key=lambda edge: (edge["from"], edge["label"], edge["to"]))
        template_ids = (
            ("graph_dispatch",) if split == "test_compositional" else ("graph_map", "graph_router")
        )
        template_id = rng.choice(template_ids)
        prompt = self._prompt(template_id, path[0], route_labels, edges)
        example = CanonicalExample(
            id="pending",
            family=self.name,
            split=split,
            seed=seed,
            depth=depth,
            template_id=template_id,
            surface_form_id=f"{template_id}:edge-list-v1:route-v1",
            prompt=prompt,
            initial_state={"node": path[0], "route_labels": route_labels},
            solution_graph={"nodes": sorted(nodes), "edges": edges},
            transitions=tuple(transitions),
            answer=path[-1],
            answer_type="exact",
            instance_hash="pending",
        )
        instance_hash = stable_hash(example.semantic_payload())
        return replace(
            example,
            id=f"graph-{split}-d{depth}-{seed:016x}-{instance_hash[:10]}",
            instance_hash=instance_hash,
        )

    def _prompt(
        self,
        template_id: str,
        start: str,
        route_labels: list[str],
        edges: list[dict[str, str]],
    ) -> str:
        edge_text = "; ".join(f"{edge['from']} -{edge['label']}-> {edge['to']}" for edge in edges)
        route_text = ", then ".join(route_labels)
        if template_id == "graph_map":
            return (
                f"A directed map contains these labeled edges: {edge_text}. Start at {start} and "
                f"follow, in order, the {route_text} edges. Which node is reached?"
            )
        if template_id == "graph_router":
            return (
                f"A router table has directed links: {edge_text}. A packet begins at {start}. "
                f"Route it using labels {route_text}, in that order. Report its final node."
            )
        if template_id == "graph_dispatch":
            return (
                f"A dispatch network lists one-way labeled connections: {edge_text}. Begin at "
                f"{start}; take connections labeled {route_text}, in order. Where do you finish?"
            )
        raise ValueError(f"unknown graph template: {template_id!r}")

    def render(self, transitions: tuple[Transition, ...], representation: str) -> str:
        lines: list[str] = []
        for item in transitions:
            if representation == "r0":
                lines.append(
                    f"Step {item.index}: From {item.before}, following {item.argument} reaches "
                    f"{item.after}."
                )
            elif representation == "r1":
                lines.append(
                    f"Step {item.index}: {item.before} via {item.argument} -> {item.after}."
                )
            elif representation == "r2":
                lines.append(f"{item.index}:{item.before}-{item.argument}>{item.after}")
            elif representation == "r3":
                lines.append(f"e{item.index}({item.before},{item.argument},{item.after})")
            else:
                raise ValueError(f"unknown representation: {representation!r}")
        return "\n".join(lines)

    def parse(self, text: str, representation: str) -> tuple[Transition, ...]:
        node = r"N\d{2}"
        label = r"[a-z]+"
        patterns = {
            "r0": re.compile(
                rf"Step (?P<i>\d+): From (?P<b>{node}), following (?P<x>{label}) "
                rf"reaches (?P<z>{node})\."
            ),
            "r1": re.compile(
                rf"Step (?P<i>\d+): (?P<b>{node}) via (?P<x>{label}) -> (?P<z>{node})\."
            ),
            "r2": re.compile(rf"(?P<i>\d+):(?P<b>{node})-(?P<x>{label})>(?P<z>{node})"),
            "r3": re.compile(rf"e(?P<i>\d+)\((?P<b>{node}),(?P<x>{label}),(?P<z>{node})\)"),
        }
        if representation not in patterns:
            raise ValueError(f"unknown representation: {representation!r}")
        parsed: list[Transition] = []
        for line in text.splitlines():
            match = patterns[representation].fullmatch(line)
            if match is None:
                raise ValueError(f"invalid {representation.upper()} graph line: {line!r}")
            parsed.append(
                Transition(
                    index=int(match.group("i")),
                    kind="graph_edge",
                    before=match.group("b"),
                    action="follow",
                    argument=match.group("x"),
                    after=match.group("z"),
                )
            )
        if not parsed:
            raise ValueError("graph trace must not be empty")
        return tuple(parsed)

    def replay(self, example: CanonicalExample, transitions: tuple[Transition, ...]) -> str:
        state = str(example.initial_state["node"])
        route_labels = example.initial_state["route_labels"]
        if len(transitions) != len(route_labels):
            raise ValueError("graph trace does not contain every required route transition")
        edge_lookup = self._edge_lookup(example)
        for index, (transition, label) in enumerate(
            zip(transitions, route_labels, strict=True), start=1
        ):
            if (
                transition.index != index
                or transition.kind != "graph_edge"
                or transition.action != "follow"
            ):
                raise ValueError(f"invalid graph transition metadata at step {index}")
            if transition.before != state or transition.argument != label:
                raise ValueError(f"graph route mismatch at step {index}")
            destination = edge_lookup.get((state, label))
            if destination is None or transition.after != destination:
                raise ValueError(f"invalid graph edge at step {index}")
            state = destination
        return state

    def solve(self, example: CanonicalExample) -> str:
        state = str(example.initial_state["node"])
        edge_lookup = self._edge_lookup(example)
        for index, label in enumerate(example.initial_state["route_labels"], start=1):
            try:
                state = edge_lookup[(state, label)]
            except KeyError as error:
                raise ValueError(f"route has no graph edge at step {index}") from error
        return state

    def _edge_lookup(self, example: CanonicalExample) -> dict[tuple[str, str], str]:
        edge_lookup: dict[tuple[str, str], str] = {}
        for edge in example.solution_graph["edges"]:
            key = (edge["from"], edge["label"])
            if key in edge_lookup:
                raise ValueError(f"ambiguous graph edge: {key}")
            edge_lookup[key] = edge["to"]
        return edge_lookup
