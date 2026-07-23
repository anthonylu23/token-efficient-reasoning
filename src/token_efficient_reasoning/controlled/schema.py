"""Versioned data contracts and stable serialization for controlled examples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPRESENTATIONS = ("r0", "r1", "r2", "r3")


def stable_json_bytes(value: Any) -> bytes:
    """Serialize JSON data canonically for hashing and reproducibility checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class Transition:
    """One executable transition shared by every rendered representation."""

    index: int
    kind: str
    before: int | str
    action: str
    argument: int | str
    after: int | str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Transition:
        required = {"index", "kind", "before", "action", "argument", "after"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"transition missing fields: {', '.join(missing)}")
        transition = cls(**value)
        if transition.index < 1:
            raise ValueError("transition index must be positive")
        if not transition.kind or not transition.action:
            raise ValueError("transition kind and action must be non-empty")
        return transition

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalExample:
    """A generated problem plus the canonical semantics behind all traces."""

    id: str
    family: str
    split: str
    seed: int
    depth: int
    template_id: str
    surface_form_id: str
    prompt: str
    initial_state: dict[str, Any]
    solution_graph: dict[str, Any]
    transitions: tuple[Transition, ...]
    answer: str
    answer_type: str
    instance_hash: str
    renderings: dict[str, str] = field(default_factory=dict)
    token_measurements: dict[str, dict[str, int]] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CanonicalExample:
        required = {
            "schema_version",
            "id",
            "family",
            "split",
            "seed",
            "depth",
            "template_id",
            "surface_form_id",
            "prompt",
            "initial_state",
            "solution_graph",
            "canonical_transitions",
            "answer",
            "answer_type",
            "instance_hash",
            "renderings",
            "token_measurements",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"controlled item missing fields: {', '.join(missing)}")
        example = cls(
            schema_version=value["schema_version"],
            id=value["id"],
            family=value["family"],
            split=value["split"],
            seed=value["seed"],
            depth=value["depth"],
            template_id=value["template_id"],
            surface_form_id=value["surface_form_id"],
            prompt=value["prompt"],
            initial_state=value["initial_state"],
            solution_graph=value["solution_graph"],
            transitions=tuple(
                Transition.from_dict(transition) for transition in value["canonical_transitions"]
            ),
            answer=value["answer"],
            answer_type=value["answer_type"],
            instance_hash=value["instance_hash"],
            renderings=value["renderings"],
            token_measurements=value["token_measurements"],
        )
        example.validate_basic()
        return example

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "initial_state": self.initial_state,
            "solution_graph": self.solution_graph,
            "canonical_transitions": [item.to_dict() for item in self.transitions],
            "answer": self.answer,
            "answer_type": self.answer_type,
        }

    def validate_basic(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported controlled item schema: {self.schema_version}")
        if self.depth <= 0 or self.depth != len(self.transitions):
            raise ValueError(f"item {self.id!r} has inconsistent depth")
        if tuple(item.index for item in self.transitions) != tuple(range(1, self.depth + 1)):
            raise ValueError(f"item {self.id!r} has non-contiguous transition indices")
        if self.instance_hash != stable_hash(self.semantic_payload()):
            raise ValueError(f"item {self.id!r} has an invalid instance hash")
        unknown = sorted(set(self.renderings) - set(REPRESENTATIONS))
        if unknown:
            raise ValueError(f"item {self.id!r} has unknown renderings: {unknown}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "family": self.family,
            "split": self.split,
            "seed": self.seed,
            "depth": self.depth,
            "template_id": self.template_id,
            "surface_form_id": self.surface_form_id,
            "prompt": self.prompt,
            "initial_state": self.initial_state,
            "solution_graph": self.solution_graph,
            "canonical_transitions": [item.to_dict() for item in self.transitions],
            "answer": self.answer,
            "answer_type": self.answer_type,
            "instance_hash": self.instance_hash,
            "renderings": self.renderings,
            "token_measurements": self.token_measurements,
        }


def load_examples(path: str | Path) -> list[CanonicalExample]:
    examples: list[CanonicalExample] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            example = CanonicalExample.from_dict(json.loads(line))
            if example.id in seen:
                raise ValueError(f"duplicate controlled item {example.id!r} on line {line_number}")
            seen.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError(f"no controlled examples found in {path}")
    return examples
