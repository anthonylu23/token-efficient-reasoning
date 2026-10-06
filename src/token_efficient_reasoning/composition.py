"""Development-only arithmetic composition holdouts, independent of the M1 corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .controlled.audit import value_signature
from .controlled.families.arithmetic import ArithmeticFamily
from .controlled.generation import derived_seed
from .controlled.rendering import render_example
from .controlled.schema import CanonicalExample, stable_hash, stable_json_bytes
from .controlled.verification import validate_examples, verify_example
from .provenance import sha256_file
from .training.artifacts import tree_manifest

TEMPLATES = ("arith_counter", "arith_machine")
SPLITS = ("train", "validation", "test_iid", "test_composition")
OPERATIONS = {"add", "sub", "mul", "div"}


@dataclass(frozen=True)
class CompositionConfig:
    master_seed: int
    depths: tuple[int, ...]
    heldout_pairs: tuple[tuple[str, str], ...]
    items_per_cell: dict[str, int]
    max_attempts_per_cell: int = 100_000
    schema_version: int = 1
    dataset_name: str = "arithmetic-composition-dev-v1"
    role: str = "development-only-not-evaluation-frozen"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CompositionConfig:
        if set(value) - cls.__dataclass_fields__.keys():
            raise ValueError("unknown composition configuration fields")
        try:
            config = cls(
                **{
                    **value,
                    "depths": tuple(value["depths"]),
                    "heldout_pairs": tuple(tuple(pair) for pair in value["heldout_pairs"]),
                }
            )
        except (KeyError, TypeError) as error:
            raise ValueError("invalid composition configuration") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1 or self.role != "development-only-not-evaluation-frozen":
            raise ValueError("only the version 1 development design is supported")
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")
        if (
            not self.depths
            or len(set(self.depths)) != len(self.depths)
            or any(type(d) is not int or not 2 <= d <= 32 for d in self.depths)
        ):
            raise ValueError("depths must be unique integers from 2 through 32")
        if (
            not self.heldout_pairs
            or len(set(self.heldout_pairs)) != len(self.heldout_pairs)
            or any(len(pair) != 2 or not set(pair) <= OPERATIONS for pair in self.heldout_pairs)
        ):
            raise ValueError("heldout_pairs must be unique pairs of arithmetic operations")
        if (
            set(self.items_per_cell) != set(SPLITS)
            or any(type(n) is not int or n <= 0 for n in self.items_per_cell.values())
            or type(self.max_attempts_per_cell) is not int
            or self.max_attempts_per_cell <= 0
        ):
            raise ValueError("positive counts and attempt limit are required for every split")

    def cells(self):
        for split in SPLITS:
            pairs = self.heldout_pairs if split == "test_composition" else (None,)
            for depth in self.depths:
                for template in TEMPLATES:
                    for pair in pairs:
                        yield split, depth, template, pair


def action_sequence(example: CanonicalExample) -> tuple[str, ...]:
    return tuple(str(op["action"]) for op in example.solution_graph["operations"])


def reserved_pairs(example: CanonicalExample, config: CompositionConfig) -> set[tuple[str, str]]:
    sequence = action_sequence(example)
    return set(zip(sequence, sequence[1:], strict=False)) & set(config.heldout_pairs)


def _cell_name(split, depth, template, pair) -> str:
    label = ">".join(pair) if pair else "none"
    return f"{split}/d{depth}/{template}/{label}"


def generate(config: CompositionConfig) -> tuple[dict[str, list[CanonicalExample]], dict]:
    """Rejection-sample exact quotas with global semantic deduplication and bounded work."""
    config.validate()
    family = ArithmeticFamily()
    by_split: dict[str, list[CanonicalExample]] = {split: [] for split in SPLITS}
    signatures: set[str] = set()
    seeds: set[int] = set()
    rejection_counts: Counter = Counter()
    attempts = {}
    for split, depth, template, pair in config.cells():
        cell = _cell_name(split, depth, template, pair)
        accepted = 0
        for attempt in range(config.max_attempts_per_cell):
            seed = derived_seed(config.master_seed, cell, family.name, depth, attempt)
            example = family.generate(seed=seed, depth=depth, split=split)
            signature = value_signature(example)
            desired = {pair} if pair else set()
            if example.template_id != template:
                rejection_counts["template"] += 1
            elif reserved_pairs(example, config) != desired:
                rejection_counts["operation_pair_partition"] += 1
            elif signature in signatures or seed in seeds:
                rejection_counts["duplicate"] += 1
            else:
                example = render_example(example)
                verify_example(example)
                by_split[split].append(example)
                signatures.add(signature)
                seeds.add(seed)
                accepted += 1
                if accepted == config.items_per_cell[split]:
                    attempts[cell] = attempt + 1
                    break
        else:
            raise RuntimeError(f"generation exhausted for {cell}: accepted {accepted} items")
    return by_split, {"attempts_per_cell": attempts, "rejection_counts": dict(rejection_counts)}


def audit(by_split: dict[str, list[CanonicalExample]], config: CompositionConfig) -> dict:
    """Fail on leakage, duplicates, missing quotas, or invalid executable semantics."""
    config.validate()
    errors = []
    if set(by_split) != set(SPLITS):
        errors.append("split set differs from configuration")
    seen: dict[str, set] = {key: set() for key in ("id", "seed", "semantic", "prompt")}
    counts: Counter = Counter()
    coverage = {}
    train_sequences = {action_sequence(e) for e in by_split.get("train", [])}
    primitives = {a for e in by_split.get("train", []) for a in action_sequence(e)}
    if primitives != OPERATIONS:
        errors.append("training does not expose all four primitive operations")
    verification = validate_examples([e for rows in by_split.values() for e in rows])
    if not verification["passed"]:
        errors.append("semantic or counterfactual verification failed")
    for split, examples in by_split.items():
        answers: Counter = Counter()
        pair_counts: Counter = Counter()
        revisit_count = 0
        distractor_matches = 0
        for example in examples:
            if example.split != split or example.family != "multi_operation_arithmetic":
                errors.append(f"incorrect split/family metadata: {example.id}")
            pairs = reserved_pairs(example, config)
            if split == "test_composition":
                if len(pairs) != 1:
                    errors.append(
                        f"composition item must contain exactly one reserved pair: {example.id}"
                    )
            elif pairs:
                errors.append(f"reserved pair leaked into {split}: {example.id}")
            pair = next(iter(pairs)) if len(pairs) == 1 else None
            counts[_cell_name(split, example.depth, example.template_id, pair)] += 1
            pair_counts.update(">".join(p) for p in pairs)
            values = {
                "id": example.id,
                "seed": example.seed,
                "semantic": value_signature(example),
                "prompt": stable_hash(example.prompt),
            }
            for key, value in values.items():
                if value in seen[key]:
                    errors.append(f"duplicate {key}: {example.id}")
                seen[key].add(value)
            states = [example.initial_state["value"], *(t.after for t in example.transitions)]
            revisit_count += len(set(states)) < len(states)
            distractor_matches += (
                str(example.initial_state["irrelevant_fact"]["count"]) == example.answer
            )
            answers[example.answer] += 1
        sequences = {action_sequence(e) for e in examples}
        overlap = len(sequences & train_sequences)
        if split == "test_composition" and overlap:
            errors.append("held-out complete action sequences overlap training")
        coverage[split] = {
            "item_count": len(examples),
            "answer_histogram": dict(sorted(answers.items())),
            "reserved_pair_counts": dict(sorted(pair_counts.items())),
            "unique_action_sequences": len(sequences),
            "action_sequences_shared_with_train": overlap,
            "items_with_revisited_state": revisit_count,
            "items_with_distractor_equal_to_answer": distractor_matches,
        }
    expected = {_cell_name(*cell): config.items_per_cell[cell[0]] for cell in config.cells()}
    if dict(counts) != expected:
        errors.append("depth/template/pair quotas differ from configuration")
    return {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "coverage": coverage,
        "cells": dict(counts),
        "expected_cells": expected,
        "training_primitives": sorted(primitives),
        "verification": verification,
        "distribution_policy": "report cancellations and distractor coincidences without filtering",
        "token_filtering": False,
    }


def prepare(config_path: Path, output: Path) -> dict:
    config = CompositionConfig.from_dict(json.loads(config_path.read_text()))
    if output.exists():
        raise FileExistsError(output)
    by_split, sampling = generate(config)
    report = audit(by_split, config)
    if not report["passed"]:
        raise ValueError(f"composition audit failed: {report['errors']}")
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in by_split.items():
        path = output / "splits" / f"{split}.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(stable_json_bytes(row.to_dict()).decode() + "\n")
    atomic_write_json(output / "resolved_config.json", asdict(config))
    atomic_write_json(output / "audit.json", report)
    atomic_write_json(output / "sampling.json", sampling)
    package = Path(__file__).parent
    atomic_write_json(
        output / "provenance.json",
        {
            "config_sha256": sha256_file(config_path),
            "source_sha256": {
                name: sha256_file(package / name)
                for name in (
                    "composition.py",
                    "controlled/families/arithmetic.py",
                    "controlled/generation.py",
                    "controlled/rendering.py",
                    "controlled/schema.py",
                    "controlled/audit.py",
                    "controlled/verification.py",
                )
            },
            "role": config.role,
            "model_evaluation_performed": False,
            "limitations": [
                "Development artifact; freeze final counts, seeds, budgets and contrasts later.",
                "Held-out pairs change operation distributions; not a matched-distribution test.",
                "Shared templates; not evidence of unseen-template or unseen-task transfer.",
            ],
        },
    )
    tree_manifest(output, config.dataset_name)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/followup/composition_dev.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.config, args.output_dir)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "counts": {
                    split: entry["item_count"] for split, entry in report["coverage"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
