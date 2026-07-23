"""Cross-split overlap and distribution auditing."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from .schema import CanonicalExample, stable_hash


def _overlaps(
    by_split: dict[str, list[CanonicalExample]], key: Callable[[CanonicalExample], Any]
) -> list[dict[str, Any]]:
    owners: dict[Any, set[str]] = defaultdict(set)
    for split, examples in by_split.items():
        for example in examples:
            owners[key(example)].add(split)
    return [
        {"value": value, "splits": sorted(splits)}
        for value, splits in owners.items()
        if len(splits) > 1
    ]


def value_signature(example: CanonicalExample) -> str:
    if example.family == "multi_operation_arithmetic":
        values = {
            "initial": example.initial_state["value"],
            "operations": example.solution_graph["operations"],
        }
    else:
        values = {
            "start": example.initial_state["node"],
            "route_labels": example.initial_state["route_labels"],
            "edges": example.solution_graph["edges"],
        }
    return stable_hash(values)


def audit_splits(by_split: dict[str, list[CanonicalExample]]) -> dict[str, Any]:
    seed_overlap = _overlaps(by_split, lambda example: example.seed)
    instance_overlap = _overlaps(by_split, lambda example: example.instance_hash)
    prompt_overlap = _overlaps(by_split, lambda example: stable_hash(example.prompt))
    value_signature_overlap = _overlaps(by_split, value_signature)

    train_templates = {
        (example.family, example.template_id) for example in by_split.get("train", [])
    }
    compositional_templates = {
        (example.family, example.template_id) for example in by_split.get("test_compositional", [])
    }
    compositional_template_overlap = sorted(train_templates & compositional_templates)
    train_depths = {example.depth for example in by_split.get("train", [])}
    depth_test_depths = {example.depth for example in by_split.get("test_depth", [])}
    depth_overlap = sorted(train_depths & depth_test_depths)

    coverage = {
        split: {
            "item_count": len(examples),
            "families": dict(sorted(Counter(item.family for item in examples).items())),
            "depths": dict(sorted(Counter(str(item.depth) for item in examples).items())),
            "templates": dict(
                sorted(Counter(f"{item.family}:{item.template_id}" for item in examples).items())
            ),
        }
        for split, examples in by_split.items()
    }
    passed = not any(
        (
            seed_overlap,
            instance_overlap,
            prompt_overlap,
            value_signature_overlap,
            compositional_template_overlap,
            depth_overlap,
        )
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "coverage": coverage,
        "overlap": {
            "seeds": seed_overlap,
            "instances": instance_overlap,
            "prompts": prompt_overlap,
            "value_signatures": value_signature_overlap,
            "train_compositional_templates": [
                {"family": family, "template_id": template}
                for family, template in compositional_template_overlap
            ],
            "train_depth_test_depths": depth_overlap,
        },
    }


def audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# M1 split-overlap audit",
        "",
        f"Overall result: **{'pass' if audit['passed'] else 'fail'}**",
        "",
        "## Coverage",
        "",
        "| Split | Items | Families | Depths |",
        "|---|---:|---|---|",
    ]
    for split, coverage in audit["coverage"].items():
        families = ", ".join(f"{family}: {count}" for family, count in coverage["families"].items())
        depths = ", ".join(f"{depth}: {count}" for depth, count in coverage["depths"].items())
        lines.append(f"| {split} | {coverage['item_count']} | {families} | {depths} |")
    lines.extend(
        [
            "",
            "## Cross-split collisions",
            "",
            "| Audit | Collisions |",
            "|---|---:|",
        ]
    )
    for key, values in audit["overlap"].items():
        lines.append(f"| {key} | {len(values)} |")
    return "\n".join(lines) + "\n"
