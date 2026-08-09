"""Saved-artifact-only bridge and primary reporting."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.controlled.schema import CanonicalExample, load_examples
from token_efficient_reasoning.provenance import sha256_file
from token_efficient_reasoning.training.artifacts import tree_manifest
from token_efficient_reasoning.training.diagnostics import diagnose_trace
from token_efficient_reasoning.training.reporting import (
    _aggregate_group,
    _bootstrap_interval,
    _training_gate,
)

from .config import M3Config

_ANALYSIS_ONLY_PROTOCOL_FIELDS = {
    "analysis_version",
    "primary_inference_split",
    "central_primary_comparators",
    "primary_comparators",
    "h5_comparator",
    "secondary_comparators",
    "bootstrap_unit",
    "familywise_alpha",
    "multiplicity_method",
    "verified_density_zero_control_rule",
    "primary_density_comparator",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _recompute_trace_diagnostics(
    rows: list[dict[str, Any]],
    examples: dict[str, CanonicalExample],
    expected_representation: str | None,
) -> list[dict[str, Any]]:
    """Return report-only rows with diagnostics recomputed from immutable saved text."""

    corrected: list[dict[str, Any]] = []
    for row in rows:
        item_id = row["item_id"]
        try:
            example = examples[item_id]
        except KeyError as error:
            raise ValueError(f"saved evaluation contains unknown item ID: {item_id}") from error
        copied = dict(row)
        copied["trace_diagnostics"] = diagnose_trace(
            example,
            row["reasoning_text"],
            expected_representation,
            allow_trailing_fragment=bool(row.get("forced_reasoning_transition", False)),
        )
        corrected.append(copied)
    return corrected


def _paired_comparison(
    indexed: dict[str, dict[tuple[str, int], dict[str, Any]]],
    *,
    depth: int,
    budget: int,
    margin: float,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    keys = [
        key
        for key, row in indexed["c4"].items()
        if row["depth"] == depth and row["reasoning_budget"] == budget
    ]
    if not keys:
        raise ValueError(f"no C4 rows for depth={depth} budget={budget}")
    control_accuracies = {
        condition: statistics.fmean(float(indexed[condition][key]["correct"]) for key in keys)
        for condition in ("c2", "c3")
    }
    control = max(control_accuracies, key=control_accuracies.get)
    differences = [
        float(indexed["c4"][key]["correct"]) - float(indexed[control][key]["correct"])
        for key in keys
    ]
    lower, upper = _bootstrap_interval(
        differences,
        seed=bootstrap_seed + depth * 10_000 + budget,
        samples=bootstrap_samples,
    )
    difference = statistics.fmean(differences)
    return {
        "depth": depth,
        "budget": budget,
        "item_count": len(keys),
        "strongest_control": control,
        "control_accuracy": control_accuracies[control],
        "c4_accuracy": statistics.fmean(
            float(indexed["c4"][key]["correct"]) for key in keys
        ),
        "paired_difference": difference,
        "bootstrap_95_ci": [lower, upper],
        "noninferiority_margin": margin,
        "point_estimate_passed": difference >= margin,
    }


def _protocol_compatibility(
    bridge_protocol: dict[str, Any],
    report_protocol: dict[str, Any],
) -> dict[str, Any]:
    bridge_protocol = json.loads(json.dumps(bridge_protocol, sort_keys=True))
    report_protocol = json.loads(json.dumps(report_protocol, sort_keys=True))
    changed = {
        key: {
            "bridge": bridge_protocol.get(key),
            "report": report_protocol.get(key),
        }
        for key in sorted(bridge_protocol.keys() | report_protocol.keys())
        if bridge_protocol.get(key) != report_protocol.get(key)
    }
    behavioral_changes = sorted(
        set(changed) - _ANALYSIS_ONLY_PROTOCOL_FIELDS
    )
    return {
        "schema_version": 1,
        "changed_fields": changed,
        "analysis_only_fields": sorted(_ANALYSIS_ONLY_PROTOCOL_FIELDS),
        "behavioral_changes": behavioral_changes,
        "passed": not behavioral_changes,
    }


def build_bridge_report(
    protocol_path: str | Path,
    evaluation_dirs: list[str | Path],
    training_dirs: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate the preregistered validation-only 1.5B bridge gate."""

    config = M3Config.from_json(protocol_path)
    _, _, split_paths = config.validate_references(protocol_path)
    validation_examples = {
        example.id: example for example in load_examples(split_paths[config.validation_split])
    }
    eval_roots = [Path(path).resolve() for path in evaluation_dirs]
    train_roots = [Path(path).resolve() for path in training_dirs]
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    by_condition: dict[str, list[dict[str, Any]]] = {}
    bridge_protocols: list[dict[str, Any]] = []
    bridge_protocol_sha256: set[str] = set()
    test_data_inspected = False
    for root in eval_roots:
        resolved = _load_json(root / "resolved_config.json")
        condition = resolved["condition"]["id"]
        if condition in by_condition:
            raise ValueError(f"duplicate bridge evaluation condition: {condition}")
        split = resolved.get("evaluation_split")
        if split != config.validation_split:
            test_data_inspected = True
        context = resolved.get("run_context", {})
        if context.get("phase") != "bridge":
            raise ValueError(f"evaluation run is not a bridge run: {root}")
        bridge_protocols.append(resolved["protocol"])
        bridge_protocol_sha256.add(resolved["protocol_sha256"])
        saved_rows = _load_jsonl(root / "items.jsonl")
        by_condition[condition] = _recompute_trace_diagnostics(
            saved_rows,
            validation_examples,
            resolved.get("expected_representation"),
        )
    if set(by_condition) != {"c0", "c1", "c2", "c3", "c4"}:
        raise ValueError("bridge report requires exactly C0-C4 evaluation runs")
    identities = {
        condition: {
            (row["item_id"], row["reasoning_budget"], row["base_seed"]) for row in rows
        }
        for condition, rows in by_condition.items()
    }
    if any(value != identities["c0"] for value in identities.values()):
        raise ValueError("bridge conditions do not contain identical item/budget/seed rows")
    if any(protocol != bridge_protocols[0] for protocol in bridge_protocols[1:]):
        raise ValueError("bridge conditions were evaluated under different protocol payloads")
    if len(bridge_protocol_sha256) != 1:
        raise ValueError("bridge conditions were evaluated under different protocol hashes")
    protocol_compatibility = _protocol_compatibility(
        bridge_protocols[0],
        config.to_dict(),
    )
    atomic_write_json(
        output_root / "protocol_compatibility.json",
        protocol_compatibility,
    )

    grouped: dict[tuple[str, int, str | None, int | None], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for condition, rows in by_condition.items():
        for row in rows:
            grouped[(condition, row["reasoning_budget"], None, None)].append(row)
            grouped[
                (
                    condition,
                    row["reasoning_budget"],
                    row["task_family"],
                    row["depth"],
                )
            ].append(row)
    aggregate_rows = [
        {
            "condition_id": condition,
            "reasoning_budget": budget,
            "task_family": family,
            "depth": depth,
            **_aggregate_group(rows),
        }
        for (condition, budget, family, depth), rows in sorted(
            grouped.items(),
            key=lambda item: tuple("" if value is None else str(value) for value in item[0]),
        )
    ]
    atomic_write_json(
        output_root / "aggregate.json",
        {"schema_version": 1, "rows": aggregate_rows},
    )

    indexed = {
        condition: {
            (row["item_id"], row["reasoning_budget"]): row for row in rows
        }
        for condition, rows in by_condition.items()
    }
    comparisons = [
        _paired_comparison(
            indexed,
            depth=depth,
            budget=budget,
            margin=config.noninferiority_margin,
            bootstrap_seed=config.bootstrap_seed,
            bootstrap_samples=config.bootstrap_samples,
        )
        for depth in (2, 4)
        for budget in config.primary_budgets
    ]
    atomic_write_json(
        output_root / "shallow_noninferiority.json",
        {"schema_version": 1, "comparisons": comparisons},
    )

    training_gate = _training_gate(train_roots)
    atomic_write_json(output_root / "training_gate.json", training_gate)
    feasible_c4 = [
        row
        for row in by_condition["c4"]
        if row["reasoning_budget"] >= min(config.primary_budgets)
    ]
    shallow_cells = [
        [
            row
            for row in feasible_c4
            if row["depth"] == depth and row["reasoning_budget"] == budget
        ]
        for depth in (2, 4)
        for budget in config.evaluation_budgets
        if budget >= min(config.primary_budgets)
    ]
    if any(not cell for cell in shallow_cells):
        raise ValueError("bridge evaluation lacks one or more feasible shallow C4 cells")
    feasible_metrics = _aggregate_group(feasible_c4)
    trace_cell_rates = [
        sum(row["trace_diagnostics"]["trace_parse_success"] for row in cell) / len(cell)
        for cell in shallow_cells
    ]
    budget_violations = sum(
        row["token_counts"]["reasoning"] > row["reasoning_budget"]
        for rows in by_condition.values()
        for row in rows
    )
    collapse_passed = (
        feasible_metrics["empty_reasoning_rate"] <= 0.01
        and feasible_metrics["prompt_copy_rate"] <= 0.01
        and feasible_metrics["repeated_line_rate"] <= 0.01
    )
    gate = {
        "schema_version": 1,
        "test_data_inspected": test_data_inspected,
        "trace_diagnostics_recomputed_from_saved_reasoning": True,
        "bridge_protocol_sha256": next(iter(bridge_protocol_sha256)),
        "report_protocol_sha256": sha256_file(Path(protocol_path).resolve()),
        "protocol_compatibility_passed": protocol_compatibility["passed"],
        "analysis_only_protocol_delta": sorted(
            protocol_compatibility["changed_fields"]
        ),
        "training_runs_passed": training_gate["passed"],
        "budget_violation_count": budget_violations,
        "budget_enforcement_passed": budget_violations == 0,
        "c4_feasible_region_parse_rate": feasible_metrics["region_parse_success_rate"],
        "c4_feasible_region_parse_passed": (
            feasible_metrics["region_parse_success_rate"] >= 0.99
        ),
        "c4_shallow_trace_parse_cell_rates": trace_cell_rates,
        "c4_shallow_trace_parse_passed": min(trace_cell_rates) >= 0.99,
        "shallow_noninferiority_passed": all(
            comparison["point_estimate_passed"] for comparison in comparisons
        ),
        "collapse_metrics": {
            "empty_reasoning_rate": feasible_metrics["empty_reasoning_rate"],
            "prompt_copy_rate": feasible_metrics["prompt_copy_rate"],
            "repeated_line_rate": feasible_metrics["repeated_line_rate"],
        },
        "collapse_passed": collapse_passed,
    }
    gate["passed"] = (
        not gate["test_data_inspected"]
        and gate["protocol_compatibility_passed"]
        and gate["training_runs_passed"]
        and gate["budget_enforcement_passed"]
        and gate["c4_feasible_region_parse_passed"]
        and gate["c4_shallow_trace_parse_passed"]
        and gate["shallow_noninferiority_passed"]
        and gate["collapse_passed"]
    )
    atomic_write_json(output_root / "gate_summary.json", gate)
    manifest = tree_manifest(output_root, output_root.name)
    return {
        "output_dir": str(output_root),
        "passed": gate["passed"],
        "test_data_inspected": gate["test_data_inspected"],
        "manifest_files": len(manifest["files"]),
    }
