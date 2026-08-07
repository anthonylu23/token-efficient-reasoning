"""Saved-artifact-only aggregation, plots, failure analysis, and M2 gate review."""

from __future__ import annotations

import csv
import html
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.provenance import sha256_file

from .artifacts import tree_manifest


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean_boolean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning_tokens = [row["token_counts"]["reasoning"] for row in rows]
    verified_transitions = [
        row["trace_diagnostics"]["verified_prefix_transitions"] for row in rows
    ]
    total_reasoning = sum(reasoning_tokens)
    return {
        "generation_count": len(rows),
        "accuracy": _mean_boolean(rows, "correct"),
        "region_parse_success_rate": sum(
            row["parse_status"] == "valid" for row in rows
        )
        / len(rows),
        "trace_parse_success_rate": sum(
            row["trace_diagnostics"]["trace_parse_success"] for row in rows
        )
        / len(rows),
        "complete_replay_rate": sum(
            row["trace_diagnostics"]["replay_valid"] for row in rows
        )
        / len(rows),
        "invalid_transition_event_rate": sum(
            diagnostics.get(
                "invalid_transition_event",
                not diagnostics["trace_parse_success"]
                or diagnostics.get("invalid_transition_count", 0) > 0,
            )
            for diagnostics in (row["trace_diagnostics"] for row in rows)
        )
        / len(rows),
        "canonical_prefix_rate": sum(value > 0 for value in verified_transitions)
        / len(rows),
        "verified_transitions": sum(verified_transitions),
        "verified_transitions_per_reasoning_token": (
            sum(verified_transitions) / total_reasoning if total_reasoning else 0.0
        ),
        "forced_transition_rate": _mean_boolean(rows, "forced_reasoning_transition"),
        "budget_violation_count": sum(
            row["token_counts"]["reasoning"] > row["reasoning_budget"] for row in rows
        ),
        "empty_reasoning_rate": sum(
            row["trace_diagnostics"]["reasoning_empty"] for row in rows
        )
        / len(rows),
        "prompt_copy_rate": sum(
            row["trace_diagnostics"]["prompt_copied"] for row in rows
        )
        / len(rows),
        "repeated_line_rate": sum(
            row["trace_diagnostics"]["repeated_line_count"] > 0 for row in rows
        )
        / len(rows),
        "median_reasoning_tokens": statistics.median(reasoning_tokens),
        "mean_latency_seconds": statistics.fmean(
            row["latency_seconds"]["total"] for row in rows
        ),
    }


def _bootstrap_interval(
    differences: list[float],
    *,
    seed: int,
    samples: int = 2_000,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("cannot bootstrap an empty paired sample")
    rng = random.Random(seed)
    count = len(differences)
    estimates = []
    for _ in range(samples):
        estimates.append(
            sum(differences[rng.randrange(count)] for _ in range(count)) / count
        )
    estimates.sort()
    return estimates[int(samples * 0.025)], estimates[min(int(samples * 0.975), samples - 1)]


def _shallow_comparisons(
    by_condition: dict[str, list[dict[str, Any]]],
    budgets: list[int],
) -> list[dict[str, Any]]:
    indexed = {
        condition: {
            (row["item_id"], row["reasoning_budget"]): row for row in rows
        }
        for condition, rows in by_condition.items()
    }
    comparisons: list[dict[str, Any]] = []
    for depth in (2, 4):
        for budget in budgets:
            keys = [
                key
                for key, row in indexed["c4"].items()
                if row["depth"] == depth and row["reasoning_budget"] == budget
            ]
            control_accuracies = {
                condition: sum(indexed[condition][key]["correct"] for key in keys)
                / len(keys)
                for condition in ("c2", "c3")
            }
            control = max(control_accuracies, key=control_accuracies.get)
            differences = [
                float(indexed["c4"][key]["correct"])
                - float(indexed[control][key]["correct"])
                for key in keys
            ]
            difference = statistics.fmean(differences)
            lower, upper = _bootstrap_interval(
                differences,
                seed=depth * 10_000 + budget,
            )
            comparisons.append(
                {
                    "depth": depth,
                    "budget": budget,
                    "item_count": len(keys),
                    "strongest_control": control,
                    "control_accuracy": control_accuracies[control],
                    "c4_accuracy": sum(
                        indexed["c4"][key]["correct"] for key in keys
                    )
                    / len(keys),
                    "paired_difference": difference,
                    "bootstrap_95_ci": [lower, upper],
                    "noninferiority_margin": -0.01,
                    "point_estimate_passed": difference >= -0.01,
                }
            )
    return comparisons


def _write_svg_plot(
    path: Path,
    title: str,
    y_label: str,
    series: dict[str, list[tuple[int, float]]],
    *,
    y_min: float = 0.0,
    y_max: float | None = None,
    x_label: str = "Reasoning-token budget",
) -> None:
    width, height = 900, 540
    left, right, top, bottom = 90, 30, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [value for points in series.values() for _, value in points]
    maximum = y_max if y_max is not None else max(values + [1e-9]) * 1.05
    maximum = max(maximum, y_min + 1e-9)
    budgets = sorted({budget for points in series.values() for budget, _ in points})

    def x_coordinate(budget: int) -> float:
        if len(budgets) == 1:
            return left + plot_width / 2
        return left + budgets.index(budget) * plot_width / (len(budgets) - 1)

    def y_coordinate(value: float) -> float:
        return top + (maximum - value) * plot_height / (maximum - y_min)

    colors = {
        "c0": "#6b7280",
        "c1": "#8b5cf6",
        "c2": "#2563eb",
        "c3": "#f59e0b",
        "c4": "#16a34a",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        f'stroke="#111827"/>',
    ]
    for tick in range(6):
        value = y_min + (maximum - y_min) * tick / 5
        y = y_coordinate(value)
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                'stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{value:.2f}</text>',
            ]
        )
    for budget in budgets:
        x = x_coordinate(budget)
        svg.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{budget}</text>'
        )
    for index, (condition, points) in enumerate(sorted(series.items())):
        color = colors.get(condition, "#111827")
        coordinates = " ".join(
            f"{x_coordinate(budget):.2f},{y_coordinate(value):.2f}"
            for budget, value in points
        )
        svg.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2.5"/>'
        )
        for budget, value in points:
            svg.append(
                f'<circle cx="{x_coordinate(budget):.2f}" cy="{y_coordinate(value):.2f}" '
                f'r="4" fill="{color}"/>'
            )
        legend_x = left + 15 + index * 145
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="{height - 18}" x2="{legend_x + 24}" '
                f'y2="{height - 18}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 30}" y="{height - 14}" font-family="sans-serif" '
                f'font-size="13">{html.escape(condition.upper())}</text>',
            ]
        )
    svg.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 38}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" '
            f'transform="rotate(-90 20 {top + plot_height / 2})">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _failure_sample(by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    indexed = {
        condition: {
            (row["item_id"], row["reasoning_budget"]): row for row in rows
        }
        for condition, rows in by_condition.items()
    }
    selected: list[dict[str, Any]] = []
    seen_strata: set[tuple[Any, ...]] = set()
    for key in sorted(indexed["c4"]):
        c4 = indexed["c4"][key]
        controls = [indexed[condition][key] for condition in ("c2", "c3")]
        reason = None
        if not c4["trace_diagnostics"]["trace_parse_success"]:
            reason = "c4_trace_parse_failure"
        elif not c4["trace_diagnostics"]["replay_valid"]:
            reason = "c4_incomplete_or_invalid_trace"
        elif not c4["correct"] and any(row["correct"] for row in controls):
            reason = "c4_accuracy_regression"
        elif c4["correct"] and not any(row["correct"] for row in controls):
            reason = "c4_accuracy_gain"
        if reason is None:
            continue
        stratum = (
            reason,
            c4["task_family"],
            c4["depth"],
            c4["reasoning_budget"],
        )
        if stratum in seen_strata:
            continue
        seen_strata.add(stratum)
        selected.append(
            {
                "reason": reason,
                "item_id": c4["item_id"],
                "task_family": c4["task_family"],
                "depth": c4["depth"],
                "budget": c4["reasoning_budget"],
                "conditions": {
                    condition: {
                        "correct": indexed[condition][key]["correct"],
                        "reasoning_text": indexed[condition][key]["reasoning_text"],
                        "parsed_answer": indexed[condition][key]["parsed_answer"],
                        "trace_diagnostics": indexed[condition][key]["trace_diagnostics"],
                    }
                    for condition in sorted(indexed)
                },
            }
        )
        if len(selected) >= 80:
            break
    return {
        "schema_version": 1,
        "selection": "first deterministic same-item contrast per reason/family/depth/budget",
        "items": selected,
    }


def _training_gate(training_dirs: list[Path]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for root in training_dirs:
        resolved = _load_json(root / "resolved_config.json")
        condition = resolved["condition"]["id"]
        telemetry = _load_json(root / "telemetry.json")
        reload_result = _load_json(root / "reload_verification.json")
        loss_values = [
            entry["loss"]
            for entry in telemetry["log_history"]
            if "loss" in entry and isinstance(entry["loss"], (int, float))
        ]
        finite_loss = bool(loss_values) and all(math.isfinite(value) for value in loss_values)
        by_condition[condition] = {
            "run_id": resolved["run_id"],
            "run_dir": str(root),
            "global_step": telemetry["global_step"],
            "finite_loss": finite_loss,
            "trainable_parameters_changed": telemetry["trainable_parameters_changed"],
            "reload_identical": reload_result["identical"],
            "peak_allocated_bytes": telemetry["torch_cuda_peak_allocated_bytes"],
            "wall_time_seconds": telemetry["wall_time_seconds"],
            "adapter_size_bytes": telemetry["adapter_size_bytes"],
            "passed": (
                finite_loss
                and telemetry["trainable_parameters_changed"]
                and reload_result["identical"]
            ),
        }
    expected = {"c2", "c3", "c4"}
    return {
        "schema_version": 1,
        "conditions": by_condition,
        "all_conditions_present": set(by_condition) == expected,
        "passed": set(by_condition) == expected
        and all(value["passed"] for value in by_condition.values()),
    }


def build_m2_report(
    evaluation_dirs: list[str | Path],
    training_dirs: list[str | Path],
    output_dir: str | Path,
    gate_review_path: str | Path,
    *,
    m1_gate_path: str | Path = "data/m1/gate_summary.json",
) -> dict[str, Any]:
    """Build every M2 aggregate and gate artifact without loading a model."""

    run_roots = [Path(path).resolve() for path in evaluation_dirs]
    train_roots = [Path(path).resolve() for path in training_dirs]
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    by_condition: dict[str, list[dict[str, Any]]] = {}
    run_metadata: dict[str, Any] = {}
    for root in run_roots:
        resolved = _load_json(root / "resolved_config.json")
        condition = resolved["condition"]["id"]
        if condition in by_condition:
            raise ValueError(f"duplicate evaluation condition: {condition}")
        by_condition[condition] = _load_jsonl(root / "items.jsonl")
        run_metadata[condition] = {
            "run_id": resolved["run_id"],
            "run_dir": str(root),
            "manifest_sha256": sha256_file(root / "manifest.json"),
        }
    expected_conditions = {"c0", "c1", "c2", "c3", "c4"}
    if set(by_condition) != expected_conditions:
        raise ValueError("M2 report requires exactly C0-C4 evaluation runs")
    identities = {
        condition: {
            (row["item_id"], row["reasoning_budget"], row["base_seed"]) for row in rows
        }
        for condition, rows in by_condition.items()
    }
    baseline_identities = identities["c0"]
    if any(value != baseline_identities for value in identities.values()):
        raise ValueError("evaluation conditions do not contain identical item/budget/seed rows")
    budgets = sorted({row["reasoning_budget"] for row in by_condition["c0"]})

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
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
    aggregates = []
    for (condition, budget, family, depth), rows in sorted(
        grouped.items(),
        key=lambda item: tuple(
            "" if value is None else str(value) for value in item[0]
        ),
    ):
        aggregates.append(
            {
                "condition_id": condition,
                "reasoning_budget": budget,
                "task_family": family,
                "depth": depth,
                **_aggregate_group(rows),
            }
        )
    atomic_write_json(
        output_root / "aggregate.json",
        {
            "schema_version": 1,
            "runs": run_metadata,
            "rows": aggregates,
        },
    )
    with (output_root / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)

    overall = [
        row for row in aggregates if row["task_family"] is None and row["depth"] is None
    ]
    accuracy_series = {
        condition: [
            (budget, next(
                row["accuracy"]
                for row in overall
                if row["condition_id"] == condition
                and row["reasoning_budget"] == budget
            ))
            for budget in budgets
        ]
        for condition in sorted(by_condition)
    }
    density_series = {
        condition: [
            (budget, next(
                row["verified_transitions_per_reasoning_token"]
                for row in overall
                if row["condition_id"] == condition
                and row["reasoning_budget"] == budget
            ))
            for budget in budgets
        ]
        for condition in sorted(by_condition)
    }
    _write_svg_plot(
        output_root / "accuracy_frontier.svg",
        "M2 validation accuracy frontier",
        "Exact-answer accuracy",
        accuracy_series,
        y_max=1.0,
    )
    _write_svg_plot(
        output_root / "verified_density_frontier.svg",
        "M2 verified reasoning density",
        "Verified transitions / reasoning token",
        density_series,
    )

    comparisons = _shallow_comparisons(by_condition, budgets)
    atomic_write_json(
        output_root / "shallow_noninferiority.json",
        {"schema_version": 1, "comparisons": comparisons},
    )
    failure_sample = _failure_sample(by_condition)
    atomic_write_json(output_root / "failure_analysis.json", failure_sample)
    training_gate = _training_gate(train_roots)
    atomic_write_json(output_root / "training_gate.json", training_gate)

    c4_rows = by_condition["c4"]
    c4_metrics = _aggregate_group(c4_rows)
    structured_usable = (
        c4_metrics["trace_parse_success_rate"] >= 0.90
        and c4_metrics["canonical_prefix_rate"] >= 0.80
        and c4_metrics["empty_reasoning_rate"] <= 0.01
        and c4_metrics["prompt_copy_rate"] <= 0.01
    )
    noninferiority_passed = all(
        comparison["point_estimate_passed"] for comparison in comparisons
    )
    m1_gate = _load_json(Path(m1_gate_path).resolve())
    gate = {
        "schema_version": 1,
        "training_runs_passed": training_gate["passed"],
        "c4_region_parse_rate": c4_metrics["region_parse_success_rate"],
        "c4_region_parse_passed": c4_metrics["region_parse_success_rate"] >= 0.99,
        "budget_violation_count": sum(
            row["token_counts"]["reasoning"] > row["reasoning_budget"]
            for rows in by_condition.values()
            for row in rows
        ),
        "budget_enforcement_passed": all(
            row["token_counts"]["reasoning"] <= row["reasoning_budget"]
            for rows in by_condition.values()
            for row in rows
        ),
        "c4_structured_trace_metrics": c4_metrics,
        "c4_structured_trace_usable": structured_usable,
        "shallow_noninferiority_passed": noninferiority_passed,
        "pipeline_artifacts_complete": True,
        "m1_human_review_complete": m1_gate.get("human_review_complete", False),
    }
    gate["engineering_passed"] = all(
        (
            gate["training_runs_passed"],
            gate["c4_region_parse_passed"],
            gate["budget_enforcement_passed"],
            gate["c4_structured_trace_usable"],
            gate["pipeline_artifacts_complete"],
        )
    )
    gate["advance_to_m3"] = (
        gate["engineering_passed"]
        and gate["shallow_noninferiority_passed"]
        and gate["m1_human_review_complete"]
    )
    atomic_write_json(output_root / "gate_summary.json", gate)

    decision = "ADVANCE" if gate["advance_to_m3"] else "REVISE OR HOLD"
    lines = [
        "# Milestone 2 gate review",
        "",
        f"**Decision:** {decision}",
        "",
        "## Scope",
        "",
        "Qwen3-0.6B smoke-test training and fixed-budget validation for C0-C4. "
        "This is an engineering and format-learnability gate, not the primary scientific result.",
        "",
        "## Run evidence",
        "",
    ]
    for condition in sorted(run_metadata):
        metadata = run_metadata[condition]
        lines.append(
            f"- {condition.upper()}: `{metadata['run_id']}`; manifest "
            f"`{metadata['manifest_sha256']}`."
        )
    lines.extend(
        [
            "",
            "## Acceptance criteria",
            "",
            f"- C2-C4 training/reload gate: **{training_gate['passed']}**.",
            f"- C4 region parseability: **{c4_metrics['region_parse_success_rate']:.3%}** "
            f"(required at least 99%).",
            f"- Post-enforcement budget violations: **{gate['budget_violation_count']}**.",
            f"- C4 structured-trace usability: **{structured_usable}** "
            f"(trace parse {c4_metrics['trace_parse_success_rate']:.3%}; canonical-prefix "
            f"{c4_metrics['canonical_prefix_rate']:.3%}; complete replay "
            f"{c4_metrics['complete_replay_rate']:.3%}).",
            f"- Shallow non-inferiority point-estimate gate: **{noninferiority_passed}**.",
            f"- M1 human review complete: **{gate['m1_human_review_complete']}**.",
            "",
            "## Known deviations and hold reasons",
            "",
            "- The all-budget structured-usability calculation includes deliberately truncated "
            "low-budget generations; budget-stratified metrics remain in `aggregate.csv`.",
            "- Failed shallow comparisons and their paired bootstrap intervals remain in "
            "`shallow_noninferiority.json`.",
            "- The prerequisite M1 human review is unsigned.",
            "- No M3 training configuration is selected while the advancement gate is held.",
            "",
            "## Interpretation",
            "",
            "A failed scientific threshold does not erase successful pipeline execution. "
            "The complete paired comparisons and bootstrap intervals are retained in "
            "`shallow_noninferiority.json`; same-item failure contrasts are retained in "
            "`failure_analysis.json`.",
            "",
            "## Reproduction",
            "",
            "All tables and SVG plots in the report directory are generated from saved item-level "
            "evaluation artifacts without loading a model or rerunning training.",
            "",
        ]
    )
    gate_review = Path(gate_review_path).resolve()
    gate_review.parent.mkdir(parents=True, exist_ok=True)
    gate_review.write_text("\n".join(lines), encoding="utf-8")
    report_manifest = tree_manifest(output_root, output_root.name)
    return {
        "output_dir": str(output_root),
        "gate_review": str(gate_review),
        "engineering_passed": gate["engineering_passed"],
        "advance_to_m3": gate["advance_to_m3"],
        "report_manifest_files": len(report_manifest["files"]),
    }
