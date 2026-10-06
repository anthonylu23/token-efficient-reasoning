"""Descriptive budget-feasibility analysis, separate from frozen M3 inference."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .controlled.schema import CanonicalExample
from .provenance import sha256_file
from .training.diagnostics import diagnose_trace

REPRESENTATIONS = {"c2": "r0", "c3": "r2", "c4": "r3"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_manifest(root: Path) -> str:
    """Reject incomplete or modified input artifacts before reading their results."""
    if (root / "failure.json").exists():
        raise ValueError(f"failed artifact: {root}")
    manifest = read_json(root / "manifest.json")
    if not manifest.get("files"):
        raise ValueError(f"empty manifest: {root}")
    for relative, metadata in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("manifest path escapes artifact directory")
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
    return sha256_file(root / "manifest.json")


def diagnose_budget(row: dict[str, Any], example: CanonicalExample) -> dict[str, Any]:
    """Treat canonical length as representation feasibility, not a reasoning lower bound."""
    condition = row["condition_id"]
    representation = REPRESENTATIONS[condition]
    if row["source_instance_hash"] != example.instance_hash:
        raise ValueError("generation does not match canonical instance")
    if row["task_family"] != example.family or row["depth"] != example.depth:
        raise ValueError("generation task metadata differs from canonical instance")
    budget = row["reasoning_budget"]
    tokens = row["token_counts"]["reasoning"]
    if budget < 1 or not 0 <= tokens <= budget:
        raise ValueError("invalid reasoning budget or token count")
    canonical_tokens = example.token_measurements[row["checkpoint_id"]][representation]
    diagnostics = diagnose_trace(
        example,
        row["reasoning_text"],
        representation,
        allow_trailing_fragment=row["forced_reasoning_transition"],
    )
    return {
        "canonical_fits": canonical_tokens <= budget,
        "canonical_tokens": canonical_tokens,
        "correct": row["correct"],
        "forced": row["forced_reasoning_transition"],
        "replay_valid": diagnostics["replay_valid"],
        "trace_parse_success": diagnostics["trace_parse_success"],
        "verified_prefix_transitions": diagnostics["verified_prefix_transitions"],
        "reasoning_tokens": tokens,
        "parsed_transition_error": diagnostics["invalid_transition_count"] > 0,
        "trailing_fragment_recovered": diagnostics["trailing_fragment_recovered"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(row["reasoning_tokens"] for row in rows)
    return {
        "generation_count": len(rows),
        **{
            f"{field}_rate": sum(row[field] for row in rows) / len(rows)
            for field in (
                "correct",
                "forced",
                "replay_valid",
                "trace_parse_success",
                "parsed_transition_error",
                "trailing_fragment_recovered",
            )
        },
        "mean_verified_prefix_transitions": (
            sum(row["verified_prefix_transitions"] for row in rows) / len(rows)
        ),
        "verified_transitions_per_token": (
            sum(row["verified_prefix_transitions"] for row in rows) / total_tokens
            if total_tokens
            else None
        ),
    }


def analyze(
    source: Path, phase: str, output: Path, *, primary_report: Path | None = None
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.is_relative_to(source):
        raise ValueError("supplemental output must be outside the frozen source checkout")
    if primary_report is not None and phase != "primary":
        raise ValueError("primary_report is only valid for primary analysis")
    if output.exists():
        raise FileExistsError(output)
    evidence = {}
    dataset = source / "data/m1"
    dataset_manifest = read_json(dataset / "manifest.json")
    protocol = read_json(source / "configs/m3/protocol.json")
    if phase == "primary":
        report = (primary_report or source / "artifacts/m3/primary/report").resolve()
        if output.is_relative_to(report):
            raise ValueError("supplemental output must be outside the primary report")
        evidence[str(report)] = verify_manifest(report)
        inputs = read_json(report / "input_summary.json")
        freeze = source / "artifacts/m3/freeze"
        evidence[str(freeze)] = verify_manifest(freeze)
        if inputs["freeze_manifest_sha256"] != evidence[str(freeze)]:
            raise ValueError("primary report is bound to a different freeze")
        plan = read_json(freeze / "run_plan.json")
        if (
            inputs["generation_count"] != plan["counts"]["generations"]
            or inputs["evaluation_run_count"] != len(plan["evaluation_runs"])
            or inputs["training_run_count"] != len(plan["training_runs"])
            or inputs["run_plan_sha256"] != plan["run_plan_sha256"]
            or inputs["all_training_runs_passed"] is not True
        ):
            raise ValueError("primary report does not cover the completed frozen matrix")
        if (
            Path(inputs["evaluation_root"]).resolve() != source / "artifacts/m3/primary/evaluation"
            or Path(inputs["training_root"]).resolve() != source / "artifacts/m3/primary/training"
            or Path(inputs["freeze_dir"]).resolve() != freeze
        ):
            raise ValueError("primary report references different source artifacts")
        frozen = read_json(freeze / "freeze.json")
        if sha256_file(dataset / "manifest.json") != frozen["dataset_manifest_sha256"]:
            raise ValueError("dataset differs from freeze")
        protocol = frozen["protocol"]
        runs = [run for run in plan["evaluation_runs"] if run["condition_id"] in REPRESENTATIONS]
    elif phase == "bridge":
        report = source / "artifacts/m3/bridge/report"
        evidence[str(report)] = verify_manifest(report)
        runs = [
            {"run_id": condition, "condition_id": condition, "split": "validation"}
            for condition in REPRESENTATIONS
        ]
    else:
        raise ValueError("phase must be bridge or primary")
    if sha256_file(dataset / "manifest.json") != protocol["dataset_manifest_sha256"]:
        raise ValueError("dataset differs from protocol")
    groups = defaultdict(list)
    examples_by_split = {}
    for run in runs:
        split = run["split"]
        if split not in examples_by_split:
            relative = f"splits/{split}.jsonl"
            path = dataset / relative
            if sha256_file(path) != dataset_manifest["files"][relative]["sha256"]:
                raise ValueError(f"split hash mismatch: {split}")
            with path.open() as handle:
                examples = [CanonicalExample.from_dict(json.loads(line)) for line in handle]
            examples_by_split[split] = {example.id: example for example in examples}
            if len(examples_by_split[split]) != len(examples):
                raise ValueError("duplicate canonical item")
        examples = examples_by_split[split]
        root = source / f"artifacts/m3/{phase}/evaluation" / run["run_id"]
        evidence[str(root)] = verify_manifest(root)
        seen = set()
        with (root / "items.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["item_id"], row["reasoning_budget"])
                if key in seen or row["condition_id"] != run["condition_id"]:
                    raise ValueError("duplicate or wrong-condition generation")
                seen.add(key)
                example = examples[row["item_id"]]
                result = diagnose_budget(row, example)
                for family, depth in (("all", 0), (example.family, example.depth)):
                    group = (
                        split,
                        row["condition_id"],
                        row["reasoning_budget"],
                        family,
                        depth,
                        result["canonical_fits"],
                    )
                    groups[group].append(result)
        expected = {
            (item_id, budget) for item_id in examples for budget in protocol["evaluation_budgets"]
        }
        if seen != expected:
            raise ValueError(f"incomplete or foreign item-budget matrix: {root}")
    result = {
        "schema_version": 1,
        "phase": phase,
        "analysis_type": "descriptive supplemental; not part of frozen primary inference",
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "limitations": [
            "Canonical length is not a lower bound on valid alternative reasoning.",
            "Feasible subsets differ across representations; "
            "between-subset rates are not causal effects.",
            "Forced transitions and parsed errors can overlap; "
            "categories are not mutually exclusive.",
            "C0/C1 are omitted because their free-form traces lack comparable canonical grammars.",
        ],
        "source_manifests": evidence,
        "rows": [
            dict(
                zip(
                    ("split", "condition", "budget", "family", "depth", "canonical_fits"),
                    key,
                    strict=True,
                ),
                **summarize(rows),
            )
            for key, rows in sorted(groups.items())
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "budget_feasibility.json", result)
    lines = [
        "# Budget feasibility diagnostics",
        "",
        f"Phase: {phase}. Descriptive supplemental analysis; frozen inference is unchanged.",
        "",
        "| Condition | Budget | Canonical fits | Generations | "
        "Accuracy | Complete replay | Forced |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        if row["family"] == "all":
            lines.append(
                f"| {row['condition']} ({row['split']}) | {row['budget']} | "
                f"{row['canonical_fits']} | {row['generation_count']} | "
                f"{row['correct_rate']:.1%} | {row['replay_valid_rate']:.1%} | "
                f"{row['forced_rate']:.1%} |"
            )
    lines.extend(["", *[f"- {note}" for note in result["limitations"]], ""])
    (output / "REPORT.md").write_text("\n".join(lines))
    atomic_write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "files": {
                name: {"sha256": sha256_file(output / name)}
                for name in ("budget_feasibility.json", "REPORT.md")
            },
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("bridge", "primary"), default="bridge")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--primary-report", type=Path,
        help="Completed recovered primary report; all freeze/input checks still apply",
    )
    args = parser.parse_args()
    result = analyze(
        args.source_root, args.phase, args.output_dir, primary_report=args.primary_report
    )
    print(json.dumps({"phase": args.phase, "cells": len(result["rows"])}))


if __name__ == "__main__":
    main()
