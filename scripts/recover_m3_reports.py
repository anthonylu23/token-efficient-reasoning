#!/usr/bin/env python3
"""Regenerate CPU-only M3 reports with a recorded reader fix and immutable source runs."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from token_efficient_reasoning import budget_analysis
from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.primary import final_reporting
from token_efficient_reasoning.primary.freeze import validate_frozen_workspace
from token_efficient_reasoning.provenance import sha256_file
from token_efficient_reasoning.training.artifacts import tree_manifest, write_failure


def recover(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("recovered reports must be outside the frozen source checkout")
    if output.exists():
        raise FileExistsError(output)
    previous_directory = Path.cwd()
    try:
        # Keep the frozen-workspace guard intact while importing the fixed report reader.
        os.chdir(source)
        authorization = validate_frozen_workspace(source / "artifacts/m3/freeze")
        primary = source / "artifacts/m3/primary"
        registry = source / "artifacts/m3/experiment_registry.jsonl"
        freeze = source / "artifacts/m3/freeze"
        plan = json.loads((freeze / "run_plan.json").read_text())
        inputs = [freeze / "manifest.json", registry]
        inputs.extend(primary / "training" / run["run_id"] / "manifest.json"
                      for run in plan["training_runs"])
        inputs.extend(primary / "evaluation" / run["run_id"] / "manifest.json"
                      for run in plan["evaluation_runs"])
        # Retain the unsuccessful original report as evidence, without editing it.
        inputs.extend(path for path in sorted((primary / "report").rglob("*")) if path.is_file())
        before = {str(path): sha256_file(path) for path in inputs}
        output.mkdir(parents=True, exist_ok=False)
        try:
            package = Path(final_reporting.__file__).resolve().parents[1]
            snapshot = output / "implementation" / "token_efficient_reasoning"
            for path in sorted(package.rglob("*.py")):
                target = snapshot / path.relative_to(package)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            shutil.copyfile(__file__, output / "implementation" / "recover_m3_reports.py")
            relative = Path("src/token_efficient_reasoning/primary/final_reporting.py")
            original = source / relative
            patched = Path(final_reporting.__file__)
            patch = "".join(difflib.unified_diff(
                original.read_text().splitlines(keepends=True),
                patched.read_text().splitlines(keepends=True),
                fromfile="frozen/" + str(relative), tofile="recovery/" + str(relative),
            ))
            (output / "report_reader.patch").write_text(patch)
            provenance = {
                "schema_version": 1, "started_at": datetime.now(UTC).isoformat(),
                "source_root": str(source), "authorization": authorization,
                "reason": "Evaluation manifests use run_id; frozen reader expected artifact_id.",
                "frozen_report_reader_sha256": sha256_file(original),
                "executed_report_reader_sha256": sha256_file(patched),
                "input_hashes_before": before,
                "model_execution": False,
                "statistical_contract": "unchanged; use frozen protocol and analysis seeds",
            }
            atomic_write_json(output / "recovery.json", provenance)
            print("Validating saved inputs and building primary report...", flush=True)
            primary_output = output / "primary"
            result = final_reporting.build_primary_report(
                freeze, primary / "evaluation", primary / "training", registry, primary_output
            )
            budget_analysis.verify_manifest(primary_output)
            print("Primary report complete. Building supplemental diagnostics...", flush=True)
            supplemental = budget_analysis.analyze(
                source, "primary", output / "budget", primary_report=primary_output
            )
            budget_analysis.verify_manifest(output / "budget")
            after = {str(path): sha256_file(path) for path in inputs}
            if after != before:
                raise ValueError("source manifests or original report changed during recovery")
            final_authorization = validate_frozen_workspace(freeze)
            if final_authorization != authorization:
                raise ValueError("frozen workspace authorization changed during recovery")
            provenance.update({
                "completed_at": datetime.now(UTC).isoformat(),
                "source_artifacts_unchanged": True, "primary_result": result,
                "supplemental_cells": len(supplemental["rows"]),
                "report_manifest_sha256": sha256_file(primary_output / "manifest.json"),
                "supplemental_manifest_sha256": sha256_file(output / "budget/manifest.json"),
            })
            atomic_write_json(output / "recovery.json", provenance)
            tree_manifest(output, output.name)
            return provenance
        except BaseException as error:
            write_failure(output, error)
            raise
    finally:
        os.chdir(previous_directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = recover(args.source_root, args.output_dir)
    print(json.dumps(result["primary_result"], indent=2))


if __name__ == "__main__":
    main()
