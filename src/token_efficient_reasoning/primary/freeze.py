"""Hash-addressed M3 protocol freeze and test-access guard."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.controlled.schema import load_examples
from token_efficient_reasoning.provenance import (
    collect_environment,
    collect_hardware,
    collect_repository_state,
    sha256_file,
)
from token_efficient_reasoning.training.artifacts import tree_manifest
from token_efficient_reasoning.training.config import QLoRAConfig

from .config import M3Config, M3ConfigError
from .planning import build_run_plan, refresh_run_plan_hash


def create_freeze(
    protocol_path: str | Path,
    qlora_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create an immutable snapshot after every prerequisite has passed."""

    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    config = M3Config.from_json(protocol_source)
    checkpoint, dataset_dir, split_paths = config.validate_references(protocol_source)
    qlora = QLoRAConfig.from_json(qlora_source)
    readiness = config.readiness(protocol_source)
    if not readiness["ready_to_freeze"]:
        missing = [
            name
            for name, passed in (
                ("M1 human review", readiness["m1_human_review_passed"]),
                ("validation bridge", readiness["bridge_gate_passed"]),
            )
            if not passed
        ]
        raise M3ConfigError(
            "M3 cannot be frozen until prerequisites pass: " + ", ".join(missing)
        )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = build_run_plan(config)
    item_counts = {
        split: len(load_examples(path))
        for split, path in split_paths.items()
        if split in config.evaluation_splits
    }
    plan["counts"]["items_by_split"] = item_counts
    plan["counts"]["generations"] = (
        len(config.training_seeds)
        * 5
        * sum(item_counts.values())
        * len(config.evaluation_budgets)
    )
    plan["counts"]["primary_generations_without_compositional"] = (
        len(config.training_seeds)
        * 5
        * (item_counts["test"] + item_counts["test_depth"])
        * len(config.evaluation_budgets)
    )
    refresh_run_plan_hash(plan)
    atomic_write_json(root / "run_plan.json", plan)
    atomic_write_json(
        root / "freeze.json",
        {
            "schema_version": 1,
            "experiment_name": config.experiment_name,
            "protocol": config.to_dict(),
            "protocol_path": str(protocol_source),
            "protocol_sha256": sha256_file(protocol_source),
            "qlora": qlora.to_dict(),
            "qlora_path": str(qlora_source),
            "qlora_sha256": sha256_file(qlora_source),
            "checkpoint": asdict(checkpoint),
            "dataset_dir": str(dataset_dir),
            "dataset_manifest_sha256": config.dataset_manifest_sha256,
            "split_sha256": {
                split: sha256_file(path) for split, path in split_paths.items()
            },
            "readiness": readiness,
            "run_plan_sha256": plan["run_plan_sha256"],
            "repository": collect_repository_state(Path.cwd()),
            "environment": collect_environment(),
            "hardware": collect_hardware(),
            "test_access_authorized": True,
        },
    )
    manifest = tree_manifest(root, root.name)
    return {
        "freeze_dir": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "run_plan_sha256": plan["run_plan_sha256"],
        "training_runs": len(plan["training_runs"]),
        "evaluation_runs": len(plan["evaluation_runs"]),
        "generations": plan["counts"]["generations"],
        "manifest_files": len(manifest["files"]),
    }


def validate_freeze(freeze_dir: str | Path) -> dict[str, Any]:
    """Validate the freeze manifest and return its authorization metadata."""

    root = Path(freeze_dir).resolve()
    manifest_path = root / "manifest.json"
    freeze_path = root / "freeze.json"
    plan_path = root / "run_plan.json"
    if not all(path.is_file() for path in (manifest_path, freeze_path, plan_path)):
        raise M3ConfigError("freeze directory lacks freeze.json, run_plan.json, or manifest.json")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for name in ("freeze.json", "run_plan.json"):
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if expected != sha256_file(root / name):
            raise M3ConfigError(f"frozen file hash mismatch: {name}")
    with freeze_path.open(encoding="utf-8") as handle:
        frozen = json.load(handle)
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if not frozen.get("test_access_authorized"):
        raise M3ConfigError("freeze does not authorize test access")
    if frozen.get("run_plan_sha256") != plan.get("run_plan_sha256"):
        raise M3ConfigError("frozen run-plan identity mismatch")
    return {
        "schema_version": 1,
        "freeze_dir": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "run_plan_sha256": plan["run_plan_sha256"],
        "test_access_authorized": True,
    }


def validate_frozen_workspace(freeze_dir: str | Path) -> dict[str, Any]:
    """Fail unless the current repository content exactly matches the frozen workspace."""

    root = Path(freeze_dir).resolve()
    authorization = validate_freeze(root)
    with (root / "freeze.json").open(encoding="utf-8") as handle:
        frozen = json.load(handle)
    expected = frozen.get("repository")
    current = collect_repository_state(Path.cwd())
    if expected != current:
        expected_digest = expected.get("workspace_sha256") if isinstance(expected, dict) else None
        raise M3ConfigError(
            "current repository workspace differs from the M3 freeze "
            f"(expected {expected_digest}, observed {current['workspace_sha256']})"
        )
    return {
        **authorization,
        "workspace_matches_current": True,
        "workspace_sha256": current["workspace_sha256"],
    }
