"""Hash-addressed M3 protocol freeze and test-access guard."""

from __future__ import annotations

import hashlib
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_bridge_report_bundle(
    gate_path: Path,
) -> tuple[Path, bytes, dict[str, bytes]]:
    """Load and validate the saved-artifact-only bridge report as one immutable bundle."""

    report_root = gate_path.parent
    manifest_path = report_root / "manifest.json"
    if not manifest_path.is_file():
        raise M3ConfigError(
            f"bridge report manifest is required beside the gate summary: {manifest_path}"
        )
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise M3ConfigError(f"invalid bridge report manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise M3ConfigError("bridge report manifest must be a JSON object")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise M3ConfigError("bridge report manifest lacks a files mapping")
    try:
        gate_relative = gate_path.relative_to(report_root).as_posix()
    except ValueError as error:
        raise M3ConfigError("bridge gate must be inside its report directory") from error
    if gate_relative not in files:
        raise M3ConfigError("bridge report manifest does not bind the configured gate summary")

    bundle: dict[str, bytes] = {}
    for relative, expected in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise M3ConfigError(f"unsafe path in bridge report manifest: {relative}")
        source = report_root / relative_path
        if not source.is_file():
            raise M3ConfigError(f"bridge report file is missing: {source}")
        content = source.read_bytes()
        if not isinstance(expected, dict) or expected.get("sha256") != _sha256_bytes(content):
            raise M3ConfigError(f"bridge report file hash mismatch: {relative}")
        if expected.get("size_bytes") != len(content):
            raise M3ConfigError(f"bridge report file size mismatch: {relative}")
        bundle[relative_path.as_posix()] = content
    return manifest_path, manifest_bytes, bundle


def _snapshot_prerequisites(readiness: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Return prerequisite metadata and exact bytes to place inside the freeze."""

    human_path = Path(readiness["m1_human_review_path"])
    sample_path = Path(readiness["m1_review_sample_path"])
    bridge_path = Path(readiness["bridge_gate_path"])
    human_bytes = human_path.read_bytes()
    sample_bytes = sample_path.read_bytes()
    if _sha256_bytes(human_bytes) != readiness["m1_human_review_sha256"]:
        raise M3ConfigError("M1 human-review evidence changed while creating the freeze")
    if _sha256_bytes(sample_bytes) != readiness["m1_review_sample_sha256"]:
        raise M3ConfigError("M1 review sample changed while creating the freeze")

    report_manifest_path, report_manifest_bytes, report_bundle = _load_bridge_report_bundle(
        bridge_path
    )
    gate_relative = bridge_path.relative_to(bridge_path.parent).as_posix()
    if _sha256_bytes(report_bundle[gate_relative]) != readiness["bridge_gate_sha256"]:
        raise M3ConfigError("bridge gate evidence changed while creating the freeze")

    human_frozen = "prerequisites/m1_human_review.json"
    sample_frozen = "prerequisites/m1_review_sample.json"
    bridge_root = "prerequisites/bridge_report"
    gate_frozen = f"{bridge_root}/{gate_relative}"
    # tree_manifest reserves the basename manifest.json at every depth, so retain the source
    # bytes under an unambiguous name that the outer freeze manifest will include.
    report_manifest_frozen = f"{bridge_root}/report_manifest.json"
    copies = {
        human_frozen: human_bytes,
        sample_frozen: sample_bytes,
        report_manifest_frozen: report_manifest_bytes,
        **{f"{bridge_root}/{relative}": content for relative, content in report_bundle.items()},
    }
    evidence = {
        "m1_human_review": {
            "source_path": str(human_path),
            "frozen_path": human_frozen,
            "sha256": _sha256_bytes(human_bytes),
        },
        "m1_review_sample": {
            "source_path": str(sample_path),
            "frozen_path": sample_frozen,
            "sha256": _sha256_bytes(sample_bytes),
        },
        "bridge_gate": {
            "source_path": str(bridge_path),
            "frozen_path": gate_frozen,
            "sha256": _sha256_bytes(report_bundle[gate_relative]),
        },
        "bridge_report_manifest": {
            "source_path": str(report_manifest_path),
            "frozen_path": report_manifest_frozen,
            "sha256": _sha256_bytes(report_manifest_bytes),
        },
    }
    return evidence, copies


def _validate_manifest_tree(root: Path, manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise M3ConfigError("freeze manifest must be a JSON object")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise M3ConfigError("freeze manifest lacks a files mapping")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != root / "manifest.json"
    }
    if actual != set(files):
        raise M3ConfigError("freeze directory contents differ from its manifest")
    for relative, expected in files.items():
        path = root / relative
        if not isinstance(expected, dict) or expected.get("sha256") != sha256_file(path):
            raise M3ConfigError(f"frozen file hash mismatch: {relative}")
        if expected.get("size_bytes") != path.stat().st_size:
            raise M3ConfigError(f"frozen file size mismatch: {relative}")


def _validate_prerequisite_evidence(root: Path, frozen: dict[str, Any]) -> None:
    evidence = frozen.get("prerequisite_evidence")
    if not isinstance(evidence, dict):
        raise M3ConfigError("freeze lacks bound prerequisite evidence")
    for name in (
        "m1_human_review",
        "m1_review_sample",
        "bridge_gate",
        "bridge_report_manifest",
    ):
        record = evidence.get(name)
        if not isinstance(record, dict) or not record.get("frozen_path"):
            raise M3ConfigError(f"freeze lacks bound prerequisite evidence: {name}")
        relative_path = Path(record["frozen_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise M3ConfigError(f"invalid frozen prerequisite evidence path: {name}")
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise M3ConfigError(f"frozen prerequisite evidence hash mismatch: {name}")

    report_manifest_path = root / evidence["bridge_report_manifest"]["frozen_path"]
    with report_manifest_path.open(encoding="utf-8") as handle:
        report_manifest = json.load(handle)
    if not isinstance(report_manifest, dict):
        raise M3ConfigError("frozen bridge report manifest must be a JSON object")
    report_root = report_manifest_path.parent
    report_files = report_manifest.get("files")
    if not isinstance(report_files, dict):
        raise M3ConfigError("frozen bridge report manifest lacks a files mapping")
    for relative, expected in report_files.items():
        relative_path = Path(relative)
        if (
            not isinstance(expected, dict)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise M3ConfigError(f"invalid frozen bridge report entry: {relative}")
        path = report_root / relative_path
        if not path.is_file() or expected.get("sha256") != sha256_file(path):
            raise M3ConfigError(f"frozen bridge report file hash mismatch: {relative}")
        if expected.get("size_bytes") != path.stat().st_size:
            raise M3ConfigError(f"frozen bridge report file size mismatch: {relative}")



def _load_frozen_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise M3ConfigError(f"invalid frozen {label} JSON") from error
    if not isinstance(value, dict):
        raise M3ConfigError(f"frozen {label} must be a JSON object")
    return value


def _validate_prerequisite_semantics(root: Path, frozen: dict[str, Any]) -> None:
    """Revalidate the meaning and cross-bindings of copied prerequisite evidence."""

    evidence = frozen["prerequisite_evidence"]
    human_path = root / evidence["m1_human_review"]["frozen_path"]
    sample_path = root / evidence["m1_review_sample"]["frozen_path"]
    gate_path = root / evidence["bridge_gate"]["frozen_path"]
    human = _load_frozen_json(human_path, "M1 human review")
    sample = _load_frozen_json(sample_path, "M1 review sample")
    gate = _load_frozen_json(gate_path, "bridge gate")

    items = sample.get("items")
    if not isinstance(items, list) or len(items) != 34:
        raise M3ConfigError("frozen M1 review sample must contain exactly 34 items")
    item_ids = {
        item.get("item_id") for item in items if isinstance(item, dict) and item.get("item_id")
    }
    if len(item_ids) != 34:
        raise M3ConfigError("frozen M1 review sample item IDs must be unique and complete")
    failed_item_ids = human.get("failed_item_ids")
    accepted_decisions = {"accept", "accept_with_documented_limitations"}
    human_valid = (
        human.get("schema_version") == 1
        and human.get("status") == "complete"
        and human.get("decision") in accepted_decisions
        and bool(human.get("reviewer"))
        and bool(human.get("reviewed_at"))
        and bool(human.get("approval_reference"))
        and human.get("reviewed_item_count") == len(items)
        and human.get("dataset_manifest_sha256")
        == frozen.get("dataset_manifest_sha256")
        and human.get("review_sample_sha256") == sha256_file(sample_path)
        and isinstance(failed_item_ids, list)
        and set(failed_item_ids) <= item_ids
    )
    if not human_valid:
        raise M3ConfigError("frozen M1 human-review semantics or bindings are invalid")

    gate_valid = (
        gate.get("schema_version") == 1
        and gate.get("passed") is True
        and gate.get("test_data_inspected") is False
        and gate.get("protocol_compatibility_passed") is True
        and gate.get("report_protocol_sha256") == frozen.get("protocol_sha256")
    )
    if not gate_valid:
        raise M3ConfigError("frozen bridge-gate semantics or protocol binding are invalid")

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
    prerequisite_evidence, prerequisite_copies = _snapshot_prerequisites(readiness)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for relative, content in prerequisite_copies.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
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
            "schema_version": 2,
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
            "prerequisite_evidence": prerequisite_evidence,
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
    _validate_manifest_tree(root, manifest)
    with freeze_path.open(encoding="utf-8") as handle:
        frozen = json.load(handle)
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if frozen.get("schema_version") != 2:
        raise M3ConfigError("unsupported M3 freeze schema; recreate the freeze")
    readiness = frozen.get("readiness", {})
    if not (
        readiness.get("m1_human_review_passed")
        and readiness.get("m1_review_sample_matches")
        and readiness.get("bridge_gate_passed")
        and readiness.get("ready_to_freeze")
    ):
        raise M3ConfigError("freeze does not record passing prerequisite gates")
    _validate_prerequisite_evidence(root, frozen)
    _validate_prerequisite_semantics(root, frozen)
    if not frozen.get("test_access_authorized"):
        raise M3ConfigError("freeze does not authorize test access")
    if frozen.get("run_plan_sha256") != plan.get("run_plan_sha256"):
        raise M3ConfigError("frozen run-plan identity mismatch")
    return {
        "schema_version": 2,
        "freeze_dir": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "run_plan_sha256": plan["run_plan_sha256"],
        "prerequisite_evidence_bound": True,
        "prerequisite_semantics_validated": True,
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
