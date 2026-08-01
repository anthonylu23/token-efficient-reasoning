"""Saved-artifact-only reporting for the frozen M3 primary experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.provenance import sha256_file
from token_efficient_reasoning.training.artifacts import tree_manifest
from token_efficient_reasoning.training.reporting import _write_svg_plot

from .config import M3Config, M3ConfigError
from .freeze import validate_frozen_workspace
from .statistics import (
    clustered_bootstrap_mean,
    derived_analysis_seed,
    holm_adjust,
    paired_cluster_values,
    paired_randomization_pvalue,
    percentile,
)

_CONDITIONS = ("c0", "c1", "c2", "c3", "c4")


@dataclass(frozen=True, slots=True)
class PrimaryRecord:
    """Compact analysis view of one immutable generation row."""

    run_id: str
    replicate: int
    condition_id: str
    split: str
    item_id: str
    task_family: str
    depth: int
    reasoning_budget: int
    correct: bool
    region_valid: bool
    trace_parse_success: bool
    replay_valid: bool
    verified_prefix_transitions: int
    parsed_transition_count: int
    invalid_transition_count: int
    reasoning_tokens: int
    answer_tokens: int
    sampled_tokens: int
    forced_transition: bool
    reasoning_empty: bool
    prompt_copied: bool
    repeated_line: bool
    latency_seconds: float
    generation_batch_size: int

    @property
    def invalid_transition_event(self) -> bool:
        return self.invalid_transition_count > 0

    @property
    def gold_path_coverage(self) -> float:
        return min(1.0, self.verified_prefix_transitions / self.depth)


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    run_id: str
    replicate: int
    condition_id: str
    split: str
    path: str
    records: tuple[PrimaryRecord, ...]
    generation_count: int
    batch_count: int
    estimated_wall_seconds: float
    sampled_tokens: int
    decode_tokens_per_second: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    decoding_sha256: str
    tokenizer_sha256: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _verify_tree_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run lacks manifest.json: {root}")
    manifest = _load_json(manifest_path)
    if manifest.get("artifact_id") != root.name:
        raise ValueError(f"manifest artifact ID does not match run directory: {root}")
    declared = manifest.get("files")
    if not isinstance(declared, dict):
        raise ValueError(f"manifest lacks a file map: {root}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != set(declared):
        missing = sorted(set(declared) - actual)
        extra = sorted(actual - set(declared))
        raise ValueError(
            f"manifest file set mismatch for {root}; missing={missing[:3]}, extra={extra[:3]}"
        )
    for relative, metadata in declared.items():
        expected = metadata.get("sha256") if isinstance(metadata, dict) else None
        if expected != sha256_file(root / relative):
            raise ValueError(f"manifest hash mismatch for {root / relative}")
    return manifest


def _read_freeze(
    freeze_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(freeze_dir).resolve()
    authorization = validate_frozen_workspace(root)
    frozen = _load_json(root / "freeze.json")
    plan = _load_json(root / "run_plan.json")
    config = M3Config.from_dict(frozen["protocol"])
    if frozen.get("run_plan_sha256") != plan.get("run_plan_sha256"):
        raise M3ConfigError("frozen report input has a mismatched run plan")
    return root, frozen, plan, {"authorization": authorization, "config": config}


def _compact_record(
    row: dict[str, Any],
    *,
    run_spec: dict[str, Any],
    config: M3Config,
) -> PrimaryRecord:
    diagnostics = row["trace_diagnostics"]
    token_counts = row["token_counts"]
    latency = float(row["latency_seconds"]["total"])
    record = PrimaryRecord(
        run_id=row["run_id"],
        replicate=run_spec["replicate"],
        condition_id=row["condition_id"],
        split=run_spec["split"],
        item_id=row["item_id"],
        task_family=row["task_family"],
        depth=int(row["depth"]),
        reasoning_budget=int(row["reasoning_budget"]),
        correct=bool(row["correct"]),
        region_valid=row["parse_status"] == "valid",
        trace_parse_success=bool(diagnostics["trace_parse_success"]),
        replay_valid=bool(diagnostics["replay_valid"]),
        verified_prefix_transitions=int(diagnostics["verified_prefix_transitions"]),
        parsed_transition_count=int(diagnostics["parsed_transition_count"]),
        invalid_transition_count=int(diagnostics["invalid_transition_count"]),
        reasoning_tokens=int(token_counts["reasoning"]),
        answer_tokens=int(token_counts["answer"]),
        sampled_tokens=int(token_counts["sampled_total"]),
        forced_transition=bool(row["forced_reasoning_transition"]),
        reasoning_empty=bool(diagnostics["reasoning_empty"]),
        prompt_copied=bool(diagnostics["prompt_copied"]),
        repeated_line=int(diagnostics["repeated_line_count"]) > 0,
        latency_seconds=latency,
        generation_batch_size=int(row["generation_batch_size"]),
    )
    if record.run_id != run_spec["run_id"]:
        raise ValueError(f"foreign run ID in {record.run_id}")
    if record.condition_id != run_spec["condition_id"]:
        raise ValueError(f"foreign condition in {record.run_id}")
    if row.get("base_seed") != run_spec["decoding_seed"]:
        raise ValueError(f"foreign decoding seed in {record.run_id}")
    if row.get("answer_allowance") != config.answer_allowance:
        raise ValueError(f"foreign answer allowance in {record.run_id}")
    if record.reasoning_budget not in config.evaluation_budgets:
        raise ValueError(f"unfrozen budget in {record.run_id}")
    if record.reasoning_tokens > record.reasoning_budget:
        raise ValueError(f"post-enforcement budget violation in {record.run_id}")
    if record.generation_batch_size < 1:
        raise ValueError(f"invalid generation batch size in {record.run_id}")
    if record.generation_batch_size > config.batch_size_for_budget(
        record.reasoning_budget
    ):
        raise ValueError(f"oversized generation batch in {record.run_id}")
    if not math.isfinite(record.latency_seconds) or record.latency_seconds < 0:
        raise ValueError(f"invalid latency in {record.run_id}")
    return record


def _summarize_ordered_batches(
    raw_rows: list[dict[str, Any]],
) -> tuple[int, float, int, float]:
    batch_count = 0
    wall_seconds = 0.0
    sampled_tokens = 0
    cursor = 0
    while cursor < len(raw_rows):
        size = int(raw_rows[cursor]["generation_batch_size"])
        batch = raw_rows[cursor : cursor + size]
        if len(batch) != size or any(
            int(row["generation_batch_size"]) != size for row in batch
        ):
            raise ValueError("evaluation rows do not form complete ordered generation batches")
        budgets = {int(row["reasoning_budget"]) for row in batch}
        reasoning_latencies = {
            float(row["latency_seconds"]["reasoning"]) for row in batch
        }
        answer_latencies = {float(row["latency_seconds"]["answer"]) for row in batch}
        if len(budgets) != 1 or len(reasoning_latencies) != 1 or len(answer_latencies) != 1:
            raise ValueError("rows within a generation batch disagree on budget or latency")
        wall_seconds += next(iter(reasoning_latencies)) + next(iter(answer_latencies))
        sampled_tokens += sum(int(row["token_counts"]["sampled_total"]) for row in batch)
        batch_count += 1
        cursor += size
    throughput = sampled_tokens / wall_seconds if wall_seconds else 0.0
    return batch_count, wall_seconds, sampled_tokens, throughput


def _verify_saved_evaluation_summary(
    summary: dict[str, Any],
    records: tuple[PrimaryRecord, ...],
    *,
    root: Path,
) -> None:
    expected = {
        "item_generation_count": len(records),
        "accuracy": statistics.fmean(record.correct for record in records),
        "region_parse_success_rate": statistics.fmean(
            record.region_valid for record in records
        ),
        "trace_parse_success_rate": statistics.fmean(
            record.trace_parse_success for record in records
        ),
        "complete_replay_rate": statistics.fmean(
            record.replay_valid for record in records
        ),
        "forced_reasoning_rate": statistics.fmean(
            record.forced_transition for record in records
        ),
        "budget_violation_count": sum(
            record.reasoning_tokens > record.reasoning_budget for record in records
        ),
        "median_reasoning_tokens": statistics.median(
            record.reasoning_tokens for record in records
        ),
        "mean_latency_seconds": statistics.fmean(
            record.latency_seconds for record in records
        ),
    }
    for key, expected_value in expected.items():
        observed = summary.get(key)
        if isinstance(expected_value, float):
            matches = isinstance(observed, (int, float)) and math.isclose(
                float(observed),
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        else:
            matches = observed == expected_value
        if not matches:
            raise ValueError(
                f"saved evaluation summary disagrees with item rows for {root}: "
                f"{key}={observed!r}, recomputed={expected_value!r}"
            )


def _load_evaluation_run(
    root: Path,
    run_spec: dict[str, Any],
    *,
    config: M3Config,
    frozen: dict[str, Any],
    freeze_manifest_sha256: str,
    training_root: Path,
) -> EvaluationRun:
    if root.name != run_spec["run_id"]:
        raise ValueError(f"evaluation path does not match frozen run ID: {root}")
    if (root / "failure.json").is_file():
        raise ValueError(f"evaluation run retains a failure artifact: {root}")
    _verify_tree_manifest(root)
    resolved = _load_json(root / "resolved_config.json")
    context = resolved.get("run_context", {})
    if context.get("phase") != "primary" or context.get("run_spec") != run_spec:
        raise ValueError(f"evaluation run does not match its frozen specification: {root}")
    if context.get("freeze", {}).get("manifest_sha256") != freeze_manifest_sha256:
        raise ValueError(f"evaluation run references a different freeze: {root}")
    if resolved.get("protocol_sha256") != frozen["protocol_sha256"]:
        raise ValueError(f"evaluation run references a different protocol: {root}")
    if resolved.get("qlora_sha256") != frozen["qlora_sha256"]:
        raise ValueError(f"evaluation run references a different QLoRA config: {root}")
    if resolved.get("evaluation_split") != run_spec["split"]:
        raise ValueError(f"evaluation split mismatch: {root}")
    if resolved.get("evaluation_sha256") != frozen["split_sha256"][run_spec["split"]]:
        raise ValueError(f"evaluation split hash mismatch: {root}")
    if resolved.get("base_seed") != run_spec["decoding_seed"]:
        raise ValueError(f"evaluation decoding seed mismatch: {root}")
    if resolved.get("checkpoint") != frozen["checkpoint"]:
        raise ValueError(f"evaluation checkpoint configuration mismatch: {root}")
    if not isinstance(resolved.get("decoding"), dict):
        raise ValueError(f"evaluation run lacks decoding configuration: {root}")
    if not isinstance(resolved.get("tokenizer"), dict):
        raise ValueError(f"evaluation run lacks tokenizer provenance: {root}")
    if tuple(resolved.get("budgets", ())) != config.evaluation_budgets:
        raise ValueError(f"evaluation budgets differ from the frozen protocol: {root}")
    expected_batch_sizes = {
        str(budget): config.batch_size_for_budget(budget)
        for budget in config.evaluation_budgets
    }
    if (
        resolved.get("batch_size") is not None
        or resolved.get("batch_sizes_by_budget") != expected_batch_sizes
    ):
        raise ValueError(
            f"evaluation batch-size schedule differs from the frozen protocol: {root}"
        )
    adapter_run_id = run_spec["adapter_run_id"]
    if adapter_run_id is None:
        if (
            resolved.get("adapter_path") is not None
            or resolved.get("adapter_manifest_sha256") is not None
        ):
            raise ValueError(f"frozen control unexpectedly references an adapter: {root}")
    else:
        expected_adapter = training_root / adapter_run_id / "adapter"
        expected_manifest = training_root / adapter_run_id / "manifest.json"
        if Path(resolved.get("adapter_path", "")).resolve() != expected_adapter.resolve():
            raise ValueError(f"evaluation adapter path mismatch: {root}")
        if (
            not expected_manifest.is_file()
            or resolved.get("adapter_manifest_sha256")
            != sha256_file(expected_manifest)
        ):
            raise ValueError(f"evaluation adapter manifest mismatch: {root}")
    item_ids = tuple(resolved.get("item_ids", ()))
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"resolved evaluation item IDs are not unique: {root}")
    raw_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with (root / "items.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["item_id"], int(row["reasoning_budget"]))
            if key in seen:
                raise ValueError(f"duplicate evaluation key at {root}:{line_number}: {key}")
            seen.add(key)
            raw_rows.append(row)
    expected = {
        (item_id, budget)
        for budget in config.evaluation_budgets
        for item_id in item_ids
    }
    if seen != expected:
        raise ValueError(f"evaluation rows are incomplete or foreign: {root}")
    records = tuple(
        _compact_record(row, run_spec=run_spec, config=config) for row in raw_rows
    )
    batch_count, wall_seconds, sampled_tokens, throughput = _summarize_ordered_batches(
        raw_rows
    )
    hardware = _load_json(root / "hardware_after.json")
    summary = _load_json(root / "summary.json")
    _verify_saved_evaluation_summary(summary, records, root=root)
    return EvaluationRun(
        run_id=run_spec["run_id"],
        replicate=run_spec["replicate"],
        condition_id=run_spec["condition_id"],
        split=run_spec["split"],
        path=str(root),
        records=records,
        generation_count=len(records),
        batch_count=batch_count,
        estimated_wall_seconds=wall_seconds,
        sampled_tokens=sampled_tokens,
        decode_tokens_per_second=throughput,
        peak_allocated_bytes=hardware.get("torch_cuda_peak_allocated_bytes"),
        peak_reserved_bytes=hardware.get("torch_cuda_peak_reserved_bytes"),
        decoding_sha256=_stable_sha256(resolved.get("decoding")),
        tokenizer_sha256=_stable_sha256(resolved.get("tokenizer")),
    )


def _load_training_run(
    root: Path,
    run_spec: dict[str, Any],
    *,
    config: M3Config,
    frozen: dict[str, Any],
    freeze_manifest_sha256: str,
) -> dict[str, Any]:
    if root.name != run_spec["run_id"]:
        raise ValueError(f"training path does not match frozen run ID: {root}")
    if (root / "failure.json").is_file():
        raise ValueError(f"training run retains a failure artifact: {root}")
    _verify_tree_manifest(root)
    resolved = _load_json(root / "resolved_config.json")
    context = _load_json(root / "m3_context.json")
    if context.get("phase") != "primary" or context.get("run_spec") != run_spec:
        raise ValueError(f"training run does not match its frozen specification: {root}")
    if context.get("freeze", {}).get("manifest_sha256") != freeze_manifest_sha256:
        raise ValueError(f"training run references a different freeze: {root}")
    if resolved.get("protocol_sha256") != frozen["protocol_sha256"]:
        raise ValueError(f"training run references a different protocol: {root}")
    if resolved.get("qlora_sha256") != frozen["qlora_sha256"]:
        raise ValueError(f"training run references a different QLoRA config: {root}")
    if resolved.get("run_id") != run_spec["run_id"]:
        raise ValueError(f"training run ID mismatch: {root}")
    if resolved.get("condition", {}).get("id") != run_spec["condition_id"]:
        raise ValueError(f"training condition mismatch: {root}")
    if resolved.get("qlora", {}).get("seed") != run_spec["training_seed"]:
        raise ValueError(f"training seed mismatch: {root}")
    expected_qlora = dict(frozen["qlora"])
    expected_qlora["seed"] = run_spec["training_seed"]
    if resolved.get("qlora") != expected_qlora:
        raise ValueError(f"training QLoRA configuration mismatch: {root}")
    if resolved.get("checkpoint") != frozen["checkpoint"]:
        raise ValueError(f"training checkpoint configuration mismatch: {root}")
    if resolved.get("protocol") != frozen["protocol"]:
        raise ValueError(f"training protocol payload mismatch: {root}")
    if resolved.get("source_split_sha256") != frozen["split_sha256"][config.train_split]:
        raise ValueError(f"training source split hash mismatch: {root}")
    telemetry = _load_json(root / "telemetry.json")
    reload_result = _load_json(root / "reload_verification.json")
    trainer_metrics = telemetry.get("trainer_metrics", {})
    exposure = _load_json(root / "token_exposure.json")["conditions"][
        run_spec["condition_id"]
    ]
    losses = [
        float(entry["loss"])
        for entry in telemetry.get("log_history", ())
        if isinstance(entry, dict) and isinstance(entry.get("loss"), (int, float))
    ]
    finite_loss = bool(losses) and all(math.isfinite(value) for value in losses)
    passed = (
        telemetry.get("global_step") == config.optimizer_updates
        and finite_loss
        and telemetry.get("trainable_parameters_changed") is True
        and reload_result.get("identical") is True
    )
    return {
        "run_id": run_spec["run_id"],
        "replicate": run_spec["replicate"],
        "condition_id": run_spec["condition_id"],
        "training_seed": run_spec["training_seed"],
        "item_count": resolved.get("item_count"),
        "item_ids_sha256": _stable_sha256(resolved.get("item_ids")),
        "path": str(root),
        "global_step": telemetry.get("global_step"),
        "finite_loss": finite_loss,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "trainable_parameters_changed": telemetry.get("trainable_parameters_changed"),
        "reload_identical": reload_result.get("identical"),
        "wall_time_seconds": telemetry.get("wall_time_seconds"),
        "examples_per_second": trainer_metrics.get("train_samples_per_second"),
        "tokens_per_second": trainer_metrics.get("train_tokens_per_second"),
        "peak_allocated_bytes": telemetry.get("torch_cuda_peak_allocated_bytes"),
        "peak_reserved_bytes": telemetry.get("torch_cuda_peak_reserved_bytes"),
        "adapter_size_bytes": telemetry.get("adapter_size_bytes"),
        "checkpoint_size_bytes": sum(
            path.stat().st_size
            for path in (root / "checkpoints").rglob("*")
            if path.is_file()
        ),
        "total_run_storage_bytes": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ),
        "token_exposure": exposure,
        "passed": passed,
    }


def _load_registry_attempts(
    registry_path: Path,
    run_plan: dict[str, Any],
) -> dict[str, Any]:
    planned = {
        run["run_id"]: run
        for key in ("training_runs", "evaluation_runs")
        for run in run_plan[key]
    }
    attempts: list[dict[str, Any]] = []
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with registry_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("phase") != "primary":
                continue
            run_id = record.get("run_id")
            if run_id not in planned:
                raise ValueError(
                    f"registry contains a foreign primary run at line {line_number}: {run_id}"
                )
            attempts.append(record)
            by_run[run_id].append(record)
    missing = sorted(set(planned) - set(by_run))
    if missing:
        raise ValueError(f"registry lacks planned primary runs: {missing[:3]}")
    terminal: dict[str, str] = {}
    for run_id, records in by_run.items():
        status = records[-1].get("status")
        terminal[run_id] = status
        if status != "complete":
            raise ValueError(f"primary run does not end complete in registry: {run_id}={status}")
        expected_hash = planned[run_id]["run_spec_sha256"]
        if records[-1].get("run_spec_sha256") != expected_hash:
            raise ValueError(f"registry run-spec hash mismatch: {run_id}")
    return {
        "schema_version": 1,
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "attempt_count": len(attempts),
        "unsuccessful_attempt_count": sum(
            record.get("status") in {"failed", "interrupted", "excluded"}
            for record in attempts
        ),
        "terminal_statuses": terminal,
        "attempts": attempts,
    }


def _load_primary_inputs(
    freeze_dir: str | Path,
    evaluation_root: str | Path,
    training_root: str | Path,
    registry_path: str | Path,
) -> tuple[
    M3Config,
    dict[str, Any],
    list[EvaluationRun],
    list[dict[str, Any]],
    dict[str, Any],
]:
    _freeze_root, frozen, plan, freeze_context = _read_freeze(freeze_dir)
    config: M3Config = freeze_context["config"]
    freeze_manifest_sha256 = freeze_context["authorization"]["manifest_sha256"]
    evaluation_base = Path(evaluation_root).resolve()
    training_base = Path(training_root).resolve()
    registry = Path(registry_path).resolve()
    evaluation_runs = [
        _load_evaluation_run(
            evaluation_base / run_spec["run_id"],
            run_spec,
            config=config,
            frozen=frozen,
            freeze_manifest_sha256=freeze_manifest_sha256,
            training_root=training_base,
        )
        for run_spec in plan["evaluation_runs"]
    ]
    training_runs = [
        _load_training_run(
            training_base / run_spec["run_id"],
            run_spec,
            config=config,
            frozen=frozen,
            freeze_manifest_sha256=freeze_manifest_sha256,
        )
        for run_spec in plan["training_runs"]
    ]
    registry_summary = _load_registry_attempts(registry, plan)
    expected_generation_count = plan["counts"]["generations"]
    for run in evaluation_runs:
        expected_run_count = (
            plan["counts"]["items_by_split"][run.split]
            * len(config.evaluation_budgets)
        )
        if run.generation_count != expected_run_count:
            raise ValueError(
                f"evaluation generation count mismatch for {run.run_id}: "
                f"{run.generation_count} != {expected_run_count}"
            )
    actual_generation_count = sum(run.generation_count for run in evaluation_runs)
    if actual_generation_count != expected_generation_count:
        raise ValueError(
            "primary generation count differs from the frozen plan: "
            f"{actual_generation_count} != {expected_generation_count}"
        )
    if not all(run["passed"] for run in training_runs):
        raise ValueError("one or more primary training runs failed the saved-artifact gate")
    if len({run["item_ids_sha256"] for run in training_runs}) != 1:
        raise ValueError("primary training controls do not contain identical item IDs")
    if len({run["item_count"] for run in training_runs}) != 1:
        raise ValueError("primary training controls do not contain identical item counts")
    if len({run.decoding_sha256 for run in evaluation_runs}) != 1:
        raise ValueError("primary evaluation runs use different decoding configurations")
    if len({run.tokenizer_sha256 for run in evaluation_runs}) != 1:
        raise ValueError("primary evaluation runs use different tokenizer provenance")
    for split in config.evaluation_splits:
        split_runs = [run for run in evaluation_runs if run.split == split]
        identities = [
            {
                (
                    record.item_id,
                    record.reasoning_budget,
                    record.task_family,
                    record.depth,
                )
                for record in run.records
            }
            for run in split_runs
        ]
        if not identities or any(identity != identities[0] for identity in identities[1:]):
            raise ValueError(
                f"primary conditions or replicate blocks are not item-paired on {split}"
            )
    return config, plan, evaluation_runs, training_runs, registry_summary


def _aggregate_records(records: list[PrimaryRecord]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty record group")
    reasoning_tokens = [record.reasoning_tokens for record in records]
    latencies = [record.latency_seconds for record in records]
    total_reasoning = sum(reasoning_tokens)
    total_parsed = sum(record.parsed_transition_count for record in records)
    return {
        "generation_count": len(records),
        "problem_count": len({record.item_id for record in records}),
        "accuracy": statistics.fmean(record.correct for record in records),
        "region_parse_success_rate": statistics.fmean(
            record.region_valid for record in records
        ),
        "trace_parse_success_rate": statistics.fmean(
            record.trace_parse_success for record in records
        ),
        "complete_replay_rate": statistics.fmean(
            record.replay_valid for record in records
        ),
        "verified_transitions": sum(
            record.verified_prefix_transitions for record in records
        ),
        "verified_transitions_per_reasoning_token": (
            sum(record.verified_prefix_transitions for record in records)
            / total_reasoning
            if total_reasoning
            else 0.0
        ),
        "mean_gold_path_coverage": statistics.fmean(
            record.gold_path_coverage for record in records
        ),
        "mean_verified_progress_before_first_invalid": statistics.fmean(
            record.verified_prefix_transitions for record in records
        ),
        "invalid_transition_count": sum(
            record.invalid_transition_count for record in records
        ),
        "invalid_transition_event_rate": statistics.fmean(
            record.invalid_transition_event for record in records
        ),
        "invalid_transition_fraction_of_parsed": (
            sum(record.invalid_transition_count for record in records) / total_parsed
            if total_parsed
            else 0.0
        ),
        "forced_transition_rate": statistics.fmean(
            record.forced_transition for record in records
        ),
        "empty_reasoning_rate": statistics.fmean(
            record.reasoning_empty for record in records
        ),
        "prompt_copy_rate": statistics.fmean(record.prompt_copied for record in records),
        "repeated_line_rate": statistics.fmean(
            record.repeated_line for record in records
        ),
        "mean_reasoning_tokens": statistics.fmean(reasoning_tokens),
        "median_reasoning_tokens": statistics.median(reasoning_tokens),
        "p95_reasoning_tokens": percentile(reasoning_tokens, 0.95),
        "mean_latency_seconds": statistics.fmean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
    }


def _analysis_groups(
    evaluation_runs: list[EvaluationRun],
) -> dict[tuple[int | None, str, str, int, str | None, int | None], list[PrimaryRecord]]:
    groups: dict[
        tuple[int | None, str, str, int, str | None, int | None],
        list[PrimaryRecord],
    ] = defaultdict(list)
    for run in evaluation_runs:
        for record in run.records:
            strata = (
                (None, None),
                (record.task_family, None),
                (None, record.depth),
                (record.task_family, record.depth),
            )
            for family, depth in strata:
                groups[
                    (
                        record.replicate,
                        record.condition_id,
                        record.split,
                        record.reasoning_budget,
                        family,
                        depth,
                    )
                ].append(record)
                groups[
                    (
                        None,
                        record.condition_id,
                        record.split,
                        record.reasoning_budget,
                        family,
                        depth,
                    )
                ].append(record)
    return groups


def _aggregate_table(evaluation_runs: list[EvaluationRun]) -> list[dict[str, Any]]:
    groups = _analysis_groups(evaluation_runs)

    def ordering(
        item: tuple[
            tuple[int | None, str, str, int, str | None, int | None],
            list[PrimaryRecord],
        ],
    ) -> tuple[str, ...]:
        return tuple("" if value is None else str(value) for value in item[0])

    return [
        {
            "replicate": replicate,
            "condition_id": condition,
            "split": split,
            "reasoning_budget": budget,
            "task_family": family,
            "depth": depth,
            **_aggregate_records(records),
        }
        for (
            (replicate, condition, split, budget, family, depth),
            records,
        ) in sorted(groups.items(), key=ordering)
    ]


def _records_by_cell(
    evaluation_runs: list[EvaluationRun],
) -> dict[tuple[str, str, int], list[PrimaryRecord]]:
    cells: dict[tuple[str, str, int], list[PrimaryRecord]] = defaultdict(list)
    for run in evaluation_runs:
        for record in run.records:
            cells[
                (record.condition_id, record.split, record.reasoning_budget)
            ].append(record)
    return cells


def _cluster_metric(
    records: Iterable[PrimaryRecord],
    metric: str,
) -> dict[str, list[float]]:
    clustered: dict[str, list[float]] = defaultdict(list)
    for record in records:
        clustered[record.item_id].append(float(getattr(record, metric)))
    replicate_counts = {len(values) for values in clustered.values()}
    if len(replicate_counts) != 1:
        raise ValueError("item clusters contain unequal replicate counts")
    return dict(clustered)


def _metric_index(
    records: Iterable[PrimaryRecord],
    metric: str,
    *,
    task_family: str | None = None,
    depth: int | None = None,
) -> dict[tuple[int, str], float]:
    result = {}
    for record in records:
        if task_family is not None and record.task_family != task_family:
            continue
        if depth is not None and record.depth != depth:
            continue
        key = (record.replicate, record.item_id)
        if key in result:
            raise ValueError(f"duplicate metric key: {key}")
        result[key] = float(getattr(record, metric))
    return result


def _accuracy_intervals(
    cells: dict[tuple[str, str, int], list[PrimaryRecord]],
    config: M3Config,
) -> list[dict[str, Any]]:
    intervals = []
    for (condition, split, budget), records in sorted(cells.items()):
        seed = derived_analysis_seed(
            config.bootstrap_seed, "accuracy", condition, split, budget
        )
        intervals.append(
            {
                "condition_id": condition,
                "split": split,
                "reasoning_budget": budget,
                **clustered_bootstrap_mean(
                    _cluster_metric(records, "correct"),
                    seed=seed,
                    samples=config.bootstrap_samples,
                ),
            }
        )
    return intervals


def _per_problem_aggregates(
    cells: dict[tuple[str, str, int], list[PrimaryRecord]],
) -> dict[str, Any]:
    item_rows = []
    summary_rows = []
    for (condition, split, budget), records in sorted(cells.items()):
        grouped: dict[str, list[PrimaryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.item_id].append(record)
        cell_items = []
        for item_id, item_records in sorted(grouped.items()):
            item_records.sort(key=lambda record: record.replicate)
            outcomes = [record.correct for record in item_records]
            row = {
                "condition_id": condition,
                "split": split,
                "reasoning_budget": budget,
                "item_id": item_id,
                "task_family": item_records[0].task_family,
                "depth": item_records[0].depth,
                "replicate_outcomes": outcomes,
                "replicate_count": len(outcomes),
                "per_problem_accuracy": statistics.fmean(outcomes),
                "any_generation_correct": any(outcomes),
                "all_generations_correct": all(outcomes),
                "majority_generations_correct": sum(outcomes) > len(outcomes) / 2,
            }
            item_rows.append(row)
            cell_items.append(row)
        summary_rows.append(
            {
                "condition_id": condition,
                "split": split,
                "reasoning_budget": budget,
                "problem_count": len(cell_items),
                "mean_per_problem_accuracy": statistics.fmean(
                    row["per_problem_accuracy"] for row in cell_items
                ),
                "any_generation_correct_rate": statistics.fmean(
                    row["any_generation_correct"] for row in cell_items
                ),
                "all_generations_correct_rate": statistics.fmean(
                    row["all_generations_correct"] for row in cell_items
                ),
                "majority_generations_correct_rate": statistics.fmean(
                    row["majority_generations_correct"] for row in cell_items
                ),
            }
        )
    return {
        "schema_version": 1,
        "aggregation": (
            "all three preregistered paired replicate generations; no best-sample selection"
        ),
        "summary_rows": summary_rows,
        "item_rows": item_rows,
    }


def _paired_effect(
    cells: dict[tuple[str, str, int], list[PrimaryRecord]],
    *,
    comparator: str,
    split: str,
    budget: int,
    metric: str,
    config: M3Config,
    task_family: str | None = None,
    depth: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    treatment = _metric_index(
        cells[("c4", split, budget)],
        metric,
        task_family=task_family,
        depth=depth,
    )
    control = _metric_index(
        cells[(comparator, split, budget)],
        metric,
        task_family=task_family,
        depth=depth,
    )
    differences = paired_cluster_values(treatment, control)
    seed = derived_analysis_seed(
        config.bootstrap_seed,
        "paired",
        metric,
        comparator,
        split,
        budget,
        task_family,
        depth,
    )
    interval = clustered_bootstrap_mean(
        differences,
        seed=seed,
        samples=config.bootstrap_samples,
    )
    return interval, differences


def _paired_comparisons(
    cells: dict[tuple[str, str, int], list[PrimaryRecord]],
    config: M3Config,
) -> list[dict[str, Any]]:
    comparisons = []
    primary_p_values: dict[str, float] = {}
    for split in config.evaluation_splits:
        for budget in config.evaluation_budgets:
            for comparator in (*config.primary_comparators, *config.secondary_comparators):
                accuracy, differences = _paired_effect(
                    cells,
                    comparator=comparator,
                    split=split,
                    budget=budget,
                    metric="correct",
                    config=config,
                )
                c4_metrics = _aggregate_records(cells[("c4", split, budget)])
                control_metrics = _aggregate_records(cells[(comparator, split, budget)])
                invalid, _ = _paired_effect(
                    cells,
                    comparator=comparator,
                    split=split,
                    budget=budget,
                    metric="invalid_transition_event",
                    config=config,
                )
                hypothesis_id = None
                randomization = None
                if (
                    split == config.primary_inference_split
                    and budget in config.primary_budgets
                    and comparator in config.primary_comparators
                ):
                    hypothesis_id = f"c4_gt_{comparator}_{split}_b{budget}"
                    randomization = paired_randomization_pvalue(
                        differences,
                        seed=derived_analysis_seed(
                            config.bootstrap_seed,
                            "randomization",
                            hypothesis_id,
                        ),
                        samples=config.bootstrap_samples,
                        alternative="greater",
                    )
                    primary_p_values[hypothesis_id] = randomization["p_value"]
                comparisons.append(
                    {
                        "hypothesis_id": hypothesis_id,
                        "scope": "primary" if hypothesis_id else "descriptive_secondary",
                        "treatment": "c4",
                        "comparator": comparator,
                        "split": split,
                        "reasoning_budget": budget,
                        "task_family": None,
                        "depth": None,
                        "c4_accuracy": c4_metrics["accuracy"],
                        "comparator_accuracy": control_metrics["accuracy"],
                        "paired_accuracy_difference": accuracy,
                        "paired_randomization": randomization,
                        "c4_verified_density": c4_metrics[
                            "verified_transitions_per_reasoning_token"
                        ],
                        "comparator_verified_density": control_metrics[
                            "verified_transitions_per_reasoning_token"
                        ],
                        "verified_density_ratio": (
                            c4_metrics["verified_transitions_per_reasoning_token"]
                            / control_metrics["verified_transitions_per_reasoning_token"]
                            if control_metrics[
                                "verified_transitions_per_reasoning_token"
                            ]
                            else None
                        ),
                        "paired_invalid_event_rate_difference": invalid,
                        "holm": None,
                    }
                )
    holm = holm_adjust(primary_p_values, alpha=config.familywise_alpha)
    for comparison in comparisons:
        hypothesis_id = comparison["hypothesis_id"]
        if hypothesis_id is not None:
            comparison["holm"] = holm[hypothesis_id]
    return comparisons


def _paired_primary_breakdowns(
    cells: dict[tuple[str, str, int], list[PrimaryRecord]],
    config: M3Config,
) -> list[dict[str, Any]]:
    reference_records = cells[
        ("c4", config.primary_inference_split, config.primary_budgets[0])
    ]
    families = sorted({record.task_family for record in reference_records})
    depths = sorted({record.depth for record in reference_records})
    rows = []
    for budget in config.primary_budgets:
        for comparator in config.primary_comparators:
            for stratum, values in (("task_family", families), ("depth", depths)):
                for value in values:
                    family = value if stratum == "task_family" else None
                    depth = value if stratum == "depth" else None
                    effect, _ = _paired_effect(
                        cells,
                        comparator=comparator,
                        split=config.primary_inference_split,
                        budget=budget,
                        metric="correct",
                        config=config,
                        task_family=family,
                        depth=depth,
                    )
                    rows.append(
                        {
                            "treatment": "c4",
                            "comparator": comparator,
                            "split": config.primary_inference_split,
                            "reasoning_budget": budget,
                            "task_family": family,
                            "depth": depth,
                            "paired_accuracy_difference": effect,
                        }
                    )
    return rows


def _seed_variability(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in aggregate_rows:
        if (
            row["replicate"] is not None
            and row["task_family"] is None
            and row["depth"] is None
        ):
            cells[
                (row["condition_id"], row["split"], row["reasoning_budget"])
            ].append(row["accuracy"])
    output = []
    for (condition, split, budget), values in sorted(cells.items()):
        output.append(
            {
                "condition_id": condition,
                "split": split,
                "reasoning_budget": budget,
                "replicate_accuracies": values,
                "mean_accuracy": statistics.fmean(values),
                "sample_standard_deviation": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "minimum_accuracy": min(values),
                "maximum_accuracy": max(values),
            }
        )
    return output


def _overall_aggregate_index(
    aggregate_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (row["condition_id"], row["split"], row["reasoning_budget"]): row
        for row in aggregate_rows
        if row["replicate"] is None
        and row["task_family"] is None
        and row["depth"] is None
    }


def _matched_accuracy_reductions(
    aggregate_rows: list[dict[str, Any]],
    config: M3Config,
) -> list[dict[str, Any]]:
    indexed = _overall_aggregate_index(aggregate_rows)
    rows = []
    for split in config.evaluation_splits:
        c4_frontier = [
            indexed[("c4", split, budget)] for budget in config.evaluation_budgets
        ]
        for comparator in (*config.primary_comparators, *config.secondary_comparators):
            for comparator_budget in config.evaluation_budgets:
                baseline = indexed[(comparator, split, comparator_budget)]
                candidates = [
                    point
                    for point in c4_frontier
                    if point["accuracy"] >= baseline["accuracy"]
                ]
                selected = (
                    min(candidates, key=lambda point: point["reasoning_budget"])
                    if candidates
                    else None
                )
                token_reduction = None
                if selected is not None and baseline["median_reasoning_tokens"] > 0:
                    token_reduction = 1.0 - (
                        selected["median_reasoning_tokens"]
                        / baseline["median_reasoning_tokens"]
                    )
                rows.append(
                    {
                        "split": split,
                        "comparator": comparator,
                        "comparator_budget": comparator_budget,
                        "comparator_accuracy": baseline["accuracy"],
                        "comparator_median_reasoning_tokens": baseline[
                            "median_reasoning_tokens"
                        ],
                        "matched": selected is not None,
                        "minimum_c4_budget_at_or_above_accuracy": (
                            selected["reasoning_budget"] if selected is not None else None
                        ),
                        "c4_accuracy": selected["accuracy"] if selected is not None else None,
                        "c4_median_reasoning_tokens": (
                            selected["median_reasoning_tokens"]
                            if selected is not None
                            else None
                        ),
                        "token_reduction": token_reduction,
                        "minimum_reduction_passed": (
                            token_reduction is not None
                            and token_reduction >= config.minimum_token_reduction
                        ),
                        "method": (
                            "lowest frozen C4 budget with pooled per-generation accuracy "
                            "at least the comparator frontier point; token reduction uses "
                            "pooled median generated reasoning tokens"
                        ),
                    }
                )
    return rows


def _reliable_depths(
    aggregate_rows: list[dict[str, Any]],
    config: M3Config,
) -> list[dict[str, Any]]:
    depth_rows = [
        row
        for row in aggregate_rows
        if row["replicate"] is None
        and row["task_family"] is None
        and row["depth"] is not None
        and row["split"] in {"test", "test_depth"}
    ]
    grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in depth_rows:
        grouped[(row["condition_id"], row["reasoning_budget"])][row["depth"]] = row
    results = []
    for (condition, budget), by_depth in sorted(grouped.items()):
        ordered = sorted(by_depth)
        reliable = [
            depth
            for depth in ordered
            if by_depth[depth]["accuracy"] >= config.reliable_accuracy_threshold
        ]
        results.append(
            {
                "condition_id": condition,
                "reasoning_budget": budget,
                "accuracy_threshold": config.reliable_accuracy_threshold,
                "depth_accuracy": {
                    str(depth): by_depth[depth]["accuracy"] for depth in ordered
                },
                "maximum_reliably_solved_depth": max(reliable) if reliable else None,
                "maximum_reliable_depth_level_index": (
                    max(ordered.index(depth) for depth in reliable) if reliable else -1
                ),
            }
        )
    c0 = {
        row["reasoning_budget"]: row["maximum_reliable_depth_level_index"]
        for row in results
        if row["condition_id"] == "c0"
    }
    for row in results:
        baseline_index = c0[row["reasoning_budget"]]
        row["depth_levels_beyond_c0"] = (
            row["maximum_reliable_depth_level_index"] - baseline_index
        )
    return results


def _primary_success_summary(
    comparisons: list[dict[str, Any]],
    reliable_depths: list[dict[str, Any]],
    matched_reductions: list[dict[str, Any]],
    config: M3Config,
) -> dict[str, Any]:
    indexed = {
        (row["comparator"], row["split"], row["reasoning_budget"]): row
        for row in comparisons
    }
    depth_index = {
        (row["condition_id"], row["reasoning_budget"]): row
        for row in reliable_depths
    }
    budget_results = []
    for budget in config.primary_budgets:
        versus = {
            comparator: indexed[
                (comparator, config.primary_inference_split, budget)
            ]
            for comparator in (*config.primary_comparators, *config.secondary_comparators)
        }
        c0 = versus["c0"]
        result = {
            "reasoning_budget": budget,
            "accuracy_improvement_over_c0": c0[
                "paired_accuracy_difference"
            ]["estimate"],
            "accuracy_threshold_passed": (
                c0["paired_accuracy_difference"]["estimate"]
                >= config.primary_accuracy_improvement
            ),
            "holm_supported_gain_over_c0": c0["holm"]["reject_null"],
            "higher_than_prompt_and_full_cot": all(
                versus[comparator]["paired_accuracy_difference"]["estimate"] > 0
                for comparator in ("c1", "c2")
            ),
            "density_ratio_over_c0": c0["verified_density_ratio"],
            "density_threshold_passed": (
                c0["verified_density_ratio"] is not None
                and c0["verified_density_ratio"] >= config.minimum_density_ratio
            ),
            "depth_levels_beyond_c0": depth_index[("c4", budget)][
                "depth_levels_beyond_c0"
            ],
            "depth_threshold_passed": (
                depth_index[("c4", budget)]["depth_levels_beyond_c0"] >= 2
            ),
        }
        result["primary_success_at_budget"] = all(
            (
                result["accuracy_threshold_passed"],
                result["holm_supported_gain_over_c0"],
                result["higher_than_prompt_and_full_cot"],
                result["density_threshold_passed"],
                result["depth_threshold_passed"],
            )
        )
        budget_results.append(result)

    noninferiority_cells = []
    for budget in config.primary_budgets:
        controls = [
            indexed[(control, config.primary_inference_split, budget)]
            for control in ("c2", "c3")
        ]
        strongest = max(controls, key=lambda row: row["comparator_accuracy"])
        lower = strongest["paired_accuracy_difference"]["confidence_interval"][0]
        invalid_upper = strongest["paired_invalid_event_rate_difference"][
            "confidence_interval"
        ][1]
        noninferiority_cells.append(
            {
                "reasoning_budget": budget,
                "strongest_trained_control": strongest["comparator"],
                "accuracy_difference": strongest["paired_accuracy_difference"],
                "accuracy_noninferiority_passed": lower >= config.noninferiority_margin,
                "invalid_event_rate_difference": strongest[
                    "paired_invalid_event_rate_difference"
                ],
                "validity_non_degradation_passed": (
                    invalid_upper <= config.invalid_transition_margin
                ),
                "density_higher": (
                    strongest["verified_density_ratio"] is not None
                    and strongest["verified_density_ratio"] > 1.0
                ),
            }
        )
    primary_matched = [
        row
        for row in matched_reductions
        if row["split"] == config.primary_inference_split
        and row["comparator"] in {"c2", "c3"}
        and row["minimum_reduction_passed"]
    ]
    minimum_viable = bool(primary_matched) and all(
        row["accuracy_noninferiority_passed"]
        and row["validity_non_degradation_passed"]
        and row["density_higher"]
        for row in noninferiority_cells
    )
    return {
        "schema_version": 1,
        "primary_budget_results": budget_results,
        "primary_success": any(
            row["primary_success_at_budget"] for row in budget_results
        ),
        "minimum_viable_noninferiority_cells": noninferiority_cells,
        "matched_accuracy_reduction_cells_passing": len(primary_matched),
        "minimum_viable_result": minimum_viable,
    }


def _select_m3_decision(
    comparisons: list[dict[str, Any]],
    matched_reductions: list[dict[str, Any]],
    reliable_depths: list[dict[str, Any]],
    config: M3Config,
) -> dict[str, Any]:
    indexed = {
        (row["comparator"], row["split"], row["reasoning_budget"]): row
        for row in comparisons
    }
    primary_c0 = [
        indexed[("c0", config.primary_inference_split, budget)]
        for budget in config.primary_budgets
    ]
    shorter = any(
        row["split"] == config.primary_inference_split
        and row["comparator"] == "c0"
        and row["comparator_budget"] in config.primary_budgets
        and row["token_reduction"] is not None
        and row["token_reduction"] > 0
        for row in matched_reductions
    )
    validity_regression = any(
        row["paired_invalid_event_rate_difference"]["confidence_interval"][1]
        > config.invalid_transition_margin
        for row in primary_c0
    )
    fixed_budget_gain = any(
        row["paired_accuracy_difference"]["estimate"] > 0 for row in primary_c0
    )
    clear_token_reduction = any(
        row["split"] == config.primary_inference_split
        and row["comparator"] in {"c0", "c1", "c2", "c3"}
        and row["minimum_reduction_passed"]
        for row in matched_reductions
    )
    depth_index = {
        (row["condition_id"], row["reasoning_budget"]): row
        for row in reliable_depths
    }
    high_depth_drop = all(
        depth_index[("c4", budget)]["maximum_reliable_depth_level_index"]
        < depth_index[("c0", budget)]["maximum_reliable_depth_level_index"]
        for budget in config.primary_budgets
    )
    shallow_retained = any(
        indexed[("c0", config.primary_inference_split, budget)][
            "paired_accuracy_difference"
        ]["estimate"]
        >= config.noninferiority_margin
        for budget in config.primary_budgets
    )
    compositional_retained = any(
        indexed[("c0", "test_compositional", budget)][
            "paired_accuracy_difference"
        ]["estimate"]
        > 0
        for budget in config.primary_budgets
    )
    nondegraded_validity = not validity_regression

    if shorter and validity_regression:
        outcome = "revise_representation"
        reason = "C4 is shorter, but its invalid-transition event rate exceeds the frozen margin."
    elif shallow_retained and high_depth_drop:
        outcome = "investigate_pause_or_latent_computation"
        reason = (
            "C4 retains shallow performance but loses reliable depth, consistent with "
            "insufficient sequential computation."
        )
    elif (fixed_budget_gain or clear_token_reduction) and not compositional_retained:
        outcome = "stop_or_reframe"
        reason = (
            "The in-distribution signal does not remain positive on the uncontaminated "
            "compositional split."
        )
    elif (fixed_budget_gain or clear_token_reduction) and nondegraded_validity:
        outcome = "continue_to_generalization"
        reason = (
            "C4 improves a fixed-budget point or clears the matched-accuracy token-reduction "
            "threshold without a material validity regression."
        )
    else:
        outcome = "stop_or_reframe"
        reason = (
            "C4 does not show a clear fixed-budget or matched-accuracy advantage under the "
            "frozen decision rules."
        )
    return {
        "schema_version": 1,
        "outcome": outcome,
        "reason": reason,
        "exactly_one_outcome_selected": outcome
        in {
            "continue_to_generalization",
            "revise_representation",
            "investigate_pause_or_latent_computation",
            "stop_or_reframe",
        },
        "signals": {
            "shorter_than_c0_at_matched_accuracy": shorter,
            "validity_regression": validity_regression,
            "fixed_budget_gain": fixed_budget_gain,
            "clear_token_reduction": clear_token_reduction,
            "shallow_retained": shallow_retained,
            "high_depth_drop": high_depth_drop,
            "compositional_retained": compositional_retained,
        },
        "precedence": [
            "revise_representation when shorter traces materially regress validity",
            "investigate_pause_or_latent_computation for retained shallow but lost "
            "high-depth performance",
            "stop_or_reframe when an in-distribution signal disappears compositionally",
            "continue_to_generalization for a gain or clear reduction without validity degradation",
            "stop_or_reframe otherwise",
        ],
    }


def _system_metrics(
    evaluation_runs: list[EvaluationRun],
    training_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluation_rows = [
        {
            "run_id": run.run_id,
            "replicate": run.replicate,
            "condition_id": run.condition_id,
            "split": run.split,
            "path": run.path,
            "generation_count": run.generation_count,
            "batch_count": run.batch_count,
            "estimated_wall_seconds": run.estimated_wall_seconds,
            "sampled_tokens": run.sampled_tokens,
            "decode_tokens_per_second": run.decode_tokens_per_second,
            "peak_allocated_bytes": run.peak_allocated_bytes,
            "peak_reserved_bytes": run.peak_reserved_bytes,
            "decoding_sha256": run.decoding_sha256,
            "tokenizer_sha256": run.tokenizer_sha256,
        }
        for run in evaluation_runs
    ]
    grouped: dict[tuple[str, str], list[EvaluationRun]] = defaultdict(list)
    for run in evaluation_runs:
        grouped[(run.condition_id, run.split)].append(run)
    evaluation_summary = []
    for (condition, split), runs in sorted(grouped.items()):
        wall = sum(run.estimated_wall_seconds for run in runs)
        sampled = sum(run.sampled_tokens for run in runs)
        evaluation_summary.append(
            {
                "condition_id": condition,
                "split": split,
                "replicate_count": len(runs),
                "generation_count": sum(run.generation_count for run in runs),
                "batch_count": sum(run.batch_count for run in runs),
                "estimated_wall_seconds": wall,
                "sampled_tokens": sampled,
                "aggregate_decode_tokens_per_second": sampled / wall if wall else 0.0,
                "maximum_peak_allocated_bytes": max(
                    (
                        run.peak_allocated_bytes
                        for run in runs
                        if run.peak_allocated_bytes is not None
                    ),
                    default=None,
                ),
                "maximum_peak_reserved_bytes": max(
                    (
                        run.peak_reserved_bytes
                        for run in runs
                        if run.peak_reserved_bytes is not None
                    ),
                    default=None,
                ),
            }
        )
    return {
        "schema_version": 1,
        "latency_interpretation": (
            "Batched backend latency is common to every row in a batch. Estimated run wall time "
            "sums each ordered batch latency once; aggregate decode throughput divides all sampled "
            "tokens by that estimate. Per-generation latency summaries retain the observed batch "
            "completion latency experienced by each row."
        ),
        "evaluation_runs": evaluation_rows,
        "evaluation_condition_split_summary": evaluation_summary,
        "training_runs": training_runs,
        "total_evaluation_estimated_wall_seconds": sum(
            run.estimated_wall_seconds for run in evaluation_runs
        ),
        "total_training_wall_seconds": sum(
            float(run["wall_time_seconds"] or 0.0) for run in training_runs
        ),
        "total_adapter_size_bytes": sum(
            int(run["adapter_size_bytes"] or 0) for run in training_runs
        ),
        "total_checkpoint_size_bytes": sum(
            int(run["checkpoint_size_bytes"] or 0) for run in training_runs
        ),
        "total_training_run_storage_bytes": sum(
            int(run["total_run_storage_bytes"] or 0) for run in training_runs
        ),
    }


def _select_failure_keys(
    evaluation_runs: list[EvaluationRun],
    config: M3Config,
) -> dict[tuple[int, str, str, int], set[str]]:
    index: dict[
        tuple[int, str, str, int],
        dict[str, PrimaryRecord],
    ] = defaultdict(dict)
    for run in evaluation_runs:
        if run.split != config.primary_inference_split:
            continue
        for record in run.records:
            if record.reasoning_budget not in config.primary_budgets:
                continue
            index[
                (
                    record.replicate,
                    record.split,
                    record.item_id,
                    record.reasoning_budget,
                )
            ][record.condition_id] = record
    selected: dict[tuple[int, str, str, int], set[str]] = {}
    seen_strata: set[tuple[Any, ...]] = set()
    for key in sorted(index):
        rows = index[key]
        if set(rows) != set(_CONDITIONS):
            raise ValueError(f"failure-analysis pairing is incomplete: {key}")
        c4 = rows["c4"]
        reasons = []
        if not c4.region_valid:
            reasons.append("c4_region_parse_failure")
        if not c4.trace_parse_success:
            reasons.append("c4_trace_parse_failure")
        if c4.invalid_transition_event:
            reasons.append("c4_invalid_transition")
        any_control_correct = any(
            rows[condition].correct for condition in ("c0", "c1", "c2", "c3")
        )
        if not c4.correct and any_control_correct:
            reasons.append("c4_accuracy_regression")
        if c4.correct and not any_control_correct:
            reasons.append("c4_accuracy_gain")
        for reason in reasons:
            stratum = (
                reason,
                c4.task_family,
                c4.depth,
                c4.reasoning_budget,
            )
            if stratum in seen_strata:
                continue
            seen_strata.add(stratum)
            selected.setdefault(key, set()).add(reason)
        if len(selected) >= 80:
            break
    return selected


def _failure_examples(
    evaluation_runs: list[EvaluationRun],
    config: M3Config,
) -> dict[str, Any]:
    selected = _select_failure_keys(evaluation_runs, config)
    examples: dict[tuple[int, str, str, int], dict[str, Any]] = {
        key: {
            "replicate": key[0],
            "split": key[1],
            "item_id": key[2],
            "reasoning_budget": key[3],
            "reasons": sorted(reasons),
            "conditions": {},
        }
        for key, reasons in selected.items()
    }
    for run in evaluation_runs:
        candidate_keys = {
            key
            for key in selected
            if key[0] == run.replicate and key[1] == run.split
        }
        if not candidate_keys:
            continue
        with (Path(run.path) / "items.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    run.replicate,
                    run.split,
                    row["item_id"],
                    int(row["reasoning_budget"]),
                )
                if key not in candidate_keys:
                    continue
                examples[key]["task_family"] = row["task_family"]
                examples[key]["depth"] = row["depth"]
                examples[key]["reference_answer"] = row["reference_answer"]
                examples[key]["conditions"][run.condition_id] = {
                    "correct": row["correct"],
                    "reasoning_text": row["reasoning_text"],
                    "parsed_answer": row["parsed_answer"],
                    "parse_status": row["parse_status"],
                    "trace_diagnostics": row["trace_diagnostics"],
                    "token_counts": row["token_counts"],
                    "forced_reasoning_transition": row[
                        "forced_reasoning_transition"
                    ],
                    "reasoning_termination": row["reasoning_termination"],
                    "answer_termination": row["answer_termination"],
                }
    output = []
    for key in sorted(examples):
        example = examples[key]
        if set(example["conditions"]) != set(_CONDITIONS):
            raise ValueError(f"failure example is missing a paired condition: {key}")
        output.append(example)
    return {
        "schema_version": 1,
        "selection": (
            "first deterministic paired example per reason, task family, depth, and primary "
            "budget; capped at 80 item-budget-replicate keys"
        ),
        "item_count": len(output),
        "items": output,
    }


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    output_root: Path,
    aggregate_rows: list[dict[str, Any]],
    config: M3Config,
) -> list[str]:
    indexed = _overall_aggregate_index(aggregate_rows)
    plots = []
    for split in config.evaluation_splits:
        for metric, label, filename, y_max in (
            ("accuracy", "Exact-answer accuracy", f"accuracy_frontier_{split}.svg", 1.0),
            (
                "verified_transitions_per_reasoning_token",
                "Verified transitions / reasoning token",
                f"verified_density_frontier_{split}.svg",
                None,
            ),
            (
                "mean_latency_seconds",
                "Observed batch-completion latency (seconds)",
                f"latency_frontier_{split}.svg",
                None,
            ),
            (
                "median_reasoning_tokens",
                "Median generated reasoning tokens",
                f"reasoning_token_frontier_{split}.svg",
                None,
            ),
            (
                "forced_transition_rate",
                "Forced-answer transition rate",
                f"forced_transition_frontier_{split}.svg",
                1.0,
            ),
        ):
            series = {
                condition: [
                    (budget, indexed[(condition, split, budget)][metric])
                    for budget in config.evaluation_budgets
                ]
                for condition in _CONDITIONS
            }
            path = output_root / filename
            _write_svg_plot(
                path,
                f"M3 {split}: {label}",
                label,
                series,
                y_max=y_max,
            )
            plots.append(filename)
        accuracy_latency_series = {
            condition: [
                (
                    indexed[(condition, split, budget)]["mean_latency_seconds"],
                    indexed[(condition, split, budget)]["accuracy"],
                )
                for budget in config.evaluation_budgets
            ]
            for condition in _CONDITIONS
        }
        accuracy_latency_filename = f"accuracy_vs_latency_{split}.svg"
        _write_svg_plot(
            output_root / accuracy_latency_filename,
            f"M3 {split}: accuracy versus observed latency",
            "Exact-answer accuracy",
            accuracy_latency_series,
            y_max=1.0,
            x_label="Mean observed batch-completion latency (seconds)",
        )
        plots.append(accuracy_latency_filename)
    depth_index = {
        (row["condition_id"], row["reasoning_budget"], row["depth"]): row
        for row in aggregate_rows
        if row["replicate"] is None
        and row["task_family"] is None
        and row["depth"] is not None
        and row["split"] in {"test", "test_depth"}
    }
    depths = sorted({key[2] for key in depth_index})
    for budget in config.primary_budgets:
        series = {
            condition: [
                (depth, depth_index[(condition, budget, depth)]["accuracy"])
                for depth in depths
            ]
            for condition in _CONDITIONS
        }
        filename = f"accuracy_by_depth_b{budget}.svg"
        _write_svg_plot(
            output_root / filename,
            f"M3 accuracy by controlled depth at budget {budget}",
            "Exact-answer accuracy",
            series,
            y_max=1.0,
            x_label="Minimum required reasoning depth",
        )
        plots.append(filename)
    return plots


def _write_markdown_report(
    path: Path,
    *,
    success: dict[str, Any],
    decision: dict[str, Any],
    comparisons: list[dict[str, Any]],
    systems: dict[str, Any],
    config: M3Config,
) -> None:
    primary = [
        row
        for row in comparisons
        if row["scope"] == "primary"
    ]
    generation_count = sum(
        row["generation_count"] for row in systems["evaluation_runs"]
    )
    training_run_count = len(systems.get("training_runs", ()))
    evaluation_run_count = len(systems["evaluation_runs"])
    lines = [
        "# Milestone 3 primary experiment report",
        "",
        f"**Decision:** `{decision['outcome']}`",
        "",
        "## Engineering validity",
        "",
        f"- Complete training/reload runs: **{training_run_count}**.",
        f"- Complete evaluation runs: **{evaluation_run_count}**.",
        f"- Final generation count: **{generation_count:,}**.",
        "- Every input run passed freeze, manifest, configuration, item-pairing, summary "
        "recomputation, budget-enforcement, registry, and provenance checks.",
        "",
        "## Statistical evidence",
        "",
        f"- Minimum viable result: **{success['minimum_viable_result']}**.",
        f"- Primary success result: **{success['primary_success']}**.",
        f"- Primary family-wise alpha: **{config.familywise_alpha:.3f}**, "
        f"corrected with **{config.multiplicity_method}** over "
        f"{len(config.primary_comparators) * len(config.primary_budgets)} hypotheses.",
        "",
        "### Primary matched-budget comparisons",
        "",
        "| Budget | Comparator | C4 accuracy | Comparator accuracy | "
        "Difference (95% CI) | Holm reject | Density ratio |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in sorted(primary, key=lambda value: (value["reasoning_budget"], value["comparator"])):
        effect = row["paired_accuracy_difference"]
        lower, upper = effect["confidence_interval"]
        density = row["verified_density_ratio"]
        lines.append(
            f"| {row['reasoning_budget']} | {row['comparator'].upper()} | "
            f"{row['c4_accuracy']:.3%} | {row['comparator_accuracy']:.3%} | "
            f"{effect['estimate']:+.3%} [{lower:+.3%}, {upper:+.3%}] | "
            f"{row['holm']['reject_null']} | "
            f"{density:.3f} |" if density is not None else
            f"| {row['reasoning_budget']} | {row['comparator'].upper()} | "
            f"{row['c4_accuracy']:.3%} | {row['comparator_accuracy']:.3%} | "
            f"{effect['estimate']:+.3%} [{lower:+.3%}, {upper:+.3%}] | "
            f"{row['holm']['reject_null']} | n/a |"
        )
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            decision["reason"],
            "",
            "## Interpretation safeguards",
            "",
            "- The complete accuracy-versus-budget frontier is the headline result; no "
            "accuracy/token scalar is substituted for it.",
            "- Confidence intervals resample item IDs and average paired replicate-block "
            "outcomes within each item before resampling.",
            "- C3 and held-out depth/compositional comparisons are descriptive secondary "
            "analyses and are not added to the primary Holm family.",
            "- Null, adverse, and unsuccessful attempted runs remain in the machine-readable "
            "artifacts and experiment registry.",
            "- All report files were generated from saved artifacts without loading a model.",
            "",
            "## Reproduction",
            "",
            "Machine-readable aggregates, intervals, paired comparisons, task/depth "
            "breakdowns, seed variability, matched-accuracy reductions, system measurements, "
            "and qualitative paired examples are colocated with this report.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_primary_report(
    freeze_dir: str | Path,
    evaluation_root: str | Path,
    training_root: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build and adjudicate the complete frozen M3 report without loading a model."""

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        (
            config,
            plan,
            evaluation_runs,
            training_runs,
            registry_summary,
        ) = _load_primary_inputs(
            freeze_dir,
            evaluation_root,
            training_root,
            registry_path,
        )
        atomic_write_json(
            output_root / "analysis_contract.json",
            {
                "schema_version": 1,
                "analysis_version": config.analysis_version,
                "primary_inference_split": config.primary_inference_split,
                "primary_budgets": config.primary_budgets,
                "primary_comparators": config.primary_comparators,
                "secondary_comparators": config.secondary_comparators,
                "bootstrap_unit": config.bootstrap_unit,
                "within_item_reduction": (
                    "arithmetic mean across the three paired training/decoding blocks"
                ),
                "bootstrap_samples": config.bootstrap_samples,
                "bootstrap_seed": config.bootstrap_seed,
                "confidence_level": 0.95,
                "primary_test": "one-sided paired sign randomization, C4 greater",
                "familywise_alpha": config.familywise_alpha,
                "multiplicity_method": config.multiplicity_method,
                "primary_hypothesis_count": (
                    len(config.primary_budgets) * len(config.primary_comparators)
                ),
                "matched_accuracy_rule": (
                    "lowest frozen C4 budget whose pooled per-generation accuracy is at least "
                    "the comparator frontier point; reduction uses pooled median reasoning tokens"
                ),
                "reliable_depth_rule": (
                    "greatest frozen controlled depth with pooled per-generation accuracy at "
                    f"least {config.reliable_accuracy_threshold:.3f}"
                ),
                "diagnostic_limitations": (
                    "The saved evaluator distinguishes verified-prefix and invalid parsed "
                    "transitions and reports repetition. It does not separately classify "
                    "reversals versus other unproductive branches; those are retained in raw "
                    "generations and the invalid-transition count rather than inferred post hoc."
                ),
            },
        )
        atomic_write_json(
            output_root / "input_summary.json",
            {
                "schema_version": 1,
                "freeze_dir": str(Path(freeze_dir).resolve()),
                "freeze_manifest_sha256": sha256_file(
                    Path(freeze_dir).resolve() / "manifest.json"
                ),
                "run_plan_sha256": plan["run_plan_sha256"],
                "evaluation_root": str(Path(evaluation_root).resolve()),
                "training_root": str(Path(training_root).resolve()),
                "evaluation_run_count": len(evaluation_runs),
                "training_run_count": len(training_runs),
                "generation_count": sum(
                    run.generation_count for run in evaluation_runs
                ),
                "all_training_runs_passed": all(
                    run["passed"] for run in training_runs
                ),
            },
        )
        atomic_write_json(
            output_root / "registry_attempts.json",
            registry_summary,
        )
        aggregate_rows = _aggregate_table(evaluation_runs)
        atomic_write_json(
            output_root / "aggregate.json",
            {"schema_version": 1, "rows": aggregate_rows},
        )
        _write_aggregate_csv(output_root / "aggregate.csv", aggregate_rows)
        cells = _records_by_cell(evaluation_runs)
        per_problem = _per_problem_aggregates(cells)
        atomic_write_json(
            output_root / "per_problem_aggregates.json",
            per_problem,
        )
        intervals = _accuracy_intervals(cells, config)
        atomic_write_json(
            output_root / "accuracy_intervals.json",
            {"schema_version": 1, "rows": intervals},
        )
        comparisons = _paired_comparisons(cells, config)
        atomic_write_json(
            output_root / "paired_comparisons.json",
            {"schema_version": 1, "rows": comparisons},
        )
        breakdowns = _paired_primary_breakdowns(cells, config)
        atomic_write_json(
            output_root / "primary_task_depth_breakdowns.json",
            {"schema_version": 1, "rows": breakdowns},
        )
        variability = _seed_variability(aggregate_rows)
        atomic_write_json(
            output_root / "seed_variability.json",
            {"schema_version": 1, "rows": variability},
        )
        matched = _matched_accuracy_reductions(aggregate_rows, config)
        atomic_write_json(
            output_root / "matched_accuracy_token_reduction.json",
            {"schema_version": 1, "rows": matched},
        )
        reliable = _reliable_depths(aggregate_rows, config)
        atomic_write_json(
            output_root / "reliable_depth.json",
            {"schema_version": 1, "rows": reliable},
        )
        success = _primary_success_summary(
            comparisons,
            reliable,
            matched,
            config,
        )
        atomic_write_json(output_root / "success_summary.json", success)
        decision = _select_m3_decision(
            comparisons,
            matched,
            reliable,
            config,
        )
        if not decision["exactly_one_outcome_selected"]:
            raise ValueError("M3 decision logic did not select exactly one roadmap outcome")
        atomic_write_json(output_root / "decision.json", decision)
        systems = _system_metrics(evaluation_runs, training_runs)
        atomic_write_json(output_root / "systems.json", systems)
        failures = _failure_examples(evaluation_runs, config)
        atomic_write_json(output_root / "failure_analysis.json", failures)
        plots = _write_plots(output_root, aggregate_rows, config)
        _write_markdown_report(
            output_root / "M3_REPORT.md",
            success=success,
            decision=decision,
            comparisons=comparisons,
            systems=systems,
            config=config,
        )
        manifest = tree_manifest(output_root, output_root.name)
        return {
            "output_dir": str(output_root),
            "decision": decision["outcome"],
            "minimum_viable_result": success["minimum_viable_result"],
            "primary_success": success["primary_success"],
            "generation_count": sum(
                run.generation_count for run in evaluation_runs
            ),
            "plots": plots,
            "report_manifest_files": len(manifest["files"]),
            "report_manifest_sha256": sha256_file(output_root / "manifest.json"),
        }
    except BaseException as error:
        atomic_write_json(
            output_root / "failure.json",
            {
                "schema_version": 1,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        tree_manifest(output_root, output_root.name)
        raise
