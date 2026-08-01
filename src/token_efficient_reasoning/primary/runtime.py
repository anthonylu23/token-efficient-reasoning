"""M3 materialization, training, and guarded evaluation workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.config import ExperimentConfig
from token_efficient_reasoning.controlled.adapters import to_evaluation_item
from token_efficient_reasoning.controlled.schema import (
    load_examples,
    stable_json_bytes,
)
from token_efficient_reasoning.provenance import collect_repository_state, sha256_file
from token_efficient_reasoning.training.artifacts import tree_manifest
from token_efficient_reasoning.training.config import QLoRAConfig
from token_efficient_reasoning.training.data import (
    audit_matched_examples,
    build_training_examples,
    summarize_token_exposure,
    tokenize_training_example,
)
from token_efficient_reasoning.training.runtime import (
    load_tokenizer,
    train_condition_resolved,
)
from token_efficient_reasoning.training.workflow import evaluate_condition_resolved

from .config import M3Config
from .freeze import validate_frozen_workspace


def _load_contract(
    protocol_path: str | Path,
    qlora_path: str | Path,
) -> tuple[M3Config, QLoRAConfig, Any, Path, ExperimentConfig]:
    protocol_source = Path(protocol_path).resolve()
    config = M3Config.from_json(protocol_source)
    checkpoint, dataset_dir, _ = config.validate_references(protocol_source)
    qlora = QLoRAConfig.from_json(qlora_path)
    m0_path = Path(config.m0_config_path)
    if not m0_path.is_absolute():
        m0_path = (Path.cwd() / m0_path).resolve()
    m0 = ExperimentConfig.from_json(m0_path)
    return config, qlora, checkpoint, dataset_dir, m0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json_bytes(row).decode() + "\n")


def resolve_runtime_batch_sizes(
    config: M3Config,
    budgets: tuple[int, ...] | None,
    batch_size: int | None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Resolve frozen and optional bridge batch sizes for selected budgets."""

    selected_budgets = budgets or config.evaluation_budgets
    unknown_budgets = sorted(set(selected_budgets) - set(config.evaluation_budgets))
    if unknown_budgets:
        raise ValueError(f"budgets are not frozen by M2: {unknown_budgets}")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    frozen = {
        budget: config.batch_size_for_budget(budget) for budget in selected_budgets
    }
    resolved = (
        {budget: batch_size for budget in selected_budgets}
        if batch_size is not None
        else frozen
    )
    return frozen, resolved


def materialize_primary_data(
    protocol_path: str | Path,
    qlora_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize matched M3 targets and primary-tokenizer exposure."""

    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    config, qlora, checkpoint, dataset_dir, _m0 = _load_contract(
        protocol_source, qlora_source
    )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    train_path = dataset_dir / "splits" / f"{config.train_split}.jsonl"
    validation_path = dataset_dir / "splits" / f"{config.validation_split}.jsonl"
    train_examples = load_examples(train_path)
    validation_examples = load_examples(validation_path)
    by_condition = build_training_examples(train_examples, config, checkpoint)
    audit = audit_matched_examples(by_condition)
    if not audit["passed"]:
        raise ValueError("matched-condition audit failed")
    tokenizer = load_tokenizer(checkpoint)
    exposure_conditions: dict[str, Any] = {}
    for condition_id, rows in by_condition.items():
        _write_jsonl(
            root / "conditions" / f"{condition_id}.jsonl",
            [row.to_dict() for row in rows],
        )
        tokenized = [
            tokenize_training_example(
                row,
                tokenizer,
                checkpoint,
                max_sequence_length=qlora.max_sequence_length,
            )
            for row in rows
        ]
        exposure_conditions.update(
            summarize_token_exposure({condition_id: tokenized})["conditions"]
        )
    atomic_write_json(root / "condition_audit.json", audit)
    atomic_write_json(
        root / "token_exposure.json",
        {"schema_version": 1, "conditions": exposure_conditions},
    )
    atomic_write_json(
        root / "resolved_config.json",
        {
            "schema_version": 1,
            "protocol": config.to_dict(),
            "qlora": qlora.to_dict(),
            "checkpoint": asdict(checkpoint),
            "protocol_path": str(protocol_source),
            "protocol_sha256": sha256_file(protocol_source),
            "qlora_path": str(qlora_source),
            "qlora_sha256": sha256_file(qlora_source),
            "train_split": str(train_path),
            "train_split_sha256": sha256_file(train_path),
            "validation_split": str(validation_path),
            "validation_split_sha256": sha256_file(validation_path),
            "validation_item_count": len(validation_examples),
            "repository": collect_repository_state(Path.cwd()),
        },
    )
    _write_jsonl(
        root / "validation_items.jsonl",
        [asdict(to_evaluation_item(example)) for example in validation_examples],
    )
    manifest = tree_manifest(root, root.name)
    return {
        "output_dir": str(root),
        "train_item_count": len(train_examples),
        "validation_item_count": len(validation_examples),
        "condition_audit_passed": audit["passed"],
        "maximum_sequence_tokens": max(
            row["maximum_sequence_tokens"] for row in exposure_conditions.values()
        ),
        "manifest_files": len(manifest["files"]),
    }


def _planned_run(
    freeze_dir: str | Path, run_type: str, run_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = validate_frozen_workspace(freeze_dir)
    plan_path = Path(freeze_dir).resolve() / "run_plan.json"
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    key = "training_runs" if run_type == "training" else "evaluation_runs"
    matches = [run for run in plan[key] if run["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"run {run_id!r} is not uniquely defined by the frozen M3 plan")
    return matches[0], freeze


def train_primary_condition(
    protocol_path: str | Path,
    qlora_path: str | Path,
    condition_id: str,
    output_dir: str | Path,
    *,
    phase: str,
    training_seed: int,
    freeze_dir: str | Path | None = None,
    max_items: int | None = None,
    max_steps: int | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train a calibration, bridge, or frozen primary adapter."""

    if phase not in {"calibration", "bridge", "primary"}:
        raise ValueError("phase must be calibration, bridge, or primary")
    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    config, qlora_template, checkpoint, dataset_dir, _m0 = _load_contract(
        protocol_source, qlora_source
    )
    if training_seed not in config.training_seeds:
        raise ValueError("training seed is not declared by the M3 protocol")
    run_id = Path(output_dir).name
    run_spec: dict[str, Any] | None = None
    freeze: dict[str, Any] | None = None
    if phase == "primary":
        if freeze_dir is None:
            raise ValueError("primary training requires --freeze-dir")
        run_spec, freeze = _planned_run(freeze_dir, "training", run_id)
        if (
            run_spec["condition_id"] != condition_id
            or run_spec["training_seed"] != training_seed
        ):
            raise ValueError("primary training arguments differ from the frozen run specification")
        if max_items is not None or max_steps is not None:
            raise ValueError("primary training forbids max-items and max-steps overrides")
    elif freeze_dir is not None:
        raise ValueError("freeze-dir is only valid for primary training")
    resolved_max_steps = (
        config.optimizer_updates if phase in {"bridge", "primary"} else max_steps
    )
    if phase == "bridge" and (max_items is not None or max_steps is not None):
        raise ValueError("bridge training uses the full frozen data and update count")
    qlora = replace(qlora_template, seed=training_seed)
    result = train_condition_resolved(
        config,
        qlora,
        checkpoint,
        dataset_dir,
        condition_id,
        output_dir,
        protocol_path=protocol_source,
        qlora_path=qlora_source,
        max_items=max_items,
        max_steps=resolved_max_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    context = {
        "schema_version": 1,
        "phase": phase,
        "training_seed": training_seed,
        "run_spec": run_spec,
        "freeze": freeze,
    }
    atomic_write_json(Path(output_dir).resolve() / "m3_context.json", context)
    tree_manifest(output_dir, run_id)
    return {**result, "phase": phase, "run_spec_sha256": run_spec and run_spec["run_spec_sha256"]}


def evaluate_primary_condition(
    protocol_path: str | Path,
    qlora_path: str | Path,
    condition_id: str,
    split: str,
    output_dir: str | Path,
    *,
    phase: str,
    decoding_seed: int,
    adapter_path: str | Path | None = None,
    freeze_dir: str | Path | None = None,
    max_items: int | None = None,
    budgets: tuple[int, ...] | None = None,
    device: str = "cuda",
    progress_every: int = 25,
    resume: bool = False,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate validation during the bridge or a frozen final M3 run."""

    if phase not in {"bridge", "primary"}:
        raise ValueError("evaluation phase must be bridge or primary")
    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    config, qlora, checkpoint, dataset_dir, m0 = _load_contract(
        protocol_source, qlora_source
    )
    run_id = Path(output_dir).name
    run_spec: dict[str, Any] | None = None
    freeze: dict[str, Any] | None = None
    frozen_batch_sizes, resolved_batch_sizes = resolve_runtime_batch_sizes(
        config,
        budgets,
        batch_size,
    )
    if phase == "bridge":
        if split != config.validation_split:
            raise ValueError("bridge evaluation may access only the validation split")
        if freeze_dir is not None:
            raise ValueError("bridge evaluation does not use a final freeze")
        if decoding_seed not in config.decoding_seeds:
            raise ValueError("bridge decoding seed is not declared by the protocol")
    else:
        if freeze_dir is None:
            raise ValueError("primary evaluation requires --freeze-dir")
        if max_items is not None or budgets is not None:
            raise ValueError("primary evaluation forbids item and budget overrides")
        if resolved_batch_sizes != frozen_batch_sizes:
            raise ValueError("primary evaluation batch sizes differ from the frozen protocol")
        run_spec, freeze = _planned_run(freeze_dir, "evaluation", run_id)
        expected = (
            run_spec["condition_id"],
            run_spec["split"],
            run_spec["decoding_seed"],
        )
        if (condition_id, split, decoding_seed) != expected:
            raise ValueError(
                "primary evaluation arguments differ from the frozen run specification"
            )
        if run_spec["adapter_run_id"] is not None and (
            adapter_path is None
            or Path(adapter_path).resolve().parent.name != run_spec["adapter_run_id"]
        ):
            raise ValueError("adapter path does not match the frozen training run")
    context = {
        "schema_version": 1,
        "phase": phase,
        "run_spec": run_spec,
        "freeze": freeze,
    }
    return evaluate_condition_resolved(
        config,
        qlora,
        checkpoint,
        dataset_dir,
        m0,
        condition_id,
        split,
        output_dir,
        protocol_path=protocol_source,
        qlora_path=qlora_source,
        decoding_seed=decoding_seed,
        adapter_path=adapter_path,
        max_items=max_items,
        budgets=budgets,
        device=device,
        progress_every=progress_every,
        resume=resume,
        run_context=context,
        batch_size=resolved_batch_sizes,
    )
