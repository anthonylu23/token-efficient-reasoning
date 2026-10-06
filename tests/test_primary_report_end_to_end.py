from __future__ import annotations

import json
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import token_efficient_reasoning.primary.final_reporting as final_reporting
from token_efficient_reasoning.artifacts import RunWriter
from token_efficient_reasoning.config import ExperimentConfig
from token_efficient_reasoning.primary.config import M3Config
from token_efficient_reasoning.primary.planning import (
    build_run_plan,
    refresh_run_plan_hash,
)
from token_efficient_reasoning.provenance import sha256_file
from token_efficient_reasoning.training.artifacts import tree_manifest
from token_efficient_reasoning.training.config import QLoRAConfig


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _raw_row(
    run: dict[str, Any],
    *,
    split: str,
    item_index: int,
    budget: int,
    batch_size: int,
) -> dict[str, Any]:
    condition = run["condition_id"]
    c4 = condition == "c4"
    item_id = f"{split}-item-{item_index:02d}"
    depth = 16 if split == "test_depth" else (2, 4, 8, 12)[item_index % 4]
    reasoning_tokens = max(1, budget // 2) if c4 else budget
    return {
        "schema_version": 1,
        "run_id": run["run_id"],
        "item_id": item_id,
        "task_family": "arithmetic" if item_index % 2 else "graph",
        "depth": depth,
        "source_instance_hash": f"instance-{split}-{item_index:02d}",
        "checkpoint_id": "deepseek-r1-distill-qwen-1.5b",
        "condition_id": condition,
        "reasoning_budget": budget,
        "answer_allowance": 16,
        "base_seed": run["decoding_seed"],
        "item_seed": item_index + budget,
        "reasoning_text": "s0 -> s1" if c4 else "ordinary reasoning",
        "parsed_answer": "1" if c4 else "0",
        "reference_answer": "1",
        "correct": c4,
        "parse_status": "valid",
        "trace_diagnostics": {
            "trace_parse_success": c4,
            "replay_valid": c4,
            "verified_prefix_transitions": depth if c4 else 1,
            "parsed_transition_count": depth if c4 else 1,
            "invalid_transition_count": 0,
            "reasoning_empty": False,
            "prompt_copied": False,
            "repeated_line_count": 0,
        },
        "forced_reasoning_transition": not c4,
        "reasoning_termination": "stop" if c4 else "forced_budget",
        "answer_termination": "stop",
        "token_counts": {
            "reasoning": reasoning_tokens,
            "answer": 2,
            "sampled_total": reasoning_tokens + 3,
        },
        "latency_seconds": {
            "reasoning": 0.08,
            "answer": 0.02,
            "total": 0.10,
        },
        "generation_batch_size": batch_size,
    }


def _write_training_runs(
    root: Path,
    runs: list[dict[str, Any]],
    *,
    freeze_manifest_sha256: str,
    protocol_sha256: str,
    qlora_sha256: str,
    train_sha256: str,
    protocol: dict[str, Any],
    qlora: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    for run in runs:
        run_dir = root / run["run_id"]
        (run_dir / "adapter").mkdir(parents=True)
        (run_dir / "adapter" / "adapter.bin").write_bytes(b"adapter")
        run_qlora = dict(qlora)
        run_qlora["seed"] = run["training_seed"]
        item_ids = [f"train-item-{index:02d}" for index in range(20)]
        _write_json(
            run_dir / "resolved_config.json",
            {
                "run_id": run["run_id"],
                "condition": {"id": run["condition_id"]},
                "protocol": protocol,
                "qlora": run_qlora,
                "checkpoint": checkpoint,
                "protocol_sha256": protocol_sha256,
                "qlora_sha256": qlora_sha256,
                "source_split_sha256": train_sha256,
                "item_count": len(item_ids),
                "item_ids": item_ids,
            },
        )
        _write_json(
            run_dir / "m3_context.json",
            {
                "phase": "primary",
                "run_spec": run,
                "freeze": {"manifest_sha256": freeze_manifest_sha256},
            },
        )
        _write_json(
            run_dir / "telemetry.json",
            {
                "global_step": 500,
                "trainable_parameters_changed": True,
                "log_history": [{"loss": 1.0}, {"loss": 0.5}],
                "trainer_metrics": {
                    "train_samples_per_second": 2.0,
                    "train_tokens_per_second": 200.0,
                },
                "wall_time_seconds": 10.0,
                "torch_cuda_peak_allocated_bytes": 100,
                "torch_cuda_peak_reserved_bytes": 200,
                "adapter_size_bytes": 7,
            },
        )
        _write_json(run_dir / "reload_verification.json", {"identical": True})
        _write_json(
            run_dir / "token_exposure.json",
            {
                "conditions": {
                    run["condition_id"]: {
                        "item_count": len(item_ids),
                        "total_target_tokens": 1_000,
                        "total_reasoning_tokens": 800,
                        "median_target_tokens": 50.0,
                    }
                }
            },
        )
        tree_manifest(run_dir, run["run_id"])


def _write_evaluation_runs(
    root: Path,
    training_root: Path,
    runs: list[dict[str, Any]],
    config: M3Config,
    *,
    freeze_manifest_sha256: str,
    protocol_sha256: str,
    qlora_sha256: str,
    split_sha256: dict[str, str],
    checkpoint: dict[str, Any],
    decoding: dict[str, Any],
) -> int:
    generation_count = 0
    for run in runs:
        run_dir = root / run["run_id"]
        run_dir.mkdir(parents=True)
        item_ids = [f"{run['split']}-item-{index:02d}" for index in range(20)]
        adapter_run_id = run["adapter_run_id"]
        adapter_path = (
            training_root / adapter_run_id / "adapter"
            if adapter_run_id is not None
            else None
        )
        adapter_manifest_sha256 = (
            sha256_file(training_root / adapter_run_id / "manifest.json")
            if adapter_run_id is not None
            else None
        )
        _write_json(
            run_dir / "resolved_config.json",
            {
                "condition": {"id": run["condition_id"]},
                "checkpoint": checkpoint,
                "decoding": decoding,
                "tokenizer": {
                    "name_or_path": config.checkpoint_id,
                    "vocab_size": 10,
                    "chat_template_sha256": "1" * 64,
                    "eos_token_id": 1,
                    "pad_token_id": 1,
                },
                "protocol_sha256": protocol_sha256,
                "qlora_sha256": qlora_sha256,
                "evaluation_split": run["split"],
                "evaluation_sha256": split_sha256[run["split"]],
                "base_seed": run["decoding_seed"],
                "budgets": list(config.evaluation_budgets),
                "batch_size": None,
                "batch_sizes_by_budget": {
                    str(budget): config.batch_size_for_budget(budget)
                    for budget in config.evaluation_budgets
                },
                "item_ids": item_ids,
                "adapter_path": str(adapter_path) if adapter_path else None,
                "adapter_manifest_sha256": adapter_manifest_sha256,
                "run_context": {
                    "phase": "primary",
                    "run_spec": run,
                    "freeze": {"manifest_sha256": freeze_manifest_sha256},
                },
            },
        )
        rows = [
            _raw_row(
                run,
                split=run["split"],
                item_index=item_index,
                budget=budget,
                batch_size=config.batch_size_for_budget(budget),
            )
            for budget in config.evaluation_budgets
            for item_index in range(20)
        ]
        with (run_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        _write_json(
            run_dir / "summary.json",
            {
                "item_generation_count": len(rows),
                "accuracy": statistics.fmean(row["correct"] for row in rows),
                "region_parse_success_rate": 1.0,
                "trace_parse_success_rate": statistics.fmean(
                    row["trace_diagnostics"]["trace_parse_success"] for row in rows
                ),
                "complete_replay_rate": statistics.fmean(
                    row["trace_diagnostics"]["replay_valid"] for row in rows
                ),
                "forced_reasoning_rate": statistics.fmean(
                    row["forced_reasoning_transition"] for row in rows
                ),
                "budget_violation_count": 0,
                "median_reasoning_tokens": statistics.median(
                    row["token_counts"]["reasoning"] for row in rows
                ),
                "mean_latency_seconds": statistics.fmean(
                    row["latency_seconds"]["total"] for row in rows
                ),
            },
        )
        _write_json(
            run_dir / "hardware_after.json",
            {
                "torch_cuda_peak_allocated_bytes": 100,
                "torch_cuda_peak_reserved_bytes": 200,
            },
        )
        # Exercise the same manifest schema as the real evaluator, not the training writer.
        RunWriter(run_dir, run["run_id"], resume=True).finalize_manifest()
        generation_count += len(rows)
    return generation_count


def test_primary_report_builds_from_complete_saved_artifact_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = replace(
        M3Config.from_json("configs/m3/protocol.json"),
        bootstrap_samples=2_000,
    )
    plan = build_run_plan(config)
    plan["counts"]["items_by_split"] = {
        split: 20 for split in config.evaluation_splits
    }
    plan["counts"]["generations"] = 5_400
    refresh_run_plan_hash(plan)
    protocol_sha256 = "a" * 64
    qlora_sha256 = "b" * 64
    split_sha256 = {
        "train": "c" * 64,
        "validation": "d" * 64,
        "test": "e" * 64,
        "test_depth": "f" * 64,
        "test_compositional": "0" * 64,
    }
    experiment = ExperimentConfig.from_json("configs/m0.json")
    checkpoint = asdict(
        next(value for value in experiment.checkpoints if value.id == config.checkpoint_id)
    )
    decoding = asdict(experiment.decoding)
    qlora = QLoRAConfig.from_json("configs/m3/qlora.json").to_dict()
    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()
    _write_json(freeze_dir / "run_plan.json", plan)
    _write_json(
        freeze_dir / "freeze.json",
        {
            "protocol": config.to_dict(),
            "protocol_sha256": protocol_sha256,
            "dataset_dir": str(Path("data/m1").resolve()),
            "qlora": qlora,
            "qlora_sha256": qlora_sha256,
            "checkpoint": checkpoint,
            "split_sha256": split_sha256,
            "run_plan_sha256": plan["run_plan_sha256"],
        },
    )
    tree_manifest(freeze_dir, freeze_dir.name)
    freeze_manifest_sha256 = sha256_file(freeze_dir / "manifest.json")
    monkeypatch.setattr(
        final_reporting,
        "validate_frozen_workspace",
        lambda _path: {"manifest_sha256": freeze_manifest_sha256},
    )

    training_root = tmp_path / "training"
    evaluation_root = tmp_path / "evaluation"
    _write_training_runs(
        training_root,
        plan["training_runs"],
        freeze_manifest_sha256=freeze_manifest_sha256,
        protocol_sha256=protocol_sha256,
        qlora_sha256=qlora_sha256,
        train_sha256=split_sha256["train"],
        protocol=config.to_dict(),
        qlora=qlora,
        checkpoint=checkpoint,
    )
    generation_count = _write_evaluation_runs(
        evaluation_root,
        training_root,
        plan["evaluation_runs"],
        config,
        freeze_manifest_sha256=freeze_manifest_sha256,
        protocol_sha256=protocol_sha256,
        qlora_sha256=qlora_sha256,
        split_sha256=split_sha256,
        checkpoint=checkpoint,
        decoding=decoding,
    )
    assert generation_count == plan["counts"]["generations"]

    registry = tmp_path / "registry.jsonl"
    with registry.open("w", encoding="utf-8") as handle:
        for run in (*plan["training_runs"], *plan["evaluation_runs"]):
            handle.write(
                json.dumps(
                    {
                        "phase": "primary",
                        "run_id": run["run_id"],
                        "status": "complete",
                        "run_spec_sha256": run["run_spec_sha256"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report_dir = tmp_path / "report"
    result = final_reporting.build_primary_report(
        freeze_dir,
        evaluation_root,
        training_root,
        registry,
        report_dir,
    )

    assert result["generation_count"] == 5_400
    assert result["primary_success"] is True
    assert result["h5_supported"] is True
    assert result["decision"] == "continue_to_generalization"
    assert result["report_manifest_files"] >= 25
    assert (report_dir / "per_problem_aggregates.json").is_file()
    sensitivity = json.loads(
        (report_dir / "distractor_answer_sensitivity.json").read_text(encoding="utf-8")
    )
    assert len(sensitivity["collision_item_ids"]) == 20
    assert len(sensitivity["rows"]) == 8
    assert (report_dir / "M3_REPORT.md").is_file()
