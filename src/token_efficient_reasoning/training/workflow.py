"""M2 data materialization and adapter-aware fixed-budget validation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from token_efficient_reasoning.artifacts import RunWriter, atomic_write_json
from token_efficient_reasoning.config import ConditionConfig
from token_efficient_reasoning.controlled.adapters import to_evaluation_item
from token_efficient_reasoning.controlled.schema import load_examples, stable_json_bytes
from token_efficient_reasoning.prompting import build_user_prompt
from token_efficient_reasoning.protocol import BatchBudgetedGenerator, BudgetedGenerator
from token_efficient_reasoning.provenance import (
    collect_environment,
    collect_hardware,
    collect_repository_state,
    sha256_file,
    stable_item_seed,
)
from token_efficient_reasoning.scoring import score_answer

from .artifacts import tree_manifest
from .data import (
    audit_matched_examples,
    build_training_examples,
    summarize_token_exposure,
    tokenize_training_example,
)
from .diagnostics import diagnose_trace
from .runtime import load_m0_config, load_m2_contract, load_tokenizer


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(stable_json_bytes(value).decode() + "\n")


def materialize_training_data(
    protocol_path: str | Path,
    qlora_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write matched text targets and tokenizer exposure as an immutable artifact."""

    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    protocol, qlora, checkpoint, dataset_dir = load_m2_contract(
        protocol_source, qlora_source
    )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    train_path = dataset_dir / "splits" / f"{protocol.train_split}.jsonl"
    validation_path = dataset_dir / "splits" / f"{protocol.validation_split}.jsonl"
    train_examples = load_examples(train_path)
    validation_examples = load_examples(validation_path)
    by_condition = build_training_examples(train_examples, protocol, checkpoint)
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
    exposure = {"schema_version": 1, "conditions": exposure_conditions}
    atomic_write_json(root / "condition_audit.json", audit)
    atomic_write_json(root / "token_exposure.json", exposure)
    atomic_write_json(
        root / "resolved_config.json",
        {
            "schema_version": 1,
            "protocol": protocol.to_dict(),
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


def _seed_evaluation(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenizer_provenance(session: Any) -> dict[str, Any]:
    template = session.tokenizer.chat_template or ""
    return {
        "name_or_path": session.tokenizer.name_or_path,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "vocab_size": len(session.tokenizer),
        "eos_token_id": session.tokenizer.eos_token_id,
        "pad_token_id": session.tokenizer.pad_token_id,
    }


def _region_parse_status(generation: Any) -> tuple[str, list[str]]:
    errors: list[str] = []
    if not generation.reasoning.content_token_ids:
        errors.append("empty_reasoning")
    if not generation.answer.content_token_ids:
        errors.append("empty_answer")
    if generation.reasoning.overflow_token_ids:
        errors.append("discarded_reasoning_overflow")
    if generation.answer.overflow_token_ids:
        errors.append("discarded_answer_overflow")
    return ("valid" if not errors else "malformed", errors)


def _evaluation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    budget_violations = sum(
        record["token_counts"]["reasoning"] > record["reasoning_budget"] for record in records
    )
    return {
        "schema_version": 1,
        "item_generation_count": len(records),
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "region_parse_success_rate": sum(
            record["parse_status"] == "valid" for record in records
        )
        / len(records),
        "trace_parse_success_rate": sum(
            record["trace_diagnostics"]["trace_parse_success"] for record in records
        )
        / len(records),
        "complete_replay_rate": sum(
            record["trace_diagnostics"]["replay_valid"] for record in records
        )
        / len(records),
        "forced_reasoning_rate": sum(
            record["forced_reasoning_transition"] for record in records
        )
        / len(records),
        "budget_violation_count": budget_violations,
        "median_reasoning_tokens": statistics.median(
            record["token_counts"]["reasoning"] for record in records
        ),
        "mean_latency_seconds": statistics.fmean(
            record["latency_seconds"]["total"] for record in records
        ),
    }


def _condition_for_evaluation(protocol: Any, m0: Any, condition_id: str) -> ConditionConfig:
    frozen = {condition.id: condition for condition in m0.conditions}
    if condition_id in frozen:
        return frozen[condition_id]
    trained = {condition.id for condition in protocol.conditions}
    if condition_id in trained:
        return ConditionConfig(id=condition_id, instruction="")
    raise ValueError(f"unknown M2 evaluation condition: {condition_id!r}")


def resolve_evaluation_batch_sizes(
    budgets: tuple[int, ...],
    batch_size: int | dict[int, int],
) -> dict[int, int]:
    """Normalize and validate a scalar or per-budget evaluation batch schedule."""

    if isinstance(batch_size, int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return {budget: batch_size for budget in budgets}
    resolved = {int(key): int(value) for key, value in batch_size.items()}
    if set(resolved) != set(budgets):
        raise ValueError("batch-size schedule must define every selected budget exactly")
    if any(value <= 0 for value in resolved.values()):
        raise ValueError("batch sizes must be positive")
    return resolved


def evaluate_condition(
    protocol_path: str | Path,
    qlora_path: str | Path,
    condition_id: str,
    output_dir: str | Path,
    *,
    adapter_path: str | Path | None = None,
    max_items: int | None = None,
    budgets: tuple[int, ...] | None = None,
    device: str = "cuda",
    progress_every: int = 25,
    resume: bool = False,
    batch_size: int | dict[int, int] = 1,
) -> dict[str, Any]:
    """Evaluate one frozen or trained M2 condition over controlled validation data."""

    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    protocol, qlora, checkpoint, dataset_dir = load_m2_contract(
        protocol_source, qlora_source
    )
    m0 = load_m0_config(protocol_source, protocol)
    return evaluate_condition_resolved(
        protocol,
        qlora,
        checkpoint,
        dataset_dir,
        m0,
        condition_id,
        protocol.validation_split,
        output_dir,
        protocol_path=protocol_source,
        qlora_path=qlora_source,
        decoding_seed=protocol.training_seed,
        adapter_path=adapter_path,
        max_items=max_items,
        budgets=budgets,
        device=device,
        progress_every=progress_every,
        resume=resume,
        batch_size=batch_size,
    )


def evaluate_condition_resolved(
    protocol: Any,
    qlora: Any,
    checkpoint: Any,
    dataset_dir: str | Path,
    m0: Any,
    condition_id: str,
    split: str,
    output_dir: str | Path,
    *,
    protocol_path: str | Path,
    qlora_path: str | Path,
    decoding_seed: int,
    adapter_path: str | Path | None = None,
    max_items: int | None = None,
    budgets: tuple[int, ...] | None = None,
    device: str = "cuda",
    progress_every: int = 25,
    resume: bool = False,
    run_context: dict[str, Any] | None = None,
    batch_size: int | dict[int, int] = 1,
) -> dict[str, Any]:
    """Evaluate one resolved condition and split under an M2 or M3 contract."""

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    if decoding_seed < 0:
        raise ValueError("decoding_seed must be non-negative")
    condition = _condition_for_evaluation(protocol, m0, condition_id)
    trained_condition_ids = {item.id for item in protocol.conditions}
    if condition_id in trained_condition_ids and adapter_path is None:
        raise ValueError(f"trained condition {condition_id} requires an adapter")
    if condition_id not in trained_condition_ids and adapter_path is not None:
        raise ValueError(f"frozen condition {condition_id} must not use an adapter")
    selected_budgets = budgets or protocol.evaluation_budgets
    unknown = sorted(set(selected_budgets) - set(protocol.evaluation_budgets))
    if unknown:
        raise ValueError(f"budgets are not frozen by M2: {unknown}")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    batch_sizes_by_budget = resolve_evaluation_batch_sizes(
        tuple(selected_budgets),
        batch_size,
    )
    split_path = dataset_dir / "splits" / f"{split}.jsonl"
    if not split_path.is_file():
        raise ValueError(f"configured evaluation split does not exist: {split!r}")
    examples = load_examples(split_path)
    if max_items is not None:
        examples = examples[:max_items]
    output_root = Path(output_dir).resolve()
    run_id = output_root.name
    writer = RunWriter(output_root, run_id, resume=resume)
    try:
        from token_efficient_reasoning.hf_backend import load_model_session

        session = load_model_session(
            checkpoint,
            m0.decoding,
            device,
            adapter_path=adapter_path,
        )
        generator = BudgetedGenerator(session.backend)
        batch_generator = BatchBudgetedGenerator(session.backend)
        reasoning_close_ids = session.encode_protocol_text(checkpoint.reasoning_close)
        answer_open_ids = session.encode_protocol_text(checkpoint.answer_open)
        answer_close_ids = session.encode_protocol_text(checkpoint.answer_close)
        reasoning_close_variants = session.encode_delimiter_variants(
            checkpoint.reasoning_close
        )
        answer_close_variants = session.encode_delimiter_variants(checkpoint.answer_close)
        expected_representation = {
            condition.id: condition.representation for condition in protocol.conditions
        }.get(condition_id)
        adapter = Path(adapter_path).resolve() if adapter_path is not None else None
        resolved = {
            "schema_version": 1,
            "run_id": run_id,
            "protocol": protocol.to_dict(),
            "protocol_path": str(protocol_source),
            "protocol_sha256": sha256_file(protocol_source),
            "qlora_path": str(qlora_source),
            "qlora_sha256": sha256_file(qlora_source),
            "checkpoint": asdict(checkpoint),
            "condition": asdict(condition),
            "decoding": asdict(m0.decoding),
            "adapter_path": str(adapter) if adapter is not None else None,
            "adapter_manifest_sha256": (
                sha256_file(adapter.parent / "manifest.json")
                if adapter is not None and (adapter.parent / "manifest.json").is_file()
                else None
            ),
            "evaluation_split": split,
            "evaluation_path": str(split_path),
            "evaluation_sha256": sha256_file(split_path),
            "item_ids": [example.id for example in examples],
            "budgets": selected_budgets,
            "answer_allowance": protocol.answer_allowance,
            "base_seed": decoding_seed,
            "device": device,
            "expected_representation": expected_representation,
            "tokenizer": _tokenizer_provenance(session),
            "repository": collect_repository_state(Path.cwd()),
            "run_context": run_context or {},
            "batch_size": batch_size if isinstance(batch_size, int) else None,
            "batch_sizes_by_budget": {
                str(budget): batch_sizes_by_budget[budget]
                for budget in selected_budgets
            },
        }
        resolved_path = output_root / "resolved_config.json"
        if resume:
            if not resolved_path.is_file():
                raise ValueError("resumed evaluation lacks resolved_config.json")
            with resolved_path.open(encoding="utf-8") as handle:
                previous_resolved = json.load(handle)
            if previous_resolved != resolved:
                raise ValueError("resumed evaluation configuration differs from the original run")
            failure_path = output_root / "failure.json"
            if failure_path.is_file():
                attempts_dir = output_root / "attempts"
                attempts_dir.mkdir(parents=True, exist_ok=True)
                archived_failures = sorted(
                    attempts_dir.glob("failure_before_resume_*.json")
                )
                attempt_number = len(archived_failures) + 1
                archived = (
                    attempts_dir
                    / f"failure_before_resume_{attempt_number:03d}.json"
                )
                if archived.exists():
                    raise ValueError(f"resume failure archive already exists: {archived}")
                failure_path.replace(archived)
        else:
            writer.write_metadata("resolved_config.json", resolved)
            writer.write_metadata("environment.json", collect_environment())
            writer.write_metadata("hardware_before.json", collect_hardware())
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        records: list[dict[str, Any]] = []
        completed: set[tuple[str, int]] = set()
        if resume and writer.items_path.is_file():
            with writer.items_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = (record["item_id"], record["reasoning_budget"])
                    if key in completed:
                        raise ValueError(f"resumed evaluation contains duplicate item key: {key}")
                    if record["run_id"] != run_id or record["condition_id"] != condition_id:
                        raise ValueError("resumed evaluation contains a foreign item record")
                    completed.add(key)
                    records.append(record)
        def append_generation(
            example: Any,
            budget: int,
            item_seed: int,
            rendered_prompt: str,
            prompt_ids: tuple[int, ...],
            generation: Any,
            generation_batch_size: int,
        ) -> None:
            reasoning_text = session.decode(
                generation.reasoning.content_token_ids
            ).strip()
            answer_text = session.decode(generation.answer.content_token_ids).strip()
            correct, normalized_prediction, normalized_reference = score_answer(
                answer_text, example.answer, example.answer_type
            )
            parse_status, parse_errors = _region_parse_status(generation)
            trace_diagnostics = diagnose_trace(
                example, reasoning_text, expected_representation
            )
            total_latency = (
                generation.reasoning.latency_seconds
                + generation.answer.latency_seconds
            )
            sampled_count = len(generation.reasoning.raw_token_ids) + len(
                generation.answer.raw_token_ids
            )
            record = {
                "schema_version": 1,
                "run_id": run_id,
                "item_id": example.id,
                "task_family": example.family,
                "depth": example.depth,
                "source_instance_hash": example.instance_hash,
                "checkpoint_id": checkpoint.id,
                "condition_id": condition.id,
                "reasoning_budget": budget,
                "answer_allowance": protocol.answer_allowance,
                "base_seed": decoding_seed,
                "item_seed": item_seed,
                "rendered_prompt": rendered_prompt,
                "prompt_token_ids": prompt_ids,
                "reasoning_raw_token_ids": generation.reasoning.raw_token_ids,
                "answer_raw_token_ids": generation.answer.raw_token_ids,
                "effective_token_ids": generation.effective_token_ids,
                "reasoning_token_ids": generation.reasoning.content_token_ids,
                "answer_token_ids": generation.answer.content_token_ids,
                "reasoning_overflow_token_ids": generation.reasoning.overflow_token_ids,
                "answer_overflow_token_ids": generation.answer.overflow_token_ids,
                "reasoning_text": reasoning_text,
                "parsed_answer": answer_text,
                "reference_answer": example.answer,
                "normalized_prediction": normalized_prediction,
                "normalized_reference": normalized_reference,
                "correct": correct,
                "parse_status": parse_status,
                "parse_errors": parse_errors,
                "trace_diagnostics": trace_diagnostics,
                "reasoning_termination": generation.reasoning.termination,
                "answer_termination": generation.answer.termination,
                "forced_reasoning_transition": generation.forced_reasoning_transition,
                "delimiter_sources": {
                    "reasoning_close": (
                        "model"
                        if generation.reasoning.delimiter_was_generated
                        else "evaluator"
                    ),
                    "answer_open": "evaluator",
                    "answer_close": (
                        "model"
                        if generation.answer.delimiter_was_generated
                        else "evaluator"
                    ),
                },
                "token_counts": {
                    "prompt": len(prompt_ids),
                    "reasoning": len(generation.reasoning.content_token_ids),
                    "answer": len(generation.answer.content_token_ids),
                    "sampled_total": sampled_count,
                },
                "latency_seconds": {
                    "reasoning": generation.reasoning.latency_seconds,
                    "answer": generation.answer.latency_seconds,
                    "total": total_latency,
                },
                "decode_tokens_per_second": (
                    sampled_count / total_latency if total_latency > 0 else 0.0
                ),
                "generation_batch_size": generation_batch_size,
            }
            writer.append_item(record)
            records.append(record)
            if len(records) % progress_every == 0:
                print(
                    f"{condition_id} completed={len(records)} "
                    f"budget={budget} item={example.id} "
                    f"correct={correct} "
                    f"trace={trace_diagnostics['trace_parse_success']}",
                    flush=True,
                )

        for budget in selected_budgets:
            current_batch_size = batch_sizes_by_budget[budget]
            pending = [
                example for example in examples if (example.id, budget) not in completed
            ]
            for start in range(0, len(pending), current_batch_size):
                batch = pending[start : start + current_batch_size]
                seeds = tuple(
                    stable_item_seed(decoding_seed, example.id, budget)
                    for example in batch
                )
                rendered_and_ids = []
                for example in batch:
                    evaluation_item = to_evaluation_item(example)
                    user_prompt = build_user_prompt(evaluation_item, condition)
                    rendered_and_ids.append(session.render_prompt(user_prompt))
                if current_batch_size == 1:
                    item_seed = seeds[0]
                    _seed_evaluation(item_seed)
                    rendered_prompt, prompt_ids = rendered_and_ids[0]
                    generation = generator.generate(
                        prompt_ids,
                        reasoning_budget=budget,
                        answer_allowance=protocol.answer_allowance,
                        reasoning_close_token_ids=reasoning_close_ids,
                        answer_open_token_ids=answer_open_ids,
                        answer_close_token_ids=answer_close_ids,
                        reasoning_close_variants=reasoning_close_variants,
                        answer_close_variants=answer_close_variants,
                    )
                    append_generation(
                        batch[0],
                        budget,
                        item_seed,
                        rendered_prompt,
                        prompt_ids,
                        generation,
                        1,
                    )
                    continue
                sampling_states = session.backend.create_sampling_states(seeds)
                generations = batch_generator.generate(
                    tuple(prompt_ids for _, prompt_ids in rendered_and_ids),
                    sampling_states=sampling_states,
                    reasoning_budget=budget,
                    answer_allowance=protocol.answer_allowance,
                    reasoning_close_token_ids=reasoning_close_ids,
                    answer_open_token_ids=answer_open_ids,
                    answer_close_token_ids=answer_close_ids,
                    reasoning_close_variants=reasoning_close_variants,
                    answer_close_variants=answer_close_variants,
                )
                for example, item_seed, rendered, generation in zip(
                    batch, seeds, rendered_and_ids, generations, strict=True
                ):
                    rendered_prompt, prompt_ids = rendered
                    append_generation(
                        example,
                        budget,
                        item_seed,
                        rendered_prompt,
                        prompt_ids,
                        generation,
                        len(batch),
                    )
            if pending:
                print(
                    f"{condition_id} completed={len(records)} budget={budget} "
                    f"item={pending[-1].id}",
                    flush=True,
                )
        hardware_after = collect_hardware()
        if torch.cuda.is_available():
            hardware_after["torch_cuda_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated()
            )
            hardware_after["torch_cuda_peak_reserved_bytes"] = (
                torch.cuda.max_memory_reserved()
            )
        summary = _evaluation_summary(records)
        writer.write_metadata("hardware_after.json", hardware_after)
        writer.write_metadata("summary.json", summary)
        writer.finalize_manifest()
        return {
            "run_dir": str(output_root),
            "condition_id": condition_id,
            "run_context": run_context or {},
            **summary,
        }
    except BaseException as error:
        writer.write_metadata(
            "failure.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "condition_id": condition_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        writer.finalize_manifest()
        raise


def append_registry(
    registry_path: str | Path,
    *,
    run_type: str,
    run_id: str,
    status: str,
    path: str | Path,
    details: dict[str, Any] | None = None,
) -> None:
    """Append one auditable attempted-run record."""

    destination = Path(registry_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "run_type": run_type,
        "run_id": run_id,
        "status": status,
        "path": str(Path(path).resolve()),
        "details": details or {},
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
