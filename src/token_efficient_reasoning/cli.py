"""Command-line entry point for M0 validation, evaluation, and repeat comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import RunWriter
from .config import CheckpointConfig, ConditionConfig, ExperimentConfig
from .data import load_items
from .prompting import build_user_prompt
from .protocol import BudgetedGenerator
from .provenance import (
    collect_environment,
    collect_hardware,
    collect_repository_state,
    sha256_file,
    stable_item_seed,
)
from .scoring import score_answer


def _select(items: tuple[Any, ...], item_id: str, kind: str) -> Any:
    matches = [item for item in items if item.id == item_id]
    if not matches:
        choices = ", ".join(item.id for item in items)
        raise ValueError(f"unknown {kind} {item_id!r}; choose one of: {choices}")
    return matches[0]


def _resolved_path(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    repo_candidate = Path.cwd() / candidate
    if repo_candidate.exists() or candidate.parts[0] in {"artifacts", "data"}:
        return repo_candidate.resolve()
    return (config_path.parent / candidate).resolve()


def _seed_everything(seed: int) -> None:
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


def _parse_status(generation: Any) -> tuple[str, list[str]]:
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


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty run")
    return {
        "schema_version": 1,
        "item_count": len(records),
        "accuracy": sum(bool(record["correct"]) for record in records) / len(records),
        "parse_success_rate": sum(record["parse_status"] == "valid" for record in records)
        / len(records),
        "forced_reasoning_rate": sum(
            bool(record["forced_reasoning_transition"]) for record in records
        )
        / len(records),
        "median_reasoning_tokens": statistics.median(
            record["token_counts"]["reasoning"] for record in records
        ),
        "mean_latency_seconds": statistics.fmean(
            record["latency_seconds"]["total"] for record in records
        ),
    }


def command_validate(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = ExperimentConfig.from_json(config_path)
    data_path = _resolved_path(config_path, config.data_path)
    items = load_items(data_path)
    print(
        json.dumps(
            {
                "config": str(config_path),
                "data": str(data_path),
                "items": len(items),
                "checkpoints": [item.id for item in config.checkpoints],
                "conditions": [item.id for item in config.conditions],
                "budgets": config.budgets,
                "seeds": config.seeds,
            },
            indent=2,
        )
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    hardware_preload = collect_hardware()
    config_path = Path(args.config).resolve()
    config = ExperimentConfig.from_json(config_path)
    checkpoint: CheckpointConfig = _select(config.checkpoints, args.checkpoint, "checkpoint")
    condition: ConditionConfig = _select(config.conditions, args.condition, "condition")
    data_path = _resolved_path(config_path, config.data_path)
    items = load_items(data_path)
    if args.max_items is not None:
        items = items[: args.max_items]
    budgets = tuple(args.budgets) if args.budgets else config.budgets
    unknown_budgets = sorted(set(budgets) - set(config.budgets))
    if unknown_budgets:
        raise ValueError(f"budgets are not preregistered in config: {unknown_budgets}")
    seeds = tuple(args.seeds) if args.seeds else config.seeds

    from .hf_backend import load_model_session

    session = load_model_session(checkpoint, config.decoding, args.device or config.device)
    generator = BudgetedGenerator(session.backend)
    reasoning_close_ids = session.encode_protocol_text(checkpoint.reasoning_close)
    answer_open_ids = session.encode_protocol_text(checkpoint.answer_open)
    answer_close_ids = session.encode_protocol_text(checkpoint.answer_close)
    reasoning_close_variants = session.encode_delimiter_variants(checkpoint.reasoning_close)
    answer_close_variants = session.encode_delimiter_variants(checkpoint.answer_close)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{timestamp}-{checkpoint.id}-{condition.id}"
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else _resolved_path(config_path, config.output_root)
    )
    writer = RunWriter(output_root / run_id, run_id)
    repo_root = Path.cwd().resolve()
    resolved_config = {
        "schema_version": 1,
        "run_id": run_id,
        "source_config": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "checkpoint": asdict(checkpoint),
        "condition": asdict(condition),
        "decoding": asdict(config.decoding),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "item_ids": [item.id for item in items],
        "budgets": budgets,
        "answer_allowance": config.answer_allowance,
        "seeds": seeds,
        "device": args.device or config.device,
        "protocol_token_ids": {
            "reasoning_close": reasoning_close_ids,
            "reasoning_close_variants": reasoning_close_variants,
            "answer_open": answer_open_ids,
            "answer_close": answer_close_ids,
            "answer_close_variants": answer_close_variants,
        },
        "tokenizer": _tokenizer_provenance(session),
        "repository": collect_repository_state(repo_root),
    }
    writer.write_metadata("resolved_config.json", resolved_config)
    writer.write_metadata("experiment_config.json", config.to_dict())
    writer.write_metadata("environment.json", collect_environment())
    writer.write_metadata("hardware_preload.json", hardware_preload)
    writer.write_metadata("hardware_before_generation.json", collect_hardware())

    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    for base_seed in seeds:
        for budget in budgets:
            for item in items:
                item_seed = stable_item_seed(base_seed, item.id, budget)
                _seed_everything(item_seed)
                user_prompt = build_user_prompt(item, condition)
                rendered_prompt, prompt_ids = session.render_prompt(user_prompt)
                try:
                    generation = generator.generate(
                        prompt_ids,
                        reasoning_budget=budget,
                        answer_allowance=config.answer_allowance,
                        reasoning_close_token_ids=reasoning_close_ids,
                        answer_open_token_ids=answer_open_ids,
                        answer_close_token_ids=answer_close_ids,
                        reasoning_close_variants=reasoning_close_variants,
                        answer_close_variants=answer_close_variants,
                    )
                except Exception as error:
                    writer.write_metadata(
                        "failure.json",
                        {
                            "schema_version": 1,
                            "run_id": run_id,
                            "checkpoint_id": checkpoint.id,
                            "condition_id": condition.id,
                            "item_id": item.id,
                            "reasoning_budget": budget,
                            "base_seed": base_seed,
                            "item_seed": item_seed,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    writer.finalize_manifest()
                    raise
                reasoning_text = session.decode(generation.reasoning.content_token_ids).strip()
                answer_text = session.decode(generation.answer.content_token_ids).strip()
                correct, normalized_prediction, normalized_reference = score_answer(
                    answer_text, item.reference_answer, item.answer_type
                )
                parse_status, parse_errors = _parse_status(generation)
                total_latency = (
                    generation.reasoning.latency_seconds + generation.answer.latency_seconds
                )
                sampled_count = len(generation.reasoning.raw_token_ids) + len(
                    generation.answer.raw_token_ids
                )
                record = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "item_id": item.id,
                    "task_family": item.task_family,
                    "checkpoint_id": checkpoint.id,
                    "condition_id": condition.id,
                    "reasoning_budget": budget,
                    "answer_allowance": config.answer_allowance,
                    "base_seed": base_seed,
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
                    "reference_answer": item.reference_answer,
                    "normalized_prediction": normalized_prediction,
                    "normalized_reference": normalized_reference,
                    "correct": correct,
                    "parse_status": parse_status,
                    "parse_errors": parse_errors,
                    "reasoning_termination": generation.reasoning.termination,
                    "answer_termination": generation.answer.termination,
                    "forced_reasoning_transition": generation.forced_reasoning_transition,
                    "delimiter_sources": {
                        "reasoning_close": (
                            "model" if generation.reasoning.delimiter_was_generated else "evaluator"
                        ),
                        "answer_open": "evaluator",
                        "answer_close": (
                            "model" if generation.answer.delimiter_was_generated else "evaluator"
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
                    "decode_tokens_per_second": sampled_count / total_latency
                    if total_latency > 0
                    else 0.0,
                }
                writer.append_item(record)
                records.append(record)
                reasoning_count = len(generation.reasoning.content_token_ids)
                print(
                    f"{item.id} budget={budget} seed={base_seed} "
                    f"correct={correct} reasoning_tokens={reasoning_count}",
                    flush=True,
                )

    hardware_after = collect_hardware()
    if torch.cuda.is_available():
        hardware_after["torch_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        hardware_after["torch_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    writer.write_metadata("hardware_after.json", hardware_after)
    writer.write_metadata("summary.json", _summarize(records))
    writer.finalize_manifest()
    print(writer.run_dir)
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def command_compare(args: argparse.Namespace) -> int:
    left = _load_jsonl(Path(args.left) / "items.jsonl")
    right = _load_jsonl(Path(args.right) / "items.jsonl")

    def identity(row: dict[str, Any]) -> tuple[str, int, int]:
        return row["item_id"], row["reasoning_budget"], row["base_seed"]

    left_by_id = {identity(row): row for row in left}
    right_by_id = {identity(row): row for row in right}
    differences: list[dict[str, Any]] = []
    fields = (
        "checkpoint_id",
        "condition_id",
        "item_seed",
        "prompt_token_ids",
        "reasoning_raw_token_ids",
        "answer_raw_token_ids",
        "effective_token_ids",
        "reference_answer",
        "parsed_answer",
        "parse_status",
        "correct",
    )
    for key in sorted(set(left_by_id) | set(right_by_id)):
        if key not in left_by_id or key not in right_by_id:
            differences.append({"identity": key, "field": "missing_item"})
            continue
        for field in fields:
            if left_by_id[key][field] != right_by_id[key][field]:
                differences.append({"identity": key, "field": field})
    result = {
        "identical": not differences,
        "left_items": len(left),
        "right_items": len(right),
        "differences": differences,
    }
    print(json.dumps(result, indent=2))
    return 0 if not differences else 1


def command_replay(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    with (run_dir / "resolved_config.json").open(encoding="utf-8") as handle:
        resolved = json.load(handle)
    replay_config = run_dir / "experiment_config.json"
    config = ExperimentConfig.from_json(replay_config)
    data_path = _resolved_path(replay_config, config.data_path)
    if sha256_file(data_path) != resolved["data_sha256"]:
        raise ValueError("evaluation data hash differs from the saved run")
    loaded_ids = [item.id for item in load_items(data_path)]
    expected_ids = resolved["item_ids"]
    if loaded_ids[: len(expected_ids)] != expected_ids:
        raise ValueError("evaluation item order differs from the saved run")
    replay_args = argparse.Namespace(
        config=str(replay_config),
        checkpoint=resolved["checkpoint"]["id"],
        condition=resolved["condition"]["id"],
        device=args.device or resolved["device"],
        max_items=len(expected_ids),
        budgets=resolved["budgets"],
        seeds=resolved["seeds"],
        run_id=args.run_id,
        output_root=args.output_root,
    )
    return command_evaluate(replay_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ter-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("config")
    validate.set_defaults(func=command_validate)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("config")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--condition", required=True)
    evaluate.add_argument("--device")
    evaluate.add_argument("--max-items", type=int)
    evaluate.add_argument("--budgets", nargs="+", type=int)
    evaluate.add_argument("--seeds", nargs="+", type=int)
    evaluate.add_argument("--run-id")
    evaluate.add_argument("--output-root")
    evaluate.set_defaults(func=command_evaluate)

    compare = subparsers.add_parser("compare-runs")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.set_defaults(func=command_compare)

    replay = subparsers.add_parser("replay-run")
    replay.add_argument("run_dir")
    replay.add_argument("--device")
    replay.add_argument("--run-id")
    replay.add_argument("--output-root")
    replay.set_defaults(func=command_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
