"""Command-line orchestration for the complete local M2 smoke-test workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .reporting import build_m2_report
from .runtime import (
    load_m2_contract,
    train_condition,
    verify_adapter_reload,
)
from .workflow import (
    append_registry,
    evaluate_condition,
    materialize_training_data,
)

DEFAULT_PROTOCOL = "configs/m2/smoke_protocol.json"
DEFAULT_QLORA = "configs/m2/qlora.json"
DEFAULT_REGISTRY = "artifacts/m2/experiment_registry.jsonl"


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def command_validate(args: argparse.Namespace) -> int:
    protocol, qlora, checkpoint, dataset_dir = load_m2_contract(
        args.protocol, args.qlora
    )
    _print(
        {
            "protocol": str(Path(args.protocol).resolve()),
            "qlora": str(Path(args.qlora).resolve()),
            "checkpoint_id": checkpoint.id,
            "dataset_dir": str(dataset_dir),
            "conditions": {
                condition.id: condition.representation for condition in protocol.conditions
            },
            "budgets": protocol.evaluation_budgets,
            "maximum_sequence_length": qlora.max_sequence_length,
            "effective_batch_size": qlora.effective_batch_size,
        }
    )
    return 0


def command_materialize(args: argparse.Namespace) -> int:
    result = materialize_training_data(
        args.protocol,
        args.qlora,
        args.output_dir,
    )
    _print(result)
    return 0


def command_train(args: argparse.Namespace) -> int:
    run_id = Path(args.output_dir).name
    try:
        result = train_condition(
            args.protocol,
            args.qlora,
            args.condition,
            args.output_dir,
            max_items=args.max_items,
            max_steps=args.max_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except BaseException as error:
        append_registry(
            args.registry,
            run_type="training",
            run_id=run_id,
            status="failed",
            path=args.output_dir,
            details={
                "condition_id": args.condition,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    append_registry(
        args.registry,
        run_type="training",
        run_id=run_id,
        status="complete",
        path=args.output_dir,
        details=result,
    )
    _print(result)
    return 0


def command_verify_reload(args: argparse.Namespace) -> int:
    result = verify_adapter_reload(args.run_dir)
    _print(result)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    run_id = Path(args.output_dir).name
    try:
        result = evaluate_condition(
            args.protocol,
            args.qlora,
            args.condition,
            args.output_dir,
            adapter_path=args.adapter,
            max_items=args.max_items,
            budgets=tuple(args.budgets) if args.budgets else None,
            device=args.device,
            progress_every=args.progress_every,
        )
    except BaseException as error:
        append_registry(
            args.registry,
            run_type="evaluation",
            run_id=run_id,
            status="failed",
            path=args.output_dir,
            details={
                "condition_id": args.condition,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    append_registry(
        args.registry,
        run_type="evaluation",
        run_id=run_id,
        status="complete",
        path=args.output_dir,
        details=result,
    )
    _print(result)
    return 0


def command_report(args: argparse.Namespace) -> int:
    result = build_m2_report(
        args.evaluation_runs,
        args.training_runs,
        args.output_dir,
        args.gate_review,
        m1_gate_path=args.m1_gate,
    )
    _print(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ter-m2")
    parser.set_defaults(protocol=DEFAULT_PROTOCOL, qlora=DEFAULT_QLORA)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    validate.add_argument("--qlora", default=DEFAULT_QLORA)
    validate.set_defaults(func=command_validate)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    materialize.add_argument("--qlora", default=DEFAULT_QLORA)
    materialize.add_argument("--output-dir", required=True)
    materialize.set_defaults(func=command_materialize)

    train = subparsers.add_parser("train")
    train.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    train.add_argument("--qlora", default=DEFAULT_QLORA)
    train.add_argument("--condition", choices=("c2", "c3", "c4"), required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--max-items", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--resume-from-checkpoint")
    train.add_argument("--registry", default=DEFAULT_REGISTRY)
    train.set_defaults(func=command_train)

    reload_parser = subparsers.add_parser("verify-reload")
    reload_parser.add_argument("run_dir")
    reload_parser.set_defaults(func=command_verify_reload)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    evaluate.add_argument("--qlora", default=DEFAULT_QLORA)
    evaluate.add_argument("--condition", choices=("c0", "c1", "c2", "c3", "c4"), required=True)
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--max-items", type=int)
    evaluate.add_argument("--budgets", nargs="+", type=int)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--progress-every", type=int, default=25)
    evaluate.add_argument("--registry", default=DEFAULT_REGISTRY)
    evaluate.set_defaults(func=command_evaluate)

    report = subparsers.add_parser("report")
    report.add_argument("--evaluation-runs", nargs=5, required=True)
    report.add_argument("--training-runs", nargs=3, required=True)
    report.add_argument("--output-dir", required=True)
    report.add_argument("--gate-review", default="artifacts/M2_GATE_REVIEW.md")
    report.add_argument("--m1-gate", default="data/m1/gate_summary.json")
    report.set_defaults(func=command_report)
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
