"""Command-line entry point for the M3 primary experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from token_efficient_reasoning.training.config import QLoRAConfig
from token_efficient_reasoning.training.runtime import verify_adapter_reload

from .config import M3Config
from .final_reporting import build_primary_report
from .freeze import create_freeze, validate_frozen_workspace
from .planning import build_run_plan
from .registry import append_registry_event
from .reporting import build_bridge_report
from .runtime import (
    evaluate_primary_condition,
    materialize_primary_data,
    train_primary_condition,
)

DEFAULT_PROTOCOL = "configs/m3/protocol.json"
DEFAULT_QLORA = "configs/m3/qlora.json"
DEFAULT_REGISTRY = "artifacts/m3/experiment_registry.jsonl"


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def command_validate(args: argparse.Namespace) -> int:
    config = M3Config.from_json(args.protocol)
    checkpoint, dataset_dir, split_paths = config.validate_references(args.protocol)
    qlora = QLoRAConfig.from_json(args.qlora)
    _print(
        {
            "protocol": str(Path(args.protocol).resolve()),
            "qlora": str(Path(args.qlora).resolve()),
            "checkpoint_id": checkpoint.id,
            "dataset_dir": str(dataset_dir),
            "splits": {name: str(path) for name, path in split_paths.items()},
            "conditions": {
                condition.id: condition.representation for condition in config.conditions
            },
            "training_seeds": config.training_seeds,
            "decoding_seeds": config.decoding_seeds,
            "budgets": config.evaluation_budgets,
            "primary_budgets": config.primary_budgets,
            "analysis_version": config.analysis_version,
            "primary_inference_split": config.primary_inference_split,
            "primary_comparators": config.primary_comparators,
            "secondary_comparators": config.secondary_comparators,
            "familywise_alpha": config.familywise_alpha,
            "multiplicity_method": config.multiplicity_method,
            "evaluation_batch_sizes": {
                str(budget): config.batch_size_for_budget(budget)
                for budget in config.evaluation_budgets
            },
            "maximum_sequence_length": qlora.max_sequence_length,
            "effective_batch_size": qlora.effective_batch_size,
            "readiness": config.readiness(args.protocol),
        }
    )
    return 0


def command_plan(args: argparse.Namespace) -> int:
    config = M3Config.from_json(args.protocol)
    config.validate_references(args.protocol)
    _print(build_run_plan(config))
    return 0


def command_materialize(args: argparse.Namespace) -> int:
    _print(materialize_primary_data(args.protocol, args.qlora, args.output_dir))
    return 0


def command_train(args: argparse.Namespace) -> int:
    run_id = Path(args.output_dir).name
    append_registry_event(
        args.registry,
        run_id=run_id,
        run_type="training",
        phase=args.phase,
        status="running",
        path=args.output_dir,
        details={"condition_id": args.condition, "training_seed": args.training_seed},
    )
    try:
        result = train_primary_condition(
            args.protocol,
            args.qlora,
            args.condition,
            args.output_dir,
            phase=args.phase,
            training_seed=args.training_seed,
            freeze_dir=args.freeze_dir,
            max_items=args.max_items,
            max_steps=args.max_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except KeyboardInterrupt:
        append_registry_event(
            args.registry,
            run_id=run_id,
            run_type="training",
            phase=args.phase,
            status="interrupted",
            path=args.output_dir,
        )
        raise
    except BaseException as error:
        append_registry_event(
            args.registry,
            run_id=run_id,
            run_type="training",
            phase=args.phase,
            status="failed",
            path=args.output_dir,
            details={"error_type": type(error).__name__, "error": str(error)},
        )
        raise
    append_registry_event(
        args.registry,
        run_id=run_id,
        run_type="training",
        phase=args.phase,
        status="complete",
        path=args.output_dir,
        run_spec_sha256=result.get("run_spec_sha256"),
        details=result,
    )
    _print(result)
    return 0


def command_verify_reload(args: argparse.Namespace) -> int:
    _print(verify_adapter_reload(args.run_dir))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    run_id = Path(args.output_dir).name
    append_registry_event(
        args.registry,
        run_id=run_id,
        run_type="evaluation",
        phase=args.phase,
        status="running",
        path=args.output_dir,
        details={
            "condition_id": args.condition,
            "split": args.split,
            "decoding_seed": args.decoding_seed,
            "resume": args.resume,
        },
    )
    try:
        result = evaluate_primary_condition(
            args.protocol,
            args.qlora,
            args.condition,
            args.split,
            args.output_dir,
            phase=args.phase,
            decoding_seed=args.decoding_seed,
            adapter_path=args.adapter,
            freeze_dir=args.freeze_dir,
            max_items=args.max_items,
            budgets=tuple(args.budgets) if args.budgets else None,
            device=args.device,
            progress_every=args.progress_every,
            resume=args.resume,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        append_registry_event(
            args.registry,
            run_id=run_id,
            run_type="evaluation",
            phase=args.phase,
            status="interrupted",
            path=args.output_dir,
        )
        raise
    except BaseException as error:
        append_registry_event(
            args.registry,
            run_id=run_id,
            run_type="evaluation",
            phase=args.phase,
            status="failed",
            path=args.output_dir,
            details={"error_type": type(error).__name__, "error": str(error)},
        )
        raise
    run_spec = result.get("run_context", {}).get("run_spec") if result else None
    append_registry_event(
        args.registry,
        run_id=run_id,
        run_type="evaluation",
        phase=args.phase,
        status="complete",
        path=args.output_dir,
        run_spec_sha256=run_spec and run_spec.get("run_spec_sha256"),
        details=result,
    )
    _print(result)
    return 0


def command_registry_event(args: argparse.Namespace) -> int:
    record = append_registry_event(
        args.registry,
        run_id=args.run_id,
        run_type=args.run_type,
        phase=args.phase,
        status=args.status,
        path=args.path,
        run_spec_sha256=args.run_spec_sha256,
        details={"reason": args.reason} if args.reason else {},
    )
    _print(record)
    return 0


def command_bridge_report(args: argparse.Namespace) -> int:
    _print(
        build_bridge_report(
            args.protocol,
            args.evaluation_runs,
            args.training_runs,
            args.output_dir,
        )
    )
    return 0


def command_primary_report(args: argparse.Namespace) -> int:
    _print(
        build_primary_report(
            args.freeze_dir,
            args.evaluation_root,
            args.training_root,
            args.registry,
            args.output_dir,
        )
    )
    return 0


def command_freeze(args: argparse.Namespace) -> int:
    _print(create_freeze(args.protocol, args.qlora, args.output_dir))
    return 0


def command_validate_freeze(args: argparse.Namespace) -> int:
    _print(validate_frozen_workspace(args.freeze_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ter-m3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    validate.add_argument("--qlora", default=DEFAULT_QLORA)
    validate.set_defaults(func=command_validate)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    plan.set_defaults(func=command_plan)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    materialize.add_argument("--qlora", default=DEFAULT_QLORA)
    materialize.add_argument("--output-dir", required=True)
    materialize.set_defaults(func=command_materialize)

    train = subparsers.add_parser("train")
    train.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    train.add_argument("--qlora", default=DEFAULT_QLORA)
    train.add_argument("--condition", choices=("c2", "c3", "c4"), required=True)
    train.add_argument("--phase", choices=("calibration", "bridge", "primary"), required=True)
    train.add_argument("--training-seed", type=int, required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--freeze-dir")
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
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--phase", choices=("bridge", "primary"), required=True)
    evaluate.add_argument("--decoding-seed", type=int, required=True)
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--freeze-dir")
    evaluate.add_argument("--max-items", type=int)
    evaluate.add_argument("--budgets", nargs="+", type=int)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--progress-every", type=int, default=25)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--registry", default=DEFAULT_REGISTRY)
    evaluate.set_defaults(func=command_evaluate)

    bridge_report = subparsers.add_parser("bridge-report")
    bridge_report.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    bridge_report.add_argument("--evaluation-runs", nargs=5, required=True)
    bridge_report.add_argument("--training-runs", nargs=3, required=True)
    bridge_report.add_argument("--output-dir", required=True)
    bridge_report.set_defaults(func=command_bridge_report)

    primary_report = subparsers.add_parser("primary-report")
    primary_report.add_argument("--freeze-dir", required=True)
    primary_report.add_argument("--evaluation-root", required=True)
    primary_report.add_argument("--training-root", required=True)
    primary_report.add_argument("--registry", default=DEFAULT_REGISTRY)
    primary_report.add_argument("--output-dir", required=True)
    primary_report.set_defaults(func=command_primary_report)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    freeze.add_argument("--qlora", default=DEFAULT_QLORA)
    freeze.add_argument("--output-dir", required=True)
    freeze.set_defaults(func=command_freeze)

    validate_frozen = subparsers.add_parser("validate-freeze")
    validate_frozen.add_argument("freeze_dir")
    validate_frozen.set_defaults(func=command_validate_freeze)

    registry = subparsers.add_parser("registry-event")
    registry.add_argument("--registry", default=DEFAULT_REGISTRY)
    registry.add_argument("--run-id", required=True)
    registry.add_argument("--run-type", required=True)
    registry.add_argument("--phase", required=True)
    registry.add_argument(
        "--status",
        choices=("planned", "running", "complete", "failed", "interrupted", "excluded"),
        required=True,
    )
    registry.add_argument("--path", required=True)
    registry.add_argument("--run-spec-sha256")
    registry.add_argument("--reason")
    registry.set_defaults(func=command_registry_event)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
