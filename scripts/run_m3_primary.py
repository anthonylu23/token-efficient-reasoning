#!/usr/bin/env python3
"""Sequential, resumable executor for the frozen M3 primary matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=REPOSITORY_ROOT, check=True)


def _capture(arguments: list[str]) -> str:
    completed = subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def _preflight(destination: Path, *, minimum_free_gib: float, allow_busy_gpu: bool) -> None:
    free_bytes = shutil.disk_usage(REPOSITORY_ROOT).free
    gpu_processes = _capture(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if free_bytes < minimum_free_gib * 1024**3:
        raise RuntimeError(
            f"only {free_bytes / 1024**3:.1f} GiB is free; "
            f"{minimum_free_gib:.1f} GiB is required"
        )
    if gpu_processes and not allow_busy_gpu:
        raise RuntimeError(
            "another GPU compute process is active; use --allow-busy-gpu only after "
            "confirming it is safe:\n" + gpu_processes
        )
    snapshot = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "repository_root": str(REPOSITORY_ROOT),
        "uptime": _capture(["uptime"]),
        "memory": _capture(["free", "-h"]),
        "disk": _capture(["df", "-h", str(REPOSITORY_ROOT)]),
        "nvidia_smi": _capture(["nvidia-smi"]),
        "gpu_compute_processes": gpu_processes,
        "disk_free_bytes": free_bytes,
        "minimum_free_gib": minimum_free_gib,
        "allow_busy_gpu": allow_busy_gpu,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _registry_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _registry_event(
    *,
    registry: Path,
    run: dict[str, Any],
    status: str,
    path: Path,
    reason: str | None = None,
) -> None:
    command = [
        "uv",
        "run",
        "ter-m3",
        "registry-event",
        "--registry",
        str(registry),
        "--run-id",
        run["run_id"],
        "--run-type",
        run["run_type"],
        "--phase",
        "primary",
        "--status",
        status,
        "--path",
        str(path),
        "--run-spec-sha256",
        run["run_spec_sha256"],
    ]
    if reason is not None:
        command.extend(("--reason", reason))
    _run(command)


def _ensure_planned_events(
    plan: dict[str, Any],
    *,
    registry: Path,
    training_root: Path,
    evaluation_root: Path,
) -> None:
    existing = {
        record.get("run_id")
        for record in _registry_events(registry)
        if record.get("phase") == "primary"
    }
    for key, base in (
        ("training_runs", training_root),
        ("evaluation_runs", evaluation_root),
    ):
        for run in plan[key]:
            if run["run_id"] not in existing:
                _registry_event(
                    registry=registry,
                    run=run,
                    status="planned",
                    path=base / run["run_id"],
                )


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for path in (run_dir / "checkpoints").glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if path.is_dir():
            candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def _training_complete(run_dir: Path, expected_steps: int) -> bool:
    required = (
        run_dir / "telemetry.json",
        run_dir / "reload_verification.json",
        run_dir / "manifest.json",
    )
    if not all(path.is_file() for path in required) or (run_dir / "failure.json").exists():
        return False
    telemetry = _load_json(run_dir / "telemetry.json")
    reload_result = _load_json(run_dir / "reload_verification.json")
    return (
        telemetry.get("global_step") == expected_steps
        and telemetry.get("trainable_parameters_changed") is True
        and reload_result.get("identical") is True
    )


def _run_training(
    plan: dict[str, Any],
    *,
    freeze_dir: Path,
    training_root: Path,
    registry: Path,
) -> None:
    expected_steps = int(plan["training_runs"][0]["optimizer_updates"])
    training_root.mkdir(parents=True, exist_ok=True)
    for run in plan["training_runs"]:
        run_dir = training_root / run["run_id"]
        if _training_complete(run_dir, expected_steps):
            print(f"skip complete training run: {run['run_id']}", flush=True)
            continue
        command = [
            "uv",
            "run",
            "ter-m3",
            "train",
            "--phase",
            "primary",
            "--condition",
            run["condition_id"],
            "--training-seed",
            str(run["training_seed"]),
            "--output-dir",
            str(run_dir),
            "--freeze-dir",
            str(freeze_dir),
            "--registry",
            str(registry),
        ]
        if run_dir.exists():
            checkpoint = _latest_checkpoint(run_dir)
            if checkpoint is None:
                raise RuntimeError(
                    f"incomplete training run has no resumable checkpoint: {run_dir}"
                )
            command.extend(("--resume-from-checkpoint", str(checkpoint)))
        _run(command)
        try:
            _run(["uv", "run", "ter-m3", "verify-reload", str(run_dir)])
        except subprocess.CalledProcessError:
            _registry_event(
                registry=registry,
                run=run,
                status="failed",
                path=run_dir,
                reason="fresh-process reload verification failed",
            )
            raise
        _registry_event(
            registry=registry,
            run=run,
            status="complete",
            path=run_dir,
            reason="training and fresh-process reload verified",
        )


def _evaluation_complete(run_dir: Path, budget_count: int) -> bool:
    required = (
        run_dir / "resolved_config.json",
        run_dir / "summary.json",
        run_dir / "manifest.json",
    )
    if not all(path.is_file() for path in required) or (run_dir / "failure.json").exists():
        return False
    resolved = _load_json(run_dir / "resolved_config.json")
    summary = _load_json(run_dir / "summary.json")
    return summary.get("item_generation_count") == len(resolved["item_ids"]) * budget_count


def _run_evaluation(
    plan: dict[str, Any],
    *,
    freeze_dir: Path,
    training_root: Path,
    evaluation_root: Path,
    registry: Path,
) -> None:
    budget_count = len(plan["evaluation_runs"][0]["budgets"])
    evaluation_root.mkdir(parents=True, exist_ok=True)
    for run in plan["evaluation_runs"]:
        run_dir = evaluation_root / run["run_id"]
        if _evaluation_complete(run_dir, budget_count):
            print(f"skip complete evaluation run: {run['run_id']}", flush=True)
            continue
        command = [
            "uv",
            "run",
            "ter-m3",
            "evaluate",
            "--phase",
            "primary",
            "--condition",
            run["condition_id"],
            "--split",
            run["split"],
            "--decoding-seed",
            str(run["decoding_seed"]),
            "--output-dir",
            str(run_dir),
            "--freeze-dir",
            str(freeze_dir),
            "--registry",
            str(registry),
            "--progress-every",
            "100",
        ]
        if run["adapter_run_id"] is not None:
            command.extend(
                (
                    "--adapter",
                    str(training_root / run["adapter_run_id"] / "adapter"),
                )
            )
        if run_dir.exists():
            command.append("--resume")
        _run(command)


def _run_report(
    *,
    freeze_dir: Path,
    training_root: Path,
    evaluation_root: Path,
    registry: Path,
    report_dir: Path,
) -> None:
    if (report_dir / "manifest.json").is_file() and not (
        report_dir / "failure.json"
    ).exists():
        print(f"skip complete primary report: {report_dir}", flush=True)
        return
    if report_dir.exists():
        raise RuntimeError(
            f"incomplete report directory already exists; preserve and adjudicate it: {report_dir}"
        )
    _run(
        [
            "uv",
            "run",
            "ter-m3",
            "primary-report",
            "--freeze-dir",
            str(freeze_dir),
            "--training-root",
            str(training_root),
            "--evaluation-root",
            str(evaluation_root),
            "--registry",
            str(registry),
            "--output-dir",
            str(report_dir),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "train", "evaluate", "report"),
        default="all",
    )
    parser.add_argument("--freeze-dir", default="artifacts/m3/freeze")
    parser.add_argument("--primary-root", default="artifacts/m3/primary")
    parser.add_argument(
        "--registry",
        default="artifacts/m3/experiment_registry.jsonl",
    )
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    freeze_dir = (REPOSITORY_ROOT / args.freeze_dir).resolve()
    primary_root = (REPOSITORY_ROOT / args.primary_root).resolve()
    training_root = primary_root / "training"
    evaluation_root = primary_root / "evaluation"
    report_dir = primary_root / "report"
    registry = (REPOSITORY_ROOT / args.registry).resolve()
    _preflight(
        primary_root / "preflight.json",
        minimum_free_gib=args.minimum_free_gib,
        allow_busy_gpu=args.allow_busy_gpu,
    )
    _run(["uv", "run", "ter-m3", "validate-freeze", str(freeze_dir)])
    plan = _load_json(freeze_dir / "run_plan.json")
    _ensure_planned_events(
        plan,
        registry=registry,
        training_root=training_root,
        evaluation_root=evaluation_root,
    )
    if args.stage in {"all", "train"}:
        _run_training(
            plan,
            freeze_dir=freeze_dir,
            training_root=training_root,
            registry=registry,
        )
    if args.stage in {"all", "evaluate"}:
        _run_evaluation(
            plan,
            freeze_dir=freeze_dir,
            training_root=training_root,
            evaluation_root=evaluation_root,
            registry=registry,
        )
    if args.stage in {"all", "report"}:
        _run_report(
            freeze_dir=freeze_dir,
            training_root=training_root,
            evaluation_root=evaluation_root,
            registry=registry,
            report_dir=report_dir,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None
