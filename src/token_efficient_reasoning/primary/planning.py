"""Deterministic M3 run-matrix construction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .config import M3Config


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def refresh_run_plan_hash(plan: dict[str, Any]) -> str:
    """Set and return a digest that covers every other field in the plan."""

    plan.pop("run_plan_sha256", None)
    digest = _stable_hash(plan)
    plan["run_plan_sha256"] = digest
    return digest


def build_run_plan(config: M3Config) -> dict[str, Any]:
    """Build the complete paired-block training and evaluation matrix."""

    training_runs = []
    for replicate, (training_seed, decoding_seed) in enumerate(
        zip(config.training_seeds, config.decoding_seeds, strict=True),
        start=1,
    ):
        for condition in config.conditions:
            run = {
                "run_type": "training",
                "phase": "primary",
                "replicate": replicate,
                "condition_id": condition.id,
                "representation": condition.representation,
                "training_seed": training_seed,
                "decoding_seed": decoding_seed,
                "optimizer_updates": config.optimizer_updates,
            }
            run["run_id"] = (
                f"m3-train-r{replicate}-{condition.id}-seed{training_seed}"
            )
            run["run_spec_sha256"] = _stable_hash(run)
            training_runs.append(run)

    evaluation_runs = []
    trained = {condition.id for condition in config.conditions}
    for replicate, (training_seed, decoding_seed) in enumerate(
        zip(config.training_seeds, config.decoding_seeds, strict=True),
        start=1,
    ):
        for condition_id in ("c0", "c1", "c2", "c3", "c4"):
            for split in config.evaluation_splits:
                run = {
                    "run_type": "evaluation",
                    "phase": "primary",
                    "replicate": replicate,
                    "condition_id": condition_id,
                    "split": split,
                    "training_seed": training_seed if condition_id in trained else None,
                    "decoding_seed": decoding_seed,
                    "budgets": list(config.evaluation_budgets),
                    "adapter_run_id": (
                        f"m3-train-r{replicate}-{condition_id}-seed{training_seed}"
                        if condition_id in trained
                        else None
                    ),
                }
                run["run_id"] = f"m3-eval-r{replicate}-{condition_id}-{split}"
                run["run_spec_sha256"] = _stable_hash(run)
                evaluation_runs.append(run)

    plan = {
        "schema_version": 1,
        "experiment_name": config.experiment_name,
        "design": "three paired training-and-decoding seed blocks",
        "training_runs": training_runs,
        "evaluation_runs": evaluation_runs,
        "counts": {
            "training_runs": len(training_runs),
            "evaluation_runs": len(evaluation_runs),
            "condition_replicates": len(config.training_seeds) * 5,
            "evaluation_cells": len(config.training_seeds)
            * 5
            * len(config.evaluation_splits)
            * len(config.evaluation_budgets),
            "generations": None,
        },
    }
    # Item counts and the exact generation count are resolved from frozen split manifests.
    refresh_run_plan_hash(plan)
    return plan
