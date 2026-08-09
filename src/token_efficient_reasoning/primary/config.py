"""Frozen scientific contract for the Milestone 3 primary experiment."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from token_efficient_reasoning.config import CheckpointConfig, ExperimentConfig
from token_efficient_reasoning.controlled.schema import REPRESENTATIONS, load_examples
from token_efficient_reasoning.provenance import sha256_file

_CONDITION_MAPPING = {"c2": "r0", "c3": "r2", "c4": "r3"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_DEPTHS = {
    "test": {2, 4, 8, 12},
    "test_depth": {16},
    "test_compositional": {2, 4, 8, 12},
}
_ACCEPTED_HUMAN_DECISIONS = {"accept", "accept_with_documented_limitations"}


class M3ConfigError(ValueError):
    """Raised when the M3 protocol is incomplete or internally inconsistent."""


def _require_keys(data: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - data.keys())
    if missing:
        raise M3ConfigError(f"{context} is missing required keys: {', '.join(missing)}")


def _resolve_reference(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    # Repository configs use repository-root-relative paths, including output paths that may not
    # exist yet. Tests and external callers can provide absolute paths when using another root.
    return (Path.cwd() / candidate).resolve()


@dataclass(frozen=True)
class M3ConditionConfig:
    id: str
    representation: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> M3ConditionConfig:
        _require_keys(data, {"id", "representation"}, "M3 condition")
        condition = cls(id=data["id"], representation=data["representation"])
        if condition.representation not in REPRESENTATIONS:
            raise M3ConfigError(
                f"condition {condition.id!r} uses unknown representation "
                f"{condition.representation!r}"
            )
        return condition


@dataclass(frozen=True)
class M3Config:
    schema_version: int
    experiment_name: str
    dataset_dir: str
    dataset_manifest_sha256: str
    m0_config_path: str
    checkpoint_id: str
    train_split: str
    validation_split: str
    evaluation_splits: tuple[str, ...]
    conditions: tuple[M3ConditionConfig, ...]
    training_seeds: tuple[int, ...]
    decoding_seeds: tuple[int, ...]
    evaluation_budgets: tuple[int, ...]
    primary_budgets: tuple[int, ...]
    answer_allowance: int
    evaluation_batch_size: int
    evaluation_batch_sizes: tuple[int, ...]
    optimizer_updates: int
    checkpoint_selection: str
    item_order: str
    label_policy: str
    target_format_version: str
    overflow_policy: str
    bootstrap_samples: int
    bootstrap_seed: int
    analysis_version: str
    primary_inference_split: str
    central_primary_comparators: tuple[str, ...]
    primary_comparators: tuple[str, ...]
    h5_comparator: str
    secondary_comparators: tuple[str, ...]
    bootstrap_unit: str
    familywise_alpha: float
    multiplicity_method: str
    noninferiority_margin: float
    invalid_transition_margin: float
    reliable_accuracy_threshold: float
    minimum_token_reduction: float
    primary_accuracy_improvement: float
    verified_density_zero_control_rule: str
    primary_density_comparator: str
    minimum_density_ratio: float
    m1_human_review_path: str
    bridge_gate_path: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> M3Config:
        required = {
            "schema_version",
            "experiment_name",
            "dataset_dir",
            "dataset_manifest_sha256",
            "m0_config_path",
            "checkpoint_id",
            "train_split",
            "validation_split",
            "evaluation_splits",
            "conditions",
            "training_seeds",
            "decoding_seeds",
            "evaluation_budgets",
            "primary_budgets",
            "answer_allowance",
            "evaluation_batch_size",
            "evaluation_batch_sizes",
            "optimizer_updates",
            "checkpoint_selection",
            "item_order",
            "label_policy",
            "target_format_version",
            "overflow_policy",
            "bootstrap_samples",
            "bootstrap_seed",
            "analysis_version",
            "primary_inference_split",
            "central_primary_comparators",
            "primary_comparators",
            "h5_comparator",
            "secondary_comparators",
            "bootstrap_unit",
            "familywise_alpha",
            "multiplicity_method",
            "noninferiority_margin",
            "invalid_transition_margin",
            "reliable_accuracy_threshold",
            "minimum_token_reduction",
            "primary_accuracy_improvement",
            "verified_density_zero_control_rule",
            "primary_density_comparator",
            "minimum_density_ratio",
            "m1_human_review_path",
            "bridge_gate_path",
        }
        _require_keys(data, required, "M3 config")
        config = cls(
            schema_version=data["schema_version"],
            experiment_name=data["experiment_name"],
            dataset_dir=data["dataset_dir"],
            dataset_manifest_sha256=data["dataset_manifest_sha256"],
            m0_config_path=data["m0_config_path"],
            checkpoint_id=data["checkpoint_id"],
            train_split=data["train_split"],
            validation_split=data["validation_split"],
            evaluation_splits=tuple(data["evaluation_splits"]),
            conditions=tuple(M3ConditionConfig.from_dict(item) for item in data["conditions"]),
            training_seeds=tuple(data["training_seeds"]),
            decoding_seeds=tuple(data["decoding_seeds"]),
            evaluation_budgets=tuple(data["evaluation_budgets"]),
            primary_budgets=tuple(data["primary_budgets"]),
            answer_allowance=data["answer_allowance"],
            evaluation_batch_size=data["evaluation_batch_size"],
            evaluation_batch_sizes=tuple(data["evaluation_batch_sizes"]),
            optimizer_updates=data["optimizer_updates"],
            checkpoint_selection=data["checkpoint_selection"],
            item_order=data["item_order"],
            label_policy=data["label_policy"],
            target_format_version=data["target_format_version"],
            overflow_policy=data["overflow_policy"],
            bootstrap_samples=data["bootstrap_samples"],
            bootstrap_seed=data["bootstrap_seed"],
            analysis_version=data["analysis_version"],
            primary_inference_split=data["primary_inference_split"],
            central_primary_comparators=tuple(data["central_primary_comparators"]),
            primary_comparators=tuple(data["primary_comparators"]),
            h5_comparator=data["h5_comparator"],
            secondary_comparators=tuple(data["secondary_comparators"]),
            bootstrap_unit=data["bootstrap_unit"],
            familywise_alpha=float(data["familywise_alpha"]),
            multiplicity_method=data["multiplicity_method"],
            noninferiority_margin=float(data["noninferiority_margin"]),
            invalid_transition_margin=float(data["invalid_transition_margin"]),
            reliable_accuracy_threshold=float(data["reliable_accuracy_threshold"]),
            minimum_token_reduction=float(data["minimum_token_reduction"]),
            primary_accuracy_improvement=float(data["primary_accuracy_improvement"]),
            verified_density_zero_control_rule=data[
                "verified_density_zero_control_rule"
            ],
            primary_density_comparator=data["primary_density_comparator"],
            minimum_density_ratio=float(data["minimum_density_ratio"]),
            m1_human_review_path=data["m1_human_review_path"],
            bridge_gate_path=data["bridge_gate_path"],
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> M3Config:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise M3ConfigError(f"unsupported M3 schema version: {self.schema_version}")
        if not self.experiment_name:
            raise M3ConfigError("experiment_name must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.dataset_manifest_sha256):
            raise M3ConfigError("dataset_manifest_sha256 must be a lowercase SHA-256 digest")
        split_names = (self.train_split, self.validation_split, *self.evaluation_splits)
        if len(set(split_names)) != len(split_names):
            raise M3ConfigError("train, validation, and evaluation splits must be distinct")
        if set(self.evaluation_splits) != set(_EXPECTED_DEPTHS):
            raise M3ConfigError(
                "M3 evaluation_splits must contain test, test_depth, and test_compositional"
            )
        mapping = {condition.id: condition.representation for condition in self.conditions}
        if mapping != _CONDITION_MAPPING or len(mapping) != len(self.conditions):
            raise M3ConfigError("M3 requires exactly c2=R0, c3=R2, and c4=R3")
        if len(self.training_seeds) < 3 or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise M3ConfigError("M3 requires at least three unique training seeds")
        if len(self.decoding_seeds) != len(self.training_seeds) or len(
            set(self.decoding_seeds)
        ) != len(self.decoding_seeds):
            raise M3ConfigError(
                "decoding_seeds must be unique and pair one-to-one with training_seeds"
            )
        if any(seed < 0 for seed in (*self.training_seeds, *self.decoding_seeds)):
            raise M3ConfigError("training and decoding seeds must be non-negative")
        if (
            not self.evaluation_budgets
            or any(budget <= 0 for budget in self.evaluation_budgets)
            or tuple(sorted(set(self.evaluation_budgets))) != self.evaluation_budgets
        ):
            raise M3ConfigError(
                "evaluation_budgets must be unique positive values in ascending order"
            )
        if not self.primary_budgets or not set(self.primary_budgets) <= set(
            self.evaluation_budgets
        ):
            raise M3ConfigError("primary_budgets must be a non-empty evaluation-budget subset")
        if (
            self.answer_allowance <= 0
            or self.evaluation_batch_size <= 0
            or self.optimizer_updates <= 0
        ):
            raise M3ConfigError(
                "answer_allowance, evaluation_batch_size, and optimizer_updates must be positive"
            )
        if (
            len(self.evaluation_batch_sizes) != len(self.evaluation_budgets)
            or any(
                value <= 0 or value > self.evaluation_batch_size
                for value in self.evaluation_batch_sizes
            )
        ):
            raise M3ConfigError(
                "evaluation_batch_sizes must define one positive value no larger than "
                "evaluation_batch_size for every evaluation budget"
            )
        if self.checkpoint_selection != "fixed_final_update":
            raise M3ConfigError("M3 checkpoint_selection must be 'fixed_final_update'")
        if self.item_order != "manifest":
            raise M3ConfigError("M3 item_order must be 'manifest'")
        if self.label_policy != "assistant_target_only":
            raise M3ConfigError("M3 label_policy must be 'assistant_target_only'")
        if self.target_format_version != "reasoning-answer-v1":
            raise M3ConfigError("unsupported M3 target format")
        if self.overflow_policy != "error":
            raise M3ConfigError("M3 overflow_policy must be 'error'")
        if self.bootstrap_samples < 2_000 or self.bootstrap_seed < 0:
            raise M3ConfigError("bootstrap_samples must be at least 2000 and seed non-negative")
        if self.analysis_version != "m3-primary-v2":
            raise M3ConfigError("analysis_version must be 'm3-primary-v2'")
        if self.primary_inference_split != "test":
            raise M3ConfigError("primary_inference_split must be 'test'")
        if self.central_primary_comparators != ("c0", "c1", "c2"):
            raise M3ConfigError(
                "central_primary_comparators must be ordered as C0, C1, and C2"
            )
        if self.primary_comparators != ("c0", "c1", "c2", "c3"):
            raise M3ConfigError(
                "primary_comparators must add C3 to the ordered central comparators"
            )
        if self.h5_comparator != "c3":
            raise M3ConfigError("h5_comparator must be C3")
        if self.secondary_comparators:
            raise M3ConfigError("secondary_comparators must be empty")
        if self.bootstrap_unit != "item_id":
            raise M3ConfigError("bootstrap_unit must be 'item_id'")
        if not 0 < self.familywise_alpha < 1:
            raise M3ConfigError("familywise_alpha must lie in (0, 1)")
        if self.multiplicity_method != "holm":
            raise M3ConfigError("multiplicity_method must be 'holm'")
        unit_interval = {
            "invalid_transition_margin": self.invalid_transition_margin,
            "reliable_accuracy_threshold": self.reliable_accuracy_threshold,
            "minimum_token_reduction": self.minimum_token_reduction,
            "primary_accuracy_improvement": self.primary_accuracy_improvement,
        }
        if any(not 0 <= value <= 1 for value in unit_interval.values()):
            raise M3ConfigError("rate thresholds and margins must lie in [0, 1]")
        if not -1 <= self.noninferiority_margin <= 0:
            raise M3ConfigError("noninferiority_margin must lie in [-1, 0]")
        if self.verified_density_zero_control_rule != "positive_treatment_required":
            raise M3ConfigError(
                "verified_density_zero_control_rule must be "
                "'positive_treatment_required'"
            )
        if self.primary_density_comparator != "c2":
            raise M3ConfigError("primary_density_comparator must be C2")
        if self.minimum_density_ratio < 1:
            raise M3ConfigError("minimum_density_ratio must be at least one")

    def validate_references(
        self, config_path: str | Path
    ) -> tuple[CheckpointConfig, Path, dict[str, Path]]:
        source = Path(config_path).resolve()
        dataset_dir = _resolve_reference(source, self.dataset_dir)
        manifest_path = dataset_dir / "manifest.json"
        if sha256_file(manifest_path) != self.dataset_manifest_sha256:
            raise M3ConfigError("M1 dataset manifest differs from the frozen M3 protocol")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        split_paths: dict[str, Path] = {}
        for split in (
            self.train_split,
            self.validation_split,
            *self.evaluation_splits,
        ):
            relative = f"splits/{split}.jsonl"
            if relative not in manifest.get("files", {}):
                raise M3ConfigError(f"M1 manifest does not contain {relative}")
            split_path = dataset_dir / relative
            if sha256_file(split_path) != manifest["files"][relative]["sha256"]:
                raise M3ConfigError(f"M1 split differs from its manifest: {relative}")
            split_paths[split] = split_path
        for split, expected_depths in _EXPECTED_DEPTHS.items():
            actual_depths = {example.depth for example in load_examples(split_paths[split])}
            if actual_depths != expected_depths:
                raise M3ConfigError(
                    f"split {split!r} depths {sorted(actual_depths)} do not match "
                    f"{sorted(expected_depths)}"
                )
        m0_path = _resolve_reference(source, self.m0_config_path)
        m0 = ExperimentConfig.from_json(m0_path)
        checkpoints = {checkpoint.id: checkpoint for checkpoint in m0.checkpoints}
        if self.checkpoint_id not in checkpoints:
            raise M3ConfigError(f"checkpoint {self.checkpoint_id!r} is not defined by M0")
        if self.evaluation_budgets != m0.budgets:
            raise M3ConfigError("M3 evaluation budgets must match the frozen M0 budgets")
        if self.answer_allowance != m0.answer_allowance:
            raise M3ConfigError("M3 answer allowance must match the frozen M0 allowance")
        return checkpoints[self.checkpoint_id], dataset_dir, split_paths

    def readiness(self, config_path: str | Path) -> dict[str, Any]:
        source = Path(config_path).resolve()
        human_path = _resolve_reference(source, self.m1_human_review_path)
        bridge_path = _resolve_reference(source, self.bridge_gate_path)
        dataset_dir = _resolve_reference(source, self.dataset_dir)
        review_sample_path = dataset_dir / "review_sample.json"
        manifest_path = dataset_dir / "manifest.json"
        human: dict[str, Any] | None = None
        bridge: dict[str, Any] | None = None
        manifest_review_sample_sha256: str | None = None
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as handle:
                dataset_manifest = json.load(handle)
            manifest_review_sample_sha256 = (
                dataset_manifest.get("files", {}).get("review_sample.json", {}).get("sha256")
            )
        human_sha256 = sha256_file(human_path) if human_path.is_file() else None
        bridge_sha256 = sha256_file(bridge_path) if bridge_path.is_file() else None
        review_sample_sha256 = (
            sha256_file(review_sample_path) if review_sample_path.is_file() else None
        )
        if human_path.is_file():
            with human_path.open(encoding="utf-8") as handle:
                human = json.load(handle)
        if bridge_path.is_file():
            with bridge_path.open(encoding="utf-8") as handle:
                bridge = json.load(handle)
        review_sample_matches = bool(
            human
            and review_sample_sha256
            and human.get("review_sample_sha256") == review_sample_sha256
            and manifest_review_sample_sha256 == review_sample_sha256
        )
        human_passed = bool(
            human
            and human.get("status") == "complete"
            and human.get("decision") in _ACCEPTED_HUMAN_DECISIONS
            and human.get("dataset_manifest_sha256") == self.dataset_manifest_sha256
            and review_sample_matches
            and human.get("reviewed_item_count") == 34
            and human.get("reviewer")
            and human.get("reviewed_at")
            and human.get("approval_reference")
        )
        bridge_protocol_compatible = bool(
            bridge
            and bridge.get("protocol_compatibility_passed") is True
            and bridge.get("report_protocol_sha256") == sha256_file(source)
        )
        bridge_passed = bool(
            bridge
            and bridge.get("passed") is True
            and bridge.get("test_data_inspected") is False
            and bridge_protocol_compatible
        )
        return {
            "schema_version": 1,
            "m1_human_review_path": str(human_path),
            "m1_human_review_sha256": human_sha256,
            "m1_review_sample_path": str(review_sample_path),
            "m1_review_sample_sha256": review_sample_sha256,
            "m1_review_sample_matches": review_sample_matches,
            "m1_human_review_passed": human_passed,
            "bridge_gate_path": str(bridge_path),
            "bridge_gate_sha256": bridge_sha256,
            "bridge_protocol_compatible": bridge_protocol_compatible,
            "bridge_gate_passed": bridge_passed,
            "ready_to_freeze": human_passed and bridge_passed,
        }

    def batch_size_for_budget(self, budget: int) -> int:
        """Return the frozen hardware-safe generation batch for one reasoning budget."""

        try:
            index = self.evaluation_budgets.index(budget)
        except ValueError as error:
            raise M3ConfigError(f"budget is not declared by M3: {budget}") from error
        return self.evaluation_batch_sizes[index]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
