"""Frozen protocol configuration for the M2 smoke-test experiment."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from token_efficient_reasoning.config import CheckpointConfig, ExperimentConfig
from token_efficient_reasoning.controlled.schema import REPRESENTATIONS
from token_efficient_reasoning.provenance import sha256_file

_EXPECTED_CONDITIONS = {"c2": "r0", "c3": "r2", "c4": "r3"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class M2ConfigError(ValueError):
    """Raised when the M2 smoke protocol is incomplete or inconsistent."""


def _require_keys(data: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - data.keys())
    if missing:
        raise M2ConfigError(f"{context} is missing required keys: {', '.join(missing)}")


def _resolve_reference(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    repository_candidate = Path.cwd() / candidate
    if repository_candidate.exists():
        return repository_candidate.resolve()
    return (config_path.parent / candidate).resolve()


@dataclass(frozen=True)
class M2ConditionConfig:
    id: str
    representation: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> M2ConditionConfig:
        _require_keys(data, {"id", "representation"}, "M2 condition")
        condition = cls(id=data["id"], representation=data["representation"])
        if condition.representation not in REPRESENTATIONS:
            raise M2ConfigError(
                f"condition {condition.id!r} uses unknown representation "
                f"{condition.representation!r}"
            )
        return condition


@dataclass(frozen=True)
class M2Config:
    schema_version: int
    experiment_name: str
    dataset_dir: str
    dataset_manifest_sha256: str
    m0_config_path: str
    checkpoint_id: str
    train_split: str
    validation_split: str
    conditions: tuple[M2ConditionConfig, ...]
    training_seed: int
    evaluation_budgets: tuple[int, ...]
    answer_allowance: int
    item_order: str
    label_policy: str
    target_format_version: str
    overflow_policy: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> M2Config:
        _require_keys(
            data,
            {
                "schema_version",
                "experiment_name",
                "dataset_dir",
                "dataset_manifest_sha256",
                "m0_config_path",
                "checkpoint_id",
                "train_split",
                "validation_split",
                "conditions",
                "training_seed",
                "evaluation_budgets",
                "answer_allowance",
                "item_order",
                "label_policy",
                "target_format_version",
                "overflow_policy",
            },
            "M2 config",
        )
        config = cls(
            schema_version=data["schema_version"],
            experiment_name=data["experiment_name"],
            dataset_dir=data["dataset_dir"],
            dataset_manifest_sha256=data["dataset_manifest_sha256"],
            m0_config_path=data["m0_config_path"],
            checkpoint_id=data["checkpoint_id"],
            train_split=data["train_split"],
            validation_split=data["validation_split"],
            conditions=tuple(M2ConditionConfig.from_dict(item) for item in data["conditions"]),
            training_seed=data["training_seed"],
            evaluation_budgets=tuple(data["evaluation_budgets"]),
            answer_allowance=data["answer_allowance"],
            item_order=data["item_order"],
            label_policy=data["label_policy"],
            target_format_version=data["target_format_version"],
            overflow_policy=data["overflow_policy"],
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> M2Config:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise M2ConfigError(f"unsupported M2 schema version: {self.schema_version}")
        if not self.experiment_name:
            raise M2ConfigError("experiment_name must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.dataset_manifest_sha256):
            raise M2ConfigError("dataset_manifest_sha256 must be a lowercase SHA-256 digest")
        if self.train_split == self.validation_split:
            raise M2ConfigError("train and validation splits must differ")
        if self.training_seed < 0:
            raise M2ConfigError("training_seed must be non-negative")
        if (
            not self.evaluation_budgets
            or any(budget <= 0 for budget in self.evaluation_budgets)
            or len(set(self.evaluation_budgets)) != len(self.evaluation_budgets)
        ):
            raise M2ConfigError("evaluation_budgets must be non-empty, positive, and unique")
        if tuple(sorted(self.evaluation_budgets)) != self.evaluation_budgets:
            raise M2ConfigError("evaluation_budgets must be in ascending order")
        if self.answer_allowance <= 0:
            raise M2ConfigError("answer_allowance must be positive")
        condition_mapping = {item.id: item.representation for item in self.conditions}
        if len(condition_mapping) != len(self.conditions):
            raise M2ConfigError("condition ids must be unique")
        if condition_mapping != _EXPECTED_CONDITIONS:
            raise M2ConfigError("M2 requires exactly c2=R0, c3=R2, and c4=R3")
        if self.item_order != "manifest":
            raise M2ConfigError("M2 item_order must be 'manifest'")
        if self.label_policy != "assistant_target_only":
            raise M2ConfigError("M2 label_policy must be 'assistant_target_only'")
        if self.target_format_version != "reasoning-answer-v1":
            raise M2ConfigError("unsupported M2 target format")
        if self.overflow_policy != "error":
            raise M2ConfigError("M2 overflow_policy must be 'error'")

    def validate_references(self, config_path: str | Path) -> CheckpointConfig:
        source_path = Path(config_path).resolve()
        dataset_dir = _resolve_reference(source_path, self.dataset_dir)
        manifest_path = dataset_dir / "manifest.json"
        actual_manifest_hash = sha256_file(manifest_path)
        if actual_manifest_hash != self.dataset_manifest_sha256:
            raise M2ConfigError(
                "M1 dataset manifest hash differs from the frozen M2 protocol: "
                f"{actual_manifest_hash}"
            )

        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        for split in (self.train_split, self.validation_split):
            relative_path = f"splits/{split}.jsonl"
            if relative_path not in manifest.get("files", {}):
                raise M2ConfigError(f"M1 manifest does not contain {relative_path}")
            split_path = dataset_dir / relative_path
            expected_hash = manifest["files"][relative_path]["sha256"]
            if sha256_file(split_path) != expected_hash:
                raise M2ConfigError(f"M1 split hash differs from its manifest: {relative_path}")

        m0_path = _resolve_reference(source_path, self.m0_config_path)
        m0_config = ExperimentConfig.from_json(m0_path)
        checkpoints = {checkpoint.id: checkpoint for checkpoint in m0_config.checkpoints}
        if self.checkpoint_id not in checkpoints:
            raise M2ConfigError(f"checkpoint {self.checkpoint_id!r} is not defined by M0")
        if self.evaluation_budgets != m0_config.budgets:
            raise M2ConfigError("M2 evaluation budgets must match the frozen M0 budgets")
        if self.answer_allowance != m0_config.answer_allowance:
            raise M2ConfigError("M2 answer allowance must match the frozen M0 allowance")
        return checkpoints[self.checkpoint_id]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QLoRAConfig:
    """Hardware-resolved QLoRA settings shared by every trained M2 condition."""

    schema_version: int
    quantization: str
    double_quantization: bool
    compute_dtype: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    max_sequence_length: int
    pad_to_multiple_of: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    epochs: float
    warmup_ratio: float
    lr_scheduler_type: str
    optimizer: str
    gradient_checkpointing: bool
    logging_steps: int
    save_steps: int
    save_total_limit: int
    seed: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QLoRAConfig:
        required = {
            "schema_version",
            "quantization",
            "double_quantization",
            "compute_dtype",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "max_sequence_length",
            "pad_to_multiple_of",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
            "epochs",
            "warmup_ratio",
            "lr_scheduler_type",
            "optimizer",
            "gradient_checkpointing",
            "logging_steps",
            "save_steps",
            "save_total_limit",
            "seed",
        }
        _require_keys(data, required, "QLoRA config")
        config = cls(
            schema_version=data["schema_version"],
            quantization=data["quantization"],
            double_quantization=data["double_quantization"],
            compute_dtype=data["compute_dtype"],
            lora_rank=data["lora_rank"],
            lora_alpha=data["lora_alpha"],
            lora_dropout=data["lora_dropout"],
            target_modules=tuple(data["target_modules"]),
            max_sequence_length=data["max_sequence_length"],
            pad_to_multiple_of=data["pad_to_multiple_of"],
            per_device_train_batch_size=data["per_device_train_batch_size"],
            gradient_accumulation_steps=data["gradient_accumulation_steps"],
            learning_rate=data["learning_rate"],
            weight_decay=data["weight_decay"],
            max_grad_norm=data["max_grad_norm"],
            epochs=data["epochs"],
            warmup_ratio=data["warmup_ratio"],
            lr_scheduler_type=data["lr_scheduler_type"],
            optimizer=data["optimizer"],
            gradient_checkpointing=data["gradient_checkpointing"],
            logging_steps=data["logging_steps"],
            save_steps=data["save_steps"],
            save_total_limit=data["save_total_limit"],
            seed=data["seed"],
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> QLoRAConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise M2ConfigError(f"unsupported QLoRA schema version: {self.schema_version}")
        if self.quantization != "nf4":
            raise M2ConfigError("M2 QLoRA quantization must be 'nf4'")
        if self.compute_dtype not in {"bfloat16", "float16"}:
            raise M2ConfigError("compute_dtype must be bfloat16 or float16")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise M2ConfigError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise M2ConfigError("lora_dropout must be in [0, 1)")
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise M2ConfigError("target_modules must be non-empty and unique")
        positive_integers = {
            "max_sequence_length": self.max_sequence_length,
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
        }
        invalid = sorted(name for name, value in positive_integers.items() if value <= 0)
        if invalid:
            raise M2ConfigError(f"QLoRA integer settings must be positive: {', '.join(invalid)}")
        if self.max_sequence_length % self.pad_to_multiple_of:
            raise M2ConfigError("max_sequence_length must be divisible by pad_to_multiple_of")
        if self.learning_rate <= 0 or self.epochs <= 0 or self.max_grad_norm <= 0:
            raise M2ConfigError("learning_rate, epochs, and max_grad_norm must be positive")
        if self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1:
            raise M2ConfigError("weight_decay and warmup_ratio are out of range")
        if self.lr_scheduler_type not in {"linear", "cosine"}:
            raise M2ConfigError("lr_scheduler_type must be linear or cosine")
        if self.optimizer not in {"paged_adamw_8bit", "adamw_torch"}:
            raise M2ConfigError("unsupported M2 optimizer")
        if self.seed < 0:
            raise M2ConfigError("QLoRA seed must be non-negative")

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
