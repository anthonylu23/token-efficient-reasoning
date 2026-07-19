"""Serializable configuration models for reproducible M0 evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def _require_keys(data: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - data.keys())
    if missing:
        raise ConfigError(f"{context} is missing required keys: {', '.join(missing)}")


@dataclass(frozen=True)
class DecodingConfig:
    do_sample: bool
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecodingConfig:
        _require_keys(data, {"do_sample"}, "decoding config")
        config = cls(**data)
        if config.do_sample and (config.temperature is None or config.temperature <= 0):
            raise ConfigError("sampled decoding requires a positive temperature")
        return config

    def generation_kwargs(self) -> dict[str, Any]:
        values = asdict(self)
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class CheckpointConfig:
    id: str
    model_id: str
    revision: str
    tokenizer_revision: str
    dtype: str
    reasoning_prefix: str
    reasoning_close: str
    answer_open: str
    answer_close: str
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointConfig:
        _require_keys(
            data,
            {
                "id",
                "model_id",
                "revision",
                "tokenizer_revision",
                "dtype",
                "reasoning_prefix",
                "reasoning_close",
                "answer_open",
                "answer_close",
            },
            "checkpoint config",
        )
        config = cls(**data)
        if len(config.revision) != 40 or len(config.tokenizer_revision) != 40:
            raise ConfigError(f"checkpoint {config.id!r} must use 40-character commit revisions")
        if config.dtype not in {"bfloat16", "float16", "float32"}:
            raise ConfigError(f"unsupported dtype {config.dtype!r}")
        return config


@dataclass(frozen=True)
class FewShotExample:
    user: str
    assistant: str


@dataclass(frozen=True)
class ConditionConfig:
    id: str
    instruction: str
    few_shots: tuple[FewShotExample, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionConfig:
        _require_keys(data, {"id", "instruction"}, "condition config")
        shots = tuple(FewShotExample(**shot) for shot in data.get("few_shots", []))
        return cls(id=data["id"], instruction=data["instruction"], few_shots=shots)


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_name: str
    data_path: str
    output_root: str
    budgets: tuple[int, ...]
    answer_allowance: int
    seeds: tuple[int, ...]
    checkpoints: tuple[CheckpointConfig, ...]
    conditions: tuple[ConditionConfig, ...]
    decoding: DecodingConfig
    device: str = "cuda"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        _require_keys(
            data,
            {
                "schema_version",
                "experiment_name",
                "data_path",
                "output_root",
                "budgets",
                "answer_allowance",
                "seeds",
                "checkpoints",
                "conditions",
                "decoding",
            },
            "experiment config",
        )
        config = cls(
            schema_version=data["schema_version"],
            experiment_name=data["experiment_name"],
            data_path=data["data_path"],
            output_root=data["output_root"],
            budgets=tuple(data["budgets"]),
            answer_allowance=data["answer_allowance"],
            seeds=tuple(data["seeds"]),
            checkpoints=tuple(CheckpointConfig.from_dict(x) for x in data["checkpoints"]),
            conditions=tuple(ConditionConfig.from_dict(x) for x in data["conditions"]),
            decoding=DecodingConfig.from_dict(data["decoding"]),
            device=data.get("device", "cuda"),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigError(f"unsupported schema version: {self.schema_version}")
        if not self.budgets or any(budget <= 0 for budget in self.budgets):
            raise ConfigError("budgets must contain positive integers")
        if len(set(self.budgets)) != len(self.budgets):
            raise ConfigError("budgets must be unique")
        if self.answer_allowance <= 0:
            raise ConfigError("answer_allowance must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ConfigError("seeds must be non-empty and unique")
        if len({item.id for item in self.checkpoints}) != len(self.checkpoints):
            raise ConfigError("checkpoint ids must be unique")
        if len({item.id for item in self.conditions}) != len(self.conditions):
            raise ConfigError("condition ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
