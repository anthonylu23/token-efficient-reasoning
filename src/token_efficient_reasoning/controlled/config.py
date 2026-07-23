"""Configuration contracts for deterministic M1 dataset generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from token_efficient_reasoning.config import ConfigError, _require_keys


@dataclass(frozen=True)
class SplitConfig:
    name: str
    depths: tuple[int, ...]
    items_per_family_depth: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SplitConfig:
        _require_keys(value, {"name", "depths", "items_per_family_depth"}, "M1 split")
        config = cls(
            name=value["name"],
            depths=tuple(value["depths"]),
            items_per_family_depth=value["items_per_family_depth"],
        )
        if not config.name or not config.depths:
            raise ConfigError("M1 split name and depths must be non-empty")
        if any(depth <= 0 for depth in config.depths) or len(set(config.depths)) != len(
            config.depths
        ):
            raise ConfigError(f"M1 split {config.name!r} has invalid depths")
        if config.items_per_family_depth <= 0:
            raise ConfigError("items_per_family_depth must be positive")
        return config


@dataclass(frozen=True)
class M1Config:
    schema_version: int
    dataset_name: str
    generator_version: str
    master_seed: int
    output_dir: str
    m0_config_path: str
    tokenizer_checkpoint_ids: tuple[str, ...]
    families: tuple[str, ...]
    splits: tuple[SplitConfig, ...]
    minimum_r3_median_reduction: float = 0.30
    require_compression_for: tuple[str, ...] = ("r2", "r3")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> M1Config:
        _require_keys(
            value,
            {
                "schema_version",
                "dataset_name",
                "generator_version",
                "master_seed",
                "output_dir",
                "m0_config_path",
                "tokenizer_checkpoint_ids",
                "families",
                "splits",
            },
            "M1 config",
        )
        config = cls(
            schema_version=value["schema_version"],
            dataset_name=value["dataset_name"],
            generator_version=value["generator_version"],
            master_seed=value["master_seed"],
            output_dir=value["output_dir"],
            m0_config_path=value["m0_config_path"],
            tokenizer_checkpoint_ids=tuple(value["tokenizer_checkpoint_ids"]),
            families=tuple(value["families"]),
            splits=tuple(SplitConfig.from_dict(item) for item in value["splits"]),
            minimum_r3_median_reduction=value.get("minimum_r3_median_reduction", 0.30),
            require_compression_for=tuple(value.get("require_compression_for", ["r2", "r3"])),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> M1Config:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigError(f"unsupported M1 config schema: {self.schema_version}")
        if not self.dataset_name or not self.generator_version:
            raise ConfigError("dataset_name and generator_version must be non-empty")
        if not self.families or len(set(self.families)) != len(self.families):
            raise ConfigError("M1 families must be non-empty and unique")
        if not self.splits or len({item.name for item in self.splits}) != len(self.splits):
            raise ConfigError("M1 splits must be non-empty and uniquely named")
        if not self.tokenizer_checkpoint_ids or len(set(self.tokenizer_checkpoint_ids)) != len(
            self.tokenizer_checkpoint_ids
        ):
            raise ConfigError("tokenizer checkpoint ids must be non-empty and unique")
        if not 0 < self.minimum_r3_median_reduction < 1:
            raise ConfigError("minimum_r3_median_reduction must be between zero and one")
        if not set(self.require_compression_for) <= {"r1", "r2", "r3"}:
            raise ConfigError("compression requirements must name R1, R2, or R3")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "generator_version": self.generator_version,
            "master_seed": self.master_seed,
            "output_dir": self.output_dir,
            "m0_config_path": self.m0_config_path,
            "tokenizer_checkpoint_ids": list(self.tokenizer_checkpoint_ids),
            "families": list(self.families),
            "splits": [
                {
                    "name": split.name,
                    "depths": list(split.depths),
                    "items_per_family_depth": split.items_per_family_depth,
                }
                for split in self.splits
            ],
            "minimum_r3_median_reduction": self.minimum_r3_median_reduction,
            "require_compression_for": list(self.require_compression_for),
        }
