from __future__ import annotations

import json

import pytest

from token_efficient_reasoning.config import ConfigError
from token_efficient_reasoning.controlled.config import M1Config


def test_repository_m1_config_is_valid() -> None:
    config = M1Config.from_json("configs/m1.json")

    assert config.families == (
        "multi_operation_arithmetic",
        "directed_graph_traversal",
    )
    assert {split.name for split in config.splits} == {
        "train",
        "validation",
        "test",
        "test_compositional",
        "test_depth",
    }


def test_m1_config_rejects_duplicate_splits() -> None:
    with open("configs/m1.json", encoding="utf-8") as handle:
        value = json.load(handle)
    value["splits"][1]["name"] = "train"

    with pytest.raises(ConfigError, match="uniquely named"):
        M1Config.from_dict(value)
