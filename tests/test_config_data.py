from __future__ import annotations

import json

import pytest

from token_efficient_reasoning.config import ConfigError, ExperimentConfig
from token_efficient_reasoning.data import load_items


def test_repository_m0_config_and_data_are_valid() -> None:
    config = ExperimentConfig.from_json("configs/m0.json")
    items = load_items(config.data_path)

    assert config.budgets == (16, 32, 64, 128, 256, 512)
    assert {checkpoint.id for checkpoint in config.checkpoints} == {
        "qwen3-0.6b",
        "deepseek-r1-distill-qwen-1.5b",
    }
    assert len(items) == 12


def test_config_rejects_floating_revision() -> None:
    with open("configs/m0.json", encoding="utf-8") as handle:
        data = json.load(handle)
    data["checkpoints"][0]["revision"] = "main"

    with pytest.raises(ConfigError, match="commit revisions"):
        ExperimentConfig.from_dict(data)


def test_data_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "items.jsonl"
    item = {
        "id": "same",
        "prompt": "x",
        "reference_answer": "1",
        "answer_type": "integer",
        "task_family": "test",
    }
    path.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate item id"):
        load_items(path)
