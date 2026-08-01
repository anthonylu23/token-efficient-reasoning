from __future__ import annotations

import json
from pathlib import Path

import pytest

from token_efficient_reasoning.training.config import (
    M2Config,
    M2ConfigError,
    QLoRAConfig,
)
from token_efficient_reasoning.training.runtime import prepare_training_run_directory
from token_efficient_reasoning.training.workflow import resolve_evaluation_batch_sizes


def test_repository_m2_protocol_and_references_are_valid() -> None:
    path = "configs/m2/smoke_protocol.json"
    config = M2Config.from_json(path)
    checkpoint = config.validate_references(path)

    assert checkpoint.id == "qwen3-0.6b"
    assert {item.id: item.representation for item in config.conditions} == {
        "c2": "r0",
        "c3": "r2",
        "c4": "r3",
    }
    assert config.evaluation_budgets == (16, 32, 64, 128, 256, 512)


def test_m2_protocol_rejects_changed_condition_mapping() -> None:
    with open("configs/m2/smoke_protocol.json", encoding="utf-8") as handle:
        data = json.load(handle)
    data["conditions"][2]["representation"] = "r2"

    with pytest.raises(M2ConfigError, match="exactly c2=R0, c3=R2, and c4=R3"):
        M2Config.from_dict(data)


def test_m2_protocol_rejects_silent_truncation() -> None:
    with open("configs/m2/smoke_protocol.json", encoding="utf-8") as handle:
        data = json.load(handle)
    data["overflow_policy"] = "truncate"

    with pytest.raises(M2ConfigError, match="overflow_policy"):
        M2Config.from_dict(data)


def test_repository_qlora_config_matches_smoke_protocol() -> None:
    protocol = M2Config.from_json("configs/m2/smoke_protocol.json")
    qlora = QLoRAConfig.from_json("configs/m2/qlora.json")

    assert qlora.seed == protocol.training_seed
    assert qlora.max_sequence_length == 568
    assert qlora.effective_batch_size == 16
    assert qlora.per_device_train_batch_size == 2
    assert qlora.quantization == "nf4"
    assert set(qlora.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


def test_qlora_config_rejects_non_aligned_sequence_length() -> None:
    with open("configs/m2/qlora.json", encoding="utf-8") as handle:
        data = json.load(handle)
    data["max_sequence_length"] = 564

    with pytest.raises(M2ConfigError, match="divisible"):
        QLoRAConfig.from_dict(data)


def test_training_resume_must_use_checkpoint_inside_same_frozen_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "m3-train-r1-c4-seed314159"
    created, checkpoint = prepare_training_run_directory(run_dir, None)
    assert created == run_dir.resolve()
    assert checkpoint is None

    valid = run_dir / "checkpoints" / "checkpoint-100"
    valid.mkdir(parents=True)
    resumed, checkpoint = prepare_training_run_directory(run_dir, valid)
    assert resumed == run_dir.resolve()
    assert checkpoint == valid.resolve()

    foreign = tmp_path / "other" / "checkpoints" / "checkpoint-100"
    foreign.mkdir(parents=True)
    with pytest.raises(ValueError, match="inside the run"):
        prepare_training_run_directory(run_dir, foreign)


def test_evaluation_batch_schedule_must_cover_selected_budgets_exactly() -> None:
    assert resolve_evaluation_batch_sizes((16, 32, 64), 4) == {
        16: 4,
        32: 4,
        64: 4,
    }
    assert resolve_evaluation_batch_sizes(
        (16, 32, 64),
        {16: 4, 32: 2, 64: 1},
    ) == {16: 4, 32: 2, 64: 1}

    with pytest.raises(ValueError, match="every selected budget"):
        resolve_evaluation_batch_sizes((16, 32), {16: 4})
