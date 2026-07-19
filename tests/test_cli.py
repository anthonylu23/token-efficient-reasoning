from __future__ import annotations

import argparse
import json

from token_efficient_reasoning.cli import command_compare, command_validate


def test_validate_repository_config(capsys) -> None:
    result = command_validate(argparse.Namespace(config="configs/m0.json"))

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["items"] == 12
    assert output["budgets"] == [16, 32, 64, 128, 256, 512]


def _comparison_record(condition: str = "c0") -> dict[str, object]:
    return {
        "item_id": "item",
        "reasoning_budget": 16,
        "base_seed": 1,
        "item_seed": 2,
        "checkpoint_id": "model",
        "condition_id": condition,
        "prompt_token_ids": [1],
        "reasoning_raw_token_ids": [2],
        "answer_raw_token_ids": [3],
        "effective_token_ids": [1, 2, 3],
        "reference_answer": "3",
        "parsed_answer": "3",
        "parse_status": "valid",
        "correct": True,
    }


def _write_run(path, record) -> None:
    path.mkdir()
    (path / "items.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_compare_repeated_runs(capsys, tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, _comparison_record())
    _write_run(right, _comparison_record())

    result = command_compare(argparse.Namespace(left=str(left), right=str(right)))

    assert result == 0
    assert json.loads(capsys.readouterr().out)["identical"] is True


def test_compare_rejects_different_conditions(capsys, tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, _comparison_record("c0"))
    _write_run(right, _comparison_record("c1"))

    result = command_compare(argparse.Namespace(left=str(left), right=str(right)))

    output = json.loads(capsys.readouterr().out)
    assert result == 1
    assert output["differences"][0]["field"] == "condition_id"
