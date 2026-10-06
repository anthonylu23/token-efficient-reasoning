from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from token_efficient_reasoning import prontoqa
from token_efficient_reasoning.budget_analysis import verify_manifest
from token_efficient_reasoning.data import load_items
from token_efficient_reasoning.scoring import score_answer


def raw():
    return {
        "example1": {
            "test_example": {
                "question": "All dax are blue. Jo is a dax.",
                "query": "True or false: Jo is blue.",
                "chain_of_thought": ["SECRET_GOLD_PROOF"],
                "answer": "True",
            },
            "in_context_example0": {"question": "SECRET_DEMONSTRATION"},
        }
    }


def config():
    return {
        "schema_version": 1,
        "role": prontoqa.ROLE,
        "seed": 20260929,
        "hops": [1, 3, 5],
        "items_per_hop": 1,
    }


def test_converter_excludes_gold_proof_and_demonstrations():
    item = prontoqa.convert_prontoqa(raw(), hops=3, seed=1, expected_count=1)[0]
    assert item.prompt == "All dax are blue. Jo is a dax.\nTrue or false: Jo is blue."
    assert "SECRET" not in item.prompt
    assert item.id == "prontoqa-v1-h3-s1-example1"
    assert score_answer("true", item.reference_answer, item.answer_type)[0]
    assert not score_answer("false", item.reference_answer, item.answer_type)[0]
    assert prontoqa.convert_prontoqa(raw(), hops=1, seed=1, expected_count=1)[0].id != item.id


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer", "yes"),
        ("answer", True),
        ("question", ""),
        ("query", None),
        ("chain_of_thought", "not a list"),
        ("model_output", "True"),
    ],
)
def test_converter_rejects_malformed_or_model_output_data(field, value):
    data = raw()
    data["example1"]["test_example"][field] = value
    with pytest.raises(ValueError):
        prontoqa.convert_prontoqa(data, hops=1, seed=1, expected_count=1)


def test_converter_rejects_duplicates_and_count_mismatch():
    data = raw()
    data["example2"] = copy.deepcopy(data["example1"])
    with pytest.raises(ValueError, match="duplicate"):
        prontoqa.convert_prontoqa(data, hops=1, seed=1, expected_count=2)
    with pytest.raises(ValueError, match="count"):
        prontoqa.convert_prontoqa(raw(), hops=1, seed=1, expected_count=2)


def test_generation_command_is_isolated_json_only():
    command = prontoqa.generation_command(Path("/source"), config(), 3)
    assert command[command.index("--model-name") + 1] == "json"
    assert command[command.index("--few-shot-examples") + 1] == "0"
    assert "--isolated" in command and "--no-project" in command
    assert "--resume" not in command
    for dependency in prontoqa.DEPENDENCIES:
        assert dependency in command
    with pytest.raises(ValueError):
        prontoqa.generation_command(Path("/source"), {**config(), "model_name": "gpt3"}, 3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hops", []),
        ("hops", [1, 1]),
        ("hops", [0]),
        ("seed", -1),
        ("seed", 2**32),
        ("items_per_hop", 0),
        ("role", "evaluation"),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        prontoqa.validate_config({**config(), field: value})


def test_pinned_source_rejects_changes(tmp_path, monkeypatch):
    for name in prontoqa.SOURCE_FILES:
        (tmp_path / name).write_bytes(b"pinned")
    pin = "a" * 40

    def check_output(argv, **kwargs):
        return pin + "\n" if argv[-2] == "rev-parse" else b"pinned"

    monkeypatch.setattr(subprocess, "check_output", check_output)
    assert len(prontoqa.verified_source(tmp_path, pin)) == len(prontoqa.SOURCE_FILES)
    (tmp_path / "theory.py").write_bytes(b"modified")
    with pytest.raises(ValueError, match="modified"):
        prontoqa.verified_source(tmp_path, pin)


def test_preparation_roundtrip_cpu_environment_and_failure_retention(tmp_path, monkeypatch):
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps({"sources": [{"name": "prontoqa-v1", "revision": "a" * 40}]}))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config()))
    monkeypatch.setattr(
        prontoqa,
        "verified_source",
        lambda *args: {name: b"upstream fixture" for name in prontoqa.SOURCE_FILES},
    )
    monkeypatch.setattr(prontoqa.shutil, "which", lambda _: "/fake/uv")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "uv fixture")

    def run(argv, *, cwd, env, **kwargs):
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["PYTHONHASHSEED"] == "0"
        assert "PYTHONPATH" not in env
        hops = int(argv[argv.index("--min-hops") + 1])
        data = raw()
        data["example1"]["test_example"]["question"] += f" Hop fixture {hops}."
        (cwd / f"{hops}hop_seed20260929.json").write_text(json.dumps(data))

    monkeypatch.setattr(subprocess, "run", run)
    output = tmp_path / "artifact"
    report = prontoqa.prepare(tmp_path, source_path, config_path, output)
    assert report["item_count"] == 3
    assert len(load_items(output / "items.jsonl")) == 3
    verify_manifest(output)
    with pytest.raises(FileExistsError):
        prontoqa.prepare(tmp_path, source_path, config_path, output)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "uv")

    monkeypatch.setattr(subprocess, "run", fail)
    failed = tmp_path / "failure"
    with pytest.raises(subprocess.CalledProcessError):
        prontoqa.prepare(tmp_path, source_path, config_path, failed)
    assert (failed / "failure.json").exists()
    assert not (failed / "manifest.json").exists()
