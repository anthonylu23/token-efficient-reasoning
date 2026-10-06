from __future__ import annotations

import json
from pathlib import Path

import pytest

from token_efficient_reasoning.budget_analysis import (
    analyze,
    diagnose_budget,
    summarize,
    verify_manifest,
)
from token_efficient_reasoning.controlled.schema import CanonicalExample
from token_efficient_reasoning.provenance import sha256_file


@pytest.fixture
def example() -> CanonicalExample:
    with Path("data/m1/splits/validation.jsonl").open() as handle:
        return CanonicalExample.from_dict(json.loads(next(handle)))


def generation(example, *, budget, text, forced=False):
    return {
        "condition_id": "c4",
        "source_instance_hash": example.instance_hash,
        "checkpoint_id": "deepseek-r1-distill-qwen-1.5b",
        "task_family": example.family,
        "depth": example.depth,
        "reasoning_budget": budget,
        "token_counts": {"reasoning": budget},
        "reasoning_text": text,
        "forced_reasoning_transition": forced,
        "correct": False,
    }


def test_feasibility_boundary_and_correctness_are_independent(example):
    length = example.token_measurements["deepseek-r1-distill-qwen-1.5b"]["r3"]
    row = generation(example, budget=length, text=example.renderings["r3"])
    result = diagnose_budget(row, example)
    assert result["canonical_fits"]
    assert result["replay_valid"]
    assert not result["correct"]  # A correct trace does not imply a correct emitted answer.
    row["reasoning_budget"] -= 1
    row["token_counts"]["reasoning"] -= 1
    assert not diagnose_budget(row, example)["canonical_fits"]


def test_forced_fragment_preserves_prefix_but_not_complete_replay(example):
    first = example.renderings["r3"].splitlines()[0]
    row = generation(example, budget=16, text=first + "\ns2=", forced=True)
    result = diagnose_budget(row, example)
    assert result["verified_prefix_transitions"] == 1
    assert result["trailing_fragment_recovered"]
    assert not result["replay_valid"]
    assert not result["trace_parse_success"]
    row["forced_reasoning_transition"] = False
    assert diagnose_budget(row, example)["verified_prefix_transitions"] == 0


def test_errors_and_forcing_can_coexist(example):
    row = generation(example, budget=16, text="s1=5>div(5)>99", forced=True)
    result = diagnose_budget(row, example)
    assert result["forced"] and result["parsed_transition_error"]
    assert result["verified_prefix_transitions"] == 0


def test_reject_mismatched_instance_and_overflow(example):
    row = generation(example, budget=16, text="")
    row["source_instance_hash"] = "wrong"
    with pytest.raises(ValueError, match="canonical instance"):
        diagnose_budget(row, example)
    row["source_instance_hash"] = example.instance_hash
    row["token_counts"]["reasoning"] = 17
    with pytest.raises(ValueError, match="budget"):
        diagnose_budget(row, example)


def test_density_pools_tokens_instead_of_averaging_ratios(example):
    row = generation(example, budget=16, text="s1=5>div(5)>1", forced=True)
    first = diagnose_budget(row, example)
    second = {**first, "reasoning_tokens": 32, "verified_prefix_transitions": 0}
    assert summarize([first, second])["verified_transitions_per_token"] == 1 / 48


def test_manifest_tampering_and_path_escape_rejected(tmp_path):
    item = tmp_path / "items.jsonl"
    item.write_text("original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": {"items.jsonl": {"sha256": sha256_file(item)}}}))
    verify_manifest(tmp_path)
    item.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(tmp_path)
    manifest.write_text(json.dumps({"files": {"../outside": {"sha256": "bad"}}}))
    with pytest.raises(ValueError, match="escapes"):
        verify_manifest(tmp_path)


def test_cannot_write_into_frozen_checkout(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        analyze(tmp_path, "bridge", tmp_path / "output")


def test_primary_requires_completed_report_before_reading_test_examples(tmp_path):
    source = tmp_path / "source"
    (source / "data/m1").mkdir(parents=True)
    (source / "configs/m3").mkdir(parents=True)
    (source / "data/m1/manifest.json").write_text("{}")
    (source / "configs/m3/protocol.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="primary/report/manifest.json"):
        analyze(source, "primary", tmp_path / "output")


def _seal(root):
    manifest = {"schema_version": 1, "files": {
        str(path.relative_to(root)): {"sha256": sha256_file(path)}
        for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"
    }}
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def recovered_inputs(tmp_path, example):
    source = tmp_path / "source"
    dataset = source / "data/m1"
    (dataset / "splits").mkdir(parents=True)
    (dataset / "splits/test.jsonl").write_text(json.dumps(example.to_dict()) + "\n")
    _seal(dataset)
    protocol = {"dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
                "evaluation_budgets": [16]}
    config_dir = source / "configs/m3"
    config_dir.mkdir(parents=True)
    (config_dir / "protocol.json").write_text(json.dumps(protocol))
    freeze = source / "artifacts/m3/freeze"
    freeze.mkdir(parents=True)
    run = {"run_id": "test-run", "condition_id": "c4", "split": "test"}
    (freeze / "run_plan.json").write_text(json.dumps({
        "evaluation_runs": [run], "training_runs": [{"run_id": "train-run"}],
        "counts": {"generations": 1}, "run_plan_sha256": "plan-identity",
    }))
    (freeze / "freeze.json").write_text(json.dumps({
        "protocol": protocol, "dataset_manifest_sha256": protocol["dataset_manifest_sha256"],
    }))
    _seal(freeze)
    evaluation = source / "artifacts/m3/primary/evaluation"
    run_dir = evaluation / run["run_id"]
    run_dir.mkdir(parents=True)
    row = {**generation(example, budget=16, text=""), "item_id": example.id}
    (run_dir / "items.jsonl").write_text(json.dumps(row) + "\n")
    _seal(run_dir)
    report = tmp_path / "recovered-report"
    report.mkdir()
    (report / "input_summary.json").write_text(json.dumps({
        "freeze_manifest_sha256": sha256_file(freeze / "manifest.json"),
        "freeze_dir": str(freeze), "run_plan_sha256": "plan-identity",
        "evaluation_root": str(evaluation),
        "training_root": str(source / "artifacts/m3/primary/training"),
        "evaluation_run_count": 1, "training_run_count": 1,
        "generation_count": 1, "all_training_runs_passed": True,
    }))
    _seal(report)
    return source, report


def test_recovered_report_override_keeps_source_and_freeze_checks(recovered_inputs, tmp_path):
    source, report = recovered_inputs
    result = analyze(source, "primary", tmp_path / "output", primary_report=report)
    assert result["phase"] == "primary"
    assert len(result["rows"]) == 2
    verify_manifest(tmp_path / "output")
    assert not (source / "artifacts/m3/primary/report").exists()


@pytest.mark.parametrize("field,value,match", [
    ("freeze_manifest_sha256", "foreign", "different freeze"),
    ("run_plan_sha256", "foreign", "completed frozen matrix"),
    ("generation_count", 0, "completed frozen matrix"),
    ("evaluation_run_count", 0, "completed frozen matrix"),
    ("training_run_count", 0, "completed frozen matrix"),
    ("all_training_runs_passed", False, "completed frozen matrix"),
    ("evaluation_root", "/foreign", "different source artifacts"),
    ("training_root", "/foreign", "different source artifacts"),
    ("freeze_dir", "/foreign", "different source artifacts"),
])
def test_recovered_report_rejects_incomplete_or_foreign_inputs(
    recovered_inputs, tmp_path, field, value, match
):
    source, report = recovered_inputs
    path = report / "input_summary.json"
    inputs = json.loads(path.read_text())
    inputs[field] = value
    path.write_text(json.dumps(inputs))
    _seal(report)
    with pytest.raises(ValueError, match=match):
        analyze(source, "primary", tmp_path / "output", primary_report=report)
    assert not (tmp_path / "output").exists()


def test_recovered_report_rejects_failed_report_or_nested_output(recovered_inputs, tmp_path):
    source, report = recovered_inputs
    with pytest.raises(ValueError, match="outside the primary report"):
        analyze(source, "primary", report / "budget", primary_report=report)
    (report / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="failed artifact"):
        analyze(source, "primary", tmp_path / "output", primary_report=report)


def test_primary_report_override_is_not_a_bridge_option(tmp_path):
    with pytest.raises(ValueError, match="only valid for primary"):
        analyze(tmp_path / "source", "bridge", tmp_path / "out", primary_report=tmp_path / "report")
