from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from token_efficient_reasoning.primary.config import M3Config, M3ConfigError
from token_efficient_reasoning.primary.freeze import create_freeze, validate_freeze
from token_efficient_reasoning.primary.planning import build_run_plan
from token_efficient_reasoning.primary.reporting import _protocol_compatibility
from token_efficient_reasoning.primary.runtime import resolve_runtime_batch_sizes
from token_efficient_reasoning.provenance import sha256_file
from token_efficient_reasoning.training.artifacts import tree_manifest

PROTOCOL_PATH = Path("configs/m3/protocol.json")
QLORA_PATH = Path("configs/m3/qlora.json")


def _load_protocol() -> dict[str, object]:
    with PROTOCOL_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_repository_m3_protocol_and_references_are_valid() -> None:
    config = M3Config.from_json(PROTOCOL_PATH)
    checkpoint, dataset_dir, split_paths = config.validate_references(PROTOCOL_PATH)

    assert checkpoint.id == "deepseek-r1-distill-qwen-1.5b"
    assert dataset_dir == Path("data/m1").resolve()
    assert set(split_paths) == {
        "train",
        "validation",
        "test",
        "test_depth",
        "test_compositional",
    }
    assert config.primary_budgets == (64, 128)
    assert config.evaluation_batch_size == 4
    assert config.evaluation_batch_sizes == (4, 4, 4, 2, 2, 2)
    assert config.batch_size_for_budget(256) == 2
    assert config.optimizer_updates == 500
    assert config.analysis_version == "m3-primary-v2"
    assert config.primary_inference_split == "test"
    assert config.central_primary_comparators == ("c0", "c1", "c2")
    assert config.primary_comparators == ("c0", "c1", "c2", "c3")
    assert config.h5_comparator == "c3"
    assert config.primary_density_comparator == "c2"
    assert config.secondary_comparators == ()
    assert config.bootstrap_unit == "item_id"
    assert config.familywise_alpha == 0.05
    assert config.multiplicity_method == "holm"


def test_partial_budget_batch_override_covers_only_selected_budget() -> None:
    config = M3Config.from_json(PROTOCOL_PATH)

    frozen, resolved = resolve_runtime_batch_sizes(config, (512,), 2)

    assert frozen == {512: 2}
    assert resolved == {512: 2}


def test_m3_requires_three_paired_unique_seed_blocks() -> None:
    data = _load_protocol()
    data["training_seeds"] = [1, 2]
    data["decoding_seeds"] = [3, 4]

    with pytest.raises(M3ConfigError, match="at least three"):
        M3Config.from_dict(data)

    data = _load_protocol()
    data["decoding_seeds"] = [3, 4]
    with pytest.raises(M3ConfigError, match="one-to-one"):
        M3Config.from_dict(data)


def test_m3_requires_all_held_out_splits_and_fixed_checkpoint_rule() -> None:
    data = _load_protocol()
    data["evaluation_splits"] = ["test", "test_depth"]
    with pytest.raises(M3ConfigError, match="test_compositional"):
        M3Config.from_dict(data)

    data = _load_protocol()
    data["checkpoint_selection"] = "best_validation_accuracy"
    with pytest.raises(M3ConfigError, match="fixed_final_update"):
        M3Config.from_dict(data)


def test_m3_rejects_unfrozen_primary_analysis_choices() -> None:
    data = _load_protocol()
    data["primary_comparators"] = ["c0", "c2"]
    with pytest.raises(M3ConfigError, match="primary_comparators"):
        M3Config.from_dict(data)

    data = _load_protocol()
    data["multiplicity_method"] = "none"
    with pytest.raises(M3ConfigError, match="multiplicity_method"):
        M3Config.from_dict(data)

    data = _load_protocol()
    data["h5_comparator"] = "c2"
    with pytest.raises(M3ConfigError, match="h5_comparator"):
        M3Config.from_dict(data)

    data = _load_protocol()
    data["verified_density_zero_control_rule"] = "add_epsilon"
    with pytest.raises(M3ConfigError, match="verified_density_zero_control_rule"):
        M3Config.from_dict(data)

    data = _load_protocol()
    data["primary_density_comparator"] = "c0"
    with pytest.raises(M3ConfigError, match="primary_density_comparator"):
        M3Config.from_dict(data)


def test_bridge_protocol_compatibility_allows_only_analysis_contract_additions() -> None:
    current = _load_protocol()
    bridge = {
        key: value
        for key, value in current.items()
        if key
        not in {
            "analysis_version",
            "primary_inference_split",
            "central_primary_comparators",
            "primary_comparators",
            "h5_comparator",
            "secondary_comparators",
            "bootstrap_unit",
            "familywise_alpha",
            "multiplicity_method",
            "verified_density_zero_control_rule",
            "primary_density_comparator",
        }
    }

    compatible = _protocol_compatibility(bridge, current)
    assert compatible["passed"] is True
    assert compatible["behavioral_changes"] == []

    tuple_backed_current = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in current.items()
    }
    compatible = _protocol_compatibility(current, tuple_backed_current)
    assert compatible["passed"] is True
    assert compatible["changed_fields"] == {}

    bridge["evaluation_batch_size"] = 1
    incompatible = _protocol_compatibility(bridge, current)
    assert incompatible["passed"] is False
    assert incompatible["behavioral_changes"] == ["evaluation_batch_size"]


def test_run_plan_contains_nine_training_and_45_evaluation_runs() -> None:
    plan = build_run_plan(M3Config.from_json(PROTOCOL_PATH))

    assert plan["counts"] == {
        "training_runs": 9,
        "evaluation_runs": 45,
        "condition_replicates": 15,
        "evaluation_cells": 270,
        "generations": None,
    }
    assert len({run["run_id"] for run in plan["training_runs"]}) == 9
    assert len({run["run_spec_sha256"] for run in plan["evaluation_runs"]}) == 45
    trained_eval = next(
        run
        for run in plan["evaluation_runs"]
        if run["replicate"] == 2
        and run["condition_id"] == "c4"
        and run["split"] == "test_depth"
    )
    assert trained_eval["adapter_run_id"] == "m3-train-r2-c4-seed271828"
    assert trained_eval["decoding_seed"] == 2718


def test_repository_readiness_passes_after_bound_human_approval_and_bridge() -> None:
    readiness = M3Config.from_json(PROTOCOL_PATH).readiness(PROTOCOL_PATH)

    assert readiness["m1_human_review_passed"] is True
    assert readiness["m1_review_sample_matches"] is True
    assert readiness["m1_human_review_sha256"] == sha256_file(
        "artifacts/M1_HUMAN_REVIEW.json"
    )
    assert readiness["bridge_gate_sha256"] == sha256_file(
        "artifacts/m3/bridge/report/gate_summary.json"
    )
    assert readiness["bridge_protocol_compatible"] is True
    assert readiness["bridge_gate_passed"] is True
    assert readiness["ready_to_freeze"] is True
    assert readiness["bridge_gate_path"] == str(
        Path("artifacts/m3/bridge/report/gate_summary.json").resolve()
    )


def test_freeze_requires_prerequisites_and_detects_tampering(tmp_path: Path) -> None:
    data = _load_protocol()
    human_path = tmp_path / "human.json"
    bridge_dir = tmp_path / "bridge_report"
    bridge_dir.mkdir()
    bridge_path = bridge_dir / "gate_summary.json"
    protocol_path = tmp_path / "protocol.json"
    human_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "decision": "accept",
                "reviewer": "Repository owner",
                "reviewed_at": "2026-07-27T00:00:00Z",
                "approval_reference": "test-fixture",
                "dataset_manifest_sha256": data["dataset_manifest_sha256"],
                "review_sample_sha256": (
                    "5bce54eb70026ad1dc375228bbcffc5131dc697d185b7131bbb19bb0fde7752d"
                ),
                "reviewed_item_count": 34,
                "failed_item_ids": [],
            }
        ),
        encoding="utf-8",
    )
    bridge_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "test_data_inspected": False,
                "protocol_compatibility_passed": True,
                "report_protocol_sha256": "",
            }
        ),
        encoding="utf-8",
    )
    data["dataset_dir"] = str(Path("data/m1").resolve())
    data["m0_config_path"] = str(Path("configs/m0.json").resolve())
    data["m1_human_review_path"] = str(human_path)
    data["bridge_gate_path"] = str(bridge_path)
    protocol_path.write_text(json.dumps(data), encoding="utf-8")
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    bridge["report_protocol_sha256"] = sha256_file(protocol_path)
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    tree_manifest(bridge_dir, "bridge-report")

    output = tmp_path / "freeze"
    result = create_freeze(protocol_path, QLORA_PATH, output)
    validated = validate_freeze(output)

    assert result["training_runs"] == 9
    assert result["evaluation_runs"] == 45
    assert result["generations"] == 342_000
    assert validated["schema_version"] == 2
    assert validated["prerequisite_evidence_bound"] is True
    assert validated["prerequisite_semantics_validated"] is True
    assert validated["test_access_authorized"] is True
    frozen = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    evidence = frozen["prerequisite_evidence"]
    assert sha256_file(output / evidence["m1_human_review"]["frozen_path"]) == evidence[
        "m1_human_review"
    ]["sha256"]
    assert sha256_file(output / evidence["m1_review_sample"]["frozen_path"]) == evidence[
        "m1_review_sample"
    ]["sha256"]
    assert sha256_file(output / evidence["bridge_gate"]["frozen_path"]) == evidence[
        "bridge_gate"
    ]["sha256"]

    frozen_human_path = output / evidence["m1_human_review"]["frozen_path"]
    frozen_human_bytes = frozen_human_path.read_bytes()
    frozen_human_path.write_bytes(frozen_human_bytes + b" ")
    with pytest.raises(M3ConfigError, match="hash mismatch"):
        validate_freeze(output)
    frozen_human_path.write_bytes(frozen_human_bytes)

    bridge_path.write_text(bridge_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(M3ConfigError, match="bridge report file hash mismatch"):
        create_freeze(protocol_path, QLORA_PATH, tmp_path / "second-freeze")
    assert not (tmp_path / "second-freeze").exists()

    malicious_human = tmp_path / "malicious-human-freeze"
    shutil.copytree(output, malicious_human)
    malicious_frozen_path = malicious_human / "freeze.json"
    malicious_frozen = json.loads(malicious_frozen_path.read_text(encoding="utf-8"))
    malicious_human_path = malicious_human / malicious_frozen["prerequisite_evidence"][
        "m1_human_review"
    ]["frozen_path"]
    malicious_review = json.loads(malicious_human_path.read_text(encoding="utf-8"))
    malicious_review["decision"] = "reject"
    malicious_human_path.write_text(json.dumps(malicious_review), encoding="utf-8")
    malicious_frozen["prerequisite_evidence"]["m1_human_review"]["sha256"] = sha256_file(
        malicious_human_path
    )
    malicious_frozen_path.write_text(json.dumps(malicious_frozen), encoding="utf-8")
    tree_manifest(malicious_human, malicious_human.name)
    with pytest.raises(M3ConfigError, match="human-review semantics"):
        validate_freeze(malicious_human)

    malicious_gate = tmp_path / "malicious-gate-freeze"
    shutil.copytree(output, malicious_gate)
    malicious_frozen_path = malicious_gate / "freeze.json"
    malicious_frozen = json.loads(malicious_frozen_path.read_text(encoding="utf-8"))
    malicious_gate_path = malicious_gate / malicious_frozen["prerequisite_evidence"][
        "bridge_gate"
    ]["frozen_path"]
    malicious_gate_value = json.loads(malicious_gate_path.read_text(encoding="utf-8"))
    malicious_gate_value["test_data_inspected"] = True
    malicious_gate_path.write_text(json.dumps(malicious_gate_value), encoding="utf-8")
    report_manifest_path = malicious_gate / malicious_frozen["prerequisite_evidence"][
        "bridge_report_manifest"
    ]["frozen_path"]
    report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
    report_manifest["files"]["gate_summary.json"] = {
        "sha256": sha256_file(malicious_gate_path),
        "size_bytes": malicious_gate_path.stat().st_size,
    }
    report_manifest_path.write_text(json.dumps(report_manifest), encoding="utf-8")
    malicious_frozen["prerequisite_evidence"]["bridge_gate"]["sha256"] = sha256_file(
        malicious_gate_path
    )
    malicious_frozen["prerequisite_evidence"]["bridge_report_manifest"][
        "sha256"
    ] = sha256_file(report_manifest_path)
    malicious_frozen_path.write_text(json.dumps(malicious_frozen), encoding="utf-8")
    tree_manifest(malicious_gate, malicious_gate.name)
    with pytest.raises(M3ConfigError, match="bridge-gate semantics"):
        validate_freeze(malicious_gate)

    plan_path = output / "run_plan.json"
    plan_path.write_text(plan_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(M3ConfigError, match="hash mismatch"):
        validate_freeze(output)
