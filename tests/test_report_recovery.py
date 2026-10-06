from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from token_efficient_reasoning import budget_analysis
from token_efficient_reasoning.primary import final_reporting


def recovery_script():
    return runpy.run_path(str(Path("scripts/recover_m3_reports.py").resolve()))["recover"]


def test_recovery_rejects_frozen_or_existing_output(tmp_path):
    recover = recovery_script()
    with pytest.raises(ValueError, match="outside"):
        recover(tmp_path, tmp_path / "report")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        recover(tmp_path / "source", existing)


@pytest.mark.parametrize("mutate_input", [False, True])
def test_recovery_records_provenance_preserves_inputs_and_restores_cwd(
    tmp_path, monkeypatch, mutate_input
):
    recover = recovery_script()
    source = tmp_path / "source"
    output = tmp_path / "recovery"
    freeze = source / "artifacts/m3/freeze"
    freeze.mkdir(parents=True)
    (freeze / "run_plan.json").write_text(json.dumps({
        "training_runs": [], "evaluation_runs": [],
    }))
    (freeze / "manifest.json").write_text("{}")
    registry = source / "artifacts/m3/experiment_registry.jsonl"
    registry.write_text("original registry")
    original_report = source / "artifacts/m3/primary/report"
    original_report.mkdir(parents=True)
    failure = original_report / "failure.json"
    failure.write_text('{"error":"original failure"}')
    frozen_reader = source / "src/token_efficient_reasoning/primary/final_reporting.py"
    frozen_reader.parent.mkdir(parents=True)
    frozen_reader.write_text("# original frozen reader\n")
    monkeypatch.setitem(recover.__globals__, "validate_frozen_workspace", lambda _: {
        "workspace_matches_current": True, "manifest_sha256": "fixture",
    })
    def build(*args):
        assert Path.cwd() == source
        report = args[-1]
        report.mkdir()
        (report / "manifest.json").write_text("{}")
        return {"generation_count": 1}
    def analyze(source_root, phase, destination, *, primary_report):
        assert source_root == source and phase == "primary"
        assert primary_report == output / "primary"
        destination.mkdir()
        (destination / "manifest.json").write_text("{}")
        if mutate_input:
            registry.write_text("changed")
        return {"rows": [{}]}
    monkeypatch.setattr(final_reporting, "build_primary_report", build)
    monkeypatch.setattr(budget_analysis, "analyze", analyze)
    monkeypatch.setattr(budget_analysis, "verify_manifest", lambda _: "fixture")
    cwd = Path.cwd()
    if mutate_input:
        with pytest.raises(ValueError, match="changed during recovery"):
            recover(source, output)
        assert (output / "failure.json").is_file()
        assert not (output / "manifest.json").exists()
    else:
        result = recover(source, output)
        assert result["source_artifacts_unchanged"]
        assert result["model_execution"] is False
        assert result["supplemental_cells"] == 1
        assert (output / "implementation/recover_m3_reports.py").is_file()
        assert (output / "report_reader.patch").read_text()
        assert (output / "manifest.json").is_file()
        assert registry.read_text() == "original registry"
    assert failure.read_text() == '{"error":"original failure"}'
    assert Path.cwd() == cwd
