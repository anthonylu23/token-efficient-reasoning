from __future__ import annotations

import json
import runpy
import tarfile

import pytest

from token_efficient_reasoning.budget_analysis import verify_manifest
from token_efficient_reasoning.provenance import sha256_file
from token_efficient_reasoning.training.artifacts import tree_manifest


def test_results_archive_is_deterministic_complete_and_no_overwrite(tmp_path):
    archive = runpy.run_path("scripts/archive_m3_results.py")["archive"]
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    (freeze / "freeze.json").write_text("{}")
    tree_manifest(freeze, "freeze")
    recovery = tmp_path / "recovery"
    primary, budget = recovery / "primary", recovery / "budget"
    primary.mkdir(parents=True)
    budget.mkdir()
    (primary / "input_summary.json").write_text(
        json.dumps(
            {
                "freeze_manifest_sha256": sha256_file(freeze / "manifest.json"),
                "generation_count": 42,
            }
        )
    )
    (primary / "M3_REPORT.md").write_text("report")
    (primary / "per_problem_aggregates.json").write_text('{"large":true}')
    (budget / "REPORT.md").write_text("budget")
    tree_manifest(primary, "primary")
    tree_manifest(budget, "budget")
    tree_manifest(recovery, "recovery")
    a, b = tmp_path / "a", tmp_path / "b"
    first = archive(recovery, freeze, a)
    second = archive(recovery, freeze, b)
    assert first == second
    assert first["byte_for_byte_verified"]
    verify_manifest(a)
    with tarfile.open(a / first["archive"], "r:gz") as handle:
        assert "recovery/primary/per_problem_aggregates.json" in handle.getnames()
        assert "recovery/primary/manifest.json" in handle.getnames()
        assert "freeze/manifest.json" in handle.getnames()
    assert (a / "primary/M3_REPORT.md").read_text() == "report"
    assert not (a / "primary/per_problem_aggregates.json").exists()
    with pytest.raises(FileExistsError):
        archive(recovery, freeze, a)
    with pytest.raises(ValueError, match="outside"):
        archive(recovery, freeze, recovery / "nested")
    (primary / "M3_REPORT.md").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        archive(recovery, freeze, tmp_path / "bad")
