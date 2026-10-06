from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from token_efficient_reasoning.primary.config import M3Config
from token_efficient_reasoning.primary.final_reporting import (
    EvaluationRun,
    PrimaryRecord,
    _aggregate_table,
    _matched_accuracy_reductions,
    _paired_comparisons,
    _per_problem_aggregates,
    _primary_success_summary,
    _records_by_cell,
    _reliable_depths,
    _select_m3_decision,
    _summarize_ordered_batches,
    _verified_density_comparison,
    _verify_tree_manifest,
    _write_markdown_report,
    _write_plots,
)
from token_efficient_reasoning.training.artifacts import tree_manifest


def _record(
    *,
    replicate: int,
    condition: str,
    split: str,
    item_id: str,
    depth: int,
    budget: int,
) -> PrimaryRecord:
    c4 = condition == "c4"
    reasoning_tokens = max(1, budget // 2) if c4 else budget
    return PrimaryRecord(
        run_id=f"m3-eval-r{replicate}-{condition}-{split}",
        replicate=replicate,
        condition_id=condition,
        split=split,
        item_id=item_id,
        task_family="arithmetic" if int(item_id.rsplit("-", 1)[1]) % 2 else "graph",
        depth=depth,
        reasoning_budget=budget,
        correct=c4,
        region_valid=True,
        trace_parse_success=c4,
        replay_valid=c4,
        verified_prefix_transitions=depth if c4 else 1,
        parsed_transition_count=depth if c4 else 1,
        invalid_transition_count=0,
        reasoning_tokens=reasoning_tokens,
        answer_tokens=2,
        sampled_tokens=reasoning_tokens + 3,
        forced_transition=not c4,
        reasoning_empty=False,
        prompt_copied=False,
        repeated_line=False,
        latency_seconds=0.1,
        generation_batch_size=4,
    )


def _matrix() -> tuple[M3Config, list[EvaluationRun]]:
    config = replace(
        M3Config.from_json("configs/m3/protocol.json"),
        bootstrap_samples=2_000,
    )
    runs = []
    for replicate in (1, 2, 3):
        for condition in ("c0", "c1", "c2", "c3", "c4"):
            for split in config.evaluation_splits:
                records = []
                for item_index in range(20):
                    depth = (
                        16
                        if split == "test_depth"
                        else (2, 4, 8, 12)[item_index % 4]
                    )
                    item_id = f"{split}-item-{item_index}"
                    for budget in config.evaluation_budgets:
                        records.append(
                            _record(
                                replicate=replicate,
                                condition=condition,
                                split=split,
                                item_id=item_id,
                                depth=depth,
                                budget=budget,
                            )
                        )
                runs.append(
                    EvaluationRun(
                        run_id=f"m3-eval-r{replicate}-{condition}-{split}",
                        replicate=replicate,
                        condition_id=condition,
                        split=split,
                        path="/unused",
                        records=tuple(records),
                        generation_count=len(records),
                        batch_count=len(records) // 4,
                        estimated_wall_seconds=1.0,
                        sampled_tokens=sum(record.sampled_tokens for record in records),
                        decode_tokens_per_second=100.0,
                        peak_allocated_bytes=1,
                        peak_reserved_bytes=2,
                        decoding_sha256="d" * 64,
                        tokenizer_sha256="t" * 64,
                    )
                )
    return config, runs


def test_complete_synthetic_matrix_supports_known_primary_success(
    tmp_path: Path,
) -> None:
    config, runs = _matrix()
    aggregate = _aggregate_table(runs)
    cells = _records_by_cell(runs)
    comparisons = _paired_comparisons(cells, config)
    per_problem = _per_problem_aggregates(cells)
    reliable = _reliable_depths(aggregate, config)
    matched = _matched_accuracy_reductions(aggregate, config)
    success = _primary_success_summary(comparisons, reliable, matched, config)
    decision = _select_m3_decision(comparisons, matched, reliable, config)

    primary = [row for row in comparisons if row["scope"] == "primary"]
    assert len(primary) == 8
    assert all(row["holm"]["reject_null"] for row in primary)
    assert all(row["paired_accuracy_difference"]["estimate"] == 1.0 for row in primary)
    h5 = [
        row
        for row in primary
        if row["primary_claim"] == "h5_structured_vs_telegraphic"
    ]
    assert len(h5) == 2
    assert {row["comparator"] for row in h5} == {"c3"}
    assert success["h5_supported"] is True
    c4_test_64 = next(
        row
        for row in per_problem["summary_rows"]
        if row["condition_id"] == "c4"
        and row["split"] == "test"
        and row["reasoning_budget"] == 64
    )
    assert c4_test_64["all_generations_correct_rate"] == 1.0
    assert success["primary_success"] is True
    assert decision["outcome"] == "continue_to_generalization"
    assert decision["exactly_one_outcome_selected"] is True

    plots = _write_plots(tmp_path, aggregate, config)
    assert len(plots) == 20
    assert all((tmp_path / name).is_file() for name in plots)
    report_path = tmp_path / "M3_REPORT.md"
    _write_markdown_report(
        report_path,
        success=success,
        decision=decision,
        comparisons=comparisons,
        systems={"evaluation_runs": [{"generation_count": 5_400}]},
        config=config,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "**Decision:** `continue_to_generalization`" in report
    assert "5,400" in report
    assert "## Engineering validity" in report
    assert "## Statistical evidence" in report
    assert "H5 (structured R3 versus telegraphic R2)" in report
    assert "eight-test Holm family" in report
    assert "## Scientific interpretation" in report


def test_verified_density_zero_control_rule_is_explicit_and_nontrivial() -> None:
    positive = _verified_density_comparison(0.25, 0.0, minimum_ratio=1.5)
    assert positive == {
        "ratio": None,
        "ratio_status": "positive_treatment_over_zero_control",
        "threshold_passed": True,
    }

    both_zero = _verified_density_comparison(0.0, 0.0, minimum_ratio=1.5)
    assert both_zero == {
        "ratio": None,
        "ratio_status": "both_zero",
        "threshold_passed": False,
    }

    finite = _verified_density_comparison(0.3, 0.2, minimum_ratio=1.5)
    assert finite["ratio"] == pytest.approx(1.5)
    assert finite["ratio_status"] == "finite"
    assert finite["threshold_passed"] is True


def test_zero_c0_boundary_is_diagnostic_while_primary_density_uses_c2() -> None:
    config, runs = _matrix()
    zero_c0_runs = [
        replace(
            run,
            records=tuple(
                replace(
                    record,
                    verified_prefix_transitions=0,
                    parsed_transition_count=0,
                )
                for record in run.records
            ),
        )
        if run.condition_id == "c0"
        else run
        for run in runs
    ]
    aggregate = _aggregate_table(zero_c0_runs)
    comparisons = _paired_comparisons(_records_by_cell(zero_c0_runs), config)
    success = _primary_success_summary(
        comparisons,
        _reliable_depths(aggregate, config),
        _matched_accuracy_reductions(aggregate, config),
        config,
    )

    c0_64 = next(
        row
        for row in comparisons
        if row["comparator"] == "c0"
        and row["split"] == config.primary_inference_split
        and row["reasoning_budget"] == 64
    )
    assert c0_64["verified_density_ratio"] is None
    assert (
        c0_64["verified_density_ratio_status"]
        == "positive_treatment_over_zero_control"
    )
    assert c0_64["verified_density_threshold_passed"] is True
    primary_64 = success["primary_budget_results"][0]
    assert primary_64["density_comparator"] == "c2"
    assert primary_64["density_ratio"] is not None
    assert primary_64["density_threshold_passed"] is True
    assert success["primary_success"] is True



def test_zero_c0_does_not_override_failed_c2_density_threshold() -> None:
    config, runs = _matrix()
    adjusted = []
    for run in runs:
        if run.condition_id == "c0":
            records = tuple(
                replace(record, verified_prefix_transitions=0, parsed_transition_count=0)
                for record in run.records
            )
        elif run.condition_id == "c2":
            records = tuple(
                replace(
                    record,
                    verified_prefix_transitions=record.depth,
                    parsed_transition_count=record.depth,
                )
                for record in run.records
            )
        elif run.condition_id == "c4":
            records = tuple(
                replace(record, verified_prefix_transitions=1, parsed_transition_count=1)
                for record in run.records
            )
        else:
            records = run.records
        adjusted.append(replace(run, records=records))

    aggregate = _aggregate_table(adjusted)
    comparisons = _paired_comparisons(_records_by_cell(adjusted), config)
    success = _primary_success_summary(
        comparisons,
        _reliable_depths(aggregate, config),
        _matched_accuracy_reductions(aggregate, config),
        config,
    )

    assert all(
        row["density_comparator"] == "c2"
        and row["density_threshold_passed"] is False
        for row in success["primary_budget_results"]
    )
    assert success["primary_success"] is False


def test_ordered_batch_wall_time_counts_shared_latency_once() -> None:
    rows = [
        {
            "generation_batch_size": 2,
            "reasoning_budget": 64,
            "latency_seconds": {"reasoning": 0.4, "answer": 0.1},
            "token_counts": {"sampled_total": 10},
        }
        for _ in range(4)
    ]

    batch_count, wall, tokens, throughput = _summarize_ordered_batches(rows)

    assert batch_count == 2
    assert wall == pytest.approx(1.0)
    assert tokens == 40
    assert throughput == pytest.approx(40.0)


def test_manifest_verifier_rejects_post_manifest_tampering(tmp_path: Path) -> None:
    (tmp_path / "value.json").write_text("{}\n", encoding="utf-8")
    tree_manifest(tmp_path, tmp_path.name)
    _verify_tree_manifest(tmp_path)

    (tmp_path / "value.json").write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_tree_manifest(tmp_path)


@pytest.mark.parametrize(
    "identity_keys", [("artifact_id",), ("run_id",), ("artifact_id", "run_id")]
)
def test_manifest_verifier_accepts_both_writer_schemas(tmp_path, identity_keys):
    from token_efficient_reasoning.artifacts import RunWriter

    (tmp_path / "value.json").write_text("{}\n")
    manifest = RunWriter(tmp_path, tmp_path.name, resume=True).finalize_manifest()
    manifest.pop("run_id")
    manifest.update({key: tmp_path.name for key in identity_keys})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    _verify_tree_manifest(tmp_path)
    assert path.read_bytes() == original  # Compatibility is read-only.
    (tmp_path / "value.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_tree_manifest(tmp_path)


@pytest.mark.parametrize("identities", [
    {}, {"run_id": "foreign"}, {"artifact_id": "foreign"},
    {"run_id": "correct", "artifact_id": "foreign"},
    {"artifact_id": "correct", "run_id": "foreign"},
    {"run_id": None},
])
def test_manifest_verifier_rejects_missing_foreign_or_conflicting_ids(tmp_path, identities):
    root = tmp_path / "correct"
    root.mkdir()
    (root / "value.json").write_text("{}\n")
    manifest = tree_manifest(root, root.name)
    manifest.pop("artifact_id")
    manifest.update(identities)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="artifact ID"):
        _verify_tree_manifest(root)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_evaluation_manifest_verifier_still_checks_complete_file_set(tmp_path, change):
    from token_efficient_reasoning.artifacts import RunWriter

    (tmp_path / "value.json").write_text("{}\n")
    RunWriter(tmp_path, tmp_path.name, resume=True).finalize_manifest()
    if change == "missing":
        (tmp_path / "value.json").unlink()
    else:
        (tmp_path / "extra.json").write_text("{}\n")
    with pytest.raises(ValueError, match="file set mismatch"):
        _verify_tree_manifest(tmp_path)
