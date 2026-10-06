from __future__ import annotations

import json
from dataclasses import replace

import pytest

from token_efficient_reasoning.composition import (
    CompositionConfig,
    audit,
    generate,
    prepare,
    reserved_pairs,
)
from token_efficient_reasoning.controlled.audit import value_signature
from token_efficient_reasoning.controlled.schema import load_examples
from token_efficient_reasoning.controlled.verification import verify_example


def config():
    return CompositionConfig.from_dict(
        {
            "master_seed": 20260929,
            "depths": [2, 4],
            "heldout_pairs": [["mul", "sub"], ["div", "add"]],
            "items_per_cell": {"train": 5, "validation": 2, "test_iid": 2, "test_composition": 2},
        }
    )


def test_deterministic_exact_quotas_shared_templates_and_partition():
    first, sampling = generate(config())
    second, repeated = generate(config())
    assert first == second
    assert sampling == repeated
    report = audit(first, config())
    assert report["passed"], report["errors"]
    assert report["cells"] == report["expected_cells"]
    assert report["coverage"]["test_composition"]["action_sequences_shared_with_train"] == 0
    assert report["training_primitives"] == ["add", "div", "mul", "sub"]
    assert report["verification"]["counterfactual_failures"] == []
    for split, examples in first.items():
        for example in examples:
            verify_example(example)
            assert bool(reserved_pairs(example, config())) == (split == "test_composition")
            for step in example.transitions:
                if step.action == "div":
                    assert step.before % step.argument == 0


def test_audit_rejects_heldout_pair_in_training():
    rows, _ = generate(config())
    rows["train"][0] = replace(rows["test_composition"][0], split="train")
    report = audit(rows, config())
    assert not report["passed"]
    assert any("reserved pair leaked" in error for error in report["errors"])


def test_semantic_deduplication_ignores_templates_and_distractors():
    rows, _ = generate(config())
    original = rows["train"][0]
    changed = replace(
        original,
        id="different-id",
        seed=1,
        split="test_iid",
        prompt="Different surface form",
        template_id="different-template",
        initial_state={**original.initial_state, "irrelevant_fact": {"count": 999, "label": "red"}},
    )
    assert value_signature(original) == value_signature(changed)
    rows["test_iid"][0] = changed
    report = audit(rows, config())
    assert not report["passed"]
    assert any("duplicate semantic" in error for error in report["errors"])


def test_audit_rejects_missing_cell_and_invalid_trace():
    rows, _ = generate(config())
    rows["test_composition"].pop()
    original = rows["validation"][0]
    rows["validation"][0] = replace(original, answer="999999")
    report = audit(rows, config())
    assert not report["passed"]
    assert "depth/template/pair quotas differ from configuration" in report["errors"]
    assert "semantic or counterfactual verification failed" in report["errors"]


def test_generation_exhaustion_is_bounded():
    with pytest.raises(RuntimeError, match="exhausted"):
        generate(replace(config(), max_attempts_per_cell=1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("depths", [1]),
        ("depths", [2, 2]),
        ("heldout_pairs", []),
        ("heldout_pairs", [["mul", "unknown"]]),
        ("heldout_pairs", [["mul", "sub"], ["mul", "sub"]]),
        ("items_per_cell", {"train": 1}),
        ("master_seed", True),
        ("unknown", 0),
    ],
)
def test_invalid_design_fails_closed(field, value):
    from dataclasses import asdict

    payload = asdict(config())
    payload[field] = value
    with pytest.raises(ValueError):
        CompositionConfig.from_dict(payload)


def test_artifact_roundtrip_and_no_overwrite(tmp_path):
    from dataclasses import asdict

    from token_efficient_reasoning.budget_analysis import verify_manifest

    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config())))
    output = tmp_path / "dataset"
    report = prepare(path, output)
    verify_manifest(output)
    rows = {
        split: load_examples(output / "splits" / f"{split}.jsonl")
        for split in config().items_per_cell
    }
    assert audit(rows, config()) == report
    with pytest.raises(FileExistsError):
        prepare(path, output)
