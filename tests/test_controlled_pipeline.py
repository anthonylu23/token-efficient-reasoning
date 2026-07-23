from __future__ import annotations

from dataclasses import replace

from token_efficient_reasoning.controlled.adapters import to_evaluation_item
from token_efficient_reasoning.controlled.audit import audit_splits
from token_efficient_reasoning.controlled.config import M1Config
from token_efficient_reasoning.controlled.generation import (
    derived_seed,
    generate_examples,
    load_dataset,
    write_dataset,
)
from token_efficient_reasoning.controlled.reporting import compression_report, percentile
from token_efficient_reasoning.controlled.schema import CanonicalExample, stable_hash
from token_efficient_reasoning.controlled.verification import validate_examples


class WordTokenizer:
    name_or_path = "word-fixture"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        assert not add_special_tokens
        return text.split()


def small_config() -> M1Config:
    return M1Config.from_dict(
        {
            "schema_version": 1,
            "dataset_name": "test-controlled",
            "generator_version": "test",
            "master_seed": 42,
            "output_dir": "unused",
            "m0_config_path": "configs/m0.json",
            "tokenizer_checkpoint_ids": ["fixture"],
            "families": ["multi_operation_arithmetic", "directed_graph_traversal"],
            "splits": [
                {"name": "train", "depths": [2, 4], "items_per_family_depth": 2},
                {
                    "name": "test_compositional",
                    "depths": [2, 4],
                    "items_per_family_depth": 1,
                },
                {"name": "test_depth", "depths": [8], "items_per_family_depth": 1},
            ],
            "minimum_r3_median_reduction": 0.3,
            "require_compression_for": ["r2", "r3"],
        }
    )


def test_generation_is_deterministic_and_passes_all_gates(tmp_path) -> None:
    config = small_config()
    tokenizers = {"fixture": WordTokenizer()}
    first, first_rejections = generate_examples(config, tokenizers)
    second, second_rejections = generate_examples(config, tokenizers)

    assert {
        split: [example.to_dict() for example in examples] for split, examples in first.items()
    } == {split: [example.to_dict() for example in examples] for split, examples in second.items()}
    assert first_rejections == second_rejections
    examples = [example for split in first.values() for example in split]
    assert validate_examples(examples, require_tokens=True)["passed"]
    assert audit_splits(first)["passed"]
    assert compression_report(examples, 0.3)["passed"]

    output = tmp_path / "dataset"
    summary = write_dataset(
        output,
        config,
        first,
        first_rejections,
        {"fixture": {"model_id": "fixture", "tokenizer_revision": "0" * 40}},
    )
    loaded = load_dataset(output)
    assert summary["item_count"] == len(examples)
    assert [item.to_dict() for item in loaded["train"]] == [
        item.to_dict() for item in first["train"]
    ]
    assert (output / "manifest.json").is_file()
    assert (output / "review_sample.json").is_file()


def test_audit_detects_cross_split_instance_overlap() -> None:
    config = small_config()
    generated, _ = generate_examples(config, {"fixture": WordTokenizer()})
    duplicate = replace(generated["train"][0], split="test_depth")
    broken = {"train": generated["train"], "test_depth": [duplicate]}

    audit = audit_splits(broken)

    assert not audit["passed"]
    assert audit["overlap"]["instances"]
    assert audit["overlap"]["seeds"]


def test_stable_hash_ignores_mapping_order_and_seed_namespaces_differ() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert derived_seed(1, "train", "family", 2, 0) == derived_seed(1, "train", "family", 2, 0)
    assert derived_seed(1, "train", "family", 2, 0) != derived_seed(1, "test", "family", 2, 0)


def test_schema_rejects_tampered_semantic_payload() -> None:
    config = small_config()
    generated, _ = generate_examples(config, {"fixture": WordTokenizer()})
    example = generated["train"][0]
    data = example.to_dict()
    data["answer"] = "tampered"

    try:
        CanonicalExample.from_dict(data)
    except ValueError as error:
        assert "instance hash" in str(error)
    else:
        raise AssertionError("tampered item was accepted")


def test_percentile_interpolates_small_samples() -> None:
    assert percentile([0.0, 10.0], 0.1) == 1.0


def test_controlled_example_adapts_to_m0_evaluator() -> None:
    generated, _ = generate_examples(small_config(), {"fixture": WordTokenizer()})
    example = generated["train"][0]

    evaluation_item = to_evaluation_item(example)

    assert evaluation_item.id == example.id
    assert evaluation_item.prompt == example.prompt
    assert evaluation_item.reference_answer == example.answer
