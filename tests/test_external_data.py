from __future__ import annotations

import json
from decimal import Decimal

import pytest

from token_efficient_reasoning.budget_analysis import verify_manifest
from token_efficient_reasoning.data import load_items
from token_efficient_reasoning.external_data import convert_svamp, prepare_svamp
from token_efficient_reasoning.provenance import sha256_file
from token_efficient_reasoning.scoring import score_answer


def fixture_row(**updates):
    return {
        "ID": "fixture-1",
        "Body": "A container holds 7.5 liters.",
        "Question": "How much remains after using 2 liters?",
        "Answer": Decimal("5.5"),
        "Equation": "GOLD_EQUATION_SENTINEL",
        "Type": "GOLD_TYPE_SENTINEL",
        **updates,
    }


def test_only_question_fields_enter_prompt_and_ids_order_survive():
    items = convert_svamp([fixture_row(ID="b"), fixture_row(ID="a")])
    assert [item.id for item in items] == ["b", "a"]
    assert items[0].prompt == (
        "A container holds 7.5 liters.\nHow much remains after using 2 liters?"
    )
    assert "GOLD" not in items[0].prompt
    assert "5.5" not in items[0].prompt
    assert items[0].reference_answer == "5.5"
    assert items[0].answer_type == "decimal"
    assert score_answer("5.50", "5.5", items[0].answer_type)[0]
    assert not score_answer("5", "5.5", items[0].answer_type)[0]


@pytest.mark.parametrize(
    ("answer", "reference"),
    [
        (Decimal("1E-7"), "0.0000001"),
        (Decimal("-0.0"), "0"),
        (12, "12"),
        (Decimal("120.00"), "120"),
        ("-2.50", "-2.5"),
    ],
)
def test_decimal_references_are_lossless(answer, reference):
    assert convert_svamp([fixture_row(Answer=answer)])[0].reference_answer == reference


@pytest.mark.parametrize("answer", [True, None, [], "five", "NaN", "Infinity"])
def test_invalid_answers_rejected(answer):
    with pytest.raises(ValueError):
        convert_svamp([fixture_row(Answer=answer)])


def test_duplicate_ids_and_empty_questions_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        convert_svamp([fixture_row(), fixture_row()])
    with pytest.raises(ValueError, match="Question"):
        convert_svamp([fixture_row(Question=" ")])
    with pytest.raises(ValueError, match="nonempty"):
        convert_svamp([])


def test_prepare_roundtrip_hash_binding_and_no_overwrite(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([fixture_row(Answer=5.5)]))
    license_path = tmp_path / "LICENSE"
    license_path.write_text("Artificial test license; not upstream data")
    config = tmp_path / "sources.json"
    config.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "svamp",
                        "revision": "a" * 40,
                        "data_sha256": sha256_file(raw),
                        "evaluation_item_count": 1,
                    }
                ]
            }
        )
    )
    output = tmp_path / "prepared"
    report = prepare_svamp(raw, license_path, config, output)
    assert report["item_count"] == 1
    assert report["model_evaluation_performed"] is False
    verify_manifest(output)
    assert load_items(output / "items.jsonl")[0].reference_answer == "5.5"
    with pytest.raises(FileExistsError):
        prepare_svamp(raw, license_path, config, output)
    raw.write_text("[]")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_svamp(raw, license_path, config, tmp_path / "bad-output")
    assert not (tmp_path / "bad-output").exists()


def test_wrong_item_count_rejected_before_creating_output(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([fixture_row(Answer=5.5)]))
    license_path = tmp_path / "LICENSE"
    license_path.write_text("fixture")
    config = tmp_path / "sources.json"
    config.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "name": "svamp",
                        "revision": "a" * 40,
                        "data_sha256": sha256_file(raw),
                        "evaluation_item_count": 2,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="item count"):
        prepare_svamp(raw, license_path, config, tmp_path / "output")
    assert not (tmp_path / "output").exists()
