from __future__ import annotations

import json

import pytest

from token_efficient_reasoning.artifacts import REQUIRED_ITEM_FIELDS, RunWriter


def _record() -> dict[str, object]:
    record = {field: "value" for field in REQUIRED_ITEM_FIELDS}
    record.update(
        {
            "schema_version": 1,
            "run_id": "run",
            "correct": True,
            "forced_reasoning_transition": False,
            "prompt_token_ids": [1],
            "reasoning_raw_token_ids": [2],
            "answer_raw_token_ids": [3],
            "effective_token_ids": [1, 2, 3],
            "reasoning_token_ids": [2],
            "answer_token_ids": [3],
            "latency_seconds": {"total": 1.0},
        }
    )
    return record


def test_writer_rejects_missing_required_field(tmp_path) -> None:
    writer = RunWriter(tmp_path / "run", "run")
    record = _record()
    del record["parsed_answer"]

    with pytest.raises(ValueError, match="missing required fields"):
        writer.append_item(record)


def test_writer_creates_hashed_manifest(tmp_path) -> None:
    writer = RunWriter(tmp_path / "run", "run")
    writer.write_metadata("resolved_config.json", {"run_id": "run"})
    writer.append_item(_record())

    manifest = writer.finalize_manifest()

    assert set(manifest["files"]) == {"items.jsonl", "resolved_config.json"}
    saved = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
