"""Fail-closed run artifact writing and manifest generation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REQUIRED_ITEM_FIELDS = {
    "schema_version",
    "run_id",
    "item_id",
    "checkpoint_id",
    "condition_id",
    "reasoning_budget",
    "answer_allowance",
    "rendered_prompt",
    "prompt_token_ids",
    "reasoning_raw_token_ids",
    "answer_raw_token_ids",
    "effective_token_ids",
    "reasoning_token_ids",
    "answer_token_ids",
    "reasoning_text",
    "parsed_answer",
    "parse_status",
    "forced_reasoning_transition",
    "correct",
    "latency_seconds",
}


def validate_item_record(record: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_ITEM_FIELDS - record.keys())
    if missing:
        raise ValueError(f"item record missing required fields: {', '.join(missing)}")
    null_fields = sorted(key for key in REQUIRED_ITEM_FIELDS if record[key] is None)
    if null_fields:
        raise ValueError(f"required item fields must not be null: {', '.join(null_fields)}")


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RunWriter:
    def __init__(self, run_dir: str | Path, run_id: str, *, resume: bool = False) -> None:
        self.run_dir = Path(run_dir)
        if resume:
            if not self.run_dir.is_dir():
                raise ValueError(f"cannot resume missing run directory: {self.run_dir}")
        else:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        self.run_id = run_id
        self.items_path = self.run_dir / "items.jsonl"

    def write_metadata(self, name: str, value: Any) -> None:
        atomic_write_json(self.run_dir / name, value)

    def append_item(self, record: dict[str, Any]) -> None:
        validate_item_record(record)
        if record["run_id"] != self.run_id:
            raise ValueError("item run_id does not match writer run_id")
        with self.items_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def finalize_manifest(self) -> dict[str, Any]:
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(self.run_dir.iterdir()):
            if not path.is_file() or path.name == "manifest.json":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[path.name] = {"sha256": digest, "size_bytes": path.stat().st_size}
        manifest = {"schema_version": 1, "run_id": self.run_id, "files": files}
        atomic_write_json(self.run_dir / "manifest.json", manifest)
        return manifest
