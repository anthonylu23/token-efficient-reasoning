"""Append-only lifecycle registry for M3 run attempts."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_STATUSES = {"planned", "running", "complete", "failed", "interrupted", "excluded"}


def append_registry_event(
    registry_path: str | Path,
    *,
    run_id: str,
    run_type: str,
    phase: str,
    status: str,
    path: str | Path,
    run_spec_sha256: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append and fsync one immutable lifecycle event."""

    if not run_id or not run_type or not phase:
        raise ValueError("run_id, run_type, and phase must be non-empty")
    if status not in _STATUSES:
        raise ValueError(f"unsupported registry status: {status!r}")
    record = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "run_type": run_type,
        "phase": phase,
        "status": status,
        "path": str(Path(path).resolve()),
        "run_spec_sha256": run_spec_sha256,
        "details": details or {},
    }
    destination = Path(registry_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def load_registry(registry_path: str | Path) -> list[dict[str, Any]]:
    path = Path(registry_path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
