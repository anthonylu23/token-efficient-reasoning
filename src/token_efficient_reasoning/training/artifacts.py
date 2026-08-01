"""Recursive manifests and structured failure capture for M2 artifacts."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.provenance import sha256_file


def tree_manifest(root: str | Path, artifact_id: str) -> dict[str, Any]:
    """Hash every regular file below an artifact root except the manifest itself."""

    artifact_root = Path(root)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(artifact_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(artifact_root).as_posix()
        files[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = {"schema_version": 1, "artifact_id": artifact_id, "files": files}
    atomic_write_json(artifact_root / "manifest.json", manifest)
    return manifest


def write_failure(root: str | Path, error: BaseException, **context: Any) -> None:
    """Retain a machine-readable record for an unsuccessful or interrupted run."""

    atomic_write_json(
        Path(root) / "failure.json",
        {
            "schema_version": 1,
            **context,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        },
    )
