"""Environment, repository, and hardware provenance capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(command: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def collect_repository_state(repo_root: str | Path) -> dict[str, Any]:
    status = _command(["git", "status", "--porcelain"], repo_root)
    diff = _command(["git", "diff", "--binary", "HEAD"], repo_root)
    listed = _command(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], repo_root
    )
    root = Path(repo_root)
    if listed is None:
        excluded_parts = {
            ".git",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            "__pycache__",
            "artifacts",
            "htmlcov",
        }
        workspace_files = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if (path.is_file() or path.is_symlink())
            and not excluded_parts.intersection(path.relative_to(root).parts)
            and path.name != ".coverage"
            and path.suffix not in {".pyc", ".pyo"}
        )
    else:
        workspace_files = sorted(
            path for path in listed.split("\0") if path and not path.startswith("artifacts/")
        )
    workspace_digest = hashlib.sha256()
    for relative in workspace_files:
        path = root / relative
        workspace_digest.update(relative.encode())
        workspace_digest.update(b"\0")
        if path.is_symlink():
            workspace_digest.update(os.readlink(path).encode())
        elif path.is_file():
            workspace_digest.update(path.read_bytes())
        workspace_digest.update(b"\0")
    return {
        "commit": _command(["git", "rev-parse", "HEAD"], repo_root),
        "branch": _command(["git", "branch", "--show-current"], repo_root),
        "dirty": bool(status),
        "status_porcelain": status,
        "tracked_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
        "workspace_file_count": len(workspace_files),
        "workspace_sha256": workspace_digest.hexdigest(),
    }


def collect_environment() -> dict[str, Any]:
    packages = {}
    for name in ("accelerate", "numpy", "psutil", "safetensors", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "environment": {
            key: os.environ.get(key)
            for key in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")
        },
    }


def collect_hardware() -> dict[str, Any]:
    virtual_memory = psutil.virtual_memory()
    disk = psutil.disk_usage(Path.cwd().anchor)
    snapshot: dict[str, Any] = {
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "memory_total_bytes": virtual_memory.total,
        "memory_available_bytes": virtual_memory.available,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }
    query = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    snapshot["nvidia_smi"] = query
    return snapshot


def stable_item_seed(base_seed: int, item_id: str, budget: int) -> int:
    payload = json.dumps([base_seed, item_id, budget], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
