#!/usr/bin/env python3
"""Package verified M3 reports for Git without raw generations or model checkpoints."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.budget_analysis import verify_manifest
from token_efficient_reasoning.provenance import sha256_file
from token_efficient_reasoning.training.artifacts import tree_manifest


def archive(recovery: Path, freeze: Path, output: Path) -> dict:
    recovery, freeze, output = (path.resolve() for path in (recovery, freeze, output))
    if output.is_relative_to(recovery) or output.is_relative_to(freeze):
        raise ValueError("publication must be outside the immutable input bundles")
    hashes = {
        "recovery": verify_manifest(recovery),
        "freeze": verify_manifest(freeze),
        "primary": verify_manifest(recovery / "primary"),
        "budget": verify_manifest(recovery / "budget"),
    }
    inputs = json.loads((recovery / "primary/input_summary.json").read_text())
    if inputs["freeze_manifest_sha256"] != hashes["freeze"]:
        raise ValueError("report and freeze identities differ")
    files = {}
    for label, root in (("recovery", recovery), ("freeze", freeze)):
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"archive input contains a symlink: {path}")
            if path.is_file():
                files[f"{label}/{path.relative_to(root).as_posix()}"] = path
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "reports-and-freeze.tar.gz"
    original_hashes = {name: sha256_file(path) for name, path in files.items()}
    with (
        archive_path.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as bundle,
    ):
        for name, path in files.items():
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            info.mode = 0o644
            with path.open("rb") as handle:
                bundle.addfile(info, handle)
    with tarfile.open(archive_path, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError("archive member set differs from input bundles")
        for member in members:
            with bundle.extractfile(member) as handle:
                if (
                    hashlib.file_digest(handle, "sha256").hexdigest()
                    != original_hashes[member.name]
                ):
                    raise ValueError(f"archive bytes differ from source: {member.name}")
    # Browseable views are exact copies, but not complete independent source bundles.
    archive_only = {
        "per_problem_aggregates.json",
        "aggregate.json",
        "aggregate.csv",
        "manifest.json",
    }
    for name in ("primary", "budget"):
        for path in sorted((recovery / name).iterdir()):
            if path.is_file() and path.name not in archive_only:
                destination = output / name / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    publication = {
        "schema_version": 1,
        "archive": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "archive_members": len(files),
        "source_manifest_sha256": hashes,
        "archive_contents": "Full recovered reports, implementation snapshot, and original freeze",
        "byte_for_byte_verified": True,
        "excluded": [
            "raw generation runs",
            "training runs and model weights",
            "development corpora",
        ],
        "readable_copies": "primary/ and budget/; large aggregate tables are inside the archive",
        "generation_count": inputs["generation_count"],
    }
    atomic_write_json(output / "publication.json", publication)
    tree_manifest(output, output.name)
    return publication


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.recovery_dir, args.freeze_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
