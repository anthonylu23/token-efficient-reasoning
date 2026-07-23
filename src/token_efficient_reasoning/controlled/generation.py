"""Deterministic corpus generation and immutable artifact writing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.provenance import sha256_file

from .audit import audit_markdown, audit_splits, value_signature
from .config import M1Config
from .families import FAMILIES
from .rendering import render_example
from .reporting import compression_markdown, compression_report
from .schema import CanonicalExample, stable_hash, stable_json_bytes
from .tokenization import compression_rejection_reason, measure_example
from .verification import validate_examples, verify_example


def derived_seed(master_seed: int, split: str, family: str, depth: int, attempt_index: int) -> int:
    payload = stable_json_bytes([master_seed, split, family, depth, attempt_index])
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def generate_examples(
    config: M1Config, tokenizers: dict[str, Any]
) -> tuple[dict[str, list[CanonicalExample]], list[dict[str, Any]]]:
    unknown_families = sorted(set(config.families) - FAMILIES.keys())
    if unknown_families:
        raise ValueError(f"unknown M1 task families: {unknown_families}")
    by_split: dict[str, list[CanonicalExample]] = {}
    rejections: list[dict[str, Any]] = []
    accepted_seeds: set[int] = set()
    accepted_instances: set[str] = set()
    accepted_value_signatures: set[str] = set()
    for split in config.splits:
        split_examples: list[CanonicalExample] = []
        for family_name in config.families:
            family = FAMILIES[family_name]
            for depth in split.depths:
                accepted = 0
                attempt = 0
                maximum_attempts = split.items_per_family_depth * 100
                while accepted < split.items_per_family_depth:
                    if attempt >= maximum_attempts:
                        raise RuntimeError(
                            f"could not generate enough accepted {family_name} items at depth "
                            f"{depth} for {split.name}"
                        )
                    seed = derived_seed(config.master_seed, split.name, family_name, depth, attempt)
                    attempt += 1
                    if seed in accepted_seeds:
                        rejections.append(
                            {
                                "split": split.name,
                                "family": family_name,
                                "depth": depth,
                                "seed": seed,
                                "reason": "derived seed collision",
                            }
                        )
                        continue
                    example = family.generate(seed=seed, depth=depth, split=split.name)
                    example = render_example(example)
                    example = measure_example(example, tokenizers)
                    reason = compression_rejection_reason(example, config.require_compression_for)
                    if reason is None and example.instance_hash in accepted_instances:
                        reason = "semantic instance collision"
                    signature = value_signature(example)
                    if reason is None and signature in accepted_value_signatures:
                        reason = "value signature collision"
                    if reason is not None:
                        rejections.append(
                            {
                                "split": split.name,
                                "family": family_name,
                                "depth": depth,
                                "seed": seed,
                                "item_id": example.id,
                                "reason": reason,
                            }
                        )
                        continue
                    verify_example(example, require_tokens=True)
                    split_examples.append(example)
                    accepted_seeds.add(seed)
                    accepted_instances.add(example.instance_hash)
                    accepted_value_signatures.add(signature)
                    accepted += 1
        by_split[split.name] = sorted(split_examples, key=lambda example: example.id)
    return by_split, rejections


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(stable_json_bytes(value).decode() + "\n")


def _review_sample(by_split: dict[str, list[CanonicalExample]]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for split, examples in sorted(by_split.items()):
        strata: dict[tuple[str, int], CanonicalExample] = {}
        for example in examples:
            strata.setdefault((example.family, example.depth), example)
        for example in strata.values():
            selected.append(
                {
                    "item_id": example.id,
                    "split": split,
                    "family": example.family,
                    "depth": example.depth,
                    "prompt": example.prompt,
                    "answer": example.answer,
                    "renderings": example.renderings,
                    "review_status": "pending_human_review",
                    "review_notes": "",
                }
            )
    return {
        "schema_version": 1,
        "selection": "lexicographically first item per split, family, and depth",
        "items": selected,
    }


def write_dataset(
    output_dir: str | Path,
    config: M1Config,
    by_split: dict[str, list[CanonicalExample]],
    rejections: list[dict[str, Any]],
    tokenizer_provenance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    all_examples = [example for examples in by_split.values() for example in examples]
    for split, examples in sorted(by_split.items()):
        records = [example.to_dict() for example in examples]
        data_path = root / "splits" / f"{split}.jsonl"
        _write_jsonl(data_path, records)
        split_manifest = {
            "schema_version": 1,
            "dataset_name": config.dataset_name,
            "generator_version": config.generator_version,
            "split": split,
            "item_count": len(examples),
            "data_file": f"splits/{split}.jsonl",
            "data_sha256": sha256_file(data_path),
            "items": [
                {
                    "id": example.id,
                    "seed": example.seed,
                    "family": example.family,
                    "depth": example.depth,
                    "instance_hash": example.instance_hash,
                    "record_hash": stable_hash(example.to_dict()),
                }
                for example in examples
            ],
        }
        atomic_write_json(root / "manifests" / f"{split}.json", split_manifest)

    _write_jsonl(root / "rejections.jsonl", rejections)
    validation = validate_examples(
        all_examples,
        require_tokens=True,
        required_compression=config.require_compression_for,
    )
    audit = audit_splits(by_split)
    report = compression_report(all_examples, config.minimum_r3_median_reduction)
    atomic_write_json(root / "resolved_config.json", config.to_dict())
    atomic_write_json(
        root / "metadata.json",
        {
            "schema_version": 1,
            "dataset_name": config.dataset_name,
            "generator_version": config.generator_version,
            "item_count": len(all_examples),
            "split_counts": {split: len(examples) for split, examples in sorted(by_split.items())},
            "tokenizers": tokenizer_provenance,
            "rejection_count": len(rejections),
        },
    )
    atomic_write_json(root / "validation.json", validation)
    atomic_write_json(root / "split_audit.json", audit)
    (root / "split_audit.md").write_text(audit_markdown(audit), encoding="utf-8")
    atomic_write_json(root / "compression_report.json", report)
    (root / "compression_report.md").write_text(compression_markdown(report), encoding="utf-8")
    atomic_write_json(root / "review_sample.json", _review_sample(by_split))
    gate = {
        "schema_version": 1,
        "validation_passed": validation["passed"],
        "split_audit_passed": audit["passed"],
        "compression_passed": report["passed"],
        "human_review_complete": False,
        "passed": False,
    }
    atomic_write_json(root / "gate_summary.json", gate)

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            relative = path.relative_to(root).as_posix()
            files[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "schema_version": 1,
        "dataset_name": config.dataset_name,
        "generator_version": config.generator_version,
        "files": files,
    }
    atomic_write_json(root / "manifest.json", manifest)
    return {
        "item_count": len(all_examples),
        "rejection_count": len(rejections),
        "validation_passed": validation["passed"],
        "audit_passed": audit["passed"],
        "compression_passed": report["passed"],
        "output_dir": str(root.resolve()),
    }


def load_dataset(root: str | Path) -> dict[str, list[CanonicalExample]]:
    from .schema import load_examples

    dataset_root = Path(root)
    paths = sorted((dataset_root / "splits").glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no split files found under {dataset_root}")
    return {path.stem: load_examples(path) for path in paths}
