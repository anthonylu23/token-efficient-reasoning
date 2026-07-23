"""Command-line interface for M1 controlled dataset workflows."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from token_efficient_reasoning.provenance import sha256_file

from .audit import audit_splits
from .config import M1Config
from .generation import generate_examples, load_dataset, write_dataset
from .reporting import compression_report
from .tokenization import load_pinned_tokenizers
from .verification import validate_examples


def _resolve_from(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repository_candidate = Path.cwd() / path
    if repository_candidate.exists() or path.parts[0] in {"artifacts", "configs", "data"}:
        return repository_candidate.resolve()
    return (config_path.parent / path).resolve()


def command_validate_config(args: argparse.Namespace) -> int:
    config = M1Config.from_json(args.config)
    print(json.dumps(config.to_dict(), indent=2))
    return 0


def _generate(config_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"dataset output already exists: {output}")
    config = M1Config.from_json(config_path)
    m0_path = _resolve_from(config_path, config.m0_config_path)
    tokenizers, provenance = load_pinned_tokenizers(config, m0_config_path=str(m0_path))
    by_split, rejections = generate_examples(config, tokenizers)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ter-m1-build-", dir=output.parent) as temporary:
        temporary_output = Path(temporary) / "dataset"
        summary = write_dataset(temporary_output, config, by_split, rejections, provenance)
        temporary_output.replace(output)
    summary["output_dir"] = str(output)
    return summary


def command_generate(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = M1Config.from_json(config_path)
    output = (
        Path(args.output).resolve()
        if args.output
        else _resolve_from(config_path, config.output_dir)
    )
    summary = _generate(config_path, output)
    print(json.dumps(summary, indent=2))
    return (
        0
        if all(summary[key] for key in ("validation_passed", "audit_passed", "compression_passed"))
        else 1
    )


def command_validate_dataset(args: argparse.Namespace) -> int:
    by_split = load_dataset(args.dataset)
    examples = [example for split in by_split.values() for example in split]
    metadata = json.loads((Path(args.dataset) / "resolved_config.json").read_text())
    validation = validate_examples(
        examples,
        require_tokens=True,
        required_compression=tuple(metadata["require_compression_for"]),
    )
    audit = audit_splits(by_split)
    report = compression_report(examples, metadata["minimum_r3_median_reduction"])
    result = {"validation": validation, "audit": audit, "compression": report}
    print(json.dumps(result, indent=2))
    return 0 if validation["passed"] and audit["passed"] and report["passed"] else 1


def command_regenerate_check(args: argparse.Namespace) -> int:
    existing = Path(args.dataset).resolve()
    config_path = Path(args.config).resolve()
    with tempfile.TemporaryDirectory(prefix="ter-m1-regenerate-") as temporary:
        regenerated = Path(temporary) / "dataset"
        _generate(config_path, regenerated)
        existing_manifest = json.loads((existing / "manifest.json").read_text())
        regenerated_manifest = json.loads((regenerated / "manifest.json").read_text())
        matching = existing_manifest == regenerated_manifest
        result = {
            "matching": matching,
            "existing_manifest_sha256": sha256_file(existing / "manifest.json"),
            "regenerated_manifest_sha256": sha256_file(regenerated / "manifest.json"),
        }
        print(json.dumps(result, indent=2))
        return 0 if matching else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ter-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config = subparsers.add_parser("validate-config")
    validate_config.add_argument("config")
    validate_config.set_defaults(function=command_validate_config)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output")
    generate.set_defaults(function=command_generate)

    validate_dataset = subparsers.add_parser("validate-dataset")
    validate_dataset.add_argument("dataset")
    validate_dataset.set_defaults(function=command_validate_dataset)

    regenerate = subparsers.add_parser("regenerate-check")
    regenerate.add_argument("--config", required=True)
    regenerate.add_argument("--dataset", required=True)
    regenerate.set_defaults(function=command_regenerate_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
