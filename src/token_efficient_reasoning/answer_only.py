"""Prepare matched answer-only SFT targets on CPU, outside the frozen M3 experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .budget_analysis import verify_manifest
from .config import ExperimentConfig
from .controlled.schema import load_examples, stable_json_bytes
from .primary.config import M3Config
from .provenance import sha256_file
from .training.artifacts import tree_manifest
from .training.config import QLoRAConfig
from .training.data import (
    ANSWER_ONLY_CONDITION,
    ANSWER_ONLY_REPRESENTATION,
    audit_matched_examples,
    build_training_examples,
    summarize_token_exposure,
    tokenize_training_example,
)


@dataclass(frozen=True)
class FollowupCondition:
    id: str
    representation: str


@dataclass(frozen=True)
class AnswerOnlyProtocol:
    experiment_name: str
    checkpoint_id: str
    train_split: str
    validation_split: str
    conditions: tuple[FollowupCondition, ...]
    training_seeds: tuple[int, ...]
    optimizer_updates: int
    source_protocol_sha256: str
    target_format_version: str = "empty-reasoning-answer-v1"
    inference_family: str = "exploratory-followup-not-m3"

    @classmethod
    def from_primary(cls, primary: M3Config, name: str, source_hash: str) -> AnswerOnlyProtocol:
        return cls(
            experiment_name=name,
            checkpoint_id=primary.checkpoint_id,
            train_split=primary.train_split,
            validation_split=primary.validation_split,
            conditions=(
                *(FollowupCondition(c.id, c.representation) for c in primary.conditions),
                FollowupCondition(ANSWER_ONLY_CONDITION, ANSWER_ONLY_REPRESENTATION),
            ),
            training_seeds=primary.training_seeds,
            optimizer_updates=primary.optimizer_updates,
            source_protocol_sha256=source_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare(config_path: Path, repository: Path, output: Path, *, tokenizer: Any = None) -> dict:
    """Materialize and audit all four conditions without loading model weights."""
    config = json.loads(config_path.read_text())
    required = {
        "schema_version",
        "experiment_name",
        "source_protocol",
        "source_protocol_sha256",
        "qlora",
        "qlora_sha256",
        "m0_sha256",
    }
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("unsupported answer-only preparation configuration")
    repository = repository.resolve()
    protocol_path = repository / config["source_protocol"]
    qlora_path = repository / config["qlora"]
    for path, digest in (
        (protocol_path, config["source_protocol_sha256"]),
        (qlora_path, config["qlora_sha256"]),
    ):
        if sha256_file(path) != digest:
            raise ValueError(f"source configuration hash mismatch: {path}")
    primary = M3Config.from_json(protocol_path)
    m0_path = repository / primary.m0_config_path
    if sha256_file(m0_path) != config["m0_sha256"]:
        raise ValueError("checkpoint configuration hash mismatch")
    checkpoint = next(
        c for c in ExperimentConfig.from_json(m0_path).checkpoints if c.id == primary.checkpoint_id
    )
    qlora = QLoRAConfig.from_json(qlora_path)
    dataset = repository / primary.dataset_dir
    if verify_manifest(dataset) != primary.dataset_manifest_sha256:
        raise ValueError("dataset does not match the primary training corpus")
    protocol = AnswerOnlyProtocol.from_primary(
        primary, config["experiment_name"], config["source_protocol_sha256"]
    )
    if tokenizer is None:
        from .training.runtime import load_tokenizer

        tokenizer = load_tokenizer(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    try:
        for split in (protocol.train_split, protocol.validation_split):
            source = dataset / "splits" / f"{split}.jsonl"
            rows = build_training_examples(load_examples(source), protocol, checkpoint)
            tokenized = {
                condition: [
                    tokenize_training_example(
                        row, tokenizer, checkpoint, max_sequence_length=qlora.max_sequence_length
                    )
                    for row in examples
                ]
                for condition, examples in rows.items()
            }
            reports[split] = {
                "matching": audit_matched_examples(rows),
                "exposure": summarize_token_exposure(tokenized),
                "source_split_sha256": sha256_file(source),
            }
            for condition, examples in rows.items():
                path = output / "targets" / condition / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as handle:
                    for row in examples:
                        handle.write(stable_json_bytes(row.to_dict()).decode() + "\n")
            # Release each split's token sequences before loading the next split.
            del tokenized, rows
        report = {
            "schema_version": 1,
            "protocol": protocol.to_dict(),
            "qlora": qlora.to_dict(),
            "seed_policy": "replace QLoRA seed with each declared training seed",
            "checkpoint": asdict(checkpoint),
            "config_sha256": sha256_file(config_path),
            "dataset_manifest_sha256": primary.dataset_manifest_sha256,
            "source_sha256": {
                "answer_only.py": sha256_file(Path(__file__)),
                "training/data.py": sha256_file(Path(__file__).parent / "training/data.py"),
            },
            "splits": reports,
            "model_weights_loaded": False,
            "training_performed": False,
            "trace_scoring_policy": "absent trace is not an invalid attempted proof",
            "launch_status": "requires a separate post-M3 training launch and evaluation freeze",
        }
        atomic_write_json(output / "preparation.json", report)
        tree_manifest(output, protocol.experiment_name)
        return report
    except Exception as error:
        atomic_write_json(output / "failure.json", {"error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/followup/answer_only.json"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.config, args.repository, args.output_dir)
    print(json.dumps({"experiment": result["protocol"]["experiment_name"], "prepared": True}))


if __name__ == "__main__":
    main()
