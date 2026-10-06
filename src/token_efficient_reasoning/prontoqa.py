"""Pinned PrOntoQA v1 generation and leakage-safe conversion in an isolated CPU environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .artifacts import atomic_write_json
from .data import EvaluationItem, load_items
from .provenance import sha256_file
from .training.artifacts import tree_manifest, write_failure

SOURCE_FILES = (
    "run_experiment.py",
    "theory.py",
    "syntax.py",
    "proof.py",
    "prompt.py",
    "fol.py",
    "bad_patterns.txt",
    "LICENSE",
)
DEPENDENCIES = ("numpy==2.4.6", "scipy==1.16.3")
ROLE = "development-only-not-evaluation-frozen"


def validate_config(config: dict) -> None:
    if set(config) != {"schema_version", "role", "seed", "hops", "items_per_hop"}:
        raise ValueError("unknown or missing PrOntoQA configuration fields")
    if config["schema_version"] != 1 or config["role"] != ROLE:
        raise ValueError("only version 1 development generation is supported")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("PrOntoQA seed must be an unsigned 32-bit integer")
    hops = config["hops"]
    if (
        not isinstance(hops, list)
        or not hops
        or any(type(h) is not int or not 1 <= h <= 5 for h in hops)
        or len(set(hops)) != len(hops)
    ):
        raise ValueError("hops must be unique integers from 1 through 5")
    if type(config["items_per_hop"]) is not int or config["items_per_hop"] <= 0:
        raise ValueError("items_per_hop must be a positive integer")


def convert_prontoqa(
    raw: dict, *, hops: int, seed: int, expected_count: int
) -> list[EvaluationItem]:
    """Ignore demonstrations and gold proofs; expose question and query only."""
    if not isinstance(raw, dict) or not raw or len(raw) != expected_count:
        raise ValueError("PrOntoQA count differs from the requested generation count")
    items = []
    seen_prompts = set()
    for source_id, entry in raw.items():
        if re.fullmatch(r"example[1-9][0-9]*", source_id) is None or not isinstance(entry, dict):
            raise ValueError("invalid upstream PrOntoQA example")
        if any(
            key != "test_example" and not re.fullmatch(r"in_context_example[0-9]+", key)
            for key in entry
        ):
            raise ValueError("unexpected field or model output in PrOntoQA example")
        row = entry.get("test_example")
        if not isinstance(row, dict) or set(row) != {
            "question",
            "query",
            "chain_of_thought",
            "answer",
        }:
            raise ValueError(
                "expected raw PrOntoQA test example, not model outputs or proof-only data"
            )
        for field in ("question", "query"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"PrOntoQA {field} must be a nonempty string")
        if row["answer"] not in ("True", "False"):
            raise ValueError("PrOntoQA answer must be exactly True or False")
        if not isinstance(row["chain_of_thought"], list) or not all(
            isinstance(step, str) and step.strip() for step in row["chain_of_thought"]
        ):
            raise ValueError("invalid upstream proof metadata")
        prompt = row["question"].strip() + "\n" + row["query"].strip()
        if prompt in seen_prompts:
            raise ValueError("duplicate PrOntoQA question/query")
        seen_prompts.add(prompt)
        items.append(
            EvaluationItem(
                id=f"prontoqa-v1-h{hops}-s{seed}-{source_id}",
                prompt=prompt,
                reference_answer=row["answer"],
                answer_type="exact",
                task_family="prontoqa-v1",
            )
        )
    return items


def verified_source(checkout: Path, revision: str) -> dict[str, bytes]:
    """Compare every executed file (and license) with its pinned Git object."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("PrOntoQA revision must be a full commit hash")
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != revision:
        raise ValueError("PrOntoQA checkout HEAD differs from source pin")
    result = {}
    for name in SOURCE_FILES:
        pinned = subprocess.check_output(["git", "-C", str(checkout), "show", f"{revision}:{name}"])
        if (checkout / name).read_bytes() != pinned:
            raise ValueError(f"modified PrOntoQA source: {name}")
        result[name] = pinned
    return result


def generation_command(source: Path, config: dict, hops: int) -> list[str]:
    """No model backend, demonstrations, or implicit dependency installation into the project."""
    validate_config(config)
    if hops not in config["hops"]:
        raise ValueError("hop count is not configured")
    return [
        "uv",
        "run",
        "--no-project",
        "--isolated",
        "--python",
        sys.executable,
        "--with",
        DEPENDENCIES[0],
        "--with",
        DEPENDENCIES[1],
        str(source / "run_experiment.py"),
        "--model-name",
        "json",
        "--model-size",
        "none",
        "--few-shot-examples",
        "0",
        "--num-trials",
        str(config["items_per_hop"]),
        "--ontology",
        "fictional",
        "--ordering",
        "postorder",
        "--min-hops",
        str(hops),
        "--max-hops",
        str(hops),
        "--hops-skip",
        "1",
        "--test-hops-diff",
        "0",
        "--repetitions-per-test",
        "1",
        "--deduction-rule",
        "ModusPonens",
        "--proof-width",
        "2",
        "--prompting",
        "COT",
        "--seed",
        str(config["seed"]),
    ]


def prepare(checkout: Path, sources_path: Path, config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text())
    validate_config(config)
    pins = json.loads(sources_path.read_text())
    sources = [s for s in pins["sources"] if s["name"] == "prontoqa-v1"]
    if len(sources) != 1:
        raise ValueError("expected exactly one pinned PrOntoQA source")
    source = sources[0]
    files = verified_source(checkout.resolve(), source["revision"])
    if shutil.which("uv") is None:
        raise ValueError("uv is required for isolated CPU generation")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "upstream"
    source_dir.mkdir()
    for name, contents in files.items():
        (source_dir / name).write_bytes(contents)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    # Do not inherit local Python modules into the isolated generator.
    environment.pop("PYTHONPATH", None)
    commands = []
    items: list[EvaluationItem] = []
    item_metadata = []
    coverage = {}
    try:
        for hops in config["hops"]:
            work = output / "raw" / f"h{hops}"
            work.mkdir(parents=True)
            shutil.copyfile(source_dir / "bad_patterns.txt", work / "bad_patterns.txt")
            command = generation_command(source_dir, config, hops)
            commands.append({"hops": hops, "argv": command, "cwd": str(work)})
            with (work / "generation.log").open("x") as log:
                subprocess.run(
                    command,
                    cwd=work,
                    env=environment,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                )
            seed_suffix = f"_seed{config['seed']}" if config["seed"] != 62471893 else ""
            raw_path = work / f"{hops}hop{seed_suffix}.json"
            batch = convert_prontoqa(
                json.loads(raw_path.read_text()),
                hops=hops,
                seed=config["seed"],
                expected_count=config["items_per_hop"],
            )
            items.extend(batch)
            item_metadata.extend(
                {
                    "id": item.id,
                    "upstream_hops": hops,
                    "upstream_seed": config["seed"],
                    "source_file": str(raw_path.relative_to(output)),
                }
                for item in batch
            )
            coverage[str(hops)] = {
                "item_count": len(batch),
                "labels": dict(Counter(item.reference_answer for item in batch)),
                "raw_sha256": sha256_file(raw_path),
            }
        if len({item.prompt for item in items}) != len(items):
            raise ValueError("duplicate PrOntoQA prompt across hop files")
        with (output / "items.jsonl").open("x", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(asdict(item), sort_keys=True, ensure_ascii=False) + "\n")
        if load_items(output / "items.jsonl") != items:
            raise ValueError("PrOntoQA evaluator round-trip failed")
        atomic_write_json(output / "item_metadata.json", {"items": item_metadata})
        atomic_write_json(output / "resolved_config.json", config)
        provenance = {
            "schema_version": 1,
            "dataset": "prontoqa-v1",
            "role": config["role"],
            "source": source,
            "sources_config_sha256": sha256_file(sources_path),
            "config_sha256": sha256_file(config_path),
            "commands": commands,
            "upstream_files": {
                name: hashlib.sha256(data).hexdigest() for name, data in files.items()
            },
            "converter_source_sha256": sha256_file(Path(__file__)),
            "dependencies": list(DEPENDENCIES),
            "python_version": sys.version,
            "uv_version": subprocess.check_output(["uv", "--version"], text=True).strip(),
            "environment_overrides": {
                key: environment[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONHASHSEED",
                    "PYTHONDONTWRITEBYTECODE",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            "coverage": coverage,
            "item_count": len(items),
            "prompt_fields": ["test_example.question", "test_example.query"],
            "model_evaluation_performed": False,
            "limitations": [
                "Upstream proofs are archived, excluded from prompts, not independently verified.",
                "Hops are upstream parameters, not synthetic verifier transition counts.",
                "Development fixtures only; use separate frozen seeds for later evaluation.",
            ],
        }
        atomic_write_json(output / "provenance.json", provenance)
        tree_manifest(output, "prontoqa-v1-development")
        return provenance
    except Exception as error:
        write_failure(output, error, commands=commands)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--sources-config", type=Path, default=Path("configs/followup/external_sources.json")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/followup/prontoqa_dev.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.checkout, args.sources_config, args.config, args.output_dir)
    print(json.dumps({"dataset": result["dataset"], "item_count": result["item_count"]}))


if __name__ == "__main__":
    main()
