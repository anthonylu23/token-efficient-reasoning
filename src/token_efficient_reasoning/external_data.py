"""CPU-only, hash-bound conversion of external evaluation data."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .data import EvaluationItem, load_items
from .provenance import sha256_file


def convert_svamp(rows: list[dict[str, Any]]) -> list[EvaluationItem]:
    """Preserve upstream order/IDs and expose only Body and Question in prompts."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("SVAMP data must be a nonempty JSON array")
    items = []
    seen = set()
    for row in rows:
        required = {"ID", "Body", "Question", "Answer"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("SVAMP row lacks required ID/Body/Question/Answer fields")
        for field in ("ID", "Body", "Question"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"SVAMP {field} must be a nonempty string")
        item_id = row["ID"]
        if item_id in seen:
            raise ValueError(f"duplicate SVAMP ID: {item_id}")
        seen.add(item_id)
        answer = row["Answer"]
        if isinstance(answer, bool) or not isinstance(answer, (str, int, float, Decimal)):
            raise ValueError(f"SVAMP answer must be numeric: {item_id}")
        try:
            number = Decimal(str(answer))
        except InvalidOperation as error:
            raise ValueError(f"invalid numeric SVAMP answer: {item_id}") from error
        if not number.is_finite():
            raise ValueError(f"SVAMP answer must be finite: {item_id}")
        reference = format(number, "f")
        if "." in reference:
            reference = reference.rstrip("0").rstrip(".")
        if number == 0:
            reference = "0"
        items.append(
            EvaluationItem(
                id=item_id,
                prompt=f"{row['Body'].strip()}\n{row['Question'].strip()}",
                reference_answer=reference,
                answer_type="decimal",
                task_family="svamp",
            )
        )
    return items


def prepare_svamp(
    raw_data: Path,
    license_path: Path,
    sources_config: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate all input before creating an immutable, evaluator-compatible artifact."""
    pins = json.loads(sources_config.read_text())
    sources = [source for source in pins["sources"] if source["name"] == "svamp"]
    if len(sources) != 1:
        raise ValueError("source configuration must contain exactly one SVAMP source")
    source = sources[0]
    if re.fullmatch(r"[0-9a-f]{40}", source["revision"]) is None:
        raise ValueError("SVAMP source must pin a full commit revision")
    if sha256_file(raw_data) != source["data_sha256"]:
        raise ValueError("SVAMP raw data does not match pinned SHA-256")
    if not license_path.read_text().strip():
        raise ValueError("SVAMP upstream license is empty")
    rows = json.loads(raw_data.read_text(), parse_float=Decimal)
    items = convert_svamp(rows)
    if len(items) != source["evaluation_item_count"]:
        raise ValueError("SVAMP item count does not match source configuration")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(raw_data, output / "SVAMP.raw.json")
    shutil.copyfile(license_path, output / "LICENSE.upstream")
    with (output / "items.jsonl").open("x", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), sort_keys=True, ensure_ascii=False) + "\n")
    if load_items(output / "items.jsonl") != items:
        raise ValueError("converted SVAMP items failed evaluator round-trip validation")
    provenance = {
        "schema_version": 1,
        "dataset": "svamp",
        "source": source,
        "sources_config_sha256": sha256_file(sources_config),
        "converter_source_sha256": sha256_file(Path(__file__)),
        "item_count": len(items),
        "prompt_fields": ["Body", "Question"],
        "prompt_join": "strip each field; join with one newline",
        "id_policy": "upstream IDs and ordering preserved",
        "reference_policy": "finite decimal; no equation execution or answer correction",
        "split_role": "evaluation only; not prompt development or training",
        "model_evaluation_performed": False,
        "limitations": [
            "Upstream reference answers are preserved, not independently adjudicated.",
            "Public benchmark contamination cannot be excluded.",
            "This artifact is data preparation, not a frozen evaluation protocol.",
        ],
    }
    atomic_write_json(output / "provenance.json", provenance)
    atomic_write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "files": {
                name: {"sha256": sha256_file(output / name)}
                for name in ("SVAMP.raw.json", "LICENSE.upstream", "items.jsonl", "provenance.json")
            },
        },
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True, type=Path)
    parser.add_argument("--license", required=True, type=Path, dest="license_path")
    parser.add_argument(
        "--sources-config", type=Path, default=Path("configs/followup/external_sources.json")
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_svamp(args.raw_data, args.license_path, args.sources_config, args.output_dir)
    print(json.dumps({"dataset": result["dataset"], "item_count": result["item_count"]}))


if __name__ == "__main__":
    main()
