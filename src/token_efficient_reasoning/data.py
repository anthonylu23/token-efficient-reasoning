"""M0 development item loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationItem:
    id: str
    prompt: str
    reference_answer: str
    answer_type: str
    task_family: str

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EvaluationItem:
        required = {"id", "prompt", "reference_answer", "answer_type", "task_family"}
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"evaluation item missing fields: {', '.join(missing)}")
        item = cls(**data)  # type: ignore[arg-type]
        if item.answer_type not in {"exact", "integer", "decimal"}:
            raise ValueError(f"unsupported answer_type {item.answer_type!r} for {item.id}")
        return item


def load_items(path: str | Path) -> list[EvaluationItem]:
    items: list[EvaluationItem] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = EvaluationItem.from_dict(json.loads(line))
            if item.id in seen:
                raise ValueError(f"duplicate item id {item.id!r} on line {line_number}")
            seen.add(item.id)
            items.append(item)
    if not items:
        raise ValueError(f"no evaluation items found in {path}")
    return items
