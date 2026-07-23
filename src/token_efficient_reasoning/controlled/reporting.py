"""Compression statistics and concise Markdown reporting."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .schema import REPRESENTATIONS, CanonicalExample


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def compression_report(examples: list[CanonicalExample], threshold: float) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int, str], list[tuple[int, float]]] = defaultdict(list)
    overall_r3: dict[str, list[float]] = defaultdict(list)
    for example in examples:
        for checkpoint_id, measurements in example.token_measurements.items():
            r0 = measurements["r0"]
            for representation in REPRESENTATIONS:
                count = measurements[representation]
                reduction = (r0 - count) / r0
                grouped[(checkpoint_id, example.family, example.depth, representation)].append(
                    (count, reduction)
                )
                if representation == "r3":
                    overall_r3[checkpoint_id].append(reduction)
    rows: list[dict[str, Any]] = []
    for (checkpoint_id, family, depth, representation), values in sorted(grouped.items()):
        counts = [float(count) for count, _ in values]
        reductions = [reduction for _, reduction in values]
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "family": family,
                "depth": depth,
                "representation": representation,
                "item_count": len(values),
                "token_count": {
                    "p10": percentile(counts, 0.10),
                    "median": statistics.median(counts),
                    "p90": percentile(counts, 0.90),
                },
                "reduction_from_r0": {
                    "p10": percentile(reductions, 0.10),
                    "median": statistics.median(reductions),
                    "p90": percentile(reductions, 0.90),
                },
            }
        )
    overall = {
        checkpoint_id: {
            "r3_median_reduction": statistics.median(values),
            "threshold": threshold,
            "passed": statistics.median(values) >= threshold,
        }
        for checkpoint_id, values in sorted(overall_r3.items())
    }
    return {
        "schema_version": 1,
        "item_count": len(examples),
        "overall": overall,
        "rows": rows,
        "passed": bool(overall) and all(item["passed"] for item in overall.values()),
    }


def compression_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M1 tokenizer compression report",
        "",
        f"Items: {report['item_count']}",
        "",
        "## Overall R3 gate",
        "",
        "| Tokenizer | Median reduction | Threshold | Result |",
        "|---|---:|---:|---|",
    ]
    for checkpoint_id, result in report["overall"].items():
        lines.append(
            f"| {checkpoint_id} | {result['r3_median_reduction']:.1%} | "
            f"{result['threshold']:.1%} | {'pass' if result['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## By family, depth, and representation",
            "",
            "| Tokenizer | Family | Depth | Rep. | Tokens p10 / median / p90 | "
            "Reduction p10 / median / p90 |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in report["rows"]:
        counts = row["token_count"]
        reductions = row["reduction_from_r0"]
        lines.append(
            f"| {row['checkpoint_id']} | {row['family']} | {row['depth']} | "
            f"{row['representation'].upper()} | {counts['p10']:.1f} / "
            f"{counts['median']:.1f} / {counts['p90']:.1f} | {reductions['p10']:.1%} / "
            f"{reductions['median']:.1%} / {reductions['p90']:.1%} |"
        )
    return "\n".join(lines) + "\n"
