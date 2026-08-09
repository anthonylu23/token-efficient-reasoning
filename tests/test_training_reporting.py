from __future__ import annotations

import pytest

from token_efficient_reasoning.training.reporting import (
    _aggregate_group,
    _shallow_comparisons,
)


def _row(
    *,
    item_id: str,
    condition: str,
    depth: int,
    budget: int,
    correct: bool,
    reasoning_tokens: int = 4,
    verified_transitions: int = 2,
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "condition_id": condition,
        "depth": depth,
        "reasoning_budget": budget,
        "correct": correct,
        "parse_status": "valid",
        "forced_reasoning_transition": False,
        "token_counts": {"reasoning": reasoning_tokens},
        "latency_seconds": {"total": 0.25},
        "trace_diagnostics": {
            "trace_parse_success": True,
            "replay_valid": True,
            "verified_prefix_transitions": verified_transitions,
            "reasoning_empty": False,
            "prompt_copied": False,
            "repeated_line_count": 0,
        },
    }


def test_aggregate_group_reports_reasoning_density_and_adherence() -> None:
    rows = [
        _row(
            item_id="first",
            condition="c4",
            depth=2,
            budget=16,
            correct=True,
        ),
        _row(
            item_id="second",
            condition="c4",
            depth=2,
            budget=16,
            correct=False,
            reasoning_tokens=6,
            verified_transitions=1,
        ),
    ]

    aggregate = _aggregate_group(rows)

    assert aggregate["generation_count"] == 2
    assert aggregate["accuracy"] == 0.5
    assert aggregate["region_parse_success_rate"] == 1.0
    assert aggregate["budget_violation_count"] == 0
    assert aggregate["verified_transitions"] == 3
    assert aggregate["verified_transitions_per_reasoning_token"] == pytest.approx(0.3)
    assert aggregate["median_reasoning_tokens"] == 5
    assert aggregate["invalid_transition_event_rate"] == 0.0


def test_aggregate_group_treats_historical_parse_failure_as_invalid_event() -> None:
    row = _row(
        item_id="malformed",
        condition="c4",
        depth=2,
        budget=16,
        correct=False,
    )
    # Old saved rows have no explicit event and recorded a zero transition count.
    row["trace_diagnostics"].update(
        trace_parse_success=False,
        invalid_transition_count=0,
    )

    aggregate = _aggregate_group([row])

    assert aggregate["invalid_transition_event_rate"] == 1.0


def test_shallow_comparisons_pair_identical_items_and_choose_strongest_control() -> None:
    by_condition: dict[str, list[dict[str, object]]] = {
        condition: [] for condition in ("c2", "c3", "c4")
    }
    for depth in (2, 4):
        for budget in (16, 32):
            for index in range(4):
                item_id = f"d{depth}-b{budget}-i{index}"
                by_condition["c2"].append(
                    _row(
                        item_id=item_id,
                        condition="c2",
                        depth=depth,
                        budget=budget,
                        correct=index < 2,
                    )
                )
                by_condition["c3"].append(
                    _row(
                        item_id=item_id,
                        condition="c3",
                        depth=depth,
                        budget=budget,
                        correct=index < 3,
                    )
                )
                by_condition["c4"].append(
                    _row(
                        item_id=item_id,
                        condition="c4",
                        depth=depth,
                        budget=budget,
                        correct=True,
                    )
                )

    comparisons = _shallow_comparisons(by_condition, [16, 32])

    assert len(comparisons) == 4
    assert all(comparison["item_count"] == 4 for comparison in comparisons)
    assert all(comparison["strongest_control"] == "c3" for comparison in comparisons)
    assert all(comparison["control_accuracy"] == 0.75 for comparison in comparisons)
    assert all(comparison["c4_accuracy"] == 1.0 for comparison in comparisons)
    assert all(comparison["paired_difference"] == 0.25 for comparison in comparisons)
    assert all(comparison["point_estimate_passed"] for comparison in comparisons)
