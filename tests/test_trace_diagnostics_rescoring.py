from __future__ import annotations

import json

from token_efficient_reasoning.controlled.schema import CanonicalExample
from token_efficient_reasoning.primary.reporting import _recompute_trace_diagnostics


def _first_validation_example() -> CanonicalExample:
    with open("data/m1/splits/validation.jsonl", encoding="utf-8") as handle:
        return CanonicalExample.from_dict(json.loads(handle.readline()))


def test_saved_bridge_row_is_rescored_without_mutating_artifact_payload() -> None:
    example = _first_validation_example()
    old_diagnostics = {
        "trace_parse_success": False,
        "parsed_transition_count": 0,
        "verified_prefix_transitions": 0,
        "invalid_transition_count": 0,
    }
    row = {
        "item_id": example.id,
        "reasoning_text": "s1=5>div(5)>1\ns2=1>",
        "forced_reasoning_transition": True,
        "trace_diagnostics": old_diagnostics,
    }

    corrected = _recompute_trace_diagnostics([row], {example.id: example}, "r3")

    assert row["trace_diagnostics"] is old_diagnostics
    assert corrected[0] is not row
    diagnostics = corrected[0]["trace_diagnostics"]
    assert diagnostics["parsed_transition_count"] == 1
    assert diagnostics["verified_prefix_transitions"] == 1
    assert diagnostics["invalid_transition_event"] is True
