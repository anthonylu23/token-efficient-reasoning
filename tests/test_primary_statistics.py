from __future__ import annotations

import pytest

from token_efficient_reasoning.primary.statistics import (
    clustered_bootstrap_mean,
    derived_analysis_seed,
    holm_adjust,
    paired_cluster_values,
    paired_randomization_pvalue,
    percentile,
)


def test_clustered_bootstrap_is_deterministic_and_averages_replicates_within_item() -> None:
    values = {
        "item-a": [1.0, 1.0, 0.0],
        "item-b": [0.0, 0.0, 1.0],
    }

    first = clustered_bootstrap_mean(values, seed=17, samples=2_000)
    second = clustered_bootstrap_mean(values, seed=17, samples=2_000)

    assert first == second
    assert first["estimate"] == pytest.approx(0.5)
    assert first["cluster_count"] == 2
    assert first["replicate_observation_count"] == 6
    assert first["confidence_interval"][0] <= 0.5 <= first["confidence_interval"][1]


def test_paired_values_require_exact_keys_and_keep_item_as_bootstrap_unit() -> None:
    treatment = {(1, "a"): 1.0, (2, "a"): 0.0, (1, "b"): 1.0, (2, "b"): 1.0}
    control = {(1, "a"): 0.0, (2, "a"): 0.0, (1, "b"): 1.0, (2, "b"): 0.0}

    assert paired_cluster_values(treatment, control) == {
        "a": [1.0, 0.0],
        "b": [0.0, 1.0],
    }

    with pytest.raises(ValueError, match="different keys"):
        paired_cluster_values(treatment, {(1, "a"): 0.0})


def test_randomization_pvalue_detects_consistently_positive_paired_effect() -> None:
    differences = {f"item-{index}": [1.0, 1.0, 1.0] for index in range(20)}

    result = paired_randomization_pvalue(
        differences,
        seed=derived_analysis_seed(5, "positive"),
        samples=10_000,
    )

    assert result["observed_difference"] == 1.0
    assert result["p_value"] < 0.001


def test_holm_adjust_uses_step_down_rejection_and_monotone_adjusted_values() -> None:
    adjusted = holm_adjust(
        {"h1": 0.001, "h2": 0.02, "h3": 0.03, "h4": 0.20},
        alpha=0.05,
    )

    assert adjusted["h1"]["reject_null"] is True
    assert adjusted["h2"]["reject_null"] is False
    assert adjusted["h3"]["reject_null"] is False
    ordered = sorted(adjusted.values(), key=lambda row: row["rank"])
    assert [row["adjusted_p_value"] for row in ordered] == sorted(
        row["adjusted_p_value"] for row in ordered
    )


def test_percentile_interpolates_and_validates_inputs() -> None:
    assert percentile([1.0, 3.0], 0.5) == 2.0
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)
