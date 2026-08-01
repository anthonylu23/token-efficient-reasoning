"""Deterministic clustered inference for the frozen M3 analysis."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def derived_analysis_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable NumPy-compatible seed from a frozen analysis namespace."""

    payload = "\x1f".join((str(base_seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return the linearly interpolated quantile used by report summaries."""

    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _cluster_means(
    values: Mapping[str, Sequence[float]],
) -> tuple[tuple[str, ...], np.ndarray]:
    if not values:
        raise ValueError("clustered inference requires at least one item")
    item_ids = tuple(sorted(values))
    means = []
    for item_id in item_ids:
        item_values = values[item_id]
        if not item_values:
            raise ValueError(f"item cluster {item_id!r} is empty")
        array = np.asarray(item_values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"item cluster {item_id!r} contains a non-finite value")
        means.append(float(array.mean()))
    return item_ids, np.asarray(means, dtype=np.float64)


def clustered_bootstrap_mean(
    values: Mapping[str, Sequence[float]],
    *,
    seed: int,
    samples: int,
    confidence_level: float = 0.95,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Bootstrap item clusters after averaging paired replicate outcomes per item."""

    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    item_ids, cluster_values = _cluster_means(values)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    cluster_count = len(cluster_values)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(
            0,
            cluster_count,
            size=(stop - start, cluster_count),
            endpoint=False,
        )
        estimates[start:stop] = cluster_values[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(estimates, (alpha / 2.0, 1.0 - alpha / 2.0))
    return {
        "estimate": float(cluster_values.mean()),
        "confidence_interval": [float(lower), float(upper)],
        "confidence_level": confidence_level,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "cluster_count": len(item_ids),
        "replicate_observation_count": int(sum(len(values[item_id]) for item_id in item_ids)),
        "bootstrap_unit": "item_id",
        "within_item_reduction": "arithmetic mean across paired replicate blocks",
    }


def paired_cluster_values(
    treatment: Mapping[tuple[int, str], float],
    control: Mapping[tuple[int, str], float],
) -> dict[str, list[float]]:
    """Return paired treatment-minus-control outcomes grouped by item ID."""

    if treatment.keys() != control.keys():
        missing_treatment = sorted(control.keys() - treatment.keys())
        missing_control = sorted(treatment.keys() - control.keys())
        raise ValueError(
            "paired outcomes have different keys; "
            f"missing treatment={missing_treatment[:3]}, "
            f"missing control={missing_control[:3]}"
        )
    grouped: dict[str, list[float]] = {}
    for replicate, item_id in sorted(treatment):
        difference = float(treatment[(replicate, item_id)]) - float(
            control[(replicate, item_id)]
        )
        if not math.isfinite(difference):
            raise ValueError("paired outcome contains a non-finite difference")
        grouped.setdefault(item_id, []).append(difference)
    replicate_counts = {len(values) for values in grouped.values()}
    if len(replicate_counts) != 1:
        raise ValueError("paired item clusters do not contain equal replicate counts")
    return grouped


def paired_randomization_pvalue(
    differences: Mapping[str, Sequence[float]],
    *,
    seed: int,
    samples: int,
    alternative: str = "greater",
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Monte Carlo paired sign-randomization test over item-cluster means."""

    if samples < 1:
        raise ValueError("randomization samples must be positive")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")
    _, cluster_values = _cluster_means(differences)
    observed = float(cluster_values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    cluster_count = len(cluster_values)
    for start in range(0, samples, chunk_size):
        width = min(chunk_size, samples - start)
        signs = rng.integers(0, 2, size=(width, cluster_count), endpoint=False)
        signs = signs * 2 - 1
        estimates = (signs * cluster_values).mean(axis=1)
        if alternative == "greater":
            extreme += int(np.count_nonzero(estimates >= observed))
        elif alternative == "less":
            extreme += int(np.count_nonzero(estimates <= observed))
        else:
            extreme += int(np.count_nonzero(np.abs(estimates) >= abs(observed)))
    return {
        "observed_difference": observed,
        "p_value": (extreme + 1) / (samples + 1),
        "alternative": alternative,
        "randomization_samples": samples,
        "randomization_seed": seed,
        "cluster_count": cluster_count,
        "method": "paired sign randomization over replicate-averaged item outcomes",
    }


def holm_adjust(p_values: Mapping[str, float], *, alpha: float) -> dict[str, dict[str, Any]]:
    """Apply Holm's step-down family-wise error correction."""

    if not p_values:
        raise ValueError("Holm adjustment requires at least one hypothesis")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    for hypothesis, value in p_values.items():
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise ValueError(f"invalid p-value for {hypothesis!r}: {value}")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running_adjusted = 0.0
    rejection_open = True
    result: dict[str, dict[str, Any]] = {}
    for rank, (hypothesis, value) in enumerate(ordered, start=1):
        multiplier = count - rank + 1
        adjusted = min(1.0, max(running_adjusted, value * multiplier))
        running_adjusted = adjusted
        threshold = alpha / multiplier
        rejected = rejection_open and value <= threshold
        if not rejected:
            rejection_open = False
        result[hypothesis] = {
            "rank": rank,
            "raw_p_value": value,
            "adjusted_p_value": adjusted,
            "step_threshold": threshold,
            "reject_null": rejected,
        }
    return result
