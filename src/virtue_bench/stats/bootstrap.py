"""
Bootstrap confidence interval computation for multi-run experiments.

With N runs at temperature > 0, we get N accuracy estimates per cell.
Bootstrap percentile CIs are preferred for small N (< 30).
"""

from __future__ import annotations

import random
from math import sqrt
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..core.schema import RunResult


@dataclass
class AggregatedResult:
    """Aggregated statistics across multiple runs for one cell."""
    model: str
    virtue: str
    variant: str
    condition: str
    frame: str
    n_runs: int
    mean_accuracy: float
    std_accuracy: float
    ci_lower: float
    ci_upper: float
    accuracies: List[float]


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap percentile confidence interval."""
    if len(values) <= 1:
        val = values[0] if values else 0.0
        return (val, val)

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    lower_idx = int((alpha / 2) * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    return (means[lower_idx], means[upper_idx])


def normal_ci(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute normal approximation CI for larger samples."""
    n = len(values)
    if n <= 1:
        val = values[0] if values else 0.0
        return (val, val)

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = sqrt(variance)

    # z-score for 95% CI
    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    margin = z * std / sqrt(n)
    return (mean - margin, mean + margin)


def aggregate_runs(
    results: List[RunResult],
    confidence: float = 0.95,
    use_bootstrap: bool = True,
) -> List[AggregatedResult]:
    """Aggregate RunResults into per-cell statistics with CIs.

    Groups by (model, virtue, variant, condition, frame) and computes
    mean accuracy with confidence intervals across runs.
    """
    groups: Dict[tuple, List[RunResult]] = {}
    for r in results:
        key = (r.model, r.virtue, r.variant, r.condition, r.frame)
        groups.setdefault(key, []).append(r)

    aggregated = []
    for (model, virtue, variant, condition, frame), runs in groups.items():
        accuracies = [r.accuracy for r in runs if r.accuracy is not None]
        if not accuracies:
            continue

        n = len(accuracies)
        mean = sum(accuracies) / n
        variance = sum((x - mean) ** 2 for x in accuracies) / max(n - 1, 1)
        std = sqrt(variance)

        if use_bootstrap and n < 30:
            ci_lower, ci_upper = bootstrap_ci(accuracies, confidence=confidence)
        else:
            ci_lower, ci_upper = normal_ci(accuracies, confidence=confidence)

        aggregated.append(AggregatedResult(
            model=model,
            virtue=virtue,
            variant=variant,
            condition=condition,
            frame=frame,
            n_runs=n,
            mean_accuracy=mean,
            std_accuracy=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            accuracies=accuracies,
        ))

    return aggregated
