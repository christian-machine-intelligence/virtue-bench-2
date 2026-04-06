"""
Model version regression detection.

Flags when a new model version is significantly worse on any condition
compared to a baseline.
"""

from __future__ import annotations

from typing import Dict, List

from .bootstrap import AggregatedResult


def detect_regressions(
    baseline: List[AggregatedResult],
    candidate: List[AggregatedResult],
    threshold: float = 0.05,
) -> List[dict]:
    """Compare candidate against baseline and flag significant regressions.

    A regression is flagged when the candidate's CI upper bound is below
    the baseline's CI lower bound, indicating a statistically significant drop.
    """
    baseline_map = {
        (r.virtue, r.variant, r.condition): r for r in baseline
    }

    regressions = []
    for cand in candidate:
        key = (cand.virtue, cand.variant, cand.condition)
        base = baseline_map.get(key)
        if base is None:
            continue

        delta = cand.mean_accuracy - base.mean_accuracy
        significant = cand.ci_upper < base.ci_lower

        if delta < -threshold or significant:
            regressions.append({
                "virtue": cand.virtue,
                "variant": cand.variant,
                "condition": cand.condition,
                "baseline_model": base.model,
                "candidate_model": cand.model,
                "baseline_accuracy": base.mean_accuracy,
                "candidate_accuracy": cand.mean_accuracy,
                "delta": delta,
                "baseline_ci": (base.ci_lower, base.ci_upper),
                "candidate_ci": (cand.ci_lower, cand.ci_upper),
                "significant": significant,
            })

    return regressions
