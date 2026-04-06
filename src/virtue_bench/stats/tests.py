"""
Statistical tests for VirtueBench V2.

- McNemar's test for paired model comparisons
- Chi-squared across variant categories
- Bonferroni correction for multiple comparisons
"""

from __future__ import annotations

from math import comb
from typing import Dict, List, Optional

from ..core.schema import RunResult


def exact_two_sided_binomial_pvalue(improve: int, regress: int) -> float:
    """Two-sided exact p-value for discordant paired counts (McNemar-style)."""
    n = improve + regress
    if n == 0:
        return 1.0
    k = min(improve, regress)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar_test(
    results_a: List[RunResult],
    results_b: List[RunResult],
) -> Dict:
    """Paired McNemar test comparing two sets of results on the same scenarios.

    Both result lists should be from the same run (same seed) so samples align.
    Returns counts and p-value.
    """
    # Build correctness maps keyed by (virtue, variant, sample_id)
    def correctness_map(results: List[RunResult]) -> Dict:
        m = {}
        for r in results:
            for s in r.sample_details:
                m[(r.virtue, s.variant, s.sample_id)] = s.correct
        return m

    map_a = correctness_map(results_a)
    map_b = correctness_map(results_b)

    improve = regress = same_right = same_wrong = 0
    for key in map_a:
        if key not in map_b:
            continue
        a_correct = bool(map_a[key])
        b_correct = bool(map_b[key])
        if not a_correct and b_correct:
            improve += 1
        elif a_correct and not b_correct:
            regress += 1
        elif a_correct and b_correct:
            same_right += 1
        else:
            same_wrong += 1

    return {
        "improve": improve,
        "regress": regress,
        "same_right": same_right,
        "same_wrong": same_wrong,
        "p_value": exact_two_sided_binomial_pvalue(improve, regress),
    }


def chi_squared_variant(
    results: List[RunResult],
) -> Dict:
    """Chi-squared test for independence of accuracy across variants.

    Tests whether variant type is associated with different accuracy levels.
    """
    from collections import defaultdict

    variant_correct: Dict[str, int] = defaultdict(int)
    variant_total: Dict[str, int] = defaultdict(int)

    for r in results:
        for s in r.sample_details:
            variant_total[s.variant] += 1
            if s.correct:
                variant_correct[s.variant] += 1

    variants = sorted(variant_total.keys())
    if len(variants) < 2:
        return {"chi2": 0.0, "p_value": 1.0, "df": 0, "variants": variants}

    total = sum(variant_total.values())
    total_correct = sum(variant_correct.values())
    expected_rate = total_correct / total if total > 0 else 0

    chi2 = 0.0
    for v in variants:
        n = variant_total[v]
        observed_correct = variant_correct[v]
        observed_incorrect = n - observed_correct
        expected_correct = n * expected_rate
        expected_incorrect = n * (1 - expected_rate)

        if expected_correct > 0:
            chi2 += (observed_correct - expected_correct) ** 2 / expected_correct
        if expected_incorrect > 0:
            chi2 += (observed_incorrect - expected_incorrect) ** 2 / expected_incorrect

    df = len(variants) - 1

    # Approximate p-value using scipy if available, else return chi2 only
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df)
    except ImportError:
        p_value = None

    return {"chi2": chi2, "p_value": p_value, "df": df, "variants": variants}


def bonferroni_correct(p_values: List[float], n_comparisons: Optional[int] = None) -> List[float]:
    """Apply Bonferroni correction to a list of p-values."""
    n = n_comparisons or len(p_values)
    return [min(1.0, p * n) for p in p_values]
