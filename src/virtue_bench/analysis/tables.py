"""
Comparison tables and reporting for VirtueBench V2.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    def tabulate(rows, headers, tablefmt="github"):
        str_rows = [[str(cell) for cell in row] for row in rows]
        widths = [
            max(len(str(h)), *(len(row[i]) for row in str_rows)) if str_rows else len(str(h))
            for i, h in enumerate(headers)
        ]
        def fmt(row):
            return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"
        hdr = fmt([str(h) for h in headers])
        div = "| " + " | ".join("-" * w for w in widths) + " |"
        body = "\n".join(fmt(r) for r in str_rows)
        return "\n".join(p for p in [hdr, div, body] if p)

from ..core.schema import RunResult
from ..stats.bootstrap import AggregatedResult


def print_comparison_table(results: List[RunResult]) -> None:
    """Print per-run results."""
    headers = ["Model", "Virtue", "Variant", "Run", "Accuracy", "Samples"]
    rows = []
    for r in results:
        rows.append([
            r.model,
            r.virtue,
            r.variant,
            r.run_index,
            f"{r.accuracy:.4f}" if r.accuracy is not None else "N/A",
            r.samples,
        ])
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))


def print_aggregated_table(aggregated: List[AggregatedResult]) -> None:
    """Print aggregated results with CIs."""
    headers = ["Model", "Virtue", "Variant", "Runs", "Mean Acc", "95% CI", "Std"]
    rows = []
    for a in aggregated:
        rows.append([
            a.model,
            a.virtue,
            a.variant,
            a.n_runs,
            f"{a.mean_accuracy:.4f}",
            f"[{a.ci_lower:.4f}, {a.ci_upper:.4f}]",
            f"{a.std_accuracy:.4f}",
        ])
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))


def print_variant_grid(aggregated: List[AggregatedResult]) -> None:
    """Print a virtue × variant grid of mean accuracies."""
    from ..core.constants import VIRTUES, VARIANTS

    # Build lookup
    lookup: Dict[Tuple[str, str], AggregatedResult] = {}
    for a in aggregated:
        lookup[(a.virtue, a.variant)] = a

    headers = ["Virtue"] + VARIANTS
    rows = []
    for virtue in VIRTUES:
        row = [virtue]
        for variant in VARIANTS:
            agg = lookup.get((virtue, variant))
            if agg:
                row.append(f"{agg.mean_accuracy:.2%}")
            else:
                row.append("—")
        rows.append(row)

    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
