"""
Visualization utilities for VirtueBench V2 results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.constants import VIRTUES, VARIANTS
from ..stats.bootstrap import AggregatedResult


def plot_variant_heatmap(
    aggregated: List[AggregatedResult],
    output_path: Optional[Path] = None,
    title: str = "VirtueBench V2: Accuracy by Virtue × Variant",
) -> None:
    """Plot a virtue × variant heatmap of mean accuracies."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ImportError:
        print("matplotlib required for visualization. Install with: pip install matplotlib")
        return

    lookup: Dict[Tuple[str, str], float] = {}
    for a in aggregated:
        lookup[(a.virtue, a.variant)] = a.mean_accuracy

    data = []
    for virtue in VIRTUES:
        row = [lookup.get((virtue, variant), 0.0) for variant in VARIANTS]
        data.append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS, rotation=45, ha="right")
    ax.set_yticks(range(len(VIRTUES)))
    ax.set_yticklabels(VIRTUES)

    for i in range(len(VIRTUES)):
        for j in range(len(VARIANTS)):
            val = data[i][j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=11)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to: {output_path}")
    else:
        plt.show()

    plt.close(fig)
