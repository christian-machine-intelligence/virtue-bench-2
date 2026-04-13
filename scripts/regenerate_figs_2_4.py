#!/usr/bin/env python3
"""
Regenerate Figure 2 (courage gap) and Figure 4 (psalm injection)
for ICMI-011 with 95% bootstrap confidence intervals.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUTPUT = Path(__file__).resolve().parent.parent.parent / "Proceedings"

VIRTUES = ["prudence", "justice", "courage", "temperance"]
VARIANTS = ["ratio", "caro", "mundus", "diabolus", "ignatian"]


def bootstrap_ci(values, n_bootstrap=10000, confidence=0.95, seed=42):
    """Bootstrap percentile CI."""
    if len(values) <= 1:
        v = values[0] if values else 0.0
        return v, v
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_bootstrap)
    )
    alpha = 1 - confidence
    lo = int((alpha / 2) * n_bootstrap)
    hi = int((1 - alpha / 2) * n_bootstrap) - 1
    return means[lo], means[hi]


def load_results(*files):
    """Load and merge multiple result JSON files."""
    records = []
    for f in files:
        p = RESULTS / f
        if p.exists():
            records.extend(json.loads(p.read_text()))
    return records


def group_by(records, keys):
    """Group records by a tuple of field names, returning dict[tuple] -> [accuracy]."""
    groups = defaultdict(list)
    for r in records:
        k = tuple(r[k] for k in keys)
        if r.get("accuracy") is not None:
            groups[k].append(r["accuracy"])
    return groups


# ─── Figure 2: Courage Gap with CIs ────────────────────────────────

def make_figure_2():
    # Load all cross-variant + ratio repro data
    all_records = load_results(
        "v1_repro_gpt4o.json", "v1_repro_gpt54.json",
        "eval_caro_gpt4o.json", "eval_caro_gpt54.json",
        "eval_mundus_gpt4o.json", "eval_mundus_gpt54.json",
        "eval_diabolus_gpt4o.json", "eval_diabolus_gpt54.json",
        "eval_ignatian_gpt4o.json", "eval_ignatian_gpt54.json",
    )

    # Group by (model, variant, virtue) -> list of per-run accuracies
    grouped = group_by(all_records, ["model", "variant", "virtue"])

    models = [("openai/gpt-4o", "GPT-4o"), ("openai/gpt-5.4", "GPT-5.4")]
    model_colors = {
        "openai/gpt-4o": ("#a8c4e0", "#2166ac"),     # light blue / dark blue
        "openai/gpt-5.4": ("#d4a5a5", "#b2182b"),     # light red / dark red
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(VARIANTS))
    total_width = 0.75
    bar_w = total_width / 4  # 4 bars per variant group
    offsets = [-1.5, -0.5, 0.5, 1.5]

    legend_handles = []
    bar_idx = 0
    for model_id, model_label in models:
        light_c, dark_c = model_colors[model_id]

        # --- "Other virtues" mean bar ---
        other_means = []
        other_ci_lo = []
        other_ci_hi = []
        for variant in VARIANTS:
            # For each run, compute mean of non-courage virtues
            # We need per-run data
            run_accs = defaultdict(dict)  # run_index -> {virtue: acc}
            for r in all_records:
                if r["model"] == model_id and r["variant"] == variant:
                    run_accs[r["run_index"]][r["virtue"]] = r["accuracy"]

            per_run_other_means = []
            for run_idx, virtues in run_accs.items():
                others = [virtues[v] for v in ["prudence", "justice", "temperance"] if v in virtues]
                if others:
                    per_run_other_means.append(sum(others) / len(others))

            m = sum(per_run_other_means) / len(per_run_other_means) if per_run_other_means else 0
            lo, hi = bootstrap_ci(per_run_other_means) if per_run_other_means else (0, 0)
            other_means.append(m * 100)
            other_ci_lo.append((m - lo) * 100)
            other_ci_hi.append((hi - m) * 100)

        bars = ax.bar(
            x + offsets[bar_idx] * bar_w, other_means, bar_w,
            color=light_c, edgecolor="grey", linewidth=0.5,
            yerr=[other_ci_lo, other_ci_hi], capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1},
        )
        legend_handles.append((bars, f"{model_label} (other virtues)"))
        bar_idx += 1

        # --- Courage bar ---
        courage_means = []
        courage_ci_lo = []
        courage_ci_hi = []
        for variant in VARIANTS:
            accs = grouped.get((model_id, variant, "courage"), [])
            m = sum(accs) / len(accs) if accs else 0
            lo, hi = bootstrap_ci(accs) if accs else (0, 0)
            courage_means.append(m * 100)
            courage_ci_lo.append((m - lo) * 100)
            courage_ci_hi.append((hi - m) * 100)

        bars = ax.bar(
            x + offsets[bar_idx] * bar_w, courage_means, bar_w,
            color=dark_c, edgecolor="grey", linewidth=0.5,
            yerr=[courage_ci_lo, courage_ci_hi], capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1},
        )
        legend_handles.append((bars, f"{model_label} Courage"))
        bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels([v.capitalize() for v in VARIANTS], fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Temptation Type", fontsize=12)
    ax.set_title("The Courage Gap: Persistent Across Models and Temptation Types", fontsize=14)
    ax.set_ylim(0, 110)
    ax.legend(
        [h[0] for h in legend_handles],
        [h[1] for h in legend_handles],
        loc="upper right", fontsize=9, ncol=2,
    )
    ax.grid(axis="y", alpha=0.3)

    out = OUTPUT / "fig_courage_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ─── Figure 4: Psalm Injection with CIs ────────────────────────────

def make_figure_4():
    opus_base = load_results("icmi_a_opus_ratio_baseline.json")
    opus_psalm = load_results("icmi_a_opus_ratio_psalms.json")
    gpt_base = load_results("icmi_a_gpt54_ratio_baseline.json")
    gpt_psalm = load_results("icmi_a_gpt54_ratio_psalms.json")

    def get_stats(records):
        grouped = group_by(records, ["virtue"])
        stats = {}
        for v in VIRTUES:
            key = (v,)
            accs = grouped.get(key, [])
            m = sum(accs) / len(accs) if accs else 0
            lo, hi = bootstrap_ci(accs) if accs else (0, 0)
            stats[v] = (m * 100, (m - lo) * 100, (hi - m) * 100)
        return stats

    opus_b = get_stats(opus_base)
    opus_p = get_stats(opus_psalm)
    gpt_b = get_stats(gpt_base)
    gpt_p = get_stats(gpt_psalm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    x = np.arange(len(VIRTUES))
    w = 0.35

    for ax, base_stats, psalm_stats, title in [
        (ax1, opus_b, opus_p, "Claude Opus 4.6"),
        (ax2, gpt_b, gpt_p, "GPT-5.4"),
    ]:
        base_vals = [base_stats[v][0] for v in VIRTUES]
        base_lo = [base_stats[v][1] for v in VIRTUES]
        base_hi = [base_stats[v][2] for v in VIRTUES]

        psalm_vals = [psalm_stats[v][0] for v in VIRTUES]
        psalm_lo = [psalm_stats[v][1] for v in VIRTUES]
        psalm_hi = [psalm_stats[v][2] for v in VIRTUES]

        bars1 = ax.bar(
            x - w / 2, base_vals, w,
            color="#888888", edgecolor="grey", linewidth=0.5,
            label="Baseline",
            yerr=[base_lo, base_hi], capsize=4,
            error_kw={"elinewidth": 1, "capthick": 1},
        )
        bars2 = ax.bar(
            x + w / 2, psalm_vals, w,
            color="#d4760a", edgecolor="grey", linewidth=0.5,
            label="+ Psalms",
            yerr=[psalm_lo, psalm_hi], capsize=4,
            error_kw={"elinewidth": 1, "capthick": 1},
        )

        # Value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 2.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9,
                )

        # Overall means as dashed lines
        base_overall = sum(base_vals) / len(base_vals)
        psalm_overall = sum(psalm_vals) / len(psalm_vals)
        ax.axhline(base_overall, color="#888888", ls="--", lw=1, alpha=0.7)
        ax.axhline(psalm_overall, color="#d4760a", ls="--", lw=1, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([v.capitalize() for v in VIRTUES], fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(55, 105)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    fig.suptitle("Psalm Injection Effect on VirtueBench 2 (Ratio Variant)", fontsize=14, y=1.02)
    fig.tight_layout()

    out = OUTPUT / "fig_psalm_injection.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure_2()
    make_figure_4()
    print("Done.")
