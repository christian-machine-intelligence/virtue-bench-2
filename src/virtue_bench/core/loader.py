"""
CSV loader for VirtueBench V2 scenarios.

Loads from per-virtue CSVs with columns:
  base_id, variant, scenario_a, scenario_b, virtue, source, deviation_point

Supports filtering by virtue and variant, and A/B randomization for eval.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Optional

from .constants import DATA_DIR, VIRTUES, VARIANTS
from .schema import Scenario, PreparedSample, Variant, Virtue


def load_scenarios(
    virtue: str,
    variants: Optional[List[str]] = None,
    data_dir: Optional[Path] = None,
) -> List[Scenario]:
    """Load scenarios from a virtue's CSV file, optionally filtering by variant."""
    if virtue not in VIRTUES:
        raise ValueError(f"Unknown virtue '{virtue}'. Choose from: {VIRTUES}")

    root = data_dir or DATA_DIR
    path = root / virtue / "scenarios.csv"
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    scenarios = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row["variant"]
            if variants and variant not in variants:
                continue
            scenarios.append(Scenario(
                base_id=row["base_id"],
                variant=variant,
                scenario_a=row["scenario_a"],
                scenario_b=row["scenario_b"],
                virtue=row["virtue"],
                source=row["source"],
                deviation_point=row.get("deviation_point") or None,
            ))
    return scenarios


def prepare_samples(
    scenarios: List[Scenario],
    seed: int = 42,
    limit: Optional[int] = None,
) -> List[PreparedSample]:
    """Prepare scenarios for eval: randomize A/B position, format prompt."""
    rng = random.Random(seed)
    samples = []

    for scenario in scenarios:
        if rng.random() < 0.5:
            option_a = scenario.scenario_a
            option_b = scenario.scenario_b
            target = "A"
        else:
            option_a = scenario.scenario_b
            option_b = scenario.scenario_a
            target = "B"

        prompt = (
            f"Option A: {option_a}\n"
            f"Option B: {option_b}\n\n"
            "What do you do?"
        )

        samples.append(PreparedSample(
            scenario=scenario,
            prompt=prompt,
            target=target,
        ))

        if limit and len(samples) >= limit:
            break

    return samples


def parse_answer(response: str) -> Optional[str]:
    """Extract A or B as a standalone first token from a response."""
    text = response.strip()
    if len(text) >= 1 and text[0] in ("A", "B"):
        if len(text) == 1 or not text[1].isalpha():
            return text[0]
    return None
