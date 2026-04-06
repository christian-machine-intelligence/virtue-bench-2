"""
Psalm injection loader with theologically-supported subsets.

Psalm categories are drawn from mainstream Christian biblical scholarship
and the ICMI-002/ICMI-A research papers. Each subset has a theological
rationale documented in the PSALM_SETS dictionary.

Psalm text is loaded from a KJV JSON source file.

Usage:
    # From CLI
    virtue-bench run --psalm-set imprecatory
    virtue-bench run --psalm-set penitential --psalm-set trust
    virtue-bench run --psalm-numbers 23,51,91

    # Programmatic
    from virtue_bench.core.psalms import load_psalm_text
    text = load_psalm_text(psalm_set="imprecatory")
    text = load_psalm_text(psalm_numbers=[23, 51, 91])
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

# Default location for KJV psalm source — can be overridden
DEFAULT_PSALM_SOURCE = (
    Path(__file__).parent.parent.parent.parent.parent
    / "psalm-alignment" / "data" / "psalms_kjv.json"
)

# ---------------------------------------------------------------------------
# Theologically-supported psalm subsets
# ---------------------------------------------------------------------------

PSALM_SETS: Dict[str, Dict] = {
    # --- From ICMI-002: Imprecatory Psalms and VirtueBench ---
    "imprecatory": {
        "psalms": [5, 6, 10, 12, 35, 37, 40, 52, 54, 55, 56, 57, 58, 59,
                    69, 79, 83, 94, 109, 137, 139, 143],
        "description": "Prayers of the oppressed calling for divine justice against enemies",
        "tradition": "ICMI-002; Augustine De Civitate Dei; Bonhoeffer Psalms",
        "expected_effect": "Amplifies Courage (+11pts in ICMI-002 on Claude Sonnet 4)",
    },

    # --- Traditional liturgical category (medieval Church canon) ---
    "penitential": {
        "psalms": [6, 32, 38, 51, 102, 130, 143],
        "description": "Seven penitential psalms: confession, repentance, restoration",
        "tradition": "Catholic/Anglican liturgical tradition; Ash Wednesday",
        "expected_effect": "May amplify Temperance (self-examination, restraint)",
    },

    # --- From ICMI-A: Psalm Injection and Alignment ---
    "popular": {
        "psalms": [1, 23, 42, 51, 88, 100, 119],
        "description": "Most frequently encountered psalms in devotional practice",
        "tradition": "ICMI-A curated set; liturgical frequency analysis",
        "expected_effect": "Broad devotional effect; tested in ICMI-A on ETHICS",
    },

    # --- From ICMI-A: Random baseline ---
    "random_baseline": {
        "psalms": [7, 23, 27, 29, 36, 58, 63, 71, 109, 140],
        "description": "Pseudo-random selection (seed 42) for control condition",
        "tradition": "ICMI-A control set",
        "expected_effect": "Baseline; diverse mix including praise, trust, imprecatory",
    },

    # --- Traditional Hallel (Egyptian Hallel + Final Hallel) ---
    "praise": {
        "psalms": [29, 100, 103, 104, 113, 114, 115, 116, 117, 118,
                    145, 146, 147, 148, 149, 150],
        "description": "Hallel psalms: joy, worship, thanksgiving",
        "tradition": "Jewish Passover liturgy; Psalter closing doxology",
        "expected_effect": "May counter dejection/acedia; emotional opposite of lament",
    },

    # --- Lament psalms (largest formal Psalter category) ---
    "lament": {
        "psalms": [7, 13, 22, 25, 26, 28, 31, 35, 42, 44, 54, 55, 56, 57,
                    61, 64, 69, 70, 80, 86, 88, 102, 109, 142, 143],
        "description": "Honest expressions of suffering, complaint, and trust amid difficulty",
        "tradition": "Gunkel/Westermann form criticism; standard genre category",
        "expected_effect": "Models endurance under adversity; validates honest moral struggle",
    },

    # --- Wisdom/Sapiential psalms ---
    "wisdom": {
        "psalms": [1, 19, 37, 49, 73, 112, 119, 127, 128, 131, 139],
        "description": "Meditation on divine order, righteousness vs wickedness, fear of God",
        "tradition": "Sapiential literature tradition; overlap with Proverbs",
        "expected_effect": "May amplify Prudence (discernment, deliberation)",
    },

    # --- Royal/Messianic psalms ---
    "royal": {
        "psalms": [2, 18, 20, 21, 45, 72, 101, 110, 132, 144],
        "description": "Psalms of kingship, authority, and messianic expectation",
        "tradition": "Christian messianic interpretation; Aquinas commentary on Psalms",
        "expected_effect": "May amplify Justice (authority, righteous judgment)",
    },

    # --- Psalms of trust/confidence ---
    "trust": {
        "psalms": [23, 25, 27, 31, 42, 56, 57, 61, 62, 63, 84, 91, 121, 125, 131],
        "description": "Affirmations of God's protection, provision, and faithfulness",
        "tradition": "Pastoral/devotional tradition; comfort psalms",
        "expected_effect": "May amplify Courage (trust enables risk-taking for virtue)",
    },

    # --- Songs of Ascent (formally defined biblical category) ---
    "ascent": {
        "psalms": list(range(120, 135)),  # Psalms 120-134
        "description": "Pilgrimage psalms sung ascending to the Jerusalem temple",
        "tradition": "Biblical superscription; Second Temple pilgrimage liturgy",
        "expected_effect": "Journey/progression frame; brevity, repetition, devotion",
    },

    # --- Historical/Narrative psalms ---
    "historical": {
        "psalms": [78, 81, 105, 106, 114, 135, 136],
        "description": "Retelling of Israel's history and God's faithfulness",
        "tradition": "Narrative psalm genre; memorial/testimony function",
        "expected_effect": "Institutional memory; perseverance through adversity",
    },
}


def list_psalm_sets() -> Dict[str, str]:
    """Return a summary of available psalm sets."""
    return {name: info["description"] for name, info in PSALM_SETS.items()}


def get_psalm_numbers(
    psalm_set: Optional[str] = None,
    psalm_numbers: Optional[List[int]] = None,
    psalm_sets: Optional[List[str]] = None,
    random_n: Optional[int] = None,
    seed: int = 42,
) -> List[int]:
    """Resolve psalm numbers from set name(s), explicit numbers, or random selection.

    Args:
        psalm_set: Single set name (e.g. "imprecatory")
        psalm_numbers: Explicit list of psalm numbers
        psalm_sets: Multiple set names to combine (union)
        random_n: Select N random psalms from 1-150
        seed: Random seed for random selection

    Returns:
        Sorted list of unique psalm numbers.
    """
    numbers: Set[int] = set()

    if psalm_set:
        if psalm_set not in PSALM_SETS:
            raise ValueError(
                f"Unknown psalm set '{psalm_set}'. "
                f"Choose from: {list(PSALM_SETS.keys())}"
            )
        numbers.update(PSALM_SETS[psalm_set]["psalms"])

    if psalm_sets:
        for name in psalm_sets:
            if name not in PSALM_SETS:
                raise ValueError(
                    f"Unknown psalm set '{name}'. "
                    f"Choose from: {list(PSALM_SETS.keys())}"
                )
            numbers.update(PSALM_SETS[name]["psalms"])

    if psalm_numbers:
        for n in psalm_numbers:
            if not (1 <= n <= 150):
                raise ValueError(f"Psalm number {n} out of range (1-150)")
        numbers.update(psalm_numbers)

    if random_n:
        rng = random.Random(seed)
        available = [i for i in range(1, 151) if i not in numbers]
        selected = rng.sample(available, min(random_n, len(available)))
        numbers.update(selected)

    return sorted(numbers)


def load_psalm_text(
    psalm_set: Optional[str] = None,
    psalm_numbers: Optional[List[int]] = None,
    psalm_sets: Optional[List[str]] = None,
    random_n: Optional[int] = None,
    seed: int = 42,
    source_path: Optional[Path] = None,
) -> str:
    """Load psalm text for injection into system prompts.

    Resolves which psalms to include, loads their KJV text, and
    concatenates them with psalm headers.

    Returns:
        Formatted psalm text ready for system prompt injection.
    """
    numbers = get_psalm_numbers(
        psalm_set=psalm_set,
        psalm_numbers=psalm_numbers,
        psalm_sets=psalm_sets,
        random_n=random_n,
        seed=seed,
    )

    if not numbers:
        raise ValueError("No psalms specified. Use psalm_set, psalm_numbers, psalm_sets, or random_n.")

    path = source_path or DEFAULT_PSALM_SOURCE
    if not path.exists():
        raise FileNotFoundError(
            f"Psalm source file not found: {path}. "
            f"Expected KJV JSON at {DEFAULT_PSALM_SOURCE}"
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Build chapter lookup: psalm number (int) -> list of verse texts
    chapter_map: Dict[int, List[str]] = {}
    for chapter in data["chapters"]:
        num = int(chapter["chapter"])
        verses = [v["text"] for v in chapter["verses"]]
        chapter_map[num] = verses

    parts = []
    for num in numbers:
        if num not in chapter_map:
            continue
        verses = chapter_map[num]
        header = f"Psalm {num}"
        text = " ".join(verses)
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)
