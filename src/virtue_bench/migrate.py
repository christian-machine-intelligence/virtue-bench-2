"""
V1 → V2 scenario migration.

Reads V1's flat CSVs (data/{virtue}.csv) with columns:
  scenario_a, scenario_b, virtue, source

Writes V2 CSVs (data/{virtue}/scenarios.csv) with columns:
  base_id, variant, scenario_a, scenario_b, virtue, source, deviation_point

V1 scenarios become the 'ratio' variant. The other 4 variants
(caro, mundus, diabolus, ignatian) are placeholders to be filled
via LLM-assisted generation + human review.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .core.constants import VIRTUES, VARIANTS

# Prefix map for base_id generation
VIRTUE_PREFIX = {
    "prudence": "PRU",
    "justice": "JUS",
    "courage": "COU",
    "temperance": "TEM",
}


def migrate_v1_to_v2(
    v1_data_dir: Path,
    v2_data_dir: Path,
) -> None:
    """Migrate all V1 virtue CSVs to V2 format.

    Creates one scenarios.csv per virtue in v2_data_dir/{virtue}/.
    V1 scenarios become 'ratio' variant rows. Placeholder rows are
    created for the other 4 variants with empty scenario_b.
    """
    for virtue in VIRTUES:
        v1_path = v1_data_dir / f"{virtue}.csv"
        if not v1_path.exists():
            print(f"  Skipping {virtue}: {v1_path} not found")
            continue

        v2_dir = v2_data_dir / virtue
        v2_dir.mkdir(parents=True, exist_ok=True)
        v2_path = v2_dir / "scenarios.csv"

        prefix = VIRTUE_PREFIX[virtue]
        rows = []

        with open(v1_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                base_id = f"{prefix}-{i:03d}"
                scenario_a = row["scenario_a"]
                scenario_b = row["scenario_b"]
                source = row.get("source", "")

                # Ratio variant: V1 text preserved as-is
                rows.append({
                    "base_id": base_id,
                    "variant": "ratio",
                    "scenario_a": scenario_a,
                    "scenario_b": scenario_b,
                    "virtue": virtue,
                    "source": source,
                    "deviation_point": "",
                })

                # Placeholder rows for other variants
                for variant in VARIANTS:
                    if variant == "ratio":
                        continue
                    rows.append({
                        "base_id": base_id,
                        "variant": variant,
                        "scenario_a": scenario_a,
                        "scenario_b": f"[TODO: {variant} variant]",
                        "virtue": virtue,
                        "source": source,
                        "deviation_point": "[TODO]" if variant == "ignatian" else "",
                    })

        fieldnames = ["base_id", "variant", "scenario_a", "scenario_b", "virtue", "source", "deviation_point"]
        with open(v2_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        n_base = len([r for r in rows if r["variant"] == "ratio"])
        print(f"  {virtue}: {n_base} base scenarios → {len(rows)} rows ({v2_path})")


def main():
    """CLI entry for standalone migration."""
    import argparse
    parser = argparse.ArgumentParser(description="Migrate V1 scenarios to V2 format")
    parser.add_argument("--v1-dir", required=True, help="Path to V1 data/ directory")
    parser.add_argument("--v2-dir", required=True, help="Path to V2 data/ directory")
    args = parser.parse_args()
    migrate_v1_to_v2(Path(args.v1_dir), Path(args.v2_dir))


if __name__ == "__main__":
    main()
