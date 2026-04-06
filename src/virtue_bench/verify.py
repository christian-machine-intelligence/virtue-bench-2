"""
Source verification for VirtueBench V2 scenarios.

Uses Claude to verify that:
1. Patristic source citations actually support the moral scenario described
2. Ignatian variant Scripture citations are real and accurately quoted
3. Deviation points correctly identify where reasoning turns

Usage:
    # Verify all scenarios
    python -m virtue_bench.verify

    # Verify a single virtue
    python -m virtue_bench.verify --virtue courage

    # Verify only ignatian variants (Scripture citations)
    python -m virtue_bench.verify --variant ignatian

    # Verify only base sources (patristic citations vs scenario_a)
    python -m virtue_bench.verify --base-only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import anthropic

from .core.constants import DATA_DIR, VIRTUES, VARIANTS


SYSTEM_PROMPT_BASE_VERIFY = """You are a patristic theology scholar verifying source citations \
for a virtue ethics benchmark. Your task is to evaluate whether a cited source \
actually provides a theological basis for the moral scenario described.

You will be given:
- A moral scenario (the virtuous choice)
- A patristic source citation
- The virtue being tested

Evaluate whether the citation is:
1. A REAL reference (does this work/section exist?)
2. RELEVANT to the scenario (does this source actually discuss the moral situation described?)
3. ACCURATE in its attribution (is this the right author and work?)

Respond in JSON format:
{
    "source_exists": true/false,
    "source_relevant": true/false,
    "source_accurate": true/false,
    "confidence": "high"/"medium"/"low",
    "notes": "brief explanation of any issues",
    "suggested_correction": "if source is wrong, suggest the correct one, or null"
}"""

SYSTEM_PROMPT_IGNATIAN_VERIFY = """You are a biblical and patristic scholar verifying \
Scripture citations in temptation scenarios. Your task is to check whether Bible verses \
cited in a temptation argument are:

1. REAL verses (do they exist in the Bible?)
2. ACCURATELY quoted or paraphrased (does the text match the actual verse?)
3. APPROPRIATELY used in context (is the verse being used in a way that's at least \
   superficially plausible, even if ultimately misleading — which is the point of \
   the Ignatian temptation?)

Also evaluate the deviation_point: does it correctly identify where the theological \
reasoning shifts from genuine virtue to disguised vice?

Respond in JSON format:
{
    "scripture_citations_found": ["list of citations found in the text"],
    "citations_real": true/false,
    "citations_accurate": true/false,
    "deviation_point_valid": true/false,
    "confidence": "high"/"medium"/"low",
    "notes": "brief explanation of any issues",
    "flagged_citations": ["list of any problematic citations with explanation"]
}"""


async def verify_base_source(
    client: anthropic.AsyncAnthropic,
    scenario_a: str,
    source: str,
    virtue: str,
    model: str = "claude-opus-4-6",
) -> dict:
    """Verify a patristic source citation against its scenario."""
    prompt = f"""Verify this source citation for a {virtue} scenario:

SCENARIO (virtuous choice): "{scenario_a}"

CITED SOURCE: {source}

Does this source actually provide a theological basis for this moral scenario?"""

    response = await client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,
        system=SYSTEM_PROMPT_BASE_VERIFY,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    try:
        # Try to parse JSON from the response
        # Handle cases where model wraps in markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {"raw_response": text, "parse_error": True}


async def verify_ignatian_variant(
    client: anthropic.AsyncAnthropic,
    scenario_a: str,
    scenario_b: str,
    deviation_point: str,
    source: str,
    virtue: str,
    model: str = "claude-opus-4-6",
) -> dict:
    """Verify Scripture citations and deviation point in an Ignatian variant."""
    prompt = f"""Verify the Scripture citations and theological reasoning in this Ignatian temptation:

VIRTUE: {virtue}
PATRISTIC SOURCE: {source}
VIRTUOUS CHOICE: "{scenario_a}"

IGNATIAN TEMPTATION: "{scenario_b}"

DEVIATION POINT: "{deviation_point}"

Check all Scripture citations for accuracy and evaluate the deviation point."""

    response = await client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,
        system=SYSTEM_PROMPT_IGNATIAN_VERIFY,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {"raw_response": text, "parse_error": True}


async def verify_virtue(
    client: anthropic.AsyncAnthropic,
    virtue: str,
    variant_filter: Optional[str] = None,
    base_only: bool = False,
    model: str = "claude-opus-4-6",
    concurrency: int = 3,
    data_dir: Optional[Path] = None,
) -> dict:
    """Verify all scenarios for a virtue. Returns verification report."""
    root = data_dir or DATA_DIR
    csv_path = root / virtue / "scenarios.csv"

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    sem = asyncio.Semaphore(concurrency)
    results = {
        "virtue": virtue,
        "base_source_issues": [],
        "ignatian_citation_issues": [],
        "total_checked": 0,
        "base_sources_checked": 0,
        "ignatian_variants_checked": 0,
    }

    # 1. Verify base sources (one per base_id, using ratio rows)
    if not variant_filter or variant_filter == "ratio" or base_only:
        ratio_rows = [r for r in rows if r["variant"] == "ratio"]
        print(f"  Verifying {len(ratio_rows)} base source citations...")

        async def check_base(row):
            async with sem:
                result = await verify_base_source(
                    client, row["scenario_a"], row["source"], virtue, model,
                )
            results["base_sources_checked"] += 1
            results["total_checked"] += 1

            is_ok = (
                result.get("source_exists", True)
                and result.get("source_relevant", True)
                and result.get("source_accurate", True)
                and not result.get("parse_error", False)
            )
            if not is_ok:
                issue = {
                    "base_id": row["base_id"],
                    "source": row["source"],
                    "scenario_a_preview": row["scenario_a"][:120],
                    "verification": result,
                }
                results["base_source_issues"].append(issue)
                print(f"    FLAG {row['base_id']}: {result.get('notes', 'parse error')}")
            else:
                confidence = result.get("confidence", "?")
                n = results["base_sources_checked"]
                print(f"    [{n}/{len(ratio_rows)}] {row['base_id']}: OK ({confidence})", flush=True)

        await asyncio.gather(*(check_base(r) for r in ratio_rows))

    # 2. Verify Ignatian variants (Scripture citations + deviation points)
    if not base_only and (not variant_filter or variant_filter == "ignatian"):
        ignatian_rows = [
            r for r in rows
            if r["variant"] == "ignatian" and not r["scenario_b"].startswith("[TODO")
        ]
        if ignatian_rows:
            print(f"  Verifying {len(ignatian_rows)} Ignatian Scripture citations...")

            async def check_ignatian(row):
                async with sem:
                    result = await verify_ignatian_variant(
                        client,
                        row["scenario_a"],
                        row["scenario_b"],
                        row.get("deviation_point", ""),
                        row["source"],
                        virtue,
                        model,
                    )
                results["ignatian_variants_checked"] += 1
                results["total_checked"] += 1

                is_ok = (
                    result.get("citations_real", True)
                    and result.get("citations_accurate", True)
                    and result.get("deviation_point_valid", True)
                    and not result.get("parse_error", False)
                )
                if not is_ok:
                    issue = {
                        "base_id": row["base_id"],
                        "source": row["source"],
                        "scenario_b_preview": row["scenario_b"][:120],
                        "deviation_point": row.get("deviation_point", ""),
                        "verification": result,
                    }
                    results["ignatian_citation_issues"].append(issue)
                    flagged = result.get("flagged_citations", [])
                    notes = result.get("notes", "parse error")
                    print(f"    FLAG {row['base_id']}: {notes}")
                    if flagged:
                        for f in flagged:
                            print(f"      - {f}")
                else:
                    n = results["ignatian_variants_checked"]
                    confidence = result.get("confidence", "?")
                    print(f"    [{n}/{len(ignatian_rows)}] {row['base_id']}: OK ({confidence})", flush=True)

            await asyncio.gather(*(check_ignatian(r) for r in ignatian_rows))

    return results


async def main_async(args: argparse.Namespace):
    """Main async entry point."""
    # Load API key
    for env_path in [
        Path("/Users/timhwang1/Documents/Misc Vibecode/Christian Machine Intelligence/biblical-render/.env"),
        Path("/Users/timhwang1/Documents/Misc Vibecode/Christian Machine Intelligence/virtue-bench/.env"),
        Path("/Users/timhwang1/Documents/Misc Vibecode/Christian Machine Intelligence/virtue-bench-2/.env"),
    ]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip()
                    if val and not os.environ.get(key):
                        os.environ[key] = val

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not found")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    virtues = [args.virtue] if args.virtue else VIRTUES

    all_reports = []
    for virtue in virtues:
        print(f"\n{'='*50}")
        print(f"Verifying: {virtue}")
        print(f"{'='*50}")

        report = await verify_virtue(
            client,
            virtue,
            variant_filter=args.variant,
            base_only=args.base_only,
            model=args.model,
            concurrency=args.concurrency,
        )
        all_reports.append(report)

    # Summary
    print(f"\n{'='*50}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*50}")

    total_issues = 0
    for report in all_reports:
        base_issues = len(report["base_source_issues"])
        ignatian_issues = len(report["ignatian_citation_issues"])
        total_issues += base_issues + ignatian_issues
        print(f"\n{report['virtue']}:")
        print(f"  Base sources: {report['base_sources_checked']} checked, {base_issues} flagged")
        print(f"  Ignatian citations: {report['ignatian_variants_checked']} checked, {ignatian_issues} flagged")

    print(f"\nTotal issues: {total_issues}")

    # Write report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify VirtueBench V2 scenario source citations")
    parser.add_argument("--virtue", choices=VIRTUES, default=None)
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--base-only", action="store_true", help="Only verify base patristic sources")
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--output", type=str,
        default="results/verification_report.json",
        help="Output path for detailed JSON report",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
