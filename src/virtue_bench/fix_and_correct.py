"""
Combined fix script for VirtueBench V2:
1. Re-verify Ignatian scenarios that had parse errors (with robust JSON parsing)
2. Correct source citations for flagged base scenarios
3. Write updated CSVs

Usage:
    python -m virtue_bench.fix_and_correct
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic

from .core.constants import DATA_DIR, VIRTUES


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from model response, handling various wrapping formats."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Try finding JSON object boundaries
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break

    return None


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

IGNATIAN_VERIFY_PROMPT = """You are a biblical scholar verifying Scripture citations.

Check whether Bible verses cited in this temptation argument are REAL and ACCURATELY quoted.
Also evaluate whether the deviation_point correctly identifies where reasoning turns from virtue to vice.

You MUST respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{"citations_real": true, "citations_accurate": true, "deviation_point_valid": true, "confidence": "high", "notes": "...", "flagged_citations": []}"""

SOURCE_CORRECT_PROMPT = """You are a patristic theology scholar. A VirtueBench scenario has been \
flagged because its source citation does not properly support the moral scenario described.

Your task: suggest the BEST patristic source citation for this scenario. Choose from:
- Thomas Aquinas, Summa Theologiae (cite as "Aquinas, ST [part] Q.[num] a.[num]")
- Augustine of Hippo (cite specific work and section)
- Ambrose of Milan (cite specific work and section)
- Gregory the Great, Moralia in Job or Regula Pastoralis
- John Chrysostom (cite specific homily or work)
- Basil the Great (cite specific work)

Respond with ONLY a JSON object:
{"corrected_source": "Author, Work Section", "justification": "one sentence explaining why this source fits"}"""


async def reverify_ignatian(
    client: anthropic.AsyncAnthropic,
    scenario_a: str,
    scenario_b: str,
    deviation_point: str,
    virtue: str,
    model: str = "claude-opus-4-6",
) -> dict:
    """Re-verify an Ignatian variant with robust parsing."""
    prompt = f"""Verify Scripture citations in this {virtue} Ignatian temptation:

VIRTUOUS CHOICE: "{scenario_a}"
TEMPTATION: "{scenario_b}"
DEVIATION POINT: "{deviation_point}"

Respond with ONLY JSON."""

    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=512,
                temperature=0,
                system=IGNATIAN_VERIFY_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=60,
        )
    except (asyncio.TimeoutError, Exception) as e:
        return {"parse_error": True, "raw_response": f"timeout/error: {e}"}

    text = response.content[0].text.strip()
    result = extract_json(text)
    if result is None:
        return {"parse_error": True, "raw_response": text[:300]}
    return result


async def correct_source(
    client: anthropic.AsyncAnthropic,
    scenario_a: str,
    current_source: str,
    virtue: str,
    flag_notes: str,
    model: str = "claude-opus-4-6",
) -> dict:
    """Get a corrected source citation for a flagged scenario."""
    prompt = f"""This {virtue} scenario needs a corrected patristic source citation.

SCENARIO (virtuous choice): "{scenario_a}"
CURRENT (INCORRECT) SOURCE: {current_source}
REASON FLAGGED: {flag_notes}

Suggest the best patristic source. Respond with ONLY JSON."""

    try:
        response = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=256,
                temperature=0,
                system=SOURCE_CORRECT_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=60,
        )
    except (asyncio.TimeoutError, Exception) as e:
        return {"parse_error": True, "raw_response": f"timeout/error: {e}"}

    text = response.content[0].text.strip()
    result = extract_json(text)
    if result is None:
        return {"parse_error": True, "raw_response": text[:300]}
    return result


async def main_async(args: argparse.Namespace):
    """Main entry point."""
    # Load API key
    for env_path in [
        Path("/Users/timhwang1/Documents/Misc Vibecode/Christian Machine Intelligence/biblical-render/.env"),
        Path("/Users/timhwang1/Documents/Misc Vibecode/Christian Machine Intelligence/virtue-bench/.env"),
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

    # Load verification report
    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"Verification report not found: {report_path}")

    with open(report_path) as f:
        reports = json.load(f)

    sem = asyncio.Semaphore(args.concurrency)

    # -----------------------------------------------------------------------
    # Step 1: Re-verify Ignatian parse errors
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Re-verifying Ignatian scenarios with parse errors")
    print("=" * 60)

    ignatian_parse_errors = []
    for report in reports:
        for issue in report.get("ignatian_citation_issues", []):
            if issue.get("verification", {}).get("parse_error"):
                ignatian_parse_errors.append((report["virtue"], issue))

    print(f"  Found {len(ignatian_parse_errors)} parse errors to re-verify")

    # Load all scenario data for lookup
    all_scenarios: Dict[str, Dict[str, dict]] = {}  # virtue -> base_id -> row
    for virtue in VIRTUES:
        csv_path = DATA_DIR / virtue / "scenarios.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["variant"] == "ignatian":
                    all_scenarios.setdefault(virtue, {})[row["base_id"]] = row

    reverify_results = {}
    reverify_count = 0
    still_parse_error = 0
    new_ignatian_flags = []

    async def do_reverify(virtue: str, issue: dict):
        nonlocal reverify_count, still_parse_error
        base_id = issue["base_id"]
        row = all_scenarios.get(virtue, {}).get(base_id)
        if not row:
            print(f"    SKIP {base_id}: not found in CSV")
            return

        async with sem:
            result = await reverify_ignatian(
                client,
                row["scenario_a"],
                row["scenario_b"],
                row.get("deviation_point", ""),
                virtue,
            )

        reverify_count += 1
        if result.get("parse_error"):
            still_parse_error += 1
            print(f"    [{reverify_count}/{len(ignatian_parse_errors)}] {base_id}: still parse error")
        else:
            is_ok = (
                result.get("citations_real", True)
                and result.get("citations_accurate", True)
                and result.get("deviation_point_valid", True)
            )
            if is_ok:
                confidence = result.get("confidence", "?")
                print(f"    [{reverify_count}/{len(ignatian_parse_errors)}] {base_id}: OK ({confidence})")
            else:
                new_ignatian_flags.append({"base_id": base_id, "virtue": virtue, "verification": result})
                notes = result.get("notes", "issue found")
                print(f"    [{reverify_count}/{len(ignatian_parse_errors)}] FLAG {base_id}: {notes}")

    await asyncio.gather(*(do_reverify(v, i) for v, i in ignatian_parse_errors))

    print(f"\n  Re-verified: {reverify_count}")
    print(f"  Now OK: {reverify_count - still_parse_error - len(new_ignatian_flags)}")
    print(f"  New flags: {len(new_ignatian_flags)}")
    print(f"  Still parse error: {still_parse_error}")

    # -----------------------------------------------------------------------
    # Step 2: Correct flagged base sources
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Correcting flagged base source citations")
    print("=" * 60)

    source_flags = []
    for report in reports:
        for issue in report.get("base_source_issues", []):
            # Handle both full verification format and simplified format
            v = issue.get("verification", {})
            if not v.get("parse_error"):
                source_flags.append((report["virtue"], issue))

    print(f"  Found {len(source_flags)} source citations to correct")

    corrections = []
    correct_count = 0

    async def do_correct(virtue: str, issue: dict):
        nonlocal correct_count
        base_id = issue["base_id"]
        # Handle both report formats
        notes = issue.get("notes", "") or issue.get("verification", {}).get("notes", "")

        # Find scenario_a and current source from CSV
        csv_path = DATA_DIR / virtue / "scenarios.csv"
        scenario_a = ""
        current_source = issue.get("source", "")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["base_id"] == base_id and row["variant"] == "ratio":
                    scenario_a = row["scenario_a"]
                    if not current_source:
                        current_source = row["source"]
                    break

        async with sem:
            result = await correct_source(
                client,
                scenario_a,
                current_source,
                virtue,
                notes,
            )

        correct_count += 1
        if result.get("parse_error"):
            print(f"    [{correct_count}/{len(source_flags)}] {base_id}: correction parse error")
            corrections.append({
                "base_id": base_id, "virtue": virtue,
                "old_source": current_source,
                "corrected_source": None,
                "error": "parse_error",
            })
        else:
            new_source = result.get("corrected_source", "")
            justification = result.get("justification", "")
            print(f"    [{correct_count}/{len(source_flags)}] {base_id}: {current_source} → {new_source}")
            print(f"      Reason: {justification}")
            corrections.append({
                "base_id": base_id, "virtue": virtue,
                "old_source": current_source,
                "corrected_source": new_source,
                "justification": justification,
            })

    await asyncio.gather(*(do_correct(v, i) for v, i in source_flags))

    # -----------------------------------------------------------------------
    # Step 3: Apply corrections to CSVs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Applying source corrections to CSVs")
    print("=" * 60)

    # Build correction map: (virtue, base_id) -> new_source
    correction_map: Dict[Tuple[str, str], str] = {}
    for c in corrections:
        if c.get("corrected_source"):
            correction_map[(c["virtue"], c["base_id"])] = c["corrected_source"]

    print(f"  {len(correction_map)} corrections to apply")

    for virtue in VIRTUES:
        csv_path = DATA_DIR / virtue / "scenarios.csv"
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)

        applied = 0
        for row in rows:
            key = (virtue, row["base_id"])
            if key in correction_map:
                old = row["source"]
                row["source"] = correction_map[key]
                applied += 1

        if applied > 0:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  {virtue}: {applied} sources corrected")
        else:
            print(f"  {virtue}: no corrections needed")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Ignatian re-verified: {reverify_count} (parse errors resolved: {reverify_count - still_parse_error})")
    print(f"  New Ignatian flags: {len(new_ignatian_flags)}")
    print(f"  Source corrections applied: {len(correction_map)}")
    print(f"  Source corrections failed: {len(corrections) - len(correction_map)}")

    # Write detailed fix report
    fix_report = {
        "ignatian_reverify": {
            "total": reverify_count,
            "resolved": reverify_count - still_parse_error - len(new_ignatian_flags),
            "new_flags": new_ignatian_flags,
            "still_parse_error": still_parse_error,
        },
        "source_corrections": corrections,
    }
    fix_report_path = Path("results/fix_report.json")
    fix_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fix_report_path, "w") as f:
        json.dump(fix_report, f, indent=2, default=str)
    print(f"\n  Detailed report: {fix_report_path}")


def main():
    parser = argparse.ArgumentParser(description="Fix and correct VirtueBench V2 scenarios")
    parser.add_argument(
        "--report", default="results/verification_report.json",
        help="Path to verification report JSON",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
