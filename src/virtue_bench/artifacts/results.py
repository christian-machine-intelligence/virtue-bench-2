"""
Result artifact I/O for VirtueBench V2.

Writes summary files (no per-sample detail) and detailed log files
(with per-sample responses and rationales).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from ..core.schema import RunResult


def write_results(
    results: List[RunResult],
    output_path: Path,
    write_logs: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """Write experiment results to JSON files.

    Returns (summary_path, logs_path_or_none).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Summary: strip sample_details
    summary = []
    for r in results:
        d = r.model_dump()
        d.pop("sample_details", None)
        summary.append(d)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logs_path = None
    if write_logs:
        logs_path = output_path.with_name(f"{output_path.stem}_logs.json")
        full = [r.model_dump() for r in results]
        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=2, default=str)

    return output_path, logs_path


def load_results(path: Path) -> List[dict]:
    """Load a JSON results file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a result list")
    return data
