"""
Multi-run experiment orchestrator for VirtueBench V2.

Supports:
- Multiple runs per configuration for statistical rigor
- Configurable temperature for genuine variance across runs
- Per-run seed variation for A/B randomization
- Filtering by virtue and variant
- Both batch (Inspect AI) and individual-query runner backends
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from ..core.constants import FRAMES
from ..core.loader import load_scenarios, prepare_samples
from ..core.schema import ExperimentConfig, RunResult, SampleResult
from ..runners.base import ModelRunner
from ..runners.inspect_ai import InspectAIRunner
from .scorer import score_response


RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results"


async def _run_single_query_batch(
    runner: ModelRunner,
    samples: list,
    system_prompt: str,
    temperature: float,
    concurrency: int,
    retries: int,
    timeout: int,
    detailed: bool,
) -> List[SampleResult]:
    """Run samples through a query-based runner with concurrency control."""
    sem = asyncio.Semaphore(concurrency)
    results: List[Optional[SampleResult]] = [None] * len(samples)

    async def process(i: int, sample):
        async with sem:
            outcome = await runner.query(
                sample.prompt,
                system_prompt,
                temperature=temperature,
                retries=retries,
                timeout=timeout,
            )
        result = score_response(sample, outcome["response"], outcome.get("infra_error"))
        results[i] = result
        status = "correct" if result.correct else ("infra:" + (result.infra_error or "")) if result.infra_error else "incorrect"
        print(f"  [{i+1}/{len(samples)}] {status}", flush=True)

    await asyncio.gather(*(process(i, s) for i, s in enumerate(samples)))
    return [r for r in results if r is not None]


def _run_inspect_batch(
    runner: InspectAIRunner,
    samples: list,
    system_prompt: str,
    temperature: float,
    log_dir: Optional[str],
) -> Tuple[Optional[float], Optional[float], int, str, List[SampleResult]]:
    """Run samples through the Inspect AI batch runner."""
    batch_result = runner.run_batch(
        samples, system_prompt, temperature=temperature, log_dir=log_dir,
    )

    sample_results = []
    for detail in batch_result.get("sample_details", []):
        sample_results.append(SampleResult(
            sample_id=detail.get("sample_id", ""),
            variant=detail.get("metadata", {}).get("variant", "ratio"),
            target=detail.get("target", ""),
            model_response=detail.get("model_response", ""),
            model_answer=detail.get("model_answer"),
            correct=detail.get("correct"),
            prompt=detail.get("prompt", ""),
            metadata=detail.get("metadata", {}),
        ))

    return (
        batch_result.get("accuracy"),
        batch_result.get("stderr"),
        batch_result.get("samples", len(samples)),
        batch_result.get("status", "unknown"),
        sample_results,
    )


async def run_single_condition(
    runner: ModelRunner,
    virtue: str,
    variant: str,
    frame: str,
    run_index: int,
    seed: int,
    temperature: float,
    limit: Optional[int],
    concurrency: int,
    retries: int,
    timeout: int,
    detailed: bool,
    injection_text: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> RunResult:
    """Run one virtue × variant × run combination."""
    run_seed = seed + run_index
    scenarios = load_scenarios(virtue, variants=[variant])
    samples = prepare_samples(scenarios, seed=run_seed, limit=limit)

    sys_prompt = FRAMES[frame]
    if injection_text:
        sys_prompt = injection_text + "\n\n---\n\n" + sys_prompt

    condition = frame if not injection_text else f"{frame}+injected"

    if isinstance(runner, InspectAIRunner):
        acc, stderr, n_samples, status, sample_results = _run_inspect_batch(
            runner, samples, sys_prompt, temperature, log_dir,
        )
    else:
        sample_results = await _run_single_query_batch(
            runner, samples, sys_prompt, temperature,
            concurrency, retries, timeout, detailed,
        )
        scored = [r for r in sample_results if r.infra_error is None]
        correct = sum(1 for r in scored if r.correct)
        n_samples = len(samples)
        acc = correct / n_samples if scored and len(scored) == n_samples else None
        stderr = None
        status = "success" if len(scored) == n_samples else "partial"

    return RunResult(
        model=runner.model_id(),
        virtue=virtue,
        variant=variant,
        condition=condition,
        frame=frame,
        run_index=run_index,
        seed=run_seed,
        temperature=temperature,
        accuracy=acc,
        stderr=stderr,
        samples=n_samples,
        status=status,
        sample_details=sample_results if detailed else [],
    )


async def run_experiment(
    config: ExperimentConfig,
    runner: ModelRunner,
) -> List[RunResult]:
    """Run a full experiment according to the config.

    Iterates over virtues × variants × runs, collecting RunResults.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = str(RESULTS_DIR / "logs" / timestamp)

    injection_text = None
    if config.injection_file:
        injection_text = Path(config.injection_file).read_text(encoding="utf-8")

    all_results: List[RunResult] = []

    for virtue in config.virtues:
        for variant in config.variants:
            for run_idx in range(config.runs):
                print(f"\n{'='*60}")
                print(f"Model: {runner.model_id()} | {virtue}/{variant} | Run {run_idx+1}/{config.runs}")
                print(f"{'='*60}")

                result = await run_single_condition(
                    runner=runner,
                    virtue=virtue,
                    variant=variant,
                    frame=config.frame,
                    run_index=run_idx,
                    seed=config.seed,
                    temperature=config.temperature,
                    limit=config.limit,
                    concurrency=config.concurrency,
                    retries=config.retries,
                    timeout=config.timeout,
                    detailed=config.detailed,
                    injection_text=injection_text,
                    log_dir=log_dir,
                )
                all_results.append(result)

                acc = f"{result.accuracy:.4f}" if result.accuracy is not None else "N/A"
                print(f"  Accuracy: {acc}")

    return all_results
