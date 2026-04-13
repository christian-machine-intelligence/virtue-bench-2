"""
Unified CLI entry point for VirtueBench V2.

Subcommands:
  run       — Run an experiment (from args or YAML config)
  analyze   — Analyze results with statistical tests
  migrate   — Migrate V1 scenarios to V2 format
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .core.constants import VIRTUES, VARIANTS, DEFAULT_SYSTEM_PROMPT
from .core.psalms import PSALM_SETS, load_psalm_text, list_psalm_sets
from .core.bible import BOOK_SETS, load_bible_text, list_book_sets
from .core.schema import ExperimentConfig
from .eval.experiment import run_experiment, RESULTS_DIR
from .runners import RUNNERS, OpenAIAPIRunner, AnthropicAPIRunner, ClaudeCLIRunner, PiCLIRunner
from .stats.bootstrap import aggregate_runs
from .analysis.tables import print_comparison_table, print_aggregated_table, print_variant_grid
from .artifacts.results import write_results, load_results


def cmd_run(args: argparse.Namespace) -> None:
    """Run an experiment."""
    if args.config:
        with open(args.config) as f:
            config_data = yaml.safe_load(f)
        config = ExperimentConfig(**config_data)
    else:
        config = ExperimentConfig(
            name=args.output or f"experiment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            model=args.model,
            virtues=VIRTUES if args.subset == "all" else [args.subset],
            variants=VARIANTS if args.variant == "all" else [args.variant],
            runs=args.runs,
            temperature=args.temperature,
            frame="default",
            seed=args.seed,
            limit=args.limit,
            injection_file=args.inject,
            concurrency=args.concurrency,
            retries=args.retries,
            timeout=args.timeout,
            detailed=args.detailed,
        )

    if args.quick:
        config.limit = 10

    # Handle psalm injection (overrides --inject if both specified)
    psalm_set_names = getattr(args, "psalm_set", None)
    psalm_nums = getattr(args, "psalm_numbers", None)
    psalm_random = getattr(args, "psalm_random", None)

    if psalm_set_names or psalm_nums or psalm_random:
        parsed_nums = None
        if psalm_nums:
            parsed_nums = [int(n.strip()) for n in psalm_nums.split(",")]

        from .core.psalms import load_psalm_text
        import tempfile
        psalm_text = load_psalm_text(
            psalm_sets=psalm_set_names,
            psalm_numbers=parsed_nums,
            random_n=psalm_random,
            seed=config.seed,
        )
        # Write to temp file for injection
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        )
        tmp.write(psalm_text)
        tmp.close()
        config.injection_file = tmp.name
        n_psalms = len(psalm_text.split("Psalm ")) - 1
        sets_label = ",".join(psalm_set_names) if psalm_set_names else ""
        print(f"Psalm injection: {n_psalms} psalms ({sets_label or 'custom selection'})")

    # Handle Bible book injection
    bible_books_arg = getattr(args, "bible", None)
    bible_set_arg = getattr(args, "bible_set", None)

    if bible_books_arg or bible_set_arg:
        import tempfile
        bible_text = load_bible_text(
            books=bible_books_arg,
            book_set=bible_set_arg,
        )
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        )
        tmp.write(bible_text)
        tmp.close()
        config.injection_file = tmp.name
        label = bible_set_arg or ",".join(bible_books_arg)
        print(f"Bible injection: {label} (KJV)")

    # Select runner
    # Strip provider prefix (e.g. "openai/gpt-4o" -> "gpt-4o") for SDK/CLI runners
    model_name = config.model.split("/", 1)[-1] if "/" in config.model else config.model

    if args.runner == "openai-api":
        runner = OpenAIAPIRunner(model=model_name)
    elif args.runner == "anthropic-api":
        runner = AnthropicAPIRunner(model=model_name)
    elif args.runner == "claude-cli":
        runner = ClaudeCLIRunner(model=model_name, effort=getattr(args, "effort", "low"))
    elif args.runner == "pi-cli":
        runner = PiCLIRunner(model=model_name)
    elif args.runner == "inspect" and "inspect" in RUNNERS:
        runner = RUNNERS["inspect"](model=config.model)
    elif args.runner == "hf-local":
        from .runners.hf_local import HFLocalRunner
        runner = HFLocalRunner(
            model_name=config.model,
            adapter_path=getattr(args, "hf_adapter", None),
        )
    else:
        # Auto-detect from model name
        if "claude" in config.model.lower() or "anthropic" in config.model.lower():
            runner = AnthropicAPIRunner(model=model_name)
        else:
            runner = OpenAIAPIRunner(model=model_name)

    print(f"Model: {runner.model_id()}")
    print(f"Runner: {args.runner}")
    print(f"Virtues: {config.virtues}")
    print(f"Variants: {config.variants}")
    print(f"Runs: {config.runs}")
    print(f"Temperature: {config.temperature}")

    print(f"Limit: {config.limit or 'all'}")

    # Set up checkpoint path for incremental saves
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = args.output or f"results_{timestamp}"
    if not filename.endswith(".json"):
        filename += ".json"
    output_path = RESULTS_DIR / filename
    checkpoint_path = output_path.with_name(f"{output_path.stem}_checkpoint.json")

    results = asyncio.run(run_experiment(config, runner, checkpoint_path=checkpoint_path))

    # Clean up checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    summary_path, logs_path = write_results(results, output_path, write_logs=config.detailed)
    print(f"\nResults saved to: {summary_path}")
    if logs_path:
        print(f"Detailed logs saved to: {logs_path}")

    # Print tables
    print_comparison_table(results)

    if config.runs > 1:
        aggregated = aggregate_runs(results)
        print_aggregated_table(aggregated)
        print_variant_grid(aggregated)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze existing results."""
    from .stats.bootstrap import aggregate_runs
    from .stats.tests import chi_squared_variant
    from .core.schema import RunResult

    path = Path(args.results)
    data = load_results(path)

    # Try to reconstruct RunResults
    results = []
    for d in data:
        try:
            results.append(RunResult(**d))
        except Exception:
            pass

    if not results:
        print("Could not parse results as RunResult objects. Showing raw data.")
        for d in data:
            print(json.dumps(d, indent=2))
        return

    aggregated = aggregate_runs(results)
    print_aggregated_table(aggregated)
    print_variant_grid(aggregated)

    # Chi-squared test across variants
    chi2_result = chi_squared_variant(results)
    print(f"\nChi-squared test across variants:")
    print(f"  chi2 = {chi2_result['chi2']:.4f}, df = {chi2_result['df']}")
    if chi2_result["p_value"] is not None:
        print(f"  p = {chi2_result['p_value']:.6f}")


def cmd_migrate(args: argparse.Namespace) -> None:
    """Migrate V1 scenarios to V2 format."""
    from .migrate import migrate_v1_to_v2
    migrate_v1_to_v2(
        v1_data_dir=Path(args.v1_dir),
        v2_data_dir=Path(args.v2_dir),
    )


def main():
    parser = argparse.ArgumentParser(
        description="VirtueBench V2: Multi-dimensional virtue evaluation benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--config", type=str, help="YAML config file")
    run_parser.add_argument("--model", default="anthropic/claude-sonnet-4-20250514")
    run_parser.add_argument("--runner", choices=list(RUNNERS.keys()), default="inspect")
    run_parser.add_argument("--hf-adapter", type=str, default=None,
                            help="LoRA adapter path for hf-local runner")
    run_parser.add_argument("--subset", choices=VIRTUES + ["all"], default="all")
    run_parser.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    run_parser.add_argument("--runs", type=int, default=5)
    run_parser.add_argument("--temperature", type=float, default=0.7)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--quick", action="store_true", help="10 samples per virtue")
    run_parser.add_argument("--inject", type=str, help="Injection text file path")
    run_parser.add_argument("--concurrency", type=int, default=5)
    run_parser.add_argument("--retries", type=int, default=2)
    run_parser.add_argument("--timeout", type=int, default=120)
    run_parser.add_argument("--detailed", action="store_true")
    run_parser.add_argument("--output", type=str, default=None)
    run_parser.add_argument("--effort", choices=["low", "medium", "high", "max"], default="low")
    run_parser.add_argument(
        "--deterministic", action="store_true",
        help="V1-compat: single run at temperature=0",
    )
    # Psalm injection args
    run_parser.add_argument(
        "--psalm-set", action="append", dest="psalm_set",
        choices=list(PSALM_SETS.keys()),
        help="Named psalm subset for injection (can repeat to combine)",
    )
    run_parser.add_argument(
        "--psalm-numbers", type=str, default=None,
        help="Comma-separated psalm numbers for injection (e.g. 23,51,91)",
    )
    run_parser.add_argument(
        "--psalm-random", type=int, default=None,
        help="Inject N randomly-selected psalms",
    )

    # Bible injection args
    run_parser.add_argument(
        "--bible", action="append", dest="bible",
        help="Bible book for injection (e.g., 'Romans', 'MAT:5-7'). Can repeat.",
    )
    run_parser.add_argument(
        "--bible-set", type=str, default=None,
        choices=list(BOOK_SETS.keys()),
        help="Named Bible book collection for injection",
    )

    # --- psalms (list available sets) ---
    psalms_parser = subparsers.add_parser("psalms", help="List available psalm sets")

    # --- bible (list available book sets) ---
    bible_parser = subparsers.add_parser("bible", help="List available Bible book sets")

    # --- analyze ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("results", type=str, help="Path to results JSON")

    # --- migrate ---
    migrate_parser = subparsers.add_parser("migrate", help="Migrate V1 scenarios to V2 format")
    migrate_parser.add_argument("--v1-dir", required=True, help="Path to V1 data/ directory")
    migrate_parser.add_argument("--v2-dir", required=True, help="Path to V2 data/ directory")

    args = parser.parse_args()

    if args.command == "run":
        if getattr(args, "deterministic", False):
            args.runs = 1
            args.temperature = 0.0
        cmd_run(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    elif args.command == "psalms":
        sets = list_psalm_sets()
        print("\nAvailable psalm sets:\n")
        for name, desc in sets.items():
            psalms = PSALM_SETS[name]["psalms"]
            print(f"  {name:<18} {desc}")
            print(f"  {'':18} Psalms: {','.join(str(p) for p in psalms)}")
            print()
    elif args.command == "bible":
        print("\nAvailable Bible book sets (KJV, all 66 books):\n")
        for name, desc in list_book_sets().items():
            books = BOOK_SETS[name]["books"]
            print(f"  {name:<24} {desc}")
            print(f"  {'':24} Books: {', '.join(books)}")
            print()
        print("Usage:")
        print("  virtue-bench run --bible Romans")
        print("  virtue-bench run --bible 'Matthew 5-7'")
        print("  virtue-bench run --bible-set sermon_on_the_mount")
        print("  virtue-bench run --bible Romans --bible James")
    else:
        parser.print_help()
