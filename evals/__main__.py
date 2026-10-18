from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Sequence

from evals.runner import format_results_table, run_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="WISMO Chat Assistant evaluation harness",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run evaluation scenarios and print results table",
    )
    run_parser.add_argument(
        "--scenario",
        "-s",
        dest="scenario",
        default=None,
        help="Name (or comma-separated names) of scenario(s) to run",
    )
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command not in ("run", None):
        parser.print_help()
        return 2

    scenario_filter = getattr(args, "scenario", None)
    try:
        results = await run_suite(scenario_filter=scenario_filter)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not results:
        print("No scenarios found to run.", file=sys.stderr)
        return 2

    table = format_results_table(results)
    print(table)

    any_failed = any(not r.passed for r in results)
    return 1 if any_failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    sys.exit(main())
