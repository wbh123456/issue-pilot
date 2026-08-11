"""Minimal CLI for IssuePilot.

Usage:
    python cli.py solve issue-001
    python cli.py solve issue-002 --max-steps 20
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from agent.client import default_model
from eval.runner import solve_task

# force_terminal + utf-8-safe printing avoids Windows cp1252 crashes on arrows etc.
console = Console(force_terminal=True, soft_wrap=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="IssuePilot harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="Solve one benchmark task with the agent")
    solve.add_argument("task_id", help="Task id from eval/dataset.json, e.g. issue-001")
    solve.add_argument(
        "--model",
        default=None,
        help=f"Model id (default: {default_model()})",
    )
    solve.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Max ReAct steps (default: 15)",
    )
    return parser


def _print_summary(record: dict) -> None:
    table = Table(title=f"Solve result: {record['task_id']}")
    table.add_column("Field")
    table.add_column("Value")

    rows = [
        ("success", str(record.get("success"))),
        ("difficulty", str(record.get("difficulty"))),
        ("termination", str(record.get("termination"))),
        ("steps", str(record.get("steps"))),
        ("tool_call_count", str(record.get("tool_call_count"))),
        ("file_reads", str(record.get("file_reads"))),
        ("tokens", str(record.get("tokens"))),
        ("latency_s", f"{float(record.get('latency') or 0):.2f}"),
        ("run_path", str(record.get("run_path"))),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)

    final = (record.get("final_answer") or "").strip()
    if final:
        console.print("\n[bold]Final answer[/bold]")
        try:
            console.print(final)
        except UnicodeEncodeError:
            sys.stdout.buffer.write((final + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "solve":
        console.print(f"[cyan]Solving {args.task_id}...[/cyan]")
        try:
            record = solve_task(
                args.task_id,
                model=args.model,
                max_steps=args.max_steps,
            )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
            return 1

        _print_summary(record)
        return 0 if record.get("success") else 2

    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
