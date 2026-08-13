"""Minimal CLI for IssuePilot.

Usage:
    python cli.py solve issue-001
    python cli.py solve issue-001 --harness v1
    python cli.py compare issue-001
    python cli.py sandbox doctor
    python cli.py sandbox build
    python cli.py solve issue-002 --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from agent.client import default_model
from eval.runner import solve_task
from harness.limits import MAX_AGENT_STEPS
from sandbox.image import DEFAULT_IMAGE, DockerPreflightError, build_image, doctor

# force_terminal + utf-8-safe printing avoids Windows cp1252 crashes on arrows etc.
console = Console(force_terminal=True, soft_wrap=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="IssuePilot harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="Solve one benchmark task with the agent")
    solve.add_argument("task_id", help="Task id from eval/dataset.json, e.g. issue-001")
    solve.add_argument(
        "--harness",
        choices=("v0", "v1"),
        default="v0",
        help="Harness version: v0 ReAct loop or v1 LangGraph Plan-Execute (default: v0)",
    )
    solve.add_argument(
        "--model",
        default=None,
        help=f"Model id (default: {default_model()})",
    )
    solve.add_argument(
        "--max-steps",
        type=int,
        default=MAX_AGENT_STEPS,
        help=f"Max executor ReAct steps (default: {MAX_AGENT_STEPS})",
    )

    compare = sub.add_parser(
        "compare",
        help="Run v0 and v1 on the same task with identical settings",
    )
    compare.add_argument("task_id", help="Task id from eval/dataset.json, e.g. issue-001")
    compare.add_argument(
        "--model",
        default=None,
        help=f"Model id (default: {default_model()})",
    )
    compare.add_argument(
        "--max-steps",
        type=int,
        default=MAX_AGENT_STEPS,
        help=f"Max executor ReAct steps for both harnesses (default: {MAX_AGENT_STEPS})",
    )

    sandbox = sub.add_parser(
        "sandbox",
        help="Docker sandbox image preflight and build (no host fallback)",
    )
    sandbox_sub = sandbox.add_subparsers(dest="sandbox_command", required=True)
    doctor_p = sandbox_sub.add_parser(
        "doctor",
        help="Check Docker Desktop / Linux mode / sandbox image readiness",
    )
    doctor_p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable doctor report",
    )
    doctor_p.add_argument(
        "--require-image",
        action="store_true",
        help="Fail if the sandbox image is not built yet",
    )
    build_p = sandbox_sub.add_parser(
        "build",
        help=f"Build the sandbox image ({DEFAULT_IMAGE})",
    )
    build_p.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Image tag (default: {DEFAULT_IMAGE})",
    )
    return parser


def _print_summary(record: dict) -> None:
    harness = record.get("harness_version") or "v0"
    table = Table(title=f"Solve result: {record['task_id']} ({harness})")
    table.add_column("Field")
    table.add_column("Value")

    rows = [
        ("harness", str(harness)),
        ("success", str(record.get("success"))),
        ("difficulty", str(record.get("difficulty"))),
        ("termination", str(record.get("termination"))),
        ("status", str(record.get("status") or "-")),
        ("workflow_passed", str(record.get("workflow_passed")
                                if record.get("workflow_passed") is not None
                                else "-")),
        ("steps", str(record.get("steps"))),
        ("llm_calls", str(record.get("llm_calls"))),
        ("tool_call_count", str(record.get("tool_call_count"))),
        ("file_reads", str(record.get("file_reads"))),
        ("tokens", str(record.get("tokens"))),
        ("latency_s", f"{float(record.get('latency') or 0):.2f}"),
        ("sandbox", str(record.get("sandbox_backend") or "-")),
        ("sandbox_image", str(record.get("sandbox_image") or "-")),
        ("sandbox_network", str(record.get("sandbox_network") or "-")),
        ("sandbox_commands", str(record.get("sandbox_command_count", "-"))),
        ("sandbox_cleaned_up", str(record.get("sandbox_cleaned_up", "-"))),
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


def _print_compare(v0: dict, v1: dict) -> None:
    table = Table(title=f"Compare: {v0.get('task_id')}")
    table.add_column("Metric")
    table.add_column("v0")
    table.add_column("v1")

    def fmt_latency(record: dict) -> str:
        return f"{float(record.get('latency') or 0):.2f}"

    rows = [
        ("gold_success", str(v0.get("success")), str(v1.get("success"))),
        ("workflow_passed", "-", str(v1.get("workflow_passed"))),
        ("termination", str(v0.get("termination")), str(v1.get("termination"))),
        ("status", "-", str(v1.get("status") or "-")),
        ("steps", str(v0.get("steps")), str(v1.get("steps"))),
        ("llm_calls", str(v0.get("llm_calls")), str(v1.get("llm_calls"))),
        ("tool_call_count", str(v0.get("tool_call_count")), str(v1.get("tool_call_count"))),
        ("file_reads", str(v0.get("file_reads")), str(v1.get("file_reads"))),
        ("tokens", str(v0.get("tokens")), str(v1.get("tokens"))),
        ("latency_s", fmt_latency(v0), fmt_latency(v1)),
        ("sandbox", str(v0.get("sandbox_backend") or "-"), str(v1.get("sandbox_backend") or "-")),
        ("sandbox_commands", str(v0.get("sandbox_command_count", "-")), str(v1.get("sandbox_command_count", "-"))),
        ("sandbox_cleaned_up", str(v0.get("sandbox_cleaned_up", "-")), str(v1.get("sandbox_cleaned_up", "-"))),
        ("run_path", str(v0.get("run_path")), str(v1.get("run_path"))),
    ]
    for metric, left, right in rows:
        table.add_row(metric, left, right)
    console.print(table)


def _print_doctor(report) -> None:
    table = Table(title="Sandbox doctor")
    table.add_column("Check")
    table.add_column("Value")
    rows = [
        ("ok", str(report.ok)),
        ("docker_path", str(report.docker_path or "-")),
        ("daemon_reachable", str(report.daemon_reachable)),
        ("server_os", str(report.server_os or "-")),
        ("linux_containers", str(report.linux_containers)),
        ("image", report.image),
        ("image_present", str(report.image_present)),
        ("dockerfile_present", str(report.dockerfile_present)),
        ("requirements_present", str(report.requirements_present)),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)
    for warning in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    for error in report.errors:
        console.print(f"[red]error:[/red] {error}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "solve":
        console.print(
            f"[cyan]Solving {args.task_id} with harness {args.harness}...[/cyan]"
        )
        try:
            record = solve_task(
                args.task_id,
                model=args.model,
                max_steps=args.max_steps,
                harness_version=args.harness,
            )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
            return 1

        _print_summary(record)
        return 0 if record.get("success") else 2

    if args.command == "compare":
        model = args.model
        max_steps = args.max_steps
        records: dict[str, dict] = {}

        for harness in ("v0", "v1"):
            console.print(
                f"[cyan]Running {args.task_id} with harness {harness} "
                f"(model={model or default_model()}, max_steps={max_steps})...[/cyan]"
            )
            try:
                records[harness] = solve_task(
                    args.task_id,
                    model=model,
                    max_steps=max_steps,
                    harness_version=harness,
                )
            except Exception as exc:
                console.print(
                    f"[red]Error during {harness}:[/red] "
                    f"{type(exc).__name__}: {exc}"
                )
                return 1
            _print_summary(records[harness])
            console.print()

        _print_compare(records["v0"], records["v1"])
        # Exit 0 only if both gold tests passed.
        both_ok = records["v0"].get("success") and records["v1"].get("success")
        return 0 if both_ok else 2

    if args.command == "sandbox":
        if args.sandbox_command == "doctor":
            report = doctor(require_image=args.require_image)
            if args.json:
                console.print_json(json.dumps(report.to_dict()))
            else:
                _print_doctor(report)
            return 0 if report.ok else 1

        if args.sandbox_command == "build":
            console.print(f"[cyan]Building sandbox image {args.image}...[/cyan]")
            try:
                tag = build_image(image=args.image)
            except DockerPreflightError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                return 1
            console.print(f"[green]Built[/green] {tag}")
            return 0

        parser.error(f"unknown sandbox command: {args.sandbox_command}")
        return 1

    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
