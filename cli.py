"""Minimal CLI for IssuePilot.

Usage:
    python cli.py solve issue-001
    python cli.py solve issue-001 --harness v1
    python cli.py solve issue-009 --harness v2
    python cli.py solve issue-001 --harness v1 --require-approval
    python cli.py solve issue-001 --harness v1 --pause-on-approval
    python cli.py compare issue-001
    python cli.py retrieve issue-009
    python cli.py retrieve --split hard
    python cli.py report --split hard
    python cli.py report --split hard --latest-per-cell
    python cli.py bench --split hard --harness v0,v1,v2 --n 1 --log
    python cli.py sandbox doctor
    python cli.py sandbox build
    python cli.py solve issue-001 --harness v1 --interactive-recovery
    python cli.py runs
    python cli.py review <run_id>
    python cli.py resume <run_id> --approve
    python cli.py resume <run_id> --reject
    python cli.py resume <run_id> --feedback "Drop the unrelated edit"
    python cli.py mcp serve --repo ../issue-pilot-benchmark
    python cli.py mcp demo --repo ../issue-pilot-benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent.client import default_model
from eval.retrieval import ALL_MODES, run_retrieval_eval
from harness.context import RETRIEVE_K
from harness.limits import MAX_AGENT_STEPS
from harness.progress import ConsoleReporter, format_recovery_summary, format_review
from retrieval.query import DEFAULT_QUERY_MODE, QUERY_MODES
from sandbox.image import DEFAULT_IMAGE, DockerPreflightError, build_image, doctor

# force_terminal + utf-8-safe printing avoids Windows cp1252 crashes on arrows etc.
console = Console(force_terminal=True, soft_wrap=True)


def _progress_reporter(args) -> ConsoleReporter | None:
    if getattr(args, "quiet", False):
        return None
    return ConsoleReporter(console)


def solve_task(*args, **kwargs):
    """Lazy import so ``retrieve`` does not require LangGraph."""
    from eval.runner import solve_task as _solve_task

    return _solve_task(*args, **kwargs)


def resume_task(*args, **kwargs):
    """Lazy import so ``retrieve`` does not require LangGraph."""
    from eval.runner import resume_task as _resume_task

    return _resume_task(*args, **kwargs)


def load_review_payload(*args, **kwargs):
    """Lazy import so ``retrieve`` does not require LangGraph."""
    from eval.runner import load_review_payload as _load_review_payload

    return _load_review_payload(*args, **kwargs)


def list_sessions(*args, **kwargs):
    from eval.session import list_sessions as _list_sessions

    return _list_sessions(*args, **kwargs)


def load_session(*args, **kwargs):
    from eval.session import load_session as _load_session

    return _load_session(*args, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="IssuePilot harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="Solve one benchmark task with the agent")
    solve.add_argument("task_id", help="Task id from eval/dataset.json, e.g. issue-001")
    solve.add_argument(
        "--harness",
        choices=("v0", "v1", "v2"),
        default="v0",
        help="Harness version: v0 ReAct, v1 Plan-Execute, v2 Plan-Execute + RAG (default: v0)",
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
    solve.add_argument(
        "--embedder",
        choices=("hashing", "fastembed"),
        default="hashing",
        help="V2 dense embedder (ignored by v0/v1). hashing = no download; "
        "fastembed matches `retrieve` DoD (may download)",
    )
    solve.add_argument(
        "--query-mode",
        choices=QUERY_MODES,
        default=DEFAULT_QUERY_MODE,
        help="V2 retrieve query: issue (same as offline eval) or issue+analysis",
    )
    solve.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print live stage/tool progress",
    )
    solve.add_argument(
        "--interactive-recovery",
        action="store_true",
        help="After automatic retries, prompt once for same-process recovery feedback",
    )
    solve.add_argument(
        "--require-approval",
        action="store_true",
        help="Interrupt after Layer 2 pass for human approve/reject/feedback "
        "(implies checkpointing; v1/v2 only)",
    )
    solve.add_argument(
        "--pause-on-approval",
        action="store_true",
        help="Write a paused session and exit when the approval gate fires "
        "(implies --require-approval)",
    )

    retrieve = sub.add_parser(
        "retrieve",
        help="Recall@K retrieval eval (grep vs bm25 vs dense vs hybrid). No LLM.",
    )
    retrieve.add_argument(
        "task_id",
        nargs="?",
        default=None,
        help="Task id from eval/dataset.json, e.g. issue-009",
    )
    retrieve.add_argument(
        "--split",
        choices=("smoke", "hard", "ablation"),
        default=None,
        help="Evaluate every task in this split (hard = issue-008–014)",
    )
    retrieve.add_argument(
        "--k",
        type=int,
        default=RETRIEVE_K,
        help=f"Recall@K cutoff (default: {RETRIEVE_K})",
    )
    retrieve.add_argument(
        "--embedder",
        choices=("hashing", "fastembed"),
        default="fastembed",
        help="Dense embedder: hashing (offline tests) or fastembed (DoD; may download)",
    )
    retrieve.add_argument(
        "--query-mode",
        choices=QUERY_MODES,
        default=DEFAULT_QUERY_MODE,
        help="Search query: issue (default, matches live V2) or issue+analysis",
    )
    retrieve.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip git reset to base_commit (debug only)",
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
    compare.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print live stage/tool progress",
    )

    report = sub.add_parser(
        "report",
        help="Aggregate runs/*.json into provenance-aware ablation tables",
    )
    report.add_argument(
        "--split",
        choices=("smoke", "hard", "ablation"),
        default=None,
        help="Keep only this dataset split",
    )
    report.add_argument(
        "--base-commit",
        default=None,
        help="Keep only solve records with this benchmark SHA",
    )
    report.add_argument(
        "--model",
        default=None,
        help="Keep only solve records with this model id",
    )
    report.add_argument(
        "--latest-per-cell",
        action="store_true",
        help="Keep only the newest solve per (task, harness) cell before aggregating",
    )
    report.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable report",
    )

    bench = sub.add_parser(
        "bench",
        help="Run a split × harness matrix (no approval / interactive-recovery)",
    )
    bench.add_argument(
        "--split",
        choices=("smoke", "hard", "ablation"),
        required=True,
        help="Dataset split to run",
    )
    bench.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids within the split (default: all)",
    )
    bench.add_argument(
        "--harness",
        default="v0,v1,v2",
        help="Comma-separated harness versions (default: v0,v1,v2)",
    )
    bench.add_argument(
        "--n",
        type=int,
        default=1,
        help="Repeats per (task, harness) cell (default: 1)",
    )
    bench.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a cell that already has n matching solve records",
    )
    bench.add_argument(
        "--log",
        nargs="?",
        const="",
        default=None,
        help="Write a matrix log; optional path (default: runs/matrix-<stamp>.log)",
    )
    bench.add_argument(
        "--model",
        default=None,
        help=f"Model id (default: {default_model()})",
    )
    bench.add_argument(
        "--max-steps",
        type=int,
        default=MAX_AGENT_STEPS,
        help=f"Max executor ReAct steps (default: {MAX_AGENT_STEPS})",
    )
    bench.add_argument(
        "--embedder",
        choices=("hashing", "fastembed"),
        default="hashing",
        help="V2 dense embedder (ignored by v0/v1)",
    )
    bench.add_argument(
        "--query-mode",
        choices=QUERY_MODES,
        default=DEFAULT_QUERY_MODE,
        help="V2 retrieve query (default: issue)",
    )
    bench.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print live stage/tool progress",
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

    sub.add_parser("runs", help="List paused approval sessions")

    review = sub.add_parser("review", help="Show the six-panel approval view for a paused run")
    review.add_argument("run_id", help="Paused session id from `runs`")

    resume = sub.add_parser("resume", help="Resume a paused approval run")
    resume.add_argument("run_id", help="Paused session id from `runs`")
    resume_decision = resume.add_mutually_exclusive_group(required=True)
    resume_decision.add_argument(
        "--approve",
        action="store_true",
        help="Accept the patch and finish the run",
    )
    resume_decision.add_argument(
        "--reject",
        action="store_true",
        help="Reject the patch and escalate to needs_human",
    )
    resume_decision.add_argument(
        "--feedback",
        metavar="TEXT",
        help="Send the patch back to diagnose with this note",
    )
    resume.add_argument(
        "--max-steps",
        type=int,
        default=MAX_AGENT_STEPS,
        help=f"Max executor ReAct steps (default: {MAX_AGENT_STEPS})",
    )
    resume.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print live stage/tool progress",
    )

    mcp = sub.add_parser(
        "mcp",
        help="Minimal MCP stdio server and client demo (live V1/V2 tools stay direct)",
    )
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_serve = mcp_sub.add_parser("serve", help="Serve read_file/search_code/git_diff on stdio")
    mcp_serve.add_argument(
        "--repo",
        default=None,
        help="Repository root (default: sibling issue-pilot-benchmark)",
    )
    mcp_demo = mcp_sub.add_parser(
        "demo",
        help="Client round-trip: Agent -> MCP Client -> MCP Server -> Tool",
    )
    mcp_demo.add_argument(
        "--repo",
        default=None,
        help="Repository root (default: sibling issue-pilot-benchmark)",
    )
    mcp_demo.add_argument(
        "--path",
        default="app/auth.py",
        help="Repo-relative file for the demo read_file call",
    )
    mcp_demo.add_argument(
        "--query",
        default="decode_token",
        help="Query for the demo search_code call",
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
        ("retry_count", str(record.get("retry_count") if record.get("retry_count") is not None else "-")),
        ("human_retry_count", str(record.get("human_retry_count") if record.get("human_retry_count") is not None else "-")),
        ("recovery_success", str(record.get("recovery_success") if record.get("recovery_success") is not None else "-")),
        ("embedder", str(record.get("embedder_name") or "-")),
        ("query_mode", str(record.get("query_mode") or "-")),
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
    if harness in {"v1", "v2"}:
        console.print(format_recovery_summary(record))

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


def _print_retrieval(result: dict) -> None:
    k = result.get("k")
    embedder = result.get("embedder") or "-"
    split = result.get("split") or "task"
    table = Table(title=f"Recall@{k} ({split}, {embedder})")
    table.add_column("Task")
    table.add_column("expected")
    for mode in ALL_MODES:
        table.add_column(mode)

    for row in result.get("tasks") or []:
        expected = ", ".join(row.get("expected_files") or []) or "-"
        cells = [str(row.get("task_id") or "-"), expected]
        modes = row.get("modes") or {}
        for mode in ALL_MODES:
            score = (modes.get(mode) or {}).get("recall_at_k")
            cells.append(f"{float(score):.2f}" if score is not None else "-")
        table.add_row(*cells)

    means = result.get("mean_recall_at_k") or {}
    mean_cells = ["mean", "-"]
    for mode in ALL_MODES:
        score = means.get(mode)
        mean_cells.append(f"{float(score):.2f}" if score is not None else "-")
    table.add_row(*mean_cells)
    console.print(table)

    for row in result.get("tasks") or []:
        task_id = row.get("task_id")
        console.print(f"\n[bold]{task_id}[/bold]  {row.get('issue') or ''}")
        expected = ", ".join(row.get("expected_files") or [])
        console.print(f"  expected: {expected}")
        modes = row.get("modes") or {}
        for mode in ALL_MODES:
            files = ", ".join((modes.get(mode) or {}).get("retrieved_files") or []) or "-"
            console.print(f"  {mode}: {files}")
        run_path = row.get("run_path")
        if run_path:
            console.print(f"  run_path: {run_path}")


def _print_report(report: dict) -> None:
    filters = report.get("filters") or {}
    title_bits = ["Ablation report"]
    if filters.get("split"):
        title_bits.append(f"split={filters['split']}")
    if filters.get("base_commit"):
        title_bits.append(str(filters["base_commit"])[:12])
    for cohort in report.get("solve_cohorts") or []:
        spec = str(cohort.get("benchmark_spec_sha") or "-")
        heading = (
            f"{cohort.get('base_commit') or '-'}  "
            f"model={cohort.get('model') or '-'}  "
            f"T={cohort.get('temperature')}  "
            f"image={cohort.get('sandbox_image') or '-'}  "
            f"spec={spec[:12]}  "
            f"n={cohort.get('n')}  "
            f"cells={cohort.get('n_cells')}"
        )
        table = Table(title=heading)
        table.add_column("Harness")
        table.add_column("n", justify="right")
        table.add_column("cells", justify="right")
        table.add_column("Resolve", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Reads", justify="right")
        table.add_column("Tools", justify="right")
        table.add_column("Retries", justify="right")
        table.add_column("Recovery", justify="right")
        table.add_column("Human", justify="right")
        table.add_column("Latency s", justify="right")
        table.add_column("LocPrec", justify="right")
        table.add_column("L1Fail", justify="right")
        table.add_column("Search", justify="right")
        table.add_column("1stRead", justify="right")
        for row in cohort.get("harnesses") or []:
            loc = row.get("localization_precision")
            gate = row.get("layer1_gate_rate")
            search = row.get("search_code_calls")
            first = row.get("first_expected_read_step")
            table.add_row(
                str(row.get("harness_version")),
                str(row.get("n")),
                str(row.get("n_cells") if row.get("n_cells") is not None else "-"),
                f"{float(row.get('resolve_rate') or 0):.2f}",
                f"{float(row.get('tokens') or 0):.0f}",
                f"{float(row.get('file_reads') or 0):.1f}",
                f"{float(row.get('tool_calls') or 0):.1f}",
                f"{float(row.get('retries') or 0):.2f}",
                f"{float(row.get('recovery_rate') or 0):.2f}",
                f"{float(row.get('human_retries') or 0):.2f}",
                f"{float(row.get('latency_s') or 0):.1f}",
                "-" if loc is None else f"{float(loc):.2f}",
                "-" if gate is None else f"{float(gate):.2f}",
                f"{float(search or 0):.1f}",
                "-" if first is None else f"{float(first):.1f}",
            )
        console.print(table)

    retrieval = report.get("retrieval") or []
    if retrieval:
        table = Table(title="Retrieval Recall@K (saved artifacts)")
        table.add_column("Task")
        table.add_column("embedder")
        table.add_column("query")
        for mode in ALL_MODES:
            table.add_column(mode, justify="right")
        for row in retrieval:
            cells = [
                str(row.get("task_id") or "-"),
                str(row.get("embedder") or "-"),
                str(row.get("query_mode") or "-"),
            ]
            for mode in ALL_MODES:
                score = row.get(mode)
                cells.append(f"{float(score):.2f}" if score is not None else "-")
            table.add_row(*cells)
        console.print(table)

    sha = report.get("harness_git_sha")
    if sha:
        console.print(f"[dim]harness_git_sha={sha}[/dim]")


def _print_bench(manifest: dict) -> None:
    settings = manifest.get("settings") or {}
    table = Table(title=f"Bench {settings.get('split') or '-'} n={settings.get('n')}")
    table.add_column("Task")
    table.add_column("Harness")
    table.add_column("Status")
    table.add_column("Success")
    table.add_column("Run")
    for cell in manifest.get("cells") or []:
        if cell.get("skipped"):
            status = "skip"
        elif cell.get("error"):
            status = "error"
        elif cell.get("paused"):
            status = "paused"
        else:
            status = "done"
        success = cell.get("success")
        success_s = "-" if success is None else str(success)
        table.add_row(
            str(cell.get("task_id") or "-"),
            str(cell.get("harness_version") or "-"),
            status,
            success_s,
            str(cell.get("run_path") or "-"),
        )
    console.print(table)
    log_path = manifest.get("log_path")
    manifest_path = manifest.get("manifest_path")
    if log_path:
        console.print(f"[dim]log={log_path}[/dim]")
    if manifest_path:
        console.print(f"[dim]manifest={manifest_path}[/dim]")


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


def _print_paused(record: dict) -> None:
    run_id = record.get("run_id") or "-"
    table = Table(title=f"Paused waiting for approval: {run_id}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in (
        ("run_id", str(run_id)),
        ("task_id", str(record.get("task_id") or "-")),
        ("harness", str(record.get("harness_version") or "-")),
        ("status", str(record.get("status") or "waiting_approval")),
        ("session_path", str(record.get("session_path") or "-")),
        ("resume_count", str(record.get("resume_count", 0))),
    ):
        table.add_row(key, value)
    console.print(table)
    console.print(format_review(record.get("approval_payload") or {}))
    console.print(
        f"[dim]Review: python cli.py review {run_id}[/dim]\n"
        f"[dim]Resume: python cli.py resume {run_id} "
        "--approve|--reject|--feedback TEXT[/dim]"
    )


def _print_runs(sessions: list) -> None:
    paused = [item for item in sessions if item.status == "paused"]
    if not paused:
        console.print("[dim]No paused sessions.[/dim]")
        return
    table = Table(title="Paused sessions")
    table.add_column("run_id")
    table.add_column("task_id")
    table.add_column("harness")
    table.add_column("model")
    table.add_column("created_at")
    table.add_column("resume_count", justify="right")
    for item in paused:
        table.add_row(
            item.run_id,
            item.task_id,
            item.harness,
            item.model,
            item.created_at,
            str(item.resume_count),
        )
    console.print(table)


def _resume_decision(args) -> tuple[str, str | None]:
    if args.approve:
        return "approve", None
    if args.reject:
        return "reject", None
    feedback = (args.feedback or "").strip()
    if not feedback:
        raise ValueError("--feedback requires a non-empty note")
    return "feedback", feedback


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "solve":
        require_approval = bool(args.require_approval or args.pause_on_approval)
        if require_approval and args.harness == "v0":
            parser.error("--require-approval requires --harness v1 or v2")
        console.print(
            f"[cyan]Solving {args.task_id} with harness {args.harness}...[/cyan]"
        )
        try:
            record = solve_task(
                args.task_id,
                model=args.model,
                max_steps=args.max_steps,
                harness_version=args.harness,
                embedder_name=args.embedder,
                query_mode=args.query_mode,
                progress=_progress_reporter(args),
                interactive_recovery=args.interactive_recovery,
                require_approval=require_approval,
            )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
            return 1

        if record.get("paused"):
            _print_paused(record)
            return 0
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
                    progress=_progress_reporter(args),
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

    if args.command == "retrieve":
        if not args.task_id and not args.split:
            parser.error("retrieve requires a task_id or --split")
        label = args.task_id or f"split={args.split}"
        console.print(
            f"[cyan]Retrieval eval {label} "
            f"(k={args.k}, embedder={args.embedder})...[/cyan]"
        )
        try:
            result = run_retrieval_eval(
                task_id=args.task_id,
                split=args.split,
                k=args.k,
                embedder_name=args.embedder,
                query_mode=args.query_mode,
                reset=not args.no_reset,
            )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
            return 1
        _print_retrieval(result)
        return 0

    if args.command == "report":
        from eval.report import build_report

        report = build_report(
            split=args.split,
            base_commit=args.base_commit,
            model=args.model,
            latest_per_cell=args.latest_per_cell,
        )
        if args.json:
            console.print_json(json.dumps(report))
        else:
            _print_report(report)
        return 0

    if args.command == "bench":
        from eval.matrix import parse_harness_list, run_matrix

        try:
            harnesses = parse_harness_list(args.harness)
        except ValueError as exc:
            parser.error(str(exc))
        if args.n < 1:
            parser.error("--n must be >= 1")
        console.print(
            f"[cyan]Bench split={args.split} harness={','.join(harnesses)} "
            f"n={args.n}...[/cyan]"
        )
        try:
            manifest = run_matrix(
                split=args.split,
                harnesses=harnesses,
                n=args.n,
                skip_existing=args.skip_existing,
                model=args.model,
                max_steps=args.max_steps,
                embedder_name=args.embedder,
                query_mode=args.query_mode,
                log_path=args.log or None,
                progress=_progress_reporter(args),
                task_ids=args.tasks,
            )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
            return 1
        _print_bench(manifest)
        cells = manifest.get("cells") or []
        if any(cell.get("error") for cell in cells):
            return 1
        executed = [cell for cell in cells if not cell.get("skipped")]
        if executed and not all(cell.get("success") for cell in executed):
            return 2
        return 0

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

    if args.command == "runs":
        _print_runs(list_sessions())
        return 0

    if args.command == "review":
        try:
            session = load_session(args.run_id)
            payload = load_review_payload(args.run_id)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
            return 1
        console.print(
            f"[cyan]Review {session.run_id} "
            f"({session.task_id}, {session.harness}) "
            f"status={session.status}[/cyan]"
        )
        console.print(format_review(payload))
        return 0

    if args.command == "resume":
        try:
            decision, feedback = _resume_decision(args)
        except ValueError as exc:
            parser.error(str(exc))
        console.print(
            f"[cyan]Resuming {args.run_id} ({decision})...[/cyan]"
        )
        try:
            record = resume_task(
                args.run_id,
                decision=decision,
                feedback=feedback,
                max_steps=args.max_steps,
                progress=_progress_reporter(args),
            )
        except Exception as exc:
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
            return 1
        if record.get("paused"):
            _print_paused(record)
            return 0
        _print_summary(record)
        return 0 if record.get("success") else 2

    if args.command == "mcp":
        from harness.mcp_client import run_demo
        from harness.mcp_server import default_repo_path, run_stdio_server

        repo = Path(args.repo).resolve() if args.repo else default_repo_path()
        if not repo.is_dir():
            sys.stderr.write(f"Error: repository not found: {repo}\n")
            return 1
        if args.mcp_command == "serve":
            run_stdio_server(repo)
            return 0
        if args.mcp_command == "demo":
            console.print("[cyan]Agent -> MCP Client -> MCP Server -> Tool[/cyan]")
            try:
                result = run_demo(repo, path=args.path, query=args.query)
            except Exception as exc:
                console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
                return 1
            tools = ", ".join(result.get("tools") or []) or "-"
            console.print(f"tools: {tools}")
            for call in result.get("calls") or []:
                preview = (call.get("text") or "").strip() or "(empty)"
                if len(preview) > 800:
                    preview = preview[:797].rstrip() + "..."
                console.print(f"\n[bold]{call.get('name')}[/bold] {call.get('arguments')}")
                console.print(preview)
            return 0
        parser.error(f"unknown mcp command: {args.mcp_command}")
        return 1

    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
