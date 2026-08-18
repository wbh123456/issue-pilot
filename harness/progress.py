"""Live CLI progress for solve/compare. Optional; tests pass None."""

from __future__ import annotations

import re
from typing import Any, Protocol

PREVIEW_CHARS = 4000
STAGE_RULE = "=" * 32

_EXIT_CODE_RE = re.compile(r"exit_code=(-?\d+)")
_BOLD_HEADING_RE = re.compile(r"^\*\*(.+?)\*\*:?$")


class ProgressReporter(Protocol):
    def stage(self, name: str, detail: str = "") -> None: ...
    def note(self, text: str) -> None: ...
    def tool(self, step: int, name: str, args: dict[str, Any], result: str) -> None: ...


class NullReporter:
    def stage(self, name: str, detail: str = "") -> None:
        return None

    def note(self, text: str) -> None:
        return None

    def tool(self, step: int, name: str, args: dict[str, Any], result: str) -> None:
        return None


def get_reporter(progress: ProgressReporter | None) -> ProgressReporter:
    return progress if progress is not None else NullReporter()


def _heading_label(line: str) -> str | None:
    """Turn a markdown heading line into ``Title:``. None if not a heading."""
    if line.startswith("#"):
        label = line.lstrip("#").strip()
        if not label:
            return None
        return label if label.endswith(":") else f"{label}:"
    match = _BOLD_HEADING_RE.fullmatch(line)
    if match:
        label = match.group(1).strip().rstrip(":")
        if not label:
            return None
        return f"{label}:"
    return None


def flatten_progress_text(text: str) -> str:
    """Collapse whitespace; heading lines become ``Title:`` before the body."""
    parts: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = _heading_label(line)
        parts.append(heading if heading is not None else line)
    return " ".join(parts)


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    collapsed = flatten_progress_text(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def summarize_files(paths: list[str], *, show: int = 2) -> str:
    kept = [p for p in paths if p]
    if not kept:
        return "(none)"
    head = kept[:show]
    extra = len(kept) - show
    text = ", ".join(head)
    if extra > 0:
        text += f" (+{extra})"
    return text


def format_plan_detail(plan: dict[str, Any] | None, *, retry: bool = False) -> str:
    """Full structured plan for the live CLI. No field truncation."""
    plan = plan or {}
    files = [p for p in (plan.get("files_to_inspect") or []) if p]
    steps = [str(s).strip() for s in (plan.get("steps") or []) if str(s).strip()]
    lines: list[str] = []
    if retry:
        lines.append("retry")
    problem = str(plan.get("problem") or "").strip()
    hypothesis = str(plan.get("hypothesis") or "").strip()
    if problem:
        lines.append(f"problem: {problem}")
    if hypothesis:
        lines.append(f"hypothesis: {hypothesis}")
    lines.append(f"files: {', '.join(files) if files else '(none)'}")
    if steps:
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}")
    else:
        lines.append("steps: (none)")
    return "\n".join(lines)


def format_recovery_summary(record: dict[str, Any] | None = None) -> str:
    """One-line CLI snapshot: attempts / Layer1 / Layer2 / human retry."""
    record = record or {}
    retry_count = int(record.get("retry_count") or 0)
    human_retry_count = int(record.get("human_retry_count") or 0)
    verification = record.get("verification")
    if not isinstance(verification, dict):
        verification = record.get("test_result") or {}
    layer1 = None
    if isinstance(verification, dict) and "deterministic_pass" in verification:
        layer1 = verification.get("deterministic_pass") is True
    evaluation = record.get("patch_evaluation") or {}
    layer2 = None
    if isinstance(evaluation, dict) and evaluation:
        layer2 = (
            evaluation.get("issue_resolved") is True
            and evaluation.get("patch_scope") == "appropriate"
            and evaluation.get("regression_risk") == "low"
            and evaluation.get("missing_tests") is False
        )
    return (
        f"attempts={retry_count + 1} "
        f"Layer1={_pass_fail(layer1)} "
        f"Layer2={_pass_fail(layer2)} "
        f"human_retry={human_retry_count}"
    )


def _pass_fail(value: bool | None) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "-"


def _format_review_kv(data: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        text = str(value)
        if "\n" in text:
            lines.append(f"{key}:")
            lines.extend(text.splitlines() or ["(none)"])
        else:
            lines.append(f"{key}: {text}")
    return "\n".join(lines) if lines else "(none)"


def format_review(payload: dict[str, Any] | None = None) -> str:
    """Six-panel approval view for ``review`` / paused solve.

    Sections match the Day 6 CLI: Issue, Plan, Changed Files, Git Diff,
    Test Results, Evaluator Result.
    """
    payload = payload or {}
    plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    files = [str(path) for path in (payload.get("changed_files") or []) if path]
    tests = payload.get("test_result") if isinstance(payload.get("test_result"), dict) else {}
    evaluation = (
        payload.get("evaluator_result")
        if isinstance(payload.get("evaluator_result"), dict)
        else {}
    )
    issue = str(payload.get("issue") or "").strip() or "(none)"
    diff = str(payload.get("git_diff") or "").strip() or "(none)"
    parts = [
        format_stage_line("Issue", issue),
        format_stage_line("Plan", format_plan_detail(plan)),
        format_stage_line("Changed Files", "\n".join(files) if files else "(none)"),
        format_stage_line("Git Diff", diff),
        format_stage_line("Test Results", _format_review_kv(tests)),
        format_stage_line("Evaluator Result", _format_review_kv(evaluation)),
    ]
    return "\n\n".join(parts)


def format_size(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k chars"
    return f"{n} chars"


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return preview(stripped)
    return ""


def _hit_count(result: str) -> str:
    if result.startswith("Error:") or result.startswith("(no"):
        return preview(result)
    lines = [ln for ln in result.splitlines() if ln.strip()]
    n = len(lines)
    return "1 hit" if n == 1 else f"{n} hits"


def summarize_tool(
    name: str, args: dict[str, Any], result: str
) -> tuple[str, str]:
    """Return ``(detail, outcome)`` for one live tool line. Never dump bodies."""
    args = args or {}
    result = result or ""
    path = str(args.get("path") or "").strip()
    query = str(args.get("query") or "").strip()

    if result.startswith("Error:"):
        detail = path or preview(query)
        return detail, preview(result)

    if name == "read_file":
        return path, format_size(len(result))
    if name == "edit_file":
        return path, _first_line(result) or "ok"
    if name == "list_files":
        n = len([ln for ln in result.splitlines() if ln.strip()])
        return path or ".", f"{n} entries"
    if name in {"grep_code", "search_code"}:
        return preview(query, 40), _hit_count(result)
    if name == "run_tests":
        match = _EXIT_CODE_RE.search(result)
        if match:
            return "", f"exit {match.group(1)}"
        return "", _first_line(result) or "ran"
    if name == "git_diff":
        if result.strip() == "(no changes)":
            return "", "no changes"
        return "", "has diff"
    return path or preview(query, 40), _first_line(result) or "ok"


def format_stage_line(name: str, detail: str = "") -> str:
    lines = [f"### {name}", STAGE_RULE]
    detail = (detail or "").strip()
    if not detail:
        return "\n".join(lines)
    for raw in detail.splitlines():
        line = raw.strip()
        if line:
            lines.append(f"  - {line}")
    return "\n".join(lines)


def format_note_line(text: str) -> str:
    """Print notes as indented lines. Truncate only past ``PREVIEW_CHARS``."""
    parts: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = _heading_label(line)
        parts.append(heading if heading is not None else line)
    if not parts:
        return "  -"
    body = "\n".join(parts)
    if len(body) > PREVIEW_CHARS:
        body = body[: PREVIEW_CHARS - 3].rstrip() + "..."
        parts = body.splitlines()
    lines = [f"  - {parts[0]}"]
    lines.extend(f"    {line}" for line in parts[1:])
    return "\n".join(lines)


def format_tool_line(
    step: int, name: str, args: dict[str, Any], result: str
) -> str:
    detail, outcome = summarize_tool(name, args, result)
    bits = [str(step), name]
    if detail:
        bits.append(detail)
    bits.append(outcome)
    return "  - " + "  ".join(bits)


class ConsoleReporter:
    """Rich console sink. Uses the CLI Console for Windows encoding."""

    def __init__(self, console: Any) -> None:
        self._console = console

    def stage(self, name: str, detail: str = "") -> None:
        self._console.print(format_stage_line(name, detail))

    def note(self, text: str) -> None:
        self._console.print(format_note_line(text))

    def tool(self, step: int, name: str, args: dict[str, Any], result: str) -> None:
        self._console.print(format_tool_line(step, name, args, result))
