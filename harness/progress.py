"""Live CLI progress for solve/compare. Optional; tests pass None."""

from __future__ import annotations

import re
from typing import Any, Protocol

PREVIEW_CHARS = 160

_EXIT_CODE_RE = re.compile(r"exit_code=(-?\d+)")


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


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    collapsed = " ".join((text or "").split())
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
    detail = (detail or "").strip()
    if detail:
        return f"{name}  {detail}"
    return name


def format_note_line(text: str) -> str:
    return f"  {preview(text)}"


def format_tool_line(
    step: int, name: str, args: dict[str, Any], result: str
) -> str:
    detail, outcome = summarize_tool(name, args, result)
    bits = [str(step), name]
    if detail:
        bits.append(detail)
    bits.append(outcome)
    return "  " + "  ".join(bits)


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
