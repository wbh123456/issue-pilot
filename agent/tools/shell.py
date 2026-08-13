"""Shell tools: run_tests via the active Docker sandbox."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from harness.limits import COMMAND_TIMEOUT
from harness.permissions import CommandPermissionError
from sandbox.runner import SandboxUnusableError

from ._sandbox import resolve_repo_root, truncate_output

if TYPE_CHECKING:
    from sandbox.runner import CommandResult, SandboxRunner


def _format_result(result: CommandResult) -> str:
    parts = [
        f"exit_code={result.exit_code}",
        f"command: {' '.join(result.command)}",
    ]
    stdout = (result.stdout or "").rstrip()
    stderr = (result.stderr or "").rstrip()
    if stdout:
        parts.append("--- stdout ---\n" + stdout)
    if stderr:
        parts.append("--- stderr ---\n" + stderr)
    if not stdout and not stderr:
        parts.append("(no output)")
    return truncate_output("\n".join(parts))


def run_tests(
    repo_path: str | Path,
    test_command: str,
    *,
    sandbox: SandboxRunner,
) -> str:
    """Run ``test_command`` inside the active sandbox.

    ``test_command`` comes from the task dataset (module/suite level — not a
    single gold test name). Host subprocess execution is not allowed.
    """
    if sandbox is None:
        raise RuntimeError(
            "run_tests requires an active SandboxRunner; host execution is not allowed"
        )
    if not test_command or not test_command.strip():
        return "Error: test_command is required"

    resolve_repo_root(repo_path)
    try:
        result = sandbox.run(test_command)
    except CommandPermissionError as exc:
        return f"Error: command not allowed: {exc}"
    except SandboxUnusableError as exc:
        return f"Error: sandbox unusable: {exc}"

    if result.timed_out:
        return f"Error: tests timed out after {COMMAND_TIMEOUT}s"
    return _format_result(result)
