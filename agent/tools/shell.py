"""Shell tools: run_tests via the active Docker sandbox."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from harness.limits import COMMAND_TIMEOUT
from harness.permissions import CommandPermissionError
from sandbox.runner import SandboxUnusableError

from ._sandbox import resolve_repo_root, truncate_output

if TYPE_CHECKING:
    from sandbox.runner import SandboxRunner


@dataclass
class CommandOutcome:
    """Structured sandbox command result. Routing must use fields, not prose."""

    command: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.timed_out and self.exit_code == 0

    def to_dict(self) -> dict:
        return asdict(self)

    def format(self) -> str:
        if self.error:
            return self.error
        parts = [
            f"exit_code={self.exit_code}",
            f"command: {' '.join(self.command)}",
        ]
        stdout = (self.stdout or "").rstrip()
        stderr = (self.stderr or "").rstrip()
        if stdout:
            parts.append("--- stdout ---\n" + stdout)
        if stderr:
            parts.append("--- stderr ---\n" + stderr)
        if not stdout and not stderr:
            parts.append("(no output)")
        return truncate_output("\n".join(parts))


def _argv(command: str) -> list[str]:
    return command.split() if command.strip() else []


def run_command(
    repo_path: str | Path,
    command: str,
    *,
    sandbox: SandboxRunner,
    empty_message: str = "Error: command is required",
    timeout_message: str | None = None,
) -> CommandOutcome:
    """Run ``command`` inside the sandbox and return a structured outcome."""
    if sandbox is None:
        raise RuntimeError(
            "run_command requires an active SandboxRunner; host execution is not allowed"
        )
    argv = _argv(command)
    if not argv:
        return CommandOutcome(command=[], exit_code=1, error=empty_message)

    resolve_repo_root(repo_path)
    timeout_text = timeout_message or f"Error: timed out after {COMMAND_TIMEOUT}s"
    try:
        result = sandbox.run(command)
    except CommandPermissionError as exc:
        return CommandOutcome(
            command=argv,
            exit_code=1,
            error=f"Error: command not allowed: {exc}",
        )
    except SandboxUnusableError as exc:
        return CommandOutcome(
            command=argv,
            exit_code=1,
            error=f"Error: sandbox unusable: {exc}",
        )

    if result.timed_out:
        return CommandOutcome(
            command=list(result.command or argv),
            exit_code=int(result.exit_code),
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            timed_out=True,
            error=timeout_text,
        )
    return CommandOutcome(
        command=list(result.command or argv),
        exit_code=int(result.exit_code),
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        timed_out=False,
    )


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
    outcome = run_command(
        repo_path,
        test_command,
        sandbox=sandbox,
        empty_message="Error: test_command is required",
        timeout_message=f"Error: tests timed out after {COMMAND_TIMEOUT}s",
    )
    return outcome.format()
