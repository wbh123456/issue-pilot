"""Shell tools: run_tests."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ._sandbox import resolve_repo_root, truncate_output

COMMAND_TIMEOUT = 60


def _normalize_args(args: list[str]) -> list[str]:
    """Rewrite bare ``pytest`` to ``python -m pytest`` when needed."""
    if not args:
        return args
    # Windows shlex may retain surrounding quotes on tokens.
    args = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "'\"" else a for a in args]
    head = args[0]
    if head in {"pytest", "pytest.exe"} and shutil.which(head) is None:
        return [sys.executable, "-m", "pytest", *args[1:]]
    return args


def run_tests(repo_path: str | Path, test_command: str) -> str:
    """Run ``test_command`` with cwd set to the benchmark repo root.

    ``test_command`` comes from the task dataset (module/suite level — not a
    single gold test name).
    """
    if not test_command or not test_command.strip():
        return "Error: test_command is required"

    root = resolve_repo_root(repo_path)
    try:
        args = shlex.split(test_command, posix=(os.name != "nt"))
    except ValueError as exc:
        return f"Error: could not parse test_command: {exc}"
    if not args:
        return "Error: test_command is empty after parsing"
    args = _normalize_args(args)

    try:
        proc = subprocess.run(
            args,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return f"Error: tests timed out after {COMMAND_TIMEOUT}s"
    except FileNotFoundError as exc:
        return f"Error: command not found: {exc.filename or args[0]}"
    except OSError as exc:
        return f"Error: failed to run tests: {exc}"

    parts = [
        f"exit_code={proc.returncode}",
        f"command: {' '.join(args)}",
    ]
    stdout = (proc.stdout or "").rstrip()
    stderr = (proc.stderr or "").rstrip()
    if stdout:
        parts.append("--- stdout ---\n" + stdout)
    if stderr:
        parts.append("--- stderr ---\n" + stderr)
    if not stdout and not stderr:
        parts.append("(no output)")
    return truncate_output("\n".join(parts))
