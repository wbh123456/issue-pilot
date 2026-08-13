"""Command allowlist and path checks for agent-visible executions.

Policy is evaluated against the container workspace view (``/workspace``).
This module is pure validation — it does not execute anything.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from typing import Sequence

WORKSPACE_ROOT = "/workspace"

# Bare executables the agent may invoke inside the sandbox.
ALLOWED_EXECUTABLES = frozenset({"pytest", "ruff", "mypy", "git"})

# Narrow read-only git used by git_diff / verify telemetry.
_ALLOWED_GIT_SUBCOMMANDS = frozenset({"status", "diff"})

# Shell operators / expansions that imply a shell, not an argv vector.
_SHELL_META_RE = re.compile(r"[;|&`$<>\n]|\|\||&&")

_SENSITIVE_PREFIXES = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/root",
    "/home",
    "/var/run/docker.sock",
)


class CommandPermissionError(Exception):
    """Raised when a command or path is outside the allowlist."""


def parse_command(command: str | Sequence[str]) -> list[str]:
    """Parse a command string or argv sequence into an argv list.

    Strings are split with POSIX rules (sandbox is a Linux container).
    """
    if isinstance(command, (list, tuple)):
        argv = [str(part) for part in command]
        if not argv or any(not part.strip() for part in argv):
            raise CommandPermissionError(
                "command argv must be a non-empty list of tokens"
            )
        return argv

    if not isinstance(command, str):
        raise CommandPermissionError(
            f"unsupported command type: {type(command).__name__}"
        )

    text = command.strip()
    if not text:
        raise CommandPermissionError("command is empty")

    if _SHELL_META_RE.search(text):
        raise CommandPermissionError(
            "shell operators and redirections are not allowed; pass a plain argv"
        )

    try:
        argv = shlex.split(text, posix=True)
    except ValueError as exc:
        raise CommandPermissionError(f"could not parse command: {exc}") from exc

    if not argv:
        raise CommandPermissionError("command is empty after parsing")
    return argv


def validate_command(
    command: str | Sequence[str],
    *,
    workspace: str = WORKSPACE_ROOT,
) -> list[str]:
    """Parse and validate ``command``; return the approved argv.

    Raises ``CommandPermissionError`` when the executable, git subcommand,
    shell syntax, or any path argument is not allowed.
    """
    argv = parse_command(command)
    executable = _basename(argv[0]).lower()

    if executable not in ALLOWED_EXECUTABLES:
        raise CommandPermissionError(
            f"executable not allowed: {argv[0]!r} "
            f"(allowed: {', '.join(sorted(ALLOWED_EXECUTABLES))})"
        )

    if executable == "git":
        _validate_git(argv[1:])
    else:
        for token in argv[1:]:
            if _looks_like_path(token):
                _validate_path_arg(token, workspace=workspace)

    return argv


def _basename(token: str) -> str:
    # Container paths are POSIX even when the harness host is Windows.
    return posixpath.basename(token.replace("\\", "/"))


def _validate_git(args: list[str]) -> None:
    if not args:
        raise CommandPermissionError("git requires a subcommand")

    # Reject git's own shell escapes and path-changing global options.
    filtered: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in {"-C", "--git-dir", "--work-tree"}:
            raise CommandPermissionError(f"git option not allowed: {arg}")
        if arg.startswith("-") and filtered == []:
            # No global options before the subcommand.
            raise CommandPermissionError(f"git global option not allowed: {arg}")
        filtered.append(arg)
        i += 1

    sub = filtered[0]
    if sub not in _ALLOWED_GIT_SUBCOMMANDS:
        raise CommandPermissionError(
            f"git subcommand not allowed: {sub!r} "
            f"(allowed: {', '.join(sorted(_ALLOWED_GIT_SUBCOMMANDS))})"
        )

    rest = filtered[1:]
    if sub == "status":
        _validate_git_status(rest)
    else:
        _validate_git_diff(rest)


def _validate_git_status(args: list[str]) -> None:
    allowed_flags = {"--short", "-s", "--porcelain"}
    for arg in args:
        if arg.startswith("-"):
            if arg not in allowed_flags:
                raise CommandPermissionError(f"git status flag not allowed: {arg}")
            continue
        _validate_path_arg(arg)


def _validate_git_diff(args: list[str]) -> None:
    allowed_flags = {"--stat", "--name-only", "--name-status", "--raw", "--no-color"}
    for arg in args:
        if arg == "HEAD" or arg == "--":
            continue
        if arg.startswith("-"):
            if arg not in allowed_flags:
                raise CommandPermissionError(f"git diff flag not allowed: {arg}")
            continue
        _validate_path_arg(arg)


def _looks_like_path(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    if token in {"HEAD", "--"}:
        return False
    markers = ("/", "\\", "..", "~", ".")
    return any(m in token for m in markers) or token.endswith(
        (".py", ".txt", ".md", ".toml", ".cfg", ".ini", ".yml", ".yaml")
    )


def _validate_path_arg(token: str, *, workspace: str = WORKSPACE_ROOT) -> None:
    raw = token.strip()
    if not raw:
        raise CommandPermissionError("empty path argument")

    # Never allow home/env expansion — agent must stay under workspace.
    if raw.startswith("~") or "$" in raw:
        raise CommandPermissionError(f"path not allowed: {token!r}")

    normalized = raw.replace("\\", "/")

    # Reject Windows drive paths and UNC shares (container view is POSIX-only).
    if re.match(r"^[A-Za-z]:", normalized) or normalized.startswith("//"):
        raise CommandPermissionError(f"path not allowed: {token!r}")

    if "/.ssh" in f"/{normalized.strip('/')}" or normalized.rstrip("/").endswith(".ssh"):
        raise CommandPermissionError(f"sensitive path not allowed: {token!r}")

    for prefix in _SENSITIVE_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            raise CommandPermissionError(f"sensitive path not allowed: {token!r}")

    workspace_norm = workspace.rstrip("/") or "/"
    if posixpath.isabs(normalized):
        resolved = posixpath.normpath(normalized)
        if resolved != workspace_norm and not resolved.startswith(workspace_norm + "/"):
            raise CommandPermissionError(
                f"path escapes workspace: {token!r} (workspace={workspace_norm})"
            )
        return

    # Relative paths must stay under workspace after normalization.
    resolved = posixpath.normpath(posixpath.join(workspace_norm, normalized))
    if resolved != workspace_norm and not resolved.startswith(workspace_norm + "/"):
        raise CommandPermissionError(
            f"path escapes workspace: {token!r} (workspace={workspace_norm})"
        )
