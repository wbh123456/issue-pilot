"""Verify node: Layer 1 pytest + ruff + non-empty patch (never LLM judgment)."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from agent.tools.git import inspect_worktree
from agent.tools.shell import CommandOutcome, run_command
from harness.limits import COMMAND_TIMEOUT
from harness.permissions import CommandPermissionError, parse_command

from ._runtime import get_reporter, require_config, traced

_MISSING_LINT = CommandOutcome(
    command=[],
    exit_code=1,
    error="Error: lint_command is required",
)

# Windows Docker bind mounts often mark files executable (EXE001/EXE002).
# Those are sandbox artifacts, not agent patches; docker tests already ignore them.
_SANDBOX_RUFF_IGNORES = ("EXE001", "EXE002")


def _parse_lint_argv(command: str) -> list[str] | None:
    try:
        return parse_command(command)
    except CommandPermissionError:
        return None


def _is_ruff_check(argv: list[str]) -> bool:
    exe = argv[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    return exe in {"ruff", "ruff.exe"} and len(argv) >= 2 and argv[1] == "check"


def _ignored_codes(argv: list[str]) -> set[str]:
    codes: set[str] = set()
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--ignore" and i + 1 < len(argv):
            codes.add(argv[i + 1])
            i += 2
            continue
        if token.startswith("--ignore="):
            codes.add(token.split("=", 1)[1])
        i += 1
    return codes


def _with_sandbox_ignores(argv: list[str]) -> list[str]:
    if not _is_ruff_check(argv):
        return argv
    extra: list[str] = []
    have = _ignored_codes(argv)
    for code in _SANDBOX_RUFF_IGNORES:
        if code not in have:
            extra.extend(["--ignore", code])
    return argv + extra


_RUFF_VALUE_FLAGS = frozenset(
    {
        "--ignore",
        "--select",
        "--extend-select",
        "--extend-ignore",
        "--config",
        "--exclude",
        "--extend-exclude",
        "--output-format",
    }
)


def _ruff_flags_without_targets(argv: list[str]) -> list[str]:
    """Keep ``ruff check`` and flags; drop positional targets such as ``app``."""
    kept = [argv[0], argv[1]]
    i = 2
    while i < len(argv):
        token = argv[i]
        if token.startswith("-"):
            kept.append(token)
            if (
                "=" not in token
                and token in _RUFF_VALUE_FLAGS
                and i + 1 < len(argv)
                and not argv[i + 1].startswith("-")
            ):
                kept.append(argv[i + 1])
                i += 2
                continue
            i += 1
            continue
        i += 1
    return kept


def _python_patch_paths(changed: list[str], untracked: list[str]) -> list[str]:
    """Agent-touched ``.py`` paths, in worktree order, without duplicates."""
    seen: list[str] = []
    for path in [*changed, *untracked]:
        if path.endswith(".py") and path not in seen:
            seen.append(path)
    return seen


def _with_fix(argv: list[str], paths: list[str] | None = None) -> list[str] | None:
    """Build ``ruff check --fix`` limited to ``paths`` when given.

    Whole-package ``ruff check --fix app`` rewrites untouched modules and
    poisons v1/v2 patch evaluation. Skip autofix when ``paths`` is empty.
    """
    if not _is_ruff_check(argv):
        return None
    if paths is not None:
        if not paths:
            return None
        base = _ruff_flags_without_targets(argv)
        if "--fix" not in base:
            base = base[:2] + ["--fix"] + base[2:]
        return base + paths
    if "--fix" in argv:
        return argv
    return argv[:2] + ["--fix"] + argv[2:]


def _join_argv(argv: list[str]) -> str:
    return " ".join(argv)


def _run_ruff(
    repo_path: str,
    lint_command: str,
    sandbox,
) -> tuple[CommandOutcome, bool]:
    """Check ruff; on failure apply ``--fix`` once and re-check."""
    argv = _parse_lint_argv(lint_command)
    check_cmd = (
        _join_argv(_with_sandbox_ignores(argv)) if argv else lint_command
    )
    outcome = run_command(
        repo_path,
        check_cmd,
        sandbox=sandbox,
        empty_message="Error: lint_command is required",
    )
    if outcome.passed or not argv:
        return outcome, False
    tree = inspect_worktree(repo_path, sandbox=sandbox)
    paths = _python_patch_paths(tree.changed_files, tree.untracked_files)
    fix_argv = _with_fix(_with_sandbox_ignores(argv), paths)
    if not fix_argv:
        return outcome, False
    run_command(
        repo_path,
        _join_argv(fix_argv),
        sandbox=sandbox,
        empty_message="Error: lint_command is required",
    )
    rechecked = run_command(
        repo_path,
        check_cmd,
        sandbox=sandbox,
        empty_message="Error: lint_command is required",
    )
    return rechecked, True


def deterministic_verify(state: AgentState, config: RunnableConfig) -> dict:
    cfg = require_config(config, "repo_path", "test_command")
    repo_path = cfg["repo_path"]
    test_command = cfg["test_command"]
    lint_command = str(cfg.get("lint_command") or "").strip()
    sandbox = cfg.get("sandbox")

    pytest_outcome = run_command(
        repo_path,
        test_command,
        sandbox=sandbox,
        empty_message="Error: test_command is required",
        timeout_message=f"Error: tests timed out after {COMMAND_TIMEOUT}s",
    )
    ruff_autofixed = False
    if lint_command:
        ruff_outcome, ruff_autofixed = _run_ruff(
            repo_path, lint_command, sandbox
        )
        if ruff_autofixed:
            get_reporter(config).note("ruff --fix applied")
    else:
        ruff_outcome = _MISSING_LINT
    # Inspect after autofix so the recorded patch includes lint repairs.
    patch = inspect_worktree(repo_path, sandbox=sandbox)

    pytest_passed = pytest_outcome.passed
    ruff_passed = ruff_outcome.passed
    patch_valid = patch.valid
    deterministic_pass = pytest_passed and ruff_passed and patch_valid

    label = "PASS" if deterministic_pass else "FAIL"
    get_reporter(config).stage(
        "verify",
        f"{label}  pytest={int(pytest_passed)} ruff={int(ruff_passed)} "
        f"patch={int(patch_valid)}",
    )

    return traced(
        state,
        {
            "test_result": {
                "command": test_command,
                "lint_command": lint_command,
                "exit_code": pytest_outcome.exit_code,
                "passed": deterministic_pass,
                "deterministic_pass": deterministic_pass,
                "pytest_passed": pytest_passed,
                "ruff_passed": ruff_passed,
                "ruff_autofixed": ruff_autofixed,
                "patch_valid": patch_valid,
                "changed_files": list(patch.changed_files),
                "untracked_files": list(patch.untracked_files),
                "output": pytest_outcome.format(),
                "ruff_output": ruff_outcome.format(),
                "git_diff": patch.format(),
            },
            "status": "verify_passed" if deterministic_pass else "verify_failed",
        },
        node="verify",
        detail=(
            f"{label}  pytest={int(pytest_passed)} ruff={int(ruff_passed)} "
            f"patch={int(patch_valid)}"
        ),
    )
