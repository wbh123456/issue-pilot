from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .filesystem import edit_file, list_files, read_file
from .git import git_diff
from .search import grep_code, search_code
from .shell import run_tests

if TYPE_CHECKING:
    from sandbox.runner import SandboxRunner


def execute_tool(
    name: str,
    args: dict[str, Any],
    *,
    repo_path: str,
    test_command: str,
    sandbox: SandboxRunner | None = None,
    search_code_enabled: bool = False,
) -> str:
    if name == "list_files":
        return list_files(repo_path, args.get("path", "."))
    if name == "read_file":
        return read_file(repo_path, args["path"])
    if name == "grep_code":
        return grep_code(repo_path, args["query"])
    if name == "search_code":
        if not search_code_enabled:
            return f"Error: unknown tool: {name}"
        return search_code(repo_path, args.get("query") or "")
    if name == "edit_file":
        return edit_file(repo_path, **args)
    if name == "run_tests":
        return run_tests(repo_path, test_command, sandbox=sandbox)
    if name == "git_diff":
        return git_diff(repo_path, sandbox=sandbox)

    return f"Error: unknown tool: {name}"
