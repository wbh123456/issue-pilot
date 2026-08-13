"""Verify node: deterministic test exit-code check (never LLM judgment)."""

from __future__ import annotations

import re

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from agent.tools.git import git_diff
from agent.tools.shell import run_tests

from ._runtime import require_config


def _parse_exit_code(output: str) -> int:
    match = re.search(r"exit_code=(-?\d+)", output)
    if match:
        return int(match.group(1))
    # Timeouts / tool errors are failures.
    return 1


def deterministic_verify(state: AgentState, config: RunnableConfig) -> dict:
    cfg = require_config(config, "repo_path", "test_command")
    repo_path = cfg["repo_path"]
    test_command = cfg["test_command"]

    sandbox = cfg.get("sandbox")
    output = run_tests(repo_path, test_command, sandbox=sandbox)
    exit_code = _parse_exit_code(output)
    passed = exit_code == 0
    diff = git_diff(repo_path, sandbox=sandbox)

    return {
        "test_result": {
            "command": test_command,
            "exit_code": exit_code,
            "passed": passed,
            "output": output,
            "git_diff": diff,
        },
        "status": "verify_passed" if passed else "verify_failed",
    }
