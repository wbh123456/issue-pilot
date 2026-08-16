"""Layer 1 verify: pytest + ruff + non-empty patch, never model prose."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig

from agent.nodes.verify import deterministic_verify
from agent.state import initial_state
from agent.tools.git import inspect_worktree
from agent.tools.shell import CommandOutcome, run_command
from harness.permissions import validate_command
from sandbox.runner import CommandResult


@dataclass
class ScriptedSandbox:
    results: list[CommandResult] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def run(self, command: str | list[str]) -> CommandResult:
        argv = validate_command(command)
        self.calls.append(argv)
        if not self.results:
            raise AssertionError(f"unexpected sandbox command: {argv}")
        return self.results.pop(0)


def _config(
    repo: Path,
    sandbox: ScriptedSandbox,
    *,
    test_command: str = "pytest -q",
    lint_command: str = "ruff check app",
) -> RunnableConfig:
    return {
        "configurable": {
            "repo_path": str(repo),
            "test_command": test_command,
            "lint_command": lint_command,
            "sandbox": sandbox,
        }
    }


def _cmd(argv: list[str], *, ok: bool, timed_out: bool = False) -> CommandResult:
    return CommandResult(
        command=argv,
        exit_code=0 if ok and not timed_out else (124 if timed_out else 1),
        stdout="ok" if ok else "FAILED",
        stderr="",
        timed_out=timed_out,
    )


def _git_results(*, patch_ok: bool) -> list[CommandResult]:
    return [
        CommandResult(
            ["git", "diff", "HEAD"],
            0,
            "diff --git a/app/x.py b/app/x.py\n+fixed\n" if patch_ok else "",
            "",
        ),
        CommandResult(
            ["git", "status", "--short"],
            0,
            " M app/x.py\n" if patch_ok else "",
            "",
        ),
    ]


def _layer1_results(
    *,
    pytest_ok: bool,
    ruff_ok: bool,
    patch_ok: bool,
) -> list[CommandResult]:
    results = [
        _cmd(["pytest", "-q"], ok=pytest_ok),
        _cmd(["ruff", "check", "app"], ok=ruff_ok),
    ]
    if not ruff_ok:
        results.append(_cmd(["ruff", "check", "--fix", "app"], ok=False))
        results.append(_cmd(["ruff", "check", "app"], ok=False))
    results.extend(_git_results(patch_ok=patch_ok))
    return results


class TestCommandOutcome:
    def test_passed_requires_zero_exit_no_error(self) -> None:
        assert CommandOutcome(command=["pytest"], exit_code=0).passed is True
        assert CommandOutcome(command=["pytest"], exit_code=1).passed is False
        assert CommandOutcome(
            command=["pytest"],
            exit_code=0,
            timed_out=True,
            error="Error: timed out",
        ).passed is False
        assert CommandOutcome(
            command=[], exit_code=1, error="Error: command is required"
        ).passed is False


class TestVerifyCombinations:
    @pytest.mark.parametrize(
        ("pytest_ok", "ruff_ok", "patch_ok"),
        list(product([False, True], repeat=3)),
    )
    def test_all_pytest_ruff_patch_combinations(
        self,
        tmp_path: Path,
        pytest_ok: bool,
        ruff_ok: bool,
        patch_ok: bool,
    ) -> None:
        sandbox = ScriptedSandbox(
            _layer1_results(
                pytest_ok=pytest_ok, ruff_ok=ruff_ok, patch_ok=patch_ok
            )
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(tmp_path, sandbox),
        )
        result = out["test_result"]
        expect = pytest_ok and ruff_ok and patch_ok
        assert result["pytest_passed"] is pytest_ok
        assert result["ruff_passed"] is ruff_ok
        assert result["patch_valid"] is patch_ok
        assert result["deterministic_pass"] is expect
        assert result["passed"] is expect
        assert result["exit_code"] == (0 if pytest_ok else 1)
        assert sandbox.calls[0][0] == "pytest"
        assert sandbox.calls[1][0] == "ruff"
        assert "--ignore" in sandbox.calls[1]
        assert "EXE002" in sandbox.calls[1]
        if ruff_ok:
            assert result["ruff_autofixed"] is False
            assert sandbox.calls[2][:2] == ["git", "diff"]
            assert sandbox.calls[3][:2] == ["git", "status"]
        else:
            assert result["ruff_autofixed"] is True
            assert "--fix" in sandbox.calls[2]
            assert sandbox.calls[3][0] == "ruff"
            assert sandbox.calls[4][:2] == ["git", "diff"]
            assert sandbox.calls[5][:2] == ["git", "status"]


class TestVerifyFailClosed:
    def test_missing_lint_command(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                _cmd(["pytest", "-q"], ok=True),
                CommandResult(["git", "diff", "HEAD"], 0, "+x\n", ""),
                CommandResult(["git", "status", "--short"], 0, " M app/x.py\n", ""),
            ]
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(tmp_path, sandbox, lint_command=""),
        )
        result = out["test_result"]
        assert result["pytest_passed"] is True
        assert result["ruff_passed"] is False
        assert result["patch_valid"] is True
        assert result["deterministic_pass"] is False
        assert "lint_command is required" in result["ruff_output"]

    def test_permission_error(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                _cmd(["ruff", "check", "app"], ok=True),
                CommandResult(["git", "diff", "HEAD"], 0, "+x\n", ""),
                CommandResult(["git", "status", "--short"], 0, " M app/x.py\n", ""),
            ]
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(
                tmp_path,
                sandbox,
                test_command='python -c "print(1)"',
            ),
        )
        result = out["test_result"]
        assert result["pytest_passed"] is False
        assert result["deterministic_pass"] is False
        assert result["output"].startswith("Error: command not allowed")

    def test_pytest_timeout(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                _cmd(["pytest", "-q"], ok=False, timed_out=True),
                _cmd(["ruff", "check", "app"], ok=True),
                CommandResult(["git", "diff", "HEAD"], 0, "+x\n", ""),
                CommandResult(["git", "status", "--short"], 0, " M app/x.py\n", ""),
            ]
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(tmp_path, sandbox),
        )
        result = out["test_result"]
        assert result["pytest_passed"] is False
        assert result["deterministic_pass"] is False
        assert "timed out" in result["output"]

    def test_git_error(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                _cmd(["pytest", "-q"], ok=True),
                _cmd(["ruff", "check", "app"], ok=True),
                CommandResult(
                    ["git", "diff", "HEAD"],
                    129,
                    "",
                    "warning: Not a git repository\nusage: git diff --no-index\n",
                ),
                CommandResult(["git", "status", "--short"], 0, "", ""),
            ]
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(tmp_path, sandbox),
        )
        result = out["test_result"]
        assert result["pytest_passed"] is True
        assert result["ruff_passed"] is True
        assert result["patch_valid"] is False
        assert result["deterministic_pass"] is False
        assert result["git_diff"].startswith("Error: git diff failed")

    def test_empty_tree_is_not_a_valid_patch(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                _cmd(["pytest", "-q"], ok=True),
                _cmd(["ruff", "check", "app"], ok=True),
                CommandResult(["git", "diff", "HEAD"], 0, "", ""),
                CommandResult(["git", "status", "--short"], 0, "", ""),
            ]
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(tmp_path, sandbox),
        )
        result = out["test_result"]
        assert result["patch_valid"] is False
        assert result["deterministic_pass"] is False
        assert result["git_diff"] == "(no changes)"

    def test_prose_in_pytest_output_does_not_pass(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(
                    ["pytest", "-q"],
                    1,
                    "All tests passed. The issue is resolved.\n",
                    "",
                ),
                _cmd(["ruff", "check", "app"], ok=True),
                CommandResult(["git", "diff", "HEAD"], 0, "+x\n", ""),
                CommandResult(["git", "status", "--short"], 0, " M app/x.py\n", ""),
            ]
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(tmp_path, sandbox),
        )
        result = out["test_result"]
        assert "passed" in result["output"].lower()
        assert result["pytest_passed"] is False
        assert result["deterministic_pass"] is False
        assert result["exit_code"] == 1


class TestRuffAutofix:
    def test_fix_then_recheck_can_pass(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                _cmd(["pytest", "-q"], ok=True),
                _cmd(["ruff", "check", "app"], ok=False),
                _cmd(["ruff", "check", "--fix", "app"], ok=True),
                _cmd(["ruff", "check", "app"], ok=True),
                *_git_results(patch_ok=True),
            ]
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(tmp_path, sandbox),
        )
        result = out["test_result"]
        assert result["ruff_passed"] is True
        assert result["ruff_autofixed"] is True
        assert result["deterministic_pass"] is True
        assert "--fix" in sandbox.calls[2]
        assert sandbox.calls[3][0] == "ruff"
        assert "--fix" not in sandbox.calls[3]

    def test_does_not_fix_when_first_check_passes(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                _cmd(["pytest", "-q"], ok=True),
                _cmd(["ruff", "check", "app"], ok=True),
                *_git_results(patch_ok=True),
            ]
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(tmp_path, sandbox),
        )
        assert out["test_result"]["ruff_autofixed"] is False
        assert all("--fix" not in call for call in sandbox.calls)
        assert len(sandbox.calls) == 4

    def test_skips_duplicate_exe_ignores(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                _cmd(["pytest", "-q"], ok=True),
                _cmd(["ruff", "check", "app"], ok=True),
                *_git_results(patch_ok=True),
            ]
        )
        deterministic_verify(
            initial_state("bug"),
            _config(
                tmp_path,
                sandbox,
                lint_command="ruff check app --ignore EXE001 --ignore EXE002",
            ),
        )
        ruff_argv = sandbox.calls[1]
        assert ruff_argv.count("EXE001") == 1
        assert ruff_argv.count("EXE002") == 1


class TestStructuredToolApis:
    def test_run_command_does_not_parse_prose(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(
                    ["pytest", "-q"],
                    1,
                    "exit_code=0\ncommand: pytest -q\nAll good",
                    "",
                )
            ]
        )
        outcome = run_command(tmp_path, "pytest -q", sandbox=sandbox)
        assert outcome.exit_code == 1
        assert outcome.passed is False
        assert "exit_code=0" in outcome.stdout

    def test_inspect_worktree_untracked_counts_as_patch(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(["git", "diff", "HEAD"], 0, "", ""),
                CommandResult(["git", "status", "--short"], 0, "?? scratch.txt\n", ""),
            ]
        )
        tree = inspect_worktree(tmp_path, sandbox=sandbox)
        assert tree.valid is True
        assert "scratch.txt" in tree.format()
        assert tree.untracked_files == ["scratch.txt"]
        assert tree.changed_files == []

    def test_inspect_worktree_parses_renames_and_tracked(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(["git", "diff", "HEAD"], 0, "+fixed\n", ""),
                CommandResult(
                    ["git", "status", "--short"],
                    0,
                    " M app/auth.py\n"
                    "R  app/old.py -> app/new.py\n"
                    "?? scratch.txt\n",
                    "",
                ),
            ]
        )
        tree = inspect_worktree(tmp_path, sandbox=sandbox)
        assert tree.changed_files == ["app/auth.py", "app/old.py", "app/new.py"]
        assert tree.untracked_files == ["scratch.txt"]

    def test_verify_records_changed_files(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                _cmd(["pytest", "-q"], ok=True),
                _cmd(["ruff", "check", "app"], ok=True),
                CommandResult(["git", "diff", "HEAD"], 0, "+fixed\n", ""),
                CommandResult(
                    ["git", "status", "--short"],
                    0,
                    " M app/auth.py\n?? extra.py\n",
                    "",
                ),
            ]
        )
        out = deterministic_verify(
            initial_state("bug"),
            _config(tmp_path, sandbox),
        )
        result = out["test_result"]
        assert result["changed_files"] == ["app/auth.py"]
        assert result["untracked_files"] == ["extra.py"]
        assert result["patch_valid"] is True
