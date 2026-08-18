"""Minimal MCP server: in-process tools plus one real stdio round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.mcp_client import run_roundtrip
from harness.mcp_server import TOOL_NAMES, TOOL_SPECS, ToolHost
from harness.permissions import validate_command
from sandbox.image import DockerPreflightError
from sandbox.runner import CommandResult


class ScriptedSandbox:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []
        self.cleaned_up = False

    def run(self, command: str | list[str]) -> CommandResult:
        argv = validate_command(command)
        self.calls.append(argv)
        if not self.results:
            raise AssertionError(f"unexpected sandbox command: {argv}")
        return self.results.pop(0)

    def cleanup(self) -> None:
        self.cleaned_up = True


def _repo(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "hello.py").write_text(
        "def greet(name):\n    return f'hi {name}'\n",
        encoding="utf-8",
    )
    return tmp_path


class TestToolSpecs:
    def test_lists_three_tools_with_schemas(self) -> None:
        names = [spec["name"] for spec in TOOL_SPECS]
        assert names == list(TOOL_NAMES) == ["read_file", "search_code", "git_diff"]
        by_name = {spec["name"]: spec for spec in TOOL_SPECS}
        assert by_name["read_file"]["inputSchema"]["required"] == ["path"]
        assert "path" in by_name["read_file"]["inputSchema"]["properties"]
        assert by_name["search_code"]["inputSchema"]["required"] == ["query"]
        assert by_name["git_diff"]["inputSchema"]["properties"] == {}


class TestInProcessHost:
    def test_read_and_search(self, tmp_path: Path) -> None:
        host = ToolHost(_repo(tmp_path))
        try:
            assert [spec["name"] for spec in host.specs()] == list(TOOL_NAMES)
            text = host.call_tool("read_file", {"path": "app/hello.py"})
            assert "def greet" in text
            hits = host.call_tool("search_code", {"query": "greet"})
            assert "hello.py" in hits
            assert host.call_tool("nope", {}) == "Error: unknown tool: nope"
            assert host.call_tool("read_file", {"path": "missing.py"}).startswith(
                "Error:"
            )
        finally:
            host.close()

    def test_git_diff_uses_injected_sandbox(self, tmp_path: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(
                    command=["git", "diff", "HEAD"],
                    exit_code=0,
                    stdout="+added\n",
                    stderr="",
                ),
                CommandResult(
                    command=["git", "status", "--short"],
                    exit_code=0,
                    stdout=" M app/hello.py\n",
                    stderr="",
                ),
            ]
        )
        host = ToolHost(_repo(tmp_path), sandbox_factory=lambda repo: sandbox)
        try:
            out = host.call_tool("git_diff", {})
            assert "added" in out
            assert sandbox.calls[0][:2] == ["git", "diff"]
        finally:
            host.close()
        assert sandbox.cleaned_up is True

    def test_git_diff_reports_missing_docker(self, tmp_path: Path) -> None:
        def factory(_repo_path: Path):
            raise DockerPreflightError("Docker daemon is not reachable")

        host = ToolHost(_repo(tmp_path), sandbox_factory=factory)
        try:
            out = host.call_tool("git_diff", {})
            assert out.startswith("Error: git_diff needs Docker:")
            assert "not reachable" in out
        finally:
            host.close()


class TestStdioRoundTrip:
    def test_client_lists_tools_and_reads_file(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path)
        result = run_roundtrip(
            repo,
            [("read_file", {"path": "app/hello.py"})],
            timeout=45.0,
        )
        assert result["tools"] == ["read_file", "search_code", "git_diff"]
        assert result["schemas"]["read_file"]["required"] == ["path"]
        assert result["calls"][0]["isError"] is False
        assert "def greet" in result["calls"][0]["text"]
