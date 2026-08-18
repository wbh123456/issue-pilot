"""Minimal MCP stdio server wrapping existing path-jailed tools.

Lives under ``harness/`` so a top-level ``mcp/`` package cannot shadow the
official SDK, and so the subprocess audit still covers this module. The
server process never imports ``subprocess``; ``git_diff`` uses SandboxRunner
which talks to Docker through ``sandbox.image.run_docker``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from agent.tools.filesystem import read_file
from agent.tools.git import git_diff
from agent.tools.search import search_code
from sandbox.image import DockerPreflightError
from sandbox.runner import SandboxError, SandboxRunner, SandboxUnusableError

HARNESS_ROOT = Path(__file__).resolve().parent.parent
SandboxFactory = Callable[[str | Path], Any]

TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file relative to the repository root. "
            "Paths are jailed to the repo."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repository-relative file path",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Hybrid symbol search over app/**/*.py (BM25 + dense + RRF). "
            "Host-side, same trust class as grep_code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language or identifier query",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "git_diff",
        "description": (
            "Working-tree diff against HEAD, executed inside the Docker "
            "sandbox. Returns a clean error if Docker is not ready."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
)

TOOL_NAMES = tuple(spec["name"] for spec in TOOL_SPECS)


def default_repo_path() -> Path:
    """Sibling benchmark worktree used by ``eval/dataset.json``."""
    return (HARNESS_ROOT / ".." / "issue-pilot-benchmark").resolve()


def default_sandbox_factory(repo_path: str | Path) -> SandboxRunner:
    return SandboxRunner(repo_path, task_id="mcp")


class ToolHost:
    """In-process tool dispatch. Stdio is a thin protocol wrapper around this."""

    def __init__(
        self,
        repo_path: str | Path,
        *,
        sandbox_factory: SandboxFactory | None = None,
    ) -> None:
        root = Path(repo_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"repository not found: {root}")
        self.repo_path = root
        self._sandbox_factory = sandbox_factory or default_sandbox_factory
        self._sandbox: Any | None = None
        self._sandbox_cm: Any | None = None

    def specs(self) -> list[dict[str, Any]]:
        return [dict(spec) for spec in TOOL_SPECS]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        arguments = arguments or {}
        if name == "read_file":
            return self._read_file(str(arguments.get("path") or ""))
        if name == "search_code":
            return self._search_code(str(arguments.get("query") or ""))
        if name == "git_diff":
            return self._git_diff()
        return f"Error: unknown tool: {name}"

    def close(self) -> None:
        cm = self._sandbox_cm
        sandbox = self._sandbox
        self._sandbox_cm = None
        self._sandbox = None
        if cm is not None and hasattr(cm, "__exit__"):
            cm.__exit__(None, None, None)
            return
        if sandbox is not None and hasattr(sandbox, "cleanup"):
            sandbox.cleanup()

    def _read_file(self, path: str) -> str:
        try:
            return read_file(self.repo_path, path)
        except (OSError, PermissionError, ValueError) as exc:
            return f"Error: {exc}"

    def _search_code(self, query: str) -> str:
        try:
            return search_code(self.repo_path, query)
        except (OSError, PermissionError, ValueError) as exc:
            return f"Error: {exc}"

    def _git_diff(self) -> str:
        try:
            sandbox = self._ensure_sandbox()
            return git_diff(self.repo_path, sandbox=sandbox)
        except DockerPreflightError as exc:
            return f"Error: git_diff needs Docker: {exc}"
        except (SandboxUnusableError, SandboxError) as exc:
            return f"Error: git_diff sandbox unavailable: {exc}"
        except (OSError, PermissionError, ValueError, RuntimeError) as exc:
            return f"Error: git_diff failed: {exc}"

    def _ensure_sandbox(self) -> Any:
        if self._sandbox is not None:
            return self._sandbox
        created = self._sandbox_factory(self.repo_path)
        self._sandbox_cm = created
        if hasattr(created, "__enter__"):
            self._sandbox = created.__enter__()
        elif hasattr(created, "start"):
            created.start()
            self._sandbox = created
        else:
            self._sandbox = created
        return self._sandbox


def run_stdio_server(
    repo_path: str | Path,
    *,
    sandbox_factory: SandboxFactory | None = None,
) -> None:
    """Serve tools on stdin/stdout. Must not write logs to stdout."""
    import anyio
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    host = ToolHost(repo_path, sandbox_factory=sandbox_factory)
    server = Server("issue-pilot")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [Tool(**spec) for spec in host.specs()]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]):
        text = host.call_tool(name, arguments)
        return [TextContent(type="text", text=text)]

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    try:
        anyio.run(_run)
    finally:
        host.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness.mcp_server",
        description="IssuePilot MCP stdio server",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository root the tools are jailed to",
    )
    args = parser.parse_args(argv)
    run_stdio_server(args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
