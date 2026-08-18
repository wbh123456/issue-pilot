"""Stdio MCP client adapter for the demo and tests.

Does not import ``subprocess``; the official SDK spawns the server process.
Live V1/V2 tool dispatch does not use this adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HARNESS_ROOT = Path(__file__).resolve().parent.parent


def stdio_server_params(repo_path: str | Path):
    """Launch ``python -m harness.mcp_server --repo <path>`` over stdio."""
    from mcp import StdioServerParameters

    repo = str(Path(repo_path).resolve())
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "harness.mcp_server", "--repo", repo],
        cwd=str(HARNESS_ROOT),
        env={"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONPATH": str(HARNESS_ROOT)},
    )


def _text_from_result(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "") or ""))
    return "".join(parts)


async def mcp_roundtrip(
    repo_path: str | Path,
    calls: list[tuple[str, dict[str, Any]]],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Initialize a session, list tools, then invoke ``calls`` in order."""
    import asyncio

    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = stdio_server_params(repo_path)
    async with asyncio.timeout(timeout):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                results: list[dict[str, Any]] = []
                for name, arguments in calls:
                    outcome = await session.call_tool(name, arguments)
                    results.append(
                        {
                            "name": name,
                            "arguments": arguments,
                            "text": _text_from_result(outcome),
                            "isError": bool(getattr(outcome, "isError", False)),
                        }
                    )
                return {
                    "tools": [tool.name for tool in listed.tools],
                    "schemas": {
                        tool.name: dict(tool.inputSchema or {})
                        for tool in listed.tools
                    },
                    "calls": results,
                }


def run_roundtrip(
    repo_path: str | Path,
    calls: list[tuple[str, dict[str, Any]]],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    import functools

    import anyio

    return anyio.run(
        functools.partial(mcp_roundtrip, repo_path, calls, timeout=timeout)
    )


def run_demo(
    repo_path: str | Path,
    *,
    path: str = "app/auth.py",
    query: str = "decode_token",
) -> dict[str, Any]:
    """Client demo: list tools, read a file, search, then git_diff."""
    return run_roundtrip(
        repo_path,
        [
            ("read_file", {"path": path}),
            ("search_code", {"query": query}),
            ("git_diff", {}),
        ],
        timeout=60.0,
    )
