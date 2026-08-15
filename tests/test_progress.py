"""Live progress formatters and optional reporter wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from agent.loop import run_agent
from harness.progress import (
    PREVIEW_CHARS,
    ConsoleReporter,
    NullReporter,
    format_note_line,
    format_size,
    format_stage_line,
    format_tool_line,
    get_reporter,
    preview,
    summarize_files,
    summarize_tool,
)


class RecordingReporter:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def stage(self, name: str, detail: str = "") -> None:
        self.events.append(("stage", name, detail))

    def note(self, text: str) -> None:
        self.events.append(("note", text))

    def tool(self, step: int, name: str, args: dict[str, Any], result: str) -> None:
        self.events.append(("tool", step, name, args, result))


class TestFormatters:
    def test_preview_collapses_and_truncates(self) -> None:
        assert preview("  hello\nworld  ") == "hello world"
        long = "x" * (PREVIEW_CHARS + 40)
        out = preview(long)
        assert len(out) <= PREVIEW_CHARS
        assert out.endswith("...")

    def test_read_file_shows_size_not_body(self) -> None:
        body = "def f():\n    return 1\n" * 80
        detail, outcome = summarize_tool("read_file", {"path": "app/auth.py"}, body)
        assert detail == "app/auth.py"
        assert "def f" not in outcome
        assert "chars" in outcome
        line = format_tool_line(1, "read_file", {"path": "app/auth.py"}, body)
        assert "app/auth.py" in line
        assert "def f" not in line

    def test_run_tests_exit_code_only(self) -> None:
        result = "exit_code=1\ncommand: pytest -q\n--- stdout ---\nFAILED tests/test_x.py"
        detail, outcome = summarize_tool("run_tests", {}, result)
        assert detail == ""
        assert outcome == "exit 1"
        line = format_tool_line(4, "run_tests", {}, result)
        assert "FAILED" not in line
        assert "exit 1" in line

    def test_grep_hit_count(self) -> None:
        result = "app/a.py:1:foo\napp/b.py:2:foo\napp/c.py:3:foo"
        _, outcome = summarize_tool("grep_code", {"query": "foo"}, result)
        assert outcome == "3 hits"
        _, none = summarize_tool("grep_code", {"query": "zzz"}, "(no matches)")
        assert none == "(no matches)"

    def test_git_diff_not_dumped(self) -> None:
        _, empty = summarize_tool("git_diff", {}, "(no changes)")
        assert empty == "no changes"
        huge = "--- diff ---\n" + ("+" * 4000)
        _, changed = summarize_tool("git_diff", {}, huge)
        assert changed == "has diff"
        assert "+" not in format_tool_line(1, "git_diff", {}, huge)

    def test_format_size(self) -> None:
        assert format_size(12) == "12 chars"
        assert format_size(1200) == "1.2k chars"

    def test_summarize_files(self) -> None:
        assert summarize_files([]) == "(none)"
        assert summarize_files(["a.py", "b.py", "c.py"]) == "a.py, b.py (+1)"

    def test_stage_and_note_lines(self) -> None:
        assert format_stage_line("execute") == "execute"
        assert format_stage_line("execute", "retry 1") == "execute  retry 1"
        assert format_note_line("  hello\nworld").startswith("  hello world")

    def test_error_is_previewed(self) -> None:
        _, outcome = summarize_tool(
            "read_file", {"path": "app/x.py"}, "Error: file not found: app/x.py"
        )
        assert outcome.startswith("Error:")


class TestReporter:
    def test_null_is_default(self) -> None:
        assert isinstance(get_reporter(None), NullReporter)
        rec = RecordingReporter()
        assert get_reporter(rec) is rec

    def test_console_reporter_prints_formatted_lines(self) -> None:
        printed: list[str] = []

        class FakeConsole:
            def print(self, text: str) -> None:
                printed.append(text)

        reporter = ConsoleReporter(FakeConsole())
        reporter.stage("analyze")
        reporter.note("Expired JWT returns 500")
        reporter.tool(1, "read_file", {"path": "app/auth.py"}, "x" * 50)
        assert printed[0] == "analyze"
        assert printed[1] == "  Expired JWT returns 500"
        assert "read_file" in printed[2]
        assert "app/auth.py" in printed[2]


class _Fn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, name: str, arguments: str, call_id: str = "c1") -> None:
        self.id = call_id
        self.function = _Fn(name, arguments)

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _Usage:
    prompt_tokens = 1
    completion_tokens = 1


class _Message:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.message = _Message(content, tool_calls)


class _Response:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.choices = [_Choice(content, tool_calls)]
        self.usage = _Usage()


class FakeClient:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self.chat = MagicMock()
        self.chat.completions.create.side_effect = lambda **kwargs: self._responses.pop(0)


class TestLoopProgress:
    def test_emits_tool_after_result(self) -> None:
        reporter = RecordingReporter()
        client = FakeClient(
            [
                _Response(
                    "",
                    tool_calls=[_Call("read_file", '{"path": "app/auth.py"}')],
                ),
                _Response("done"),
            ]
        )
        with patch(
            "agent.loop.execute_tool",
            return_value="def decode_token():\n    pass\n",
        ):
            run_agent(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                progress=reporter,
            )
        tools = [e for e in reporter.events if e[0] == "tool"]
        assert len(tools) == 1
        assert tools[0][1] == 1
        assert tools[0][2] == "read_file"
        assert tools[0][3]["path"] == "app/auth.py"

    def test_silent_without_reporter(self) -> None:
        client = FakeClient([_Response("done")])
        result = run_agent(
            client=client,
            issue="bug",
            repo_path="/tmp/repo",
            test_command="pytest -q",
        )
        assert result["final_answer"] == "done"
