"""OpenAI/DeepSeek-compatible tool schemas for the agent loop.

Only LLM-controlled arguments appear here. ``repo_path`` and ``test_command``
are injected by ``execute_tool``, never by the model.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories under a path relative to the "
                "repository root. Directories end with '/'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path. Defaults to '.'.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_code",
            "description": (
                "Search repository source files for a literal query string. "
                "Returns matching lines as path:line:content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Literal text to search for.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit a file inside the repository. Use exactly one mode: "
                "(1) `content` to overwrite/create the whole file, or "
                "(2) `old_str` + `new_str` to replace exactly one occurrence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path to edit.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full new file contents (full-write mode).",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact text to find (search-replace mode).",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text (search-replace mode).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the task's test suite for the current issue and return "
                "stdout/stderr plus exit code. Call after making edits."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": (
                "Show the current working-tree changes against HEAD, including "
                "untracked files via short status."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

SEARCH_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": (
            "Hybrid (BM25 + dense) search over app/*.py symbols. "
            "Start here to locate an issue: the user's words often differ "
            "from identifiers in the code. Prefer this over grep_code when "
            "you do not already know the exact symbol or file name. Returns "
            "ranked chunks with path, symbol, and line range."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language or identifier query.",
                },
            },
            "required": ["query"],
        },
    },
}

# V0/V1 keep the original six tools. V2 appends search_code for ablation isolation.
V2_TOOLS = [*TOOLS, SEARCH_CODE_TOOL]
