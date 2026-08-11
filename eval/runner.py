"""Eval runner: reset benchmark → run agent → score gold test → save run."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.client import create_client, default_model
from agent.loop import run_agent
from agent.tools.git import _find_git
from agent.tools.shell import run_tests

HARNESS_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = HARNESS_ROOT / "eval" / "dataset.json"
RUNS_DIR = HARNESS_ROOT / "runs"


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset.json must be a JSON array of tasks")
    return data


def get_task(task_id: str, path: Path = DATASET_PATH) -> dict[str, Any]:
    for task in load_dataset(path):
        if task.get("id") == task_id:
            return task
    known = ", ".join(t["id"] for t in load_dataset(path))
    raise KeyError(f"unknown task_id={task_id!r}; known: {known}")


def resolve_repo_path(task: dict[str, Any]) -> Path:
    raw = Path(task["repo_path"])
    repo = raw if raw.is_absolute() else (HARNESS_ROOT / raw).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"benchmark repo not found: {repo}")
    return repo


def reset_repo(repo_path: Path, base_commit: str) -> None:
    """Hard-reset the benchmark repo to ``base_commit`` and clean extras."""
    git = _find_git()
    if git is None:
        raise RuntimeError("git executable not found")

    commands = [
        [git, "-C", str(repo_path), "reset", "--hard", base_commit],
        [git, "-C", str(repo_path), "clean", "-fd"],
    ]
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"git command failed ({' '.join(cmd)}): {err or 'unknown error'}"
            )


def _test_file_from_command(test_command: str) -> str:
    """Extract the pytest target path from a module-level test_command."""
    # e.g. "pytest tests/test_auth.py -q" → "tests/test_auth.py"
    tokens = test_command.split()
    for token in tokens:
        if token.endswith(".py") or "/test" in token.replace("\\", "/"):
            return token
    raise ValueError(f"could not find test file in test_command={test_command!r}")


def run_gold_test(repo_path: Path, task: dict[str, Any]) -> dict[str, Any]:
    """Independently run the gold test; agent must not see this selector."""
    test_file = _test_file_from_command(task["test_command"])
    gold = task["gold_test"]
    # Keep node-id form so pytest runs only the gold assertion.
    command = f"pytest {test_file}::{gold} -q"
    output = run_tests(repo_path, command)
    match = re.search(r"exit_code=(-?\d+)", output)
    exit_code = int(match.group(1)) if match else 1
    return {
        "command": command,
        "exit_code": exit_code,
        "passed": exit_code == 0,
        "output": output,
    }


def save_run(record: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUNS_DIR / f"{record['task_id']}-{stamp}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def solve_task(
    task_id: str,
    *,
    model: str | None = None,
    max_steps: int = 15,
    client=None,
) -> dict[str, Any]:
    """Full harness cycle for one dataset task."""
    task = get_task(task_id)
    repo_path = resolve_repo_path(task)
    model_name = model or default_model()
    llm = client or create_client()

    reset_repo(repo_path, task["base_commit"])

    agent_result = run_agent(
        client=llm,
        issue=task["issue"],
        repo_path=str(repo_path),
        test_command=task["test_command"],
        model=model_name,
        max_steps=max_steps,
    )

    gold = run_gold_test(repo_path, task)

    record = {
        "task_id": task_id,
        "difficulty": task.get("difficulty"),
        "issue": task["issue"],
        "base_commit": task["base_commit"],
        "repo_path": str(repo_path),
        "expected_files": task.get("expected_files"),
        "model": model_name,
        "success": gold["passed"],
        "gold_test": gold,
        "tool_call_count": agent_result.get("tool_call_count"),
        "file_reads": agent_result.get("file_reads"),
        "steps": agent_result.get("steps"),
        "tokens": agent_result.get("tokens"),
        "prompt_tokens": agent_result.get("prompt_tokens"),
        "completion_tokens": agent_result.get("completion_tokens"),
        "estimated_cost": None,  # stretch: fill once pricing is wired
        "latency": agent_result.get("latency"),
        "termination": agent_result.get("termination"),
        "final_answer": agent_result.get("final_answer"),
        "trajectory": agent_result.get("trajectory"),
        "messages": agent_result.get("messages"),
        "python": sys.version,
    }
    record["run_path"] = str(save_run(record))
    return record
