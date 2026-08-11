"""Eval runner: reset benchmark → run harness → score gold test → save run."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from agent.client import create_client, default_model
from agent.graph import run_workflow
from agent.loop import run_agent
from agent.tools.git import _find_git
from agent.tools.shell import run_tests

HARNESS_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = HARNESS_ROOT / "eval" / "dataset.json"
RUNS_DIR = HARNESS_ROOT / "runs"

HarnessVersion = Literal["v0", "v1"]


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


def _normalize_harness(harness_version: str) -> HarnessVersion:
    value = (harness_version or "v0").strip().lower()
    if value not in {"v0", "v1"}:
        raise ValueError(
            f"harness_version must be 'v0' or 'v1', got {harness_version!r}"
        )
    return value  # type: ignore[return-value]


def save_run(record: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    harness = record.get("harness_version") or "v0"
    out = RUNS_DIR / f"{record['task_id']}-{harness}-{stamp}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _run_harness(
    *,
    harness: HarnessVersion,
    client,
    issue: str,
    repo_path: str,
    test_command: str,
    model: str,
    max_steps: int,
) -> dict[str, Any]:
    if harness == "v0":
        return run_agent(
            client=client,
            issue=issue,
            repo_path=repo_path,
            test_command=test_command,
            model=model,
            max_steps=max_steps,
        )
    return run_workflow(
        client=client,
        issue=issue,
        repo_path=repo_path,
        test_command=test_command,
        model=model,
        max_steps=max_steps,
    )


def _empty_agent_result() -> dict[str, Any]:
    return {
        "final_answer": "",
        "termination": "error",
        "steps": 0,
        "tool_call_count": 0,
        "file_reads": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tokens": 0,
        "latency": 0.0,
        "trajectory": [],
        "messages": [],
        "llm_calls": 0,
        "stage_tokens": {},
    }


def solve_task(
    task_id: str,
    *,
    model: str | None = None,
    max_steps: int = 15,
    client=None,
    harness_version: str = "v0",
) -> dict[str, Any]:
    """Full harness cycle for one dataset task.

    Same flow for both harnesses: reset repo → run harness → gold test → save.
    Gold scoring stays independent of V0/V1 workflow verification.
    Exceptions still produce a versioned run artifact.
    """
    harness = _normalize_harness(harness_version)
    task = get_task(task_id)
    repo_path = resolve_repo_path(task)
    model_name = model or default_model()
    llm = client or create_client()

    reset_repo(repo_path, task["base_commit"])

    agent_result = _empty_agent_result()
    gold: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    gold_error_type: str | None = None
    gold_error_message: str | None = None

    try:
        agent_result = _run_harness(
            harness=harness,
            client=llm,
            issue=task["issue"],
            repo_path=str(repo_path),
            test_command=task["test_command"],
            model=model_name,
            max_steps=max_steps,
        )
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        agent_result = _empty_agent_result()

    try:
        gold = run_gold_test(repo_path, task)
    except Exception as exc:
        gold_error_type = type(exc).__name__
        gold_error_message = str(exc)
        gold = {
            "command": "",
            "exit_code": 1,
            "passed": False,
            "output": f"Error running gold test: {gold_error_type}: {gold_error_message}",
        }

    # V0 has one LLM call per ReAct step; V1 reports aggregated stage calls.
    llm_calls = agent_result.get("llm_calls")
    if llm_calls is None:
        llm_calls = agent_result.get("steps")

    termination = agent_result.get("termination")
    if error_type is not None:
        termination = "error"

    record: dict[str, Any] = {
        "task_id": task_id,
        "harness_version": harness,
        "difficulty": task.get("difficulty"),
        "issue": task["issue"],
        "base_commit": task["base_commit"],
        "repo_path": str(repo_path),
        "expected_files": task.get("expected_files"),
        "model": model_name,
        "success": bool(gold and gold.get("passed")) and error_type is None,
        "gold_test": gold,
        "tool_call_count": agent_result.get("tool_call_count"),
        "file_reads": agent_result.get("file_reads"),
        "steps": agent_result.get("steps"),
        "llm_calls": llm_calls,
        "tokens": agent_result.get("tokens"),
        "prompt_tokens": agent_result.get("prompt_tokens"),
        "completion_tokens": agent_result.get("completion_tokens"),
        "estimated_cost": None,  # stretch: fill once pricing is wired
        "latency": agent_result.get("latency"),
        "termination": termination,
        "final_answer": agent_result.get("final_answer"),
        "trajectory": agent_result.get("trajectory"),
        "messages": agent_result.get("messages"),
        "python": sys.version,
    }

    if error_type is not None:
        record["error_type"] = error_type
        record["error_message"] = error_message
    if gold_error_type is not None:
        record["gold_error_type"] = gold_error_type
        record["gold_error_message"] = gold_error_message

    if harness == "v1":
        record.update(
            {
                "analysis": agent_result.get("analysis", ""),
                "plan": agent_result.get("plan", {}),
                "diagnosis": agent_result.get("diagnosis", ""),
                "verification": agent_result.get("test_result", {}),
                "status": agent_result.get("status", ""),
                "workflow_passed": agent_result.get("workflow_passed"),
                "stage_tokens": agent_result.get("stage_tokens", {}),
            }
        )

    record["run_path"] = str(save_run(record))
    return record
