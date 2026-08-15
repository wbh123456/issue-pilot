"""Eval runner: reset benchmark → sandbox → run harness → gold test → save."""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from agent.client import create_client, default_model
from agent.graph import get_v2_graph, run_workflow
from agent.loop import run_agent
from agent.tools.shell import run_tests
from eval.metrics import recall_at_k
from eval.repository import GOLD_STAGING_DIRNAME, git_sha, reset_repo
from harness.limits import AGENT_TEMPERATURE, MAX_AGENT_STEPS
from retrieval.query import DEFAULT_QUERY_MODE, normalize_query_mode
from sandbox.image import DEFAULT_IMAGE
from sandbox.runner import SandboxMetadata, SandboxRunner

HARNESS_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = HARNESS_ROOT / "eval" / "dataset.json"
GOLD_DIR = HARNESS_ROOT / "eval" / "gold"
RUNS_DIR = HARNESS_ROOT / "runs"

HarnessVersion = Literal["v0", "v1", "v2"]


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


def gold_file_name(task: dict[str, Any]) -> str:
    """Return the gold module filename for ``task`` (under eval/gold/)."""
    named = task.get("gold_file")
    if named:
        return str(named)
    return f"test_{str(task['id']).replace('-', '_')}.py"


def gold_staging_dir(repo_path: Path) -> Path:
    return Path(repo_path) / "tests" / GOLD_STAGING_DIRNAME


def cleanup_gold_staging(repo_path: Path) -> None:
    staged = gold_staging_dir(repo_path)
    if staged.exists():
        shutil.rmtree(staged, ignore_errors=True)


def stage_gold_file(repo_path: Path, task: dict[str, Any]) -> Path:
    """Copy the hidden gold module into the benchmark for scoring only."""
    src = GOLD_DIR / gold_file_name(task)
    if not src.is_file():
        raise FileNotFoundError(f"gold file not found: {src}")
    dest_dir = gold_staging_dir(repo_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def run_gold_test(
    repo_path: Path,
    task: dict[str, Any],
    *,
    sandbox,
) -> dict[str, Any]:
    """Run the hidden gold module; agent tools never see this file.

    The module is copied into ``tests/_gold/`` for the duration of scoring
    (path jail + ``norecursedirs`` hide that directory from the agent), then
    removed. ``test_command`` is not used here.
    """
    try:
        staged = stage_gold_file(repo_path, task)
        rel = staged.relative_to(repo_path).as_posix()
        gold = task.get("gold_test")
        command = f"pytest {rel}::{gold} -q" if gold else f"pytest {rel} -q"
        output = run_tests(repo_path, command, sandbox=sandbox)
        match = re.search(r"exit_code=(-?\d+)", output)
        exit_code = int(match.group(1)) if match else 1
        return {
            "command": command,
            "exit_code": exit_code,
            "passed": exit_code == 0,
            "output": output,
        }
    finally:
        cleanup_gold_staging(repo_path)


def _normalize_harness(harness_version: str) -> HarnessVersion:
    value = (harness_version or "v0").strip().lower()
    if value not in {"v0", "v1", "v2"}:
        raise ValueError(
            f"harness_version must be 'v0', 'v1', or 'v2', got {harness_version!r}"
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
    sandbox=None,
    embedder_name: str = "hashing",
    query_mode: str = DEFAULT_QUERY_MODE,
    progress=None,
) -> dict[str, Any]:
    if harness == "v0":
        return run_agent(
            client=client,
            issue=issue,
            repo_path=repo_path,
            test_command=test_command,
            model=model,
            max_steps=max_steps,
            sandbox=sandbox,
            progress=progress,
        )
    if harness == "v1":
        return run_workflow(
            client=client,
            issue=issue,
            repo_path=repo_path,
            test_command=test_command,
            model=model,
            max_steps=max_steps,
            sandbox=sandbox,
            progress=progress,
        )
    return run_workflow(
        client=client,
        issue=issue,
        repo_path=repo_path,
        test_command=test_command,
        model=model,
        max_steps=max_steps,
        sandbox=sandbox,
        graph=get_v2_graph(),
        enable_search_code=True,
        embedder_name=embedder_name,
        query_mode=query_mode,
        progress=progress,
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
        "relevant_files": [],
        "retrieval_calls": 0,
        "retry_count": 0,
    }


def _failed_gold(message: str) -> dict[str, Any]:
    return {
        "command": "",
        "exit_code": 1,
        "passed": False,
        "output": message,
    }


def _sandbox_fields(
    meta: SandboxMetadata | None,
    *,
    start_error: str | None = None,
) -> dict[str, Any]:
    """Auditable sandbox telemetry. There is no public host backend."""
    if meta is None:
        return {
            "sandbox_backend": "docker",
            "sandbox_image": DEFAULT_IMAGE,
            "sandbox_network": "none",
            "sandbox_container_name": None,
            "sandbox_command_count": 0,
            "sandbox_timeout_count": 0,
            "sandbox_truncation_count": 0,
            "sandbox_denial_count": 0,
            "sandbox_exec_latency_ms": 0.0,
            "sandbox_started": False,
            "sandbox_cleaned_up": False,
            "sandbox_start_error": start_error,
        }
    return {
        "sandbox_backend": meta.backend,
        "sandbox_image": meta.image,
        "sandbox_network": meta.network_mode,
        "sandbox_container_name": meta.container_name,
        "sandbox_command_count": meta.command_count,
        "sandbox_timeout_count": meta.timeout_count,
        "sandbox_truncation_count": meta.truncation_count,
        "sandbox_denial_count": meta.denial_count,
        "sandbox_exec_latency_ms": round(meta.total_exec_latency_ms, 2),
        "sandbox_started": meta.started,
        "sandbox_cleaned_up": meta.cleaned_up,
    }


def solve_task(
    task_id: str,
    *,
    model: str | None = None,
    max_steps: int = MAX_AGENT_STEPS,
    client=None,
    harness_version: str = "v0",
    embedder_name: str = "hashing",
    query_mode: str = DEFAULT_QUERY_MODE,
    progress=None,
) -> dict[str, Any]:
    """Full harness cycle for one dataset task.

    Flow: reset repo → enter one sandbox → run V0/V1/V2 → gold test → cleanup.
    Gold scoring stays independent of workflow verification.
    Sandbox startup or execution failures still produce a versioned run artifact.
    """
    harness = _normalize_harness(harness_version)
    task = get_task(task_id)
    repo_path = resolve_repo_path(task)
    model_name = model or default_model()
    llm = client or create_client()
    retrieve_query_mode = normalize_query_mode(query_mode)

    reset_repo(repo_path, task["base_commit"])

    agent_result = _empty_agent_result()
    gold: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    gold_error_type: str | None = None
    gold_error_message: str | None = None
    sandbox_record = _sandbox_fields(None)

    try:
        with SandboxRunner(repo_path, task_id=task_id) as sandbox:
            try:
                agent_result = _run_harness(
                    harness=harness,
                    client=llm,
                    issue=task["issue"],
                    repo_path=str(repo_path),
                    test_command=task["test_command"],
                    model=model_name,
                    max_steps=max_steps,
                    sandbox=sandbox,
                    embedder_name=embedder_name,
                    query_mode=retrieve_query_mode,
                    progress=progress,
                )
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                agent_result = _empty_agent_result()

            try:
                if not sandbox.meta.usable:
                    gold_error_type = "SandboxUnusableError"
                    gold_error_message = (
                        "sandbox is not usable; skipped gold test after "
                        "timeout or forced cleanup"
                    )
                    gold = _failed_gold(
                        f"Error running gold test: {gold_error_type}: "
                        f"{gold_error_message}"
                    )
                else:
                    gold = run_gold_test(repo_path, task, sandbox=sandbox)
            except Exception as exc:
                gold_error_type = type(exc).__name__
                gold_error_message = str(exc)
                gold = _failed_gold(
                    f"Error running gold test: {gold_error_type}: {gold_error_message}"
                )
        # Snapshot after __exit__ so cleanup outcome is included.
        sandbox_record = _sandbox_fields(sandbox.meta)
    except Exception as exc:
        # Construction or start failed; __exit__ did not run.
        if error_type is None:
            error_type = type(exc).__name__
            error_message = str(exc)
        sandbox_record = _sandbox_fields(None, start_error=str(exc))
        if gold is None:
            gold = _failed_gold(
                f"Error running gold test: sandbox unavailable: {type(exc).__name__}: {exc}"
            )
            gold_error_type = type(exc).__name__
            gold_error_message = str(exc)

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
        "split": task.get("split"),
        "issue": task["issue"],
        "base_commit": task["base_commit"],
        "repo_path": str(repo_path),
        "expected_files": task.get("expected_files"),
        "model": model_name,
        "temperature": AGENT_TEMPERATURE,
        "harness_git_sha": git_sha(HARNESS_ROOT),
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
        **sandbox_record,
    }

    if error_type is not None:
        record["error_type"] = error_type
        record["error_message"] = error_message
    if gold_error_type is not None:
        record["gold_error_type"] = gold_error_type
        record["gold_error_message"] = gold_error_message

    if harness in {"v1", "v2"}:
        record.update(
            {
                "analysis": agent_result.get("analysis", ""),
                "plan": agent_result.get("plan", {}),
                "diagnosis": agent_result.get("diagnosis", ""),
                "verification": agent_result.get("test_result", {}),
                "status": agent_result.get("status", ""),
                "workflow_passed": agent_result.get("workflow_passed"),
                "stage_tokens": agent_result.get("stage_tokens", {}),
                "retry_count": agent_result.get("retry_count", 0),
            }
        )
    if harness == "v2":
        relevant = list(agent_result.get("relevant_files") or [])
        expected = list(task.get("expected_files") or [])
        record.update(
            {
                "retrieval_mode": "hybrid",
                "retrieval_calls": agent_result.get("retrieval_calls", 0),
                "relevant_files": relevant,
                "recall_at_5": recall_at_k(relevant, expected, k=5),
                "embedder_name": embedder_name,
                "query_mode": retrieve_query_mode,
                "retrieve_query": agent_result.get("retrieve_query"),
            }
        )

    record["run_path"] = str(save_run(record))
    return record
