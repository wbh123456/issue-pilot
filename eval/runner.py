"""Eval runner: reset benchmark → sandbox → run harness → gold test → save."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from agent.client import create_client, default_model
from agent.graph import build_graph, get_v2_graph, resume_workflow, run_workflow
from agent.loop import run_agent
from agent.nodes.approve import review_payload
from agent.state import reached_checkpoint_stages
from agent.tools.shell import run_tests
from eval.metrics import recall_at_k
from eval.repository import (
    GOLD_STAGING_DIRNAME,
    capture_patch_diff,
    git_sha,
    reset_repo,
    verify_resume_worktree,
)
from eval.session import (
    RunSession,
    load_session,
    new_run_id,
    save_session,
    session_path,
    update_session,
    utc_now_iso,
)
from harness.checkpoint import open_checkpointer
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


def benchmark_spec_sha(
    *,
    dataset_path: Path = DATASET_PATH,
    gold_dir: Path = GOLD_DIR,
) -> str:
    """Fingerprint of issue text + hidden gold. Changing either splits cohorts.

    Hashes the on-disk ``dataset.json`` bytes, then each ``eval/gold/*.py``
    file in name order (name + contents). Not a hand-maintained version number.
    """
    hasher = hashlib.sha256()
    hasher.update(dataset_path.read_bytes())
    gold_root = Path(gold_dir)
    if gold_root.is_dir():
        for path in sorted(gold_root.glob("*.py"), key=lambda item: item.name):
            hasher.update(b"\n")
            hasher.update(path.name.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


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
    run_id = record.get("run_id")
    if run_id:
        out = RUNS_DIR / f"{run_id}.json"
    else:
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
    lint_command: str = "ruff check app",
    model: str,
    max_steps: int,
    sandbox=None,
    embedder_name: str = "hashing",
    query_mode: str = DEFAULT_QUERY_MODE,
    progress=None,
    feedback_provider=None,
    graph=None,
    thread_id: str | None = None,
    require_approval: bool = False,
) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if graph is not None:
        extra["graph"] = graph
    if thread_id:
        extra["thread_id"] = thread_id
    if require_approval:
        extra["require_approval"] = require_approval
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
            lint_command=lint_command,
            model=model,
            max_steps=max_steps,
            sandbox=sandbox,
            progress=progress,
            feedback_provider=feedback_provider,
            **extra,
        )
    extra.setdefault("graph", get_v2_graph())
    extra["enable_search_code"] = True
    return run_workflow(
        client=client,
        issue=issue,
        repo_path=repo_path,
        test_command=test_command,
        lint_command=lint_command,
        model=model,
        max_steps=max_steps,
        sandbox=sandbox,
        embedder_name=embedder_name,
        query_mode=query_mode,
        progress=progress,
        feedback_provider=feedback_provider,
        **extra,
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
        "human_retry_count": 0,
        "human_feedback": "",
        "attempt_history": [],
        "structured_diagnosis": {},
        "patch_evaluation": {},
        "workflow_trace": [],
        "checkpoint_stages": [],
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


def _waiting_approval(result: dict[str, Any]) -> bool:
    return (
        result.get("termination") == "waiting_approval"
        or result.get("status") == "waiting_approval"
    )


def _score_gold_or_error(
    repo_path: Path,
    task: dict[str, Any],
    sandbox,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Run gold inside an open sandbox, or synthesize a failed gold payload."""
    try:
        if not sandbox.meta.usable:
            gold_error_type = "SandboxUnusableError"
            gold_error_message = (
                "sandbox is not usable; skipped gold test after "
                "timeout or forced cleanup"
            )
            gold = _failed_gold(
                f"Error running gold test: {gold_error_type}: {gold_error_message}"
            )
            return gold, gold_error_type, gold_error_message
        return run_gold_test(repo_path, task, sandbox=sandbox), None, None
    except Exception as exc:
        gold_error_type = type(exc).__name__
        gold_error_message = str(exc)
        gold = _failed_gold(
            f"Error running gold test: {gold_error_type}: {gold_error_message}"
        )
        return gold, gold_error_type, gold_error_message


def _build_run_record(
    *,
    task_id: str,
    task: dict[str, Any],
    harness: HarnessVersion,
    repo_path: Path,
    model_name: str,
    retrieve_query_mode: str,
    embedder_name: str,
    agent_result: dict[str, Any],
    gold: dict[str, Any] | None,
    error_type: str | None,
    error_message: str | None,
    gold_error_type: str | None,
    gold_error_message: str | None,
    sandbox_record: dict[str, Any],
    run_id: str | None = None,
    thread_id: str | None = None,
    resumed: bool = False,
    resume_count: int = 0,
    sandbox_sessions: int = 1,
) -> dict[str, Any]:
    """Assemble the versioned run JSON. Gold ``success`` is independent of HITL."""
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
        "benchmark_spec_sha": benchmark_spec_sha(),
        "patch_diff": capture_patch_diff(repo_path),
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
    if run_id:
        record["run_id"] = run_id
        record["thread_id"] = thread_id or run_id

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
                "human_retry_count": agent_result.get("human_retry_count", 0),
                "human_feedback": agent_result.get("human_feedback") or "",
                "attempt_history": list(agent_result.get("attempt_history") or []),
                "structured_diagnosis": agent_result.get("structured_diagnosis") or {},
                "patch_evaluation": agent_result.get("patch_evaluation") or {},
                "recovery_success": (
                    int(agent_result.get("retry_count") or 0) > 0
                    and agent_result.get("workflow_passed") is True
                ),
                "approval_decision": agent_result.get("approval_decision") or "",
                "resumed": resumed,
                "resume_count": resume_count,
                # Counters below are from the last container only; each resume
                # starts a fresh sandbox, so they are not summed across sessions.
                "sandbox_sessions": sandbox_sessions,
                "retrieval_calls": int(agent_result.get("retrieval_calls") or 0),
                "workflow_trace": list(agent_result.get("workflow_trace") or []),
                "checkpoint_stages": list(
                    agent_result.get("checkpoint_stages")
                    or reached_checkpoint_stages(
                        agent_result.get("workflow_trace") or []
                    )
                ),
            }
        )
    if harness == "v2":
        relevant = list(agent_result.get("relevant_files") or [])
        expected = list(task.get("expected_files") or [])
        record.update(
            {
                "retrieval_mode": "hybrid",
                "relevant_files": relevant,
                "recall_at_5": recall_at_k(relevant, expected, k=5),
                "embedder_name": embedder_name,
                "query_mode": retrieve_query_mode,
                "retrieve_query": agent_result.get("retrieve_query"),
            }
        )
    return record


def _paused_record(
    *,
    task_id: str,
    harness: HarnessVersion,
    session: RunSession,
    session_path: Path,
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "harness_version": harness,
        "run_id": session.run_id,
        "thread_id": session.thread_id,
        "paused": True,
        "status": "waiting_approval",
        "termination": "waiting_approval",
        "workflow_passed": False,
        "session_path": str(session_path),
        "approval_payload": agent_result.get("approval_payload") or {},
        "approval_decision": "",
        "resume_count": session.resume_count,
        "model": session.model,
        "analysis": agent_result.get("analysis", ""),
        "plan": agent_result.get("plan", {}),
        "verification": agent_result.get("test_result", {}),
        "patch_evaluation": agent_result.get("patch_evaluation") or {},
    }


def _write_paused_session(
    *,
    run_id: str,
    thread_id: str,
    task_id: str,
    harness: HarnessVersion,
    model_name: str,
    embedder_name: str,
    query_mode: str,
    repo_path: Path,
    base_commit: str,
    resume_count: int = 0,
    sessions_dir: Path | None = None,
    existing: RunSession | None = None,
) -> tuple[RunSession, Path]:
    if existing is not None:
        session = update_session(
            existing,
            sessions_dir=sessions_dir,
            status="paused",
            resume_count=resume_count,
        )
    else:
        session = RunSession(
            run_id=run_id,
            thread_id=thread_id,
            task_id=task_id,
            harness=harness,
            model=model_name,
            embedder_name=embedder_name,
            query_mode=query_mode,
            repo_path=str(repo_path),
            base_commit=base_commit,
            status="paused",
            created_at=utc_now_iso(),
            resume_count=resume_count,
        )
        save_session(session, sessions_dir=sessions_dir)
    return session, session_path(run_id, sessions_dir=sessions_dir)


def _invoke_checkpointed_harness(
    *,
    checkpoint_path: str | Path | None,
    require_approval: bool,
    thread_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    harness = kwargs["harness"]
    with open_checkpointer(checkpoint_path) as saver:
        graph = build_graph(
            include_retrieve=harness == "v2",
            checkpointer=saver,
        )
        return _run_harness(
            graph=graph,
            thread_id=thread_id,
            require_approval=require_approval,
            **kwargs,
        )


def _run_sandbox_cycle(
    *,
    repo_path: Path,
    task_id: str,
    task: dict[str, Any],
    invoke,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, str | None, str | None, str | None, dict[str, Any]]:
    """Enter one sandbox, invoke the harness, score gold unless the graph paused.

    Returns ``(agent_result, gold, error_type, error_message, gold_error_type,
    gold_error_message, sandbox_record)``.
    """
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
                agent_result = invoke(sandbox)
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                agent_result = _empty_agent_result()

            if error_type is None and not _waiting_approval(agent_result):
                gold, gold_error_type, gold_error_message = _score_gold_or_error(
                    repo_path, task, sandbox
                )
        sandbox_record = _sandbox_fields(sandbox.meta)
    except Exception as exc:
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

    return (
        agent_result,
        gold,
        error_type,
        error_message,
        gold_error_type,
        gold_error_message,
        sandbox_record,
    )


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
    interactive_recovery: bool = False,
    feedback_provider=None,
    require_approval: bool = False,
    checkpoint: bool = False,
    checkpoint_path: str | Path | None = None,
    sessions_dir: Path | None = None,
) -> dict[str, Any]:
    """Full harness cycle for one dataset task.

    Flow: reset repo → enter one sandbox → run V0/V1/V2 → gold test → cleanup.
    Gold scoring stays independent of workflow verification.
    Sandbox startup or execution failures still produce a versioned run artifact.

    ``require_approval`` implies checkpointing. On interrupt the function writes
    a paused session under ``runs/sessions/`` and returns without a run JSON.
    """
    harness = _normalize_harness(harness_version)
    if require_approval:
        checkpoint = True
    if checkpoint and harness == "v0":
        raise ValueError("checkpoint and require_approval require harness v1 or v2")

    task = get_task(task_id)
    repo_path = resolve_repo_path(task)
    model_name = model or default_model()
    llm = client or create_client()
    retrieve_query_mode = normalize_query_mode(query_mode)
    provider = feedback_provider
    if provider is None and interactive_recovery:
        from agent.nodes.feedback import stdin_feedback_provider

        provider = stdin_feedback_provider

    reset_repo(repo_path, task["base_commit"])

    run_id = new_run_id(task_id, harness) if checkpoint else None
    thread_id = run_id

    def invoke(sandbox):
        kwargs = {
            "harness": harness,
            "client": llm,
            "issue": task["issue"],
            "repo_path": str(repo_path),
            "test_command": task["test_command"],
            "lint_command": str(task.get("lint_command") or ""),
            "model": model_name,
            "max_steps": max_steps,
            "sandbox": sandbox,
            "embedder_name": embedder_name,
            "query_mode": retrieve_query_mode,
            "progress": progress,
            "feedback_provider": provider,
        }
        if checkpoint:
            assert run_id is not None
            return _invoke_checkpointed_harness(
                checkpoint_path=checkpoint_path,
                require_approval=require_approval,
                thread_id=run_id,
                **kwargs,
            )
        return _run_harness(**kwargs)

    (
        agent_result,
        gold,
        error_type,
        error_message,
        gold_error_type,
        gold_error_message,
        sandbox_record,
    ) = _run_sandbox_cycle(
        repo_path=repo_path,
        task_id=task_id,
        task=task,
        invoke=invoke,
    )

    if error_type is None and _waiting_approval(agent_result) and run_id:
        session, path = _write_paused_session(
            run_id=run_id,
            thread_id=run_id,
            task_id=task_id,
            harness=harness,
            model_name=model_name,
            embedder_name=embedder_name,
            query_mode=retrieve_query_mode,
            repo_path=repo_path,
            base_commit=str(task["base_commit"]),
            resume_count=0,
            sessions_dir=sessions_dir,
        )
        return _paused_record(
            task_id=task_id,
            harness=harness,
            session=session,
            session_path=path,
            agent_result=agent_result,
        )

    record = _build_run_record(
        task_id=task_id,
        task=task,
        harness=harness,
        repo_path=repo_path,
        model_name=model_name,
        retrieve_query_mode=retrieve_query_mode,
        embedder_name=embedder_name,
        agent_result=agent_result,
        gold=gold,
        error_type=error_type,
        error_message=error_message,
        gold_error_type=gold_error_type,
        gold_error_message=gold_error_message,
        sandbox_record=sandbox_record,
        run_id=run_id,
        thread_id=thread_id,
        resumed=False,
        resume_count=0,
        sandbox_sessions=1,
    )
    record["run_path"] = str(save_run(record))
    return record


def resume_task(
    run_id: str,
    *,
    decision: str,
    feedback: str | None = None,
    client=None,
    max_steps: int = MAX_AGENT_STEPS,
    progress=None,
    feedback_provider=None,
    checkpoint_path: str | Path | None = None,
    sessions_dir: Path | None = None,
) -> dict[str, Any]:
    """Finish a paused approval run. Never calls ``reset_repo``.

    Loads the session, checks that HEAD is still ``base_commit`` with the
    agent patch present, opens a fresh sandbox and a fresh SqliteSaver, then
    resumes with ``Command(resume=...)``. Gold is scored only after the graph
    finishes (including reject / needs_human). Sandbox counters are from this
    container only; ``sandbox_sessions`` counts how many containers the run used.
    """
    session = load_session(run_id, sessions_dir=sessions_dir)
    if session.status != "paused":
        raise ValueError(f"session {run_id} is {session.status!r}, not paused")
    harness = _normalize_harness(session.harness)
    if harness == "v0":
        raise ValueError("resume_task does not support harness v0")

    task = get_task(session.task_id)
    repo_path = Path(session.repo_path)
    if not repo_path.is_dir():
        raise FileNotFoundError(f"benchmark repo not found: {repo_path}")
    verify_resume_worktree(repo_path, session.base_commit)

    llm = client or create_client()
    resume_count = session.resume_count + 1
    sandbox_sessions = 1 + resume_count

    def invoke(sandbox):
        with open_checkpointer(checkpoint_path) as saver:
            graph = build_graph(
                include_retrieve=harness == "v2",
                checkpointer=saver,
            )
            return resume_workflow(
                graph=graph,
                thread_id=session.thread_id,
                client=llm,
                repo_path=str(repo_path),
                test_command=task["test_command"],
                decision=decision,
                feedback=feedback or "",
                lint_command=str(task.get("lint_command") or ""),
                model=session.model,
                max_steps=max_steps,
                sandbox=sandbox,
                enable_search_code=harness == "v2",
                embedder_name=session.embedder_name,
                query_mode=session.query_mode,
                progress=progress,
                feedback_provider=feedback_provider,
                require_approval=True,
            )

    (
        agent_result,
        gold,
        error_type,
        error_message,
        gold_error_type,
        gold_error_message,
        sandbox_record,
    ) = _run_sandbox_cycle(
        repo_path=repo_path,
        task_id=session.task_id,
        task=task,
        invoke=invoke,
    )

    if error_type is None and _waiting_approval(agent_result):
        session, path = _write_paused_session(
            run_id=session.run_id,
            thread_id=session.thread_id,
            task_id=session.task_id,
            harness=harness,
            model_name=session.model,
            embedder_name=session.embedder_name,
            query_mode=session.query_mode,
            repo_path=repo_path,
            base_commit=session.base_commit,
            resume_count=resume_count,
            sessions_dir=sessions_dir,
            existing=session,
        )
        return _paused_record(
            task_id=session.task_id,
            harness=harness,
            session=session,
            session_path=path,
            agent_result=agent_result,
        )

    update_session(
        session,
        sessions_dir=sessions_dir,
        status="completed",
        resume_count=resume_count,
    )
    record = _build_run_record(
        task_id=session.task_id,
        task=task,
        harness=harness,
        repo_path=repo_path,
        model_name=session.model,
        retrieve_query_mode=session.query_mode,
        embedder_name=session.embedder_name,
        agent_result=agent_result,
        gold=gold,
        error_type=error_type,
        error_message=error_message,
        gold_error_type=gold_error_type,
        gold_error_message=gold_error_message,
        sandbox_record=sandbox_record,
        run_id=session.run_id,
        thread_id=session.thread_id,
        resumed=True,
        resume_count=resume_count,
        sandbox_sessions=sandbox_sessions,
    )
    record["run_path"] = str(save_run(record))
    return record


def load_review_payload(
    run_id: str,
    *,
    checkpoint_path: str | Path | None = None,
    sessions_dir: Path | None = None,
) -> dict[str, Any]:
    """Rebuild the six-part review bundle from the durable checkpoint.

    Session sidecars do not store the interrupt payload, so ``review`` after a
    process kill reads ``AgentState`` from a fresh SqliteSaver.
    """
    session = load_session(run_id, sessions_dir=sessions_dir)
    harness = _normalize_harness(session.harness)
    with open_checkpointer(checkpoint_path) as saver:
        graph = build_graph(
            include_retrieve=harness == "v2",
            checkpointer=saver,
        )
        snapshot = graph.get_state(
            {"configurable": {"thread_id": session.thread_id}}
        )
    values = dict(getattr(snapshot, "values", None) or {})
    if not values:
        raise ValueError(f"no checkpoint state for {run_id}")
    return review_payload(values)
