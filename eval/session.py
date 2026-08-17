"""Sidecar registry for durable solve sessions (pause / resume).

Files live in ``runs/sessions/{run_id}.json`` so they are not picked up by
``eval.report.load_run_files``, which globs only ``runs/*.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HARNESS_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = HARNESS_ROOT / "runs" / "sessions"

_SESSION_FIELDS = (
    "run_id",
    "thread_id",
    "task_id",
    "harness",
    "model",
    "embedder_name",
    "query_mode",
    "repo_path",
    "base_commit",
    "status",
    "created_at",
)


@dataclass(frozen=True)
class RunSession:
    """Runtime arguments LangGraph does not persist in the checkpointer."""

    run_id: str
    thread_id: str
    task_id: str
    harness: str
    model: str
    embedder_name: str
    query_mode: str
    repo_path: str
    base_commit: str
    status: str
    created_at: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(task_id: str, harness: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{task_id}-{harness}-{stamp}"


def _validate_run_id(run_id: str) -> str:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError(f"invalid run_id: {run_id!r}")
    return run_id


def session_path(run_id: str, *, sessions_dir: Path | None = None) -> Path:
    root = Path(sessions_dir or SESSIONS_DIR)
    return root / f"{_validate_run_id(run_id)}.json"


def session_from_dict(data: dict[str, Any]) -> RunSession:
    if not isinstance(data, dict):
        raise ValueError("session payload must be a JSON object")
    missing = [key for key in _SESSION_FIELDS if key not in data]
    if missing:
        raise ValueError("session missing fields: " + ", ".join(missing))
    values = {}
    for key in _SESSION_FIELDS:
        value = data[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"session field {key!r} must be a non-empty string")
        values[key] = value
    return RunSession(**values)


def save_session(session: RunSession, *, sessions_dir: Path | None = None) -> Path:
    path = session_path(session.run_id, sessions_dir=sessions_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(session), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_session(run_id: str, *, sessions_dir: Path | None = None) -> RunSession:
    path = session_path(run_id, sessions_dir=sessions_dir)
    if not path.is_file():
        raise FileNotFoundError(f"session not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt session file: {path}") from exc
    loaded = session_from_dict(data)
    if loaded.run_id != run_id:
        raise ValueError(
            f"session run_id {loaded.run_id!r} does not match path {run_id!r}"
        )
    return loaded


def list_sessions(*, sessions_dir: Path | None = None) -> list[RunSession]:
    root = Path(sessions_dir or SESSIONS_DIR)
    if not root.is_dir():
        return []
    sessions: list[RunSession] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append(session_from_dict(data))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return sessions
