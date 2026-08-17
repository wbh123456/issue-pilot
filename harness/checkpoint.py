"""Durable LangGraph checkpointer (SqliteSaver on a local sqlite file)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

HARNESS_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = HARNESS_ROOT / "runs" / "checkpoints.sqlite"


def checkpoint_path(path: str | Path | None = None) -> Path:
    """Return the sqlite file used for durable graph checkpoints."""
    if path is None:
        return DEFAULT_CHECKPOINT_PATH
    return Path(path)


@contextmanager
def open_checkpointer(path: str | Path | None = None) -> Iterator[SqliteSaver]:
    """Yield a ``SqliteSaver`` bound to ``path`` (default ``runs/checkpoints.sqlite``).

    The connection stays open for the duration of the ``with`` block. Resume
    across processes is: close this saver, open a fresh one on the same file.
    """
    db_path = checkpoint_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        saver.setup()
        yield saver
