"""Per-task Docker sandbox runner.

One detached, network-disabled container per task. Agent-visible commands are
policy-checked, then executed with ``docker exec`` (no shell). Cleanup always
runs on context exit. There is no host-execution fallback.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from harness.limits import COMMAND_TIMEOUT, MAX_TOOL_OUTPUT, truncate_output
from harness.permissions import (
    WORKSPACE_ROOT,
    CommandPermissionError,
    validate_command,
)
from sandbox.image import (
    DEFAULT_IMAGE,
    DockerPreflightError,
    DockerRunner,
    DockerTimeoutError,
    require_docker_ready,
    run_docker,
)

_CONTAINER_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


class SandboxError(RuntimeError):
    """Base error for sandbox lifecycle failures."""


class SandboxUnusableError(SandboxError):
    """Raised when the task container was killed or removed mid-run."""


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SandboxMetadata:
    image: str
    network_mode: str = "none"
    workspace_host: str = ""
    workspace_container: str = WORKSPACE_ROOT
    container_name: str | None = None
    container_id: str | None = None
    task_id: str | None = None
    command_count: int = 0
    timeout_count: int = 0
    truncation_count: int = 0
    denial_count: int = 0
    total_exec_latency_ms: float = 0.0
    cleaned_up: bool = False
    usable: bool = False
    started: bool = False
    backend: str = "docker"

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_container_name(task_id: str | None) -> str:
    base = _CONTAINER_NAME_RE.sub("-", (task_id or "task").strip()) or "task"
    base = base.strip(".-")[:40] or "task"
    return f"issue-pilot-{base}-{uuid.uuid4().hex[:8]}"


def _host_mount_path(host_path: Path) -> str:
    """Return a Docker Desktop-friendly absolute host path."""
    return host_path.resolve().as_posix()


class SandboxRunner:
    """Context-managed per-task Docker sandbox.

    Usage::

        with SandboxRunner(repo_path, task_id="issue-001") as sb:
            result = sb.run(["pytest", "tests/test_auth.py", "-q"])
    """

    def __init__(
        self,
        workspace_host: str | Path,
        *,
        task_id: str | None = None,
        image: str = DEFAULT_IMAGE,
        workspace_container: str = WORKSPACE_ROOT,
        command_timeout: float = COMMAND_TIMEOUT,
        max_output: int = MAX_TOOL_OUTPUT,
        docker: str | None = None,
        runner: DockerRunner | None = None,
        skip_preflight: bool = False,
    ) -> None:
        host = Path(workspace_host)
        if not host.exists() or not host.is_dir():
            raise SandboxError(f"workspace host path is not a directory: {host}")

        self._command_timeout = float(command_timeout)
        self._max_output = int(max_output)
        self._docker = docker
        self._runner = runner
        self._skip_preflight = skip_preflight
        self.meta = SandboxMetadata(
            image=image,
            workspace_host=str(host.resolve()),
            workspace_container=workspace_container,
            container_name=_safe_container_name(task_id),
            task_id=task_id,
        )

    def __enter__(self) -> SandboxRunner:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.cleanup()
        return None

    def _cli(self, args: Sequence[str], *, timeout: float):
        return run_docker(
            args,
            docker=self._docker,
            runner=self._runner,
            timeout=timeout,
        )

    def start(self) -> None:
        if self.meta.started and self.meta.usable:
            return
        if not self._skip_preflight:
            require_docker_ready(
                image=self.meta.image,
                docker=self._docker,
                runner=self._runner,
                require_image=True,
            )

        name = self.meta.container_name
        workspace = self.meta.workspace_container
        mount_source = _host_mount_path(Path(self.meta.workspace_host))
        proc = self._cli(
            [
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "--label",
                f"issue-pilot.task_id={self.meta.task_id or 'unknown'}",
                "--label",
                "issue-pilot.sandbox=1",
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--workdir",
                workspace,
                "--mount",
                f"type=bind,source={mount_source},target={workspace}",
                self.meta.image,
            ],
            timeout=120,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise SandboxError(
                f"failed to start sandbox container {name}: "
                f"{detail or f'exit {proc.returncode}'}"
            )

        cid = (proc.stdout or "").strip() or name
        self.meta.container_id = cid.splitlines()[-1].strip()
        self.meta.started = True
        self.meta.usable = True
        self.meta.cleaned_up = False

    def cleanup(self) -> None:
        """Force-remove the task container. Safe to call multiple times."""
        if self.meta.cleaned_up:
            return
        self._kill_and_remove()

    def _kill_and_remove(self) -> None:
        """Stop leftover processes, then delete the task container."""
        name_or_id = self.meta.container_id or self.meta.container_name
        if name_or_id:
            for args in (
                ["kill", str(name_or_id)],
                ["rm", "-f", str(name_or_id)],
            ):
                try:
                    self._cli(args, timeout=30)
                except DockerPreflightError:
                    pass
        self.meta.usable = False
        self.meta.cleaned_up = True
        self.meta.container_id = None

    def run(self, command: str | Sequence[str]) -> CommandResult:
        """Validate and execute ``command`` inside the task container."""
        if not self.meta.started or not self.meta.usable:
            raise SandboxUnusableError(
                "sandbox is not usable (not started, timed out, or already cleaned up)"
            )

        try:
            argv = validate_command(
                command, workspace=self.meta.workspace_container
            )
        except CommandPermissionError:
            self.meta.denial_count += 1
            raise

        started = time.perf_counter()
        timed_out = False
        exit_code = 1
        stdout = ""
        stderr = ""

        try:
            proc = self._cli(
                ["exec", str(self.meta.container_name), *argv],
                timeout=self._command_timeout,
            )
            exit_code = int(proc.returncode)
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
        except DockerTimeoutError:
            timed_out = True
            exit_code = 124
            stderr = f"Error: command timed out after {self._command_timeout:g}s"
            self.cleanup()
        except DockerPreflightError as exc:
            raise SandboxError(str(exc)) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        out_trunc = len(stdout) > self._max_output
        err_trunc = len(stderr) > self._max_output
        stdout = truncate_output(stdout, self._max_output)
        stderr = truncate_output(stderr, self._max_output)
        truncated = out_trunc or err_trunc

        self.meta.command_count += 1
        self.meta.total_exec_latency_ms += latency_ms
        if timed_out:
            self.meta.timeout_count += 1
        if truncated:
            self.meta.truncation_count += 1

        return CommandResult(
            command=list(argv),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            truncated=truncated,
            latency_ms=latency_ms,
        )
