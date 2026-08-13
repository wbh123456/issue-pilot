"""Sandbox image identity, Docker preflight, and explicit build workflow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

HARNESS_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = HARNESS_ROOT / "sandbox" / "Dockerfile"
REQUIREMENTS_PATH = HARNESS_ROOT / "sandbox" / "requirements.txt"

IMAGE_NAME = "issue-pilot-sandbox"
IMAGE_TAG = "py312"
DEFAULT_IMAGE = f"{IMAGE_NAME}:{IMAGE_TAG}"

DockerRunner = Callable[..., subprocess.CompletedProcess[str]]


class DockerPreflightError(RuntimeError):
    """Docker or the sandbox image is unavailable; do not fall back to host."""


class DockerTimeoutError(DockerPreflightError):
    """A trusted Docker CLI invocation exceeded its timeout."""


@dataclass
class DoctorReport:
    docker_path: str | None = None
    daemon_reachable: bool = False
    server_os: str | None = None
    linux_containers: bool = False
    image: str = DEFAULT_IMAGE
    image_present: bool = False
    dockerfile_present: bool = False
    requirements_present: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def find_docker() -> str | None:
    """Locate the Docker CLI. Checks PATH, then well-known Desktop install dirs."""
    found = shutil.which("docker")
    if found:
        return found

    local_app = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidates = [
        Path(program_files) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
        Path(local_app) / "Programs" / "DockerDesktop" / "resources" / "bin" / "docker.exe",
        Path(local_app) / "Docker" / "Docker" / "resources" / "bin" / "docker.exe",
        Path("/usr/bin/docker"),
        Path("/usr/local/bin/docker"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _default_runner(
    args: Sequence[str],
    *,
    timeout: float | None = 60,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


def run_docker(
    args: Sequence[str],
    *,
    docker: str | None = None,
    runner: DockerRunner | None = None,
    timeout: float | None = 60,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a trusted Docker CLI invocation (fixed argv from harness code)."""
    exe = docker or find_docker()
    if exe is None:
        raise DockerPreflightError(
            "Docker CLI not found on PATH. Install Docker Desktop and ensure "
            "it is running in Linux-container mode."
        )
    invoke = runner or _default_runner
    try:
        return invoke([exe, *args], timeout=timeout, check=check)
    except FileNotFoundError as exc:
        raise DockerPreflightError(
            "Docker CLI not found on PATH. Install Docker Desktop and ensure "
            "it is running in Linux-container mode."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerTimeoutError(
            f"docker {' '.join(args)} timed out after {timeout}s"
        ) from exc


def image_exists(
    image: str = DEFAULT_IMAGE,
    *,
    docker: str | None = None,
    runner: DockerRunner | None = None,
) -> bool:
    proc = run_docker(
        ["image", "inspect", image],
        docker=docker,
        runner=runner,
        timeout=30,
    )
    return proc.returncode == 0


def doctor(
    *,
    image: str = DEFAULT_IMAGE,
    docker: str | None = None,
    runner: DockerRunner | None = None,
    require_image: bool = False,
) -> DoctorReport:
    """Inspect Docker availability. Never silently falls back to host execution."""
    report = DoctorReport(image=image)
    report.dockerfile_present = DOCKERFILE_PATH.is_file()
    report.requirements_present = REQUIREMENTS_PATH.is_file()

    if not report.dockerfile_present:
        report.errors.append(f"missing Dockerfile: {DOCKERFILE_PATH}")
    if not report.requirements_present:
        report.errors.append(f"missing requirements: {REQUIREMENTS_PATH}")

    exe = docker or find_docker()
    report.docker_path = exe
    if exe is None:
        report.errors.append(
            "Docker CLI not found. Install Docker Desktop (Linux containers) "
            "and reopen the terminal so `docker` is on PATH."
        )
        return report

    try:
        info = run_docker(
            ["info", "--format", "{{json .}}"],
            docker=exe,
            runner=runner,
            timeout=30,
        )
    except DockerPreflightError as exc:
        report.errors.append(str(exc))
        return report

    if info.returncode != 0:
        err = (info.stderr or info.stdout or "").strip()
        report.errors.append(
            "Docker daemon is not reachable. Start Docker Desktop and wait "
            f"until it is ready. Detail: {err or f'exit {info.returncode}'}"
        )
        return report

    report.daemon_reachable = True
    server_os = _parse_server_os(info.stdout)
    report.server_os = server_os
    report.linux_containers = (server_os or "").lower() == "linux"
    if not report.linux_containers:
        report.errors.append(
            "Docker is not in Linux-container mode "
            f"(server OSType={server_os!r}). Switch Docker Desktop to Linux "
            "containers; Windows containers are not supported."
        )

    try:
        report.image_present = image_exists(image, docker=exe, runner=runner)
    except DockerPreflightError as exc:
        report.errors.append(str(exc))
        return report

    if require_image and not report.image_present:
        report.errors.append(
            f"sandbox image {image!r} is missing. Run: python cli.py sandbox build"
        )
    elif not report.image_present:
        report.warnings.append(
            f"sandbox image {image!r} is not built yet. Run: python cli.py sandbox build"
        )

    return report


def require_docker_ready(
    *,
    image: str = DEFAULT_IMAGE,
    docker: str | None = None,
    runner: DockerRunner | None = None,
    require_image: bool = True,
) -> DoctorReport:
    """Raise ``DockerPreflightError`` unless Docker (and optionally the image) is ready."""
    report = doctor(
        image=image,
        docker=docker,
        runner=runner,
        require_image=require_image,
    )
    if not report.ok:
        raise DockerPreflightError("; ".join(report.errors))
    return report


def build_image(
    *,
    image: str = DEFAULT_IMAGE,
    docker: str | None = None,
    runner: DockerRunner | None = None,
    harness_root: Path | None = None,
) -> str:
    """Build the sandbox image from the harness repo. Returns the image tag."""
    root = harness_root or HARNESS_ROOT
    dockerfile = root / "sandbox" / "Dockerfile"
    requirements = root / "sandbox" / "requirements.txt"
    if not dockerfile.is_file():
        raise DockerPreflightError(f"missing Dockerfile: {dockerfile}")
    if not requirements.is_file():
        raise DockerPreflightError(f"missing requirements: {requirements}")

    # Preflight daemon/OS before spending time on a build.
    require_docker_ready(
        image=image,
        docker=docker,
        runner=runner,
        require_image=False,
    )

    proc = run_docker(
        [
            "build",
            "-f",
            str(dockerfile),
            "-t",
            image,
            str(root),
        ],
        docker=docker,
        runner=runner,
        timeout=600,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise DockerPreflightError(
            f"docker build failed for {image}: {detail or f'exit {proc.returncode}'}"
        )
    return image


def _parse_server_os(info_stdout: str) -> str | None:
    text = (info_stdout or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text.lower() or None
    if isinstance(payload, dict):
        ostype = payload.get("OSType") or payload.get("osType")
        if isinstance(ostype, str) and ostype.strip():
            return ostype.strip().lower()
    return None
