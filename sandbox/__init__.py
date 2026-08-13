"""Docker sandbox image and per-task runner.

Trusted orchestration only — never invoked with model-controlled arguments.
"""

from __future__ import annotations

from .image import (
    DEFAULT_IMAGE,
    DockerPreflightError,
    DockerTimeoutError,
    build_image,
    doctor,
    image_exists,
    require_docker_ready,
)
from .runner import (
    CommandResult,
    SandboxError,
    SandboxMetadata,
    SandboxRunner,
    SandboxUnusableError,
)

__all__ = [
    "CommandResult",
    "DEFAULT_IMAGE",
    "DockerPreflightError",
    "DockerTimeoutError",
    "SandboxError",
    "SandboxMetadata",
    "SandboxRunner",
    "SandboxUnusableError",
    "build_image",
    "doctor",
    "image_exists",
    "require_docker_ready",
]
