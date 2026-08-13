"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from sandbox.image import doctor


@pytest.fixture(scope="session")
def docker_preflight():
    """Skip the calling test if Docker Desktop / sandbox image is unavailable."""
    report = doctor(require_image=True)
    if not report.ok:
        pytest.skip("; ".join(report.errors))
    return report
