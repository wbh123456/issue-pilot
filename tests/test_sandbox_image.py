"""Unit tests for sandbox image preflight/build (mocked Docker CLI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sandbox.image import (
    DEFAULT_IMAGE,
    DockerPreflightError,
    build_image,
    doctor,
    find_docker,
    require_docker_ready,
)


def _proc(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> Any:
    class Result:
        def __init__(self) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    return Result()


class FakeDocker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.info_os = "linux"
        self.image_present = False
        self.build_ok = True

    def __call__(self, args: list[str], *, timeout=None, check=False):  # noqa: ANN001
        self.calls.append(list(args))
        # args: [docker, ...]
        rest = args[1:]
        if rest[:1] == ["info"]:
            return _proc(0, stdout=json.dumps({"OSType": self.info_os}))
        if rest[:2] == ["image", "inspect"]:
            return _proc(0 if self.image_present else 1, stderr="No such image")
        if rest[:1] == ["build"]:
            return _proc(0 if self.build_ok else 1, stderr="build failed")
        return _proc(1, stderr=f"unexpected: {rest}")


class TestDoctor:
    def test_find_docker_checks_desktop_install_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sandbox.image.shutil.which", lambda _name: None)
        bin_dir = tmp_path / "Programs" / "DockerDesktop" / "resources" / "bin"
        bin_dir.mkdir(parents=True)
        exe = bin_dir / "docker.exe"
        exe.write_text("", encoding="utf-8")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert find_docker() == str(exe)

    def test_missing_docker_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sandbox.image.find_docker", lambda: None)
        report = doctor()
        assert not report.ok
        assert any("Docker CLI not found" in e for e in report.errors)

    def test_windows_containers_rejected(self) -> None:
        fake = FakeDocker()
        fake.info_os = "windows"
        report = doctor(docker="docker", runner=fake)
        assert not report.ok
        assert any("Linux-container" in e for e in report.errors)

    def test_linux_ready_with_image_warning(self) -> None:
        fake = FakeDocker()
        fake.image_present = False
        report = doctor(docker="docker", runner=fake)
        assert report.ok
        assert report.linux_containers
        assert report.warnings
        assert not report.image_present

    def test_require_image_fails_when_missing(self) -> None:
        fake = FakeDocker()
        with pytest.raises(DockerPreflightError, match="sandbox image"):
            require_docker_ready(docker="docker", runner=fake, require_image=True)


class TestBuild:
    def test_build_invokes_docker_build(self, tmp_path: Path) -> None:
        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        (sandbox_dir / "requirements.txt").write_text("pytest\n", encoding="utf-8")

        fake = FakeDocker()
        fake.image_present = False
        tag = build_image(
            image=DEFAULT_IMAGE,
            docker="docker",
            runner=fake,
            harness_root=tmp_path,
        )
        assert tag == DEFAULT_IMAGE
        build_calls = [c for c in fake.calls if c[1:2] == ["build"]]
        assert build_calls
        assert "-t" in build_calls[0]
        assert DEFAULT_IMAGE in build_calls[0]

    def test_build_failure_raises(self, tmp_path: Path) -> None:
        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        (sandbox_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        (sandbox_dir / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        fake = FakeDocker()
        fake.build_ok = False
        with pytest.raises(DockerPreflightError, match="docker build failed"):
            build_image(docker="docker", runner=fake, harness_root=tmp_path)
