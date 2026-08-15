# IssuePilot Phase 3 sandbox image.
# Build from the harness repo root:
#   docker build -f sandbox/Dockerfile -t issue-pilot-sandbox:py312 .
#
# Runtime policy (enforced by SandboxRunner, not this file):
#   --network none, read-only root, /tmp tmpfs, dropped caps,
#   no-new-privileges, single RW mount at /workspace.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user; bind-mounted /workspace is chown'd at container start if needed.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin sandbox

COPY sandbox/requirements.txt /tmp/sandbox-requirements.txt
RUN pip install --no-cache-dir -r /tmp/sandbox-requirements.txt \
    && rm /tmp/sandbox-requirements.txt

WORKDIR /workspace
USER sandbox
# Bind mounts from Windows/macOS often look "dubious" to git; allow any path.
# GIT_OPTIONAL_LOCKS avoids index.lock writes on the read-only rootfs.
ENV GIT_OPTIONAL_LOCKS=0
RUN git config --global --add safe.directory '*' \
    && git config --global --add safe.directory /workspace

# Detached task containers stay alive for repeated `docker exec`.
CMD ["sleep", "infinity"]
