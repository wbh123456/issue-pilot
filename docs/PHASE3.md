# Phase 3 — Day 3 Completion Record

Date: 2026-08-13  
Harness: `issue-pilot`  
Benchmark (sibling repo): `issue-pilot-benchmark`  
LLM: DeepSeek (`deepseek-v4-flash`) via OpenAI-compatible SDK  
Harness versions: **V0** (ReAct) and **V1** (LangGraph Plan-Execute) — both share one Docker execution backend  
Sandbox: `issue-pilot-sandbox:py312` (`--network none`, benchmark mounted at `/workspace`)

## Goal

Harden the completed V0/V1 harness so every **agent-visible command** follows:

```text
Agent → Policy → Docker → Result
```

Mechanical guardrails, not prompt instructions: command allowlist, path jail, per-task isolated container, timeout/truncation, deterministic cleanup. `harness_version` still describes **control flow** (v0 vs v1). Sandbox is metadata, not a new V2 (V2 remains reserved for RAG).

## Current agentic workflow architecture

Phase 3 does not change *how* the agent thinks. It changes *where commands run*. Control flow (V0 vs V1) sits on top of a shared tool + sandbox layer.

```text
CLI  solve / compare
        │
        ▼
eval/runner.solve_task
        │
        ├─ 1. eval/repository.reset_repo     [trusted host git]
        │       git reset --hard <base_commit>
        │       git clean -fd
        │
        └─ 2. with SandboxRunner(benchmark → /workspace)
                │   one detached Linux container per task
                │   --network none, --read-only, cap-drop, no-new-privileges
                │
                ├─ 3a. V0  agent/loop.run_agent          ReAct tool loop
                │  or
                ├─ 3b. V1  agent/graph.run_workflow      LangGraph
                │         analyze → plan → execute → verify
                │            │                    │
                │            │                    ├─ PASS → END
                │            │                    └─ FAIL → diagnose → END
                │            └─ execute reuses V0 run_agent + plan context
                │
                └─ 4. run_gold_test (same container; agent never sees selector)
              finally: docker rm -f
        │
        ▼
runs/{task}-{v0|v1}-{stamp}.json   (+ sandbox_* telemetry)
```

### V0 vs V1 (control flow)

```text
V0 ReAct (agent/loop.py)
  for step in 1..MAX_AGENT_STEPS:
      LLM(+tools) → execute_tool* → append result
      stop on final answer or max_steps

V1 LangGraph (agent/graph.py)
  START → analyze (1 LLM, no tools)
        → plan    (1 LLM, JSON {problem, hypothesis, files_to_inspect, steps})
        → execute (V0 ReAct loop, plan injected as workflow_context)
        → verify  (pytest exit code + git_diff; never LLM judgment)
             ├─ PASS → mark_success → END
             └─ FAIL → diagnose (1 LLM, no re-execute) → END
```

`SandboxRunner` is **runtime config**: passed into `run_agent` / LangGraph `configurable`. It is **not** part of serializable `AgentState` (checkpoint-ready for a later phase).

### Tool dispatch (shared by V0 and V1)

Six tools. Two isolation mechanisms:

| Tool | Mechanism | Where it runs |
|---|---|---|
| `list_files` / `read_file` / `edit_file` | Host path jail (`agent/tools/_sandbox.py`) | Host Python I/O, only under benchmark root |
| `grep_code` | Same path jail, **pure Python walk** | Host; no `rg` subprocess |
| `run_tests` | Allowlist then `docker exec` argv | Container `/workspace` |
| `git_diff` | Fixed `git diff HEAD` + `git status --short` | Container `/workspace` |

```text
execute_tool
    ├─ file / grep  → resolve_in_repo()  → host FS (jailed)
    └─ run_tests / git_diff
            → validate_command()          pytest|ruff|mypy|git status|diff
            → SandboxRunner.run()         docker exec (no shell)
            → bounded stdout/stderr
```

No silent host fallback: missing sandbox returns an error string / raises. The LLM never receives the harness repo, `.env`, or Docker socket.

### Trusted vs untrusted

| Untrusted (model-triggered) | Trusted host orchestration (fixed argv) |
|---|---|
| `run_tests`, `git_diff`, `grep_code`, file edits | `git reset --hard` / `git clean -fd` (`eval/repository.py`) |
| V1 verify + independent gold pytest | Docker CLI: build, `run`, `exec`, `rm` (`sandbox/image.py`) |

Same binary names (`git`, `pytest`) are **not** the same trust class. Evaluator reset is not an agent tool.

## What we built

### Limits and policy

| Area | Location | Notes |
|---|---|---|
| Limits | `harness/limits.py` | `MAX_AGENT_STEPS=15`, `MAX_RETRY=2` (config only), `MAX_TOOL_OUTPUT=10_000`, `COMMAND_TIMEOUT=60` |
| Allowlist | `harness/permissions.py` | argv parse; allow `pytest`/`ruff`/`mypy`/`git status\|diff`; reject shells, `curl`/`rm`/`python`, metacharacters, `/etc`, `~/.ssh`, workspace escape |
| Path jail | `agent/tools/_sandbox.py` | `resolve()` + `relative_to(root)`; symlink escape rejected |

### Docker image and runner

| Area | Location | Notes |
|---|---|---|
| Image | `sandbox/Dockerfile` | Python 3.12-slim, non-root `sandbox`, git + pytest/ruff/mypy + FastAPI deps |
| Manifest | `sandbox/requirements.txt`, `.dockerignore` | Build context excludes `.env`, `.git`, `runs/`, harness source |
| Preflight | `sandbox/image.py` | `doctor` / `build_image`; Linux containers required; no host fallback |
| Runner | `sandbox/runner.py` | Context manager; one container per task; `docker exec` argv; timeout → rm → unusable |

Container policy (enforced at `docker run`): `--network none`, `--read-only`, `/tmp` tmpfs, `--cap-drop ALL`, `no-new-privileges`, single bind: benchmark → `/workspace`.

### Eval / CLI

| Area | Location | Notes |
|---|---|---|
| Reset helper | `eval/repository.py` | Trusted host git only |
| Runner | `eval/runner.py` | reset → sandbox → V0/V1 → gold (same container) → cleanup; persist on sandbox/harness failure |
| CLI | `cli.py` | `solve` / `compare` plus `sandbox doctor` / `sandbox build`; summaries show sandbox fields |

Run JSON additions (no public `host` backend): `sandbox_backend=docker`, image, network, command/timeout/truncation/denial counts, exec latency, `sandbox_started`, `sandbox_cleaned_up`. `compare` starts a **fresh container per side**.

## Test evidence

Default unit suite does **not** require Docker.

| Suite | Role |
|---|---|
| `tests/test_permissions.py` | Allowlist / denials / shell / path / git subcommands |
| `tests/test_sandbox.py` | Fake Docker CLI: lifecycle, argv, timeout, truncation, cleanup |
| `tests/test_sandbox_image.py` | Mocked doctor/build |
| `tests/test_tools.py` | Sandbox injection; no host `rg`; no host fallback |
| `tests/test_graph.py` | V0/V1 contracts; sandbox in runtime config, not state |
| `tests/test_runner.py` | One container per solve; gold shares it; start failure still saved |
| `tests/test_subprocess_audit.py` | AST: only `sandbox/image.py` and `eval/repository.py` may `import subprocess` |
| `tests/test_sandbox_docker.py` (`-m docker`) | Live: mount, pytest, network none, only `/workspace` bind, timeout, truncation, cleanup |

### Live Docker smoke (2026-08-13)

Windows: Docker Desktop installed but `docker` was not on PATH until:

```powershell
$env:Path = "C:\Users\octav\AppData\Local\Programs\DockerDesktop\resources\bin;" + $env:Path
```

Then:

```text
python cli.py sandbox doctor     → ok=True, server_os=linux, image missing (warning)
python cli.py sandbox build      → Built issue-pilot-sandbox:py312
python -m pytest tests/test_sandbox_docker.py -m docker -q
                                 → 6 passed in 14.67s
```

Live Docker checks that passed:

- Repository mount visible (`/workspace/visible.txt`)
- `pytest` inside the container
- `NetworkMode=none`; curl denied by policy
- Exactly one bind mount, destination `/workspace`
- Timeout → container removed, sandbox unusable
- Output truncation
- Container gone after normal exit and after exception

Core unit + skip-without-Docker: `119 passed, 7 skipped` (6 Docker tests skipped when the engine is down; 1 symlink skip).

Live `issue-001` V0/V1 solve through Docker was **not** re-run in this phase (Phase 2 host trajectories remain the ablation baseline for resolve rate). Sandbox correctness is evidenced by the 6 live container tests plus mocked eval lifecycle tests.

## Comparison with Phase 2 (host execution)

| | Phase 2 | Phase 3 |
|---|---|---|
| Control flow | V0 ReAct / V1 graph | Unchanged |
| `run_tests` / `git_diff` | Host `subprocess` | Policy → `docker exec` |
| `grep_code` | Host `rg` if present | Pure Python only |
| Network | Host network | Container `--network none` |
| Isolation | Path jail only | Path jail + per-task container |
| Cleanup | Process exit | `finally` + `--rm`; metadata records outcome |
| Run record | Workflow telemetry | Plus `sandbox_*` fields |

Trade-off: Docker Desktop (Linux containers) is now a hard prerequisite for `solve`. There is no host fallback. CPU/memory/PID limits remain stretch (not implemented).

## Observations

1. **Agent capability ≠ model capability** — the model can *ask* for `curl`; the harness never execs it.
2. **Path jail ≠ command sandbox** — jail stops `read_file("../../.env")`; it cannot stop a host `pytest` from using the network. Phase 3 closes that gap for agent-visible commands.
3. **One container per task** — agent tests, V1 verify, and gold share the same mount so scoring sees the same worktree; `compare` isolates V0 vs V1 with two containers.
4. **Windows PATH** — Docker Desktop can be installed while `docker` is still missing from PATH; `sandbox doctor` fails closed instead of running on the host.
5. Inherited from Phase 1/2: gold tests are still readable; no retry after diagnose; reset is still before-run.

## Explicitly out of scope (Phase 3)

- Retry / replan (Day 5); `MAX_RETRY` is unused config
- RAG / embeddings (Day 4; that is V2)
- Checkpoint / HITL / MCP (Day 6)
- Docker CPU / memory / PID limits (stretch)
- A public host execution backend

## How to reproduce

```powershell
cd issue-pilot
copy .env.example .env   # set DEEPSEEK_API_KEY
python -m pip install -r requirements.txt

# Docker Desktop must be running in Linux-container mode.
# If `docker` is not found, prepend the Desktop bin, e.g.:
# $env:Path = "C:\Users\octav\AppData\Local\Programs\DockerDesktop\resources\bin;" + $env:Path

python cli.py sandbox doctor
python cli.py sandbox build
python -m pytest tests -q
python -m pytest tests/test_sandbox_docker.py -m docker -q

python cli.py solve issue-001 --harness v0
python cli.py solve issue-001 --harness v1
python cli.py compare issue-001
```

Each solve: reset benchmark → one sandbox → harness → gold → cleanup. Compare = two solves = two containers.

## Later (not this phase)

Retry after diagnose is wired in `docs/PHASE4-followup.md`. The bullets above remain what Phase 3 itself did not ship.
