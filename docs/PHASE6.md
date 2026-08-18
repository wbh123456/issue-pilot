# Phase 6 — Persistence, Approval HITL, Observability, MCP

Date: 2026-08-19  
Harness: `issue-pilot`  
Benchmark (sibling repo): `issue-pilot-benchmark`  
LLM: DeepSeek (`deepseek-v4-flash`) via OpenAI-compatible SDK  
Harness versions: **V0** unchanged. **V1** / **V2** gain checkpoint, optional approval, node traces. MCP is a stdio demo, not the live tool path.  
Sandbox: `issue-pilot-sandbox:py312` (`--network none`, benchmark mounted at `/workspace`)  
`base_commit`: `4b2258b1d16e802aa9b4a82bcb4a2b0f3911f84c`

This is **Day 6** in `docs/project-plan.md`. Phase 5 remains Day 5 (retry / dual-layer eval / same-process feedback). Gold pass/fail (`success`) is still the **only** benchmark resolve metric. An approved run whose gold fails is still a failure.

## Goal

1. Durable checkpoint so a later process can resume the same thread
2. Approval HITL after Layer 1 **and** Layer 2 pass: pause → review → approve / reject / feedback
3. Node-visit `workflow_trace` plus the five Day-6 checkpoint stages on v1/v2 run JSON
4. A minimal official-SDK MCP stdio server (`read_file`, `search_code`, `git_diff`) so the Agent → MCP Client → MCP Server → Tool chain is real, without putting live V1/V2 behind MCP

Definition of Done:

```text
Agent Run → Checkpoint → Pause → Human Approval → Resume → Finish
```

Stretch (implemented): approval `--feedback` re-enters `diagnose` (and consumes the normal retry budget).

## Current agentic workflow (V1 / V2)

```text
CLI  solve [--harness v1|v2] [--require-approval] [--interactive-recovery]
        │
        ▼
eval/runner.solve_task
        ├─ reset_repo to base_commit          ← solve only; resume never resets
        └─ SandboxRunner
                │
                ├─ V1  analyze → plan → execute → verify
                └─ V2  analyze → retrieve → plan → execute → verify
                                      │
                    verify ── Layer1 pass ──► evaluate
                         │                      │
                         │                      ├─ Layer2 pass → await_approval
                         │                      │                    │
                         │                      │     gate off / approve → mark_success
                         │                      │     reject            → mark_needs_human
                         │                      │     feedback          → diagnose
                         │                      └─ Layer2 fail ─────────► diagnose
                         └─ Layer1 fail ────────────────────────────────► diagnose
                                                                            │
                                                      retry_count < 2  → plan
                                                      else             → feedback → needs_human
        │
        ├─ interrupt + --require-approval
        │     runs/sessions/{run_id}.json  (paused; no runs/*.json, no gold yet)
        │     python cli.py review <run_id>
        │     python cli.py resume <run_id> --approve|--reject|--feedback TEXT
        │           no reset_repo; fresh checkpointer + fresh sandbox
        └─ finish
              gold  →  runs/{task}-{v1|v2}-{stamp}.json
```

`compare` still loops `("v0", "v1")` only and does **not** pass `--require-approval` or `--interactive-recovery`.

### Policy (mechanical, not a prompt)

| Limit | Value | Effect |
|---|---|---|
| Checkpointer | `SqliteSaver` at `runs/checkpoints.sqlite` | Per-node snapshots; `get_graph()` / `get_v2_graph()` stay in-memory |
| Sessions | `runs/sessions/{run_id}.json` | Sidecars; `eval.report` still globs only `runs/*.json` |
| `require_approval` | opt-in | `await_approval` is always in the graph; pass-through unless the flag is set (implies checkpointing; v1/v2 only) |
| Resume | **never** `reset_repo` | `verify_resume_worktree`: HEAD == `base_commit` and the tree is dirty |
| Gold | independent of HITL | Approve / reject do not write `success` |
| `retrieval_calls` | v1 and v2 | `0` on v1; retrieve node / `search_code` on v2 |
| MCP | stdio demo | Live executor still calls `agent.tools` directly |

## What we built

| Area | Location | Notes |
|---|---|---|
| Pins | `requirements.txt` | `langgraph>=1,<2`, `langchain-core>=1,<2`, `langgraph-checkpoint-sqlite>=3,<4`, `mcp>=1.29,<2` |
| Checkpointer | `harness/checkpoint.py` | `open_checkpointer()` → `SqliteSaver` |
| Sessions | `eval/session.py` | `RunSession` + `resume_count` / `update_session` |
| Approval | `agent/nodes/approve.py` | `interrupt()` six-part payload; unknown/blank feedback fail-closes to reject |
| Graph | `agent/graph.py` | `await_approval` on Layer 2 pass; `resume_workflow(Command(resume=...))` |
| Pause / resume | `eval/runner.py`, `eval/repository.py` | Paused session, no premature run JSON; resume verifies worktree, new sandbox |
| CLI HITL | `cli.py`, `harness/progress.py` | `--require-approval`, `runs` / `review` / `resume`; `format_review` six panels |
| Trace | `agent/state.py`, `agent/nodes/_runtime.py` | `WorkflowTraceEvent`; `append_trace` / `traced`; retries append |
| Persist | `eval/runner.py` | `workflow_trace`, `checkpoint_stages`, `retrieval_calls`, `run_id` / `resumed` / `sandbox_sessions` |
| Checkpoint size | `agent/nodes/execute.py` | Raw chat `messages` stay out of telemetry / sqlite |
| MCP | `harness/mcp_server.py`, `harness/mcp_client.py` | `read_file` / `search_code` / `git_diff`; lazy `SandboxRunner`; no `subprocess` import |
| MCP CLI | `cli.py mcp serve` / `mcp demo` | Stdio server; demo is Agent → Client → Server → Tool |

Checkpoint stages map first visits of `analyze` / `plan` / `execute` / `verify` / `await_approval` onto `Issue analyzed` … `Waiting approval`.

## Live evidence (2026-08-19)

No new billed DeepSeek solve was required for Day 6. The DoD is a **process boundary**, not a gold-rate change: pause, then finish from a **fresh** `SqliteSaver` and a **fresh** `resume_task` call (`tests/test_resume.py`). That is the same shape as killing the CLI and running `python cli.py resume <run_id> --approve`.

HITL CLI (needs `.env` `DEEPSEEK_API_KEY` and Docker):

```powershell
python cli.py solve issue-001 --harness v1 --require-approval
python cli.py runs
python cli.py review <run_id>
python cli.py resume <run_id> --approve
```

Solve resets the benchmark to `base_commit`. Resume must not. Do not commit agent patches into `issue-pilot-benchmark`.

### Checkpoint sqlite size

Local `runs/checkpoints.sqlite` from development HITL/graph tests (not a 009-class live transcript):

| Metric | Value |
|---|---|
| File size | **1,044,480 bytes** (~1.0 MiB) |
| `checkpoints` rows | 26 |
| `writes` rows | 133 |
| Largest `checkpoint` blob | **61,428 bytes** |

The Phase 5 issue-009 V2 run used **245,310 tokens**. Dumping `telemetry.messages` (raw chat dicts) into every node snapshot would have dominated sqlite. Execute therefore keeps **trajectory** and drops **messages** from checkpointed state. `adapt_result` still exposes `messages` as `[]` on v1/v2 unless some other node wrote them.

### MCP stdio

`tests/test_mcp.py::TestStdioRoundTrip` spawns `python -m harness.mcp_server --repo <tmp>` through the official client, lists the three tools, and reads `app/hello.py`. In-process tests inject a fake sandbox for `git_diff` and assert a clean `Error: git_diff needs Docker: ...` string when the factory raises `DockerPreflightError`. Live V1/V2 dispatch is unchanged.

```powershell
python cli.py mcp demo --repo ../issue-pilot-benchmark
```

`git_diff` starts a sandbox on first use and tears it down when the server exits. If Docker is down, the tool returns an error string instead of crashing the session.

## Test evidence

`python -m pytest tests -q -m "not docker"` → **358 passed**, 2 skipped.  
`python -m pytest tests -q -m docker` → not re-run for this write-up (image/daemon still required for the existing 10 docker tests).

| Suite | Role |
|---|---|
| `tests/test_checkpoint.py` | Per-node checkpoint; reload from a fresh saver; trace nodes match completed nodes; no `messages` in checkpointed telemetry |
| `tests/test_resume.py` | Pause writes no `runs/*.json`; resume skips `reset_repo`; gold independent of the decision |
| `tests/test_approve.py` | Gate off is pass-through; gate on interrupts; approve / reject / feedback routing |
| `tests/test_trace.py` | Retries append; stage labels; execute omits chat dumps |
| `tests/test_runner.py` | v1 emits `retrieval_calls == 0`; CLI `runs` / `review` / `resume` / `mcp` |
| `tests/test_mcp.py` | Schemas, in-process tools, fake-sandbox `git_diff`, real stdio round-trip |
| `tests/test_subprocess_audit.py` | Still only `sandbox/image.py` and `eval/repository.py` may import `subprocess` |

## Observations

1. **One graph, two modes.** `await_approval` is always wired after Layer 2 pass. Automated `compare` / default `solve` stay on the old path because the node is a pass-through unless `require_approval` is set.
2. **Resume is load-bearing on the bind mount.** The patch is host files, not container state. Calling `reset_repo` on resume would destroy the work the human just reviewed.
3. **Sandbox counters restart per container.** Each resume starts a new `SandboxRunner`. Record `sandbox_sessions` and treat last-container counters as last-container only.
4. **Feedback after Layer 2 still increments `retry_count`.** If the automatic budget is already spent, `route_after_diagnose` goes to `feedback` and then `needs_human`. That is the intended mechanical outcome.
5. **Traces append on retry.** A FAIL → diagnose → replan pass records two `plan` / `execute` / `verify` visits; `checkpoint_stages` keeps first-seen order of the five named stages.
6. **MCP is a capability demo, not a new harness.** Putting the executor behind MCP would be a different ablation. `search_code` on the server uses the same host-side hybrid index as the V2 tool; `git_diff` still requires Docker.
7. **Sqlite stayed small once messages were dropped.** ~61 KiB max blob vs a 245k-token chat log is the reason `telemetry.messages` is not checkpointed.

## Explicitly out of scope (Phase 6 / Day 6)

- Routing live V0/V1/V2 tool calls through MCP
- Putting `--require-approval` or V2 into `compare`
- Durable review UI beyond the six-panel CLI
- Streaming / HTTP MCP transports
- Benchmark or `eval/gold/` edits
- Day 7 (README, full ablation write-up)

## How to reproduce

```powershell
cd issue-pilot
copy .env.example .env   # set DEEPSEEK_API_KEY
python cli.py sandbox doctor --require-image

python -m pytest tests -q -m "not docker"

# Durable HITL (resets benchmark on solve only)
python cli.py solve issue-001 --harness v1 --require-approval
python cli.py runs
python cli.py review <run_id>
python cli.py resume <run_id> --approve

# MCP demo (does not reset; git_diff needs Docker)
python cli.py mcp demo --repo ../issue-pilot-benchmark
python cli.py mcp serve --repo ../issue-pilot-benchmark
```

Each **solve**: reset benchmark → sandbox → harness → (pause or gold) → cleanup.  
Each **resume**: verify worktree → new sandbox → `Command(resume=...)` → gold → `runs/*.json`.  
Do not commit agent patches into `issue-pilot-benchmark`.
