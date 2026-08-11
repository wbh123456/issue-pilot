# Phase 2 — Day 2 Completion Record

Date: 2026-08-11  
Harness: `issue-pilot`  
Benchmark (sibling repo): `issue-pilot-benchmark`  
LLM: DeepSeek (`deepseek-v4-flash`) via OpenAI-compatible SDK  
Harness versions: **V0** (ReAct loop) vs **V1** (LangGraph Plan-Execute)

## Goal

Turn the free-form Day 1 agent loop into a controllable stateful workflow, then ablate V0 vs V1 on the same issue with identical model settings:

1. Serializable `AgentState` + structured plan schema
2. LangGraph: `analyze → plan → execute → verify → (success | diagnose)`
3. Eval runner / CLI that can select harness and compare metrics

## What we built

### Workflow (`agent/`)

| Area | Location | Notes |
|---|---|---|
| State | `agent/state.py` | `AgentState`, `StructuredPlan` (`extra="forbid"`, steps 3–5), `Telemetry` + `stage_tokens` |
| Nodes | `agent/nodes/` | `analyze`, `plan`, `execute`, `verify`, `diagnose` + `_runtime` helpers |
| Graph | `agent/graph.py` | Compile once; `route_after_verify` uses process `exit_code` only |
| V0 seam | `agent/loop.py` | `run_agent(..., workflow_context=None)` — V0 unchanged when context omitted |
| Planner grounding | `agent/nodes/plan.py` | Repo inventory via `list_files`; drop nonexistent `files_to_inspect` |
| Execute context | `agent/nodes/execute.py` | Plan-only JSON + guardrail (no duplicated analysis) |

Graph shape (Day 2; no replan loop yet):

```text
START → analyze → plan → execute → verify
                                      ├─ PASS → mark_success → END
                                      └─ FAIL → diagnose → END
```

### Eval / CLI

| Area | Location | Notes |
|---|---|---|
| Runner | `eval/runner.py` | `harness_version` v0/v1; versioned `runs/{task}-{v0\|v1}-{stamp}.json`; failed harness still saves a partial record |
| CLI | `cli.py` | `solve --harness`, `compare` |
| Deps | `requirements.txt` | `langgraph`, `langchain-core`, `pydantic` |
| Tests | `tests/test_graph.py`, `tests/test_runner.py` | Graph routing, schema bounds, stage tokens, error persistence |

## Eval results (DoD)

Same task (`issue-001`), same model, `max_steps=15`. Gold scoring is independent of workflow verify.

### Post-fix compare (authoritative)

| Metric | V0 | V1 |
|---|---|---|
| Gold success | True | True |
| Workflow passed | — | True |
| Steps | 9 | 10 |
| LLM calls | 9 | 12 |
| Tool calls | 10 | 13 |
| File reads | 3 | 6 |
| Tokens | 32,100 | 54,456 |
| Latency | ~21.5s | ~36.5s |

Trajectories:

- `runs/issue-001-v0-20260811T161519Z.json`
- `runs/issue-001-v1-20260811T161557Z.json`

DoD (“run the same issue on V0 and V1 and save both architectures’ metrics”) **met**.  
Stretch (“planner replan on failed hypothesis”) **not met** — deferred (Day 5-style retry).

### Pre-fix V1 (why we tightened the planner)

First V1 smoke also gold-passed but planned **hallucinated Spring/Java paths** and burned ~115k tokens / ~98s:

| | V1 pre-fix | V1 post-fix |
|---|---|---|
| Tokens | 115,399 | 54,456 |
| Latency | ~98s | ~36.5s |
| Plan `files_to_inspect` | `src/main/java/...` (invented) | `app/auth.py`, `tests/test_auth.py`, … |
| Plan steps | 10 (overlong) | 5 (schema-capped) |

Fixes that landed before the final compare: grounded inventory, strict schema, slim execute context, `stage_tokens`, persist-on-error, pin `langchain-core`.

## Observations

1. **Resolve rate unchanged on issue-001** — both harnesses already succeed; the win is efficiency + plan grounding, not pass/fail.
2. **V1 still costs more than V0** (~1.7× tokens / latency after fixes) because analyze+plan are extra LLM calls on top of the executor ReAct loop.
3. **Deterministic verify ≠ gold** — workflow verify may run a broader suite; gold remains the score reported in `success`.
4. **Test leakage remains** (from Phase 1) — executor still reads `tests/`; plan steps must not instruct modifying tests, but reading them still helps.
5. **Stage telemetry** is now in V1 run JSON (`stage_tokens.analyze|plan|execute|diagnose`) for later ablations.

## Explicitly out of scope (Phase 2)

- Planner replan / retry loops
- RAG / embeddings
- MCP
- Docker sandbox
- Human-in-the-loop
- Changing V0 default behavior

## How to reproduce

```powershell
cd issue-pilot
copy .env.example .env   # set DEEPSEEK_API_KEY
python -m pip install -r requirements.txt
python -m pytest tests -q
python cli.py compare issue-001
# or one side:
python cli.py solve issue-001 --harness v0
python cli.py solve issue-001 --harness v1
```

Each solve resets the benchmark worktree to `base_commit` before the agent runs; compare runs v0 then v1 with identical model / `max_steps`.
