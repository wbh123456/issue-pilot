# Phase 5 — Retry, Dual-Layer Eval, Same-Process Feedback

Date: 2026-08-16  
Harness: `issue-pilot`  
Benchmark (sibling repo): `issue-pilot-benchmark`  
LLM: DeepSeek (`deepseek-v4-flash`) via OpenAI-compatible SDK  
Harness versions: **V1** / **V2** (recovery graph). V0 ReAct is unchanged.  
Sandbox: `issue-pilot-sandbox:py312` (`--network none`, benchmark mounted at `/workspace`)  
`base_commit`: `4b2258b1d16e802aa9b4a82bcb4a2b0f3911f84c`

This is **Day 5** in `docs/project-plan.md` (retry / evaluation / same-process feedback). It is **not** the week-plan label “Reliability + HITL” in full. Checkpoint/resume, approval HITL, observability expansion, and MCP stay Day 6.

Gold pass/fail (`success`) remains the **only** benchmark resolve metric. `recovery_success` is extra evidence: `retry_count > 0` and `workflow_passed`. First-try pass is not recovery.

## Goal

1. Automatic recovery: Layer 1 fail or Layer 2 reject → diagnose → **replan** (not re-execute the old plan) while `retry_count < MAX_RETRY`
2. Dual-layer evaluation: pytest + ruff + non-empty patch, then mechanical LLM-as-judge
3. Same-process feedback once after the automatic budget (`MAX_HUMAN_RETRY=1`), opt-in via `--interactive-recovery`

## Current agentic workflow (V1 / V2)

```text
CLI  solve [--harness v1|v2] [--interactive-recovery]
        │
        ▼
eval/runner.solve_task
        ├─ reset_repo to base_commit
        └─ SandboxRunner
                │
                ├─ V1  analyze → plan → execute → verify
                └─ V2  analyze → retrieve → plan → execute → verify
                                      │
                    verify ── Layer1 pass ──► evaluate
                         │                      │
                         │                      ├─ Layer2 pass → mark_success
                         │                      └─ Layer2 fail → diagnose
                         └─ Layer1 fail ────────────────────────► diagnose
                                                                    │
                                              retry_count < 2  → plan (new hypothesis)
                                              else             → feedback
                                                                    │
                                              accepted hint    → plan
                                              skip / blank / 2nd → mark_needs_human
        ▼
runs/{task}-{v1|v2}-{stamp}.json
  + attempt_history, patch_evaluation, human_retry_count,
    recovery_success  (V0 records do not get these keys)
```

`compare` still loops `("v0", "v1")` only and does **not** pass `--interactive-recovery`.

### Policy (mechanical, not a prompt)

| Limit | Value | Effect |
|---|---|---|
| `MAX_RETRY` | 2 | Diagnose increments `retry_count`. One initial execute + one automatic retry, then feedback. |
| `MAX_HUMAN_RETRY` | 1 | One accepted stdin hint may replan. Missing provider, blank, decline, or a second request → `needs_human`. |
| Layer 1 | `deterministic_pass` | AND of pytest, ruff, `patch_valid`. Empty lint command is fail-closed. Routing uses this flag only. |
| Layer 2 | `patch_evaluation_passed()` | `issue_resolved` and `patch_scope==appropriate` and `regression_risk==low` and `missing_tests==False`. The model cannot emit a pass field. |
| `workflow_passed` | Layer 1 **and** Layer 2 | Layer 2 cannot promote a Layer 1 failure to success. |

Diagnose includes Layer 2 JSON **only when this attempt’s Layer 1 passed** (stale-eval isolation). V2 evaluator-reject does not re-retrieve.

## What we built

| Area | Location | Notes |
|---|---|---|
| State | `agent/state.py` | `StructuredDiagnosis`, `PatchEvaluation`, `AttemptSummary`; `diagnosis` stays a string; no clients in state |
| Layer 1 | `agent/nodes/verify.py`, `agent/tools/shell.py`, `agent/tools/git.py` | `CommandOutcome` / `run_command()`; `WorktreeDiff` / `inspect_worktree()` |
| Diagnose + replan | `agent/nodes/diagnose.py`, `agent/nodes/plan.py` | Structured JSON; graph is `diagnose → plan`; repeated hypothesis overridden by `new_hypothesis` |
| Layer 2 | `agent/nodes/evaluate.py` | Called only after Layer 1 pass; mechanical pass |
| Feedback | `agent/nodes/feedback.py` | Runtime `configurable["feedback_provider"]`; CLI `solve --interactive-recovery` |
| Graph | `agent/graph.py` | Topology above; `get_graph()` / `get_v2_graph()` remain process-wide singletons (workflow tests use `build_graph()`) |
| Persist | `eval/runner.py` | V1/V2: `attempt_history`, `structured_diagnosis`, `patch_evaluation`, `human_retry_count`, `human_feedback`, `recovery_success` |
| Report | `eval/report.py`, `cli.py` | Cohort `recovery_rate` / `human_retries`; summary columns; `format_recovery_summary` one-liner |
| Dataset | `eval/dataset.json` | `"lint_command": "ruff check app"` (no gold / issue-text changes) |

## Live evidence (2026-08-16)

Same model, temperature 0, image `issue-pilot-sandbox:py312`, same `base_commit`. Harness SHA on these files: `aaa56fab`. No `--interactive-recovery`.

### Post-persist cohort (recovery fields on disk)

| Metric | issue-001 V1 `…T102904Z` | issue-001 V2 `…T103805Z` | **issue-009 V2 `…T104550Z`** |
|---|---|---|---|
| Gold `success` | True | True | **True** |
| `workflow_passed` | True | True | **True** |
| `retry_count` | 0 | 0 | **1** |
| `recovery_success` | False | False | **True** |
| `human_retry_count` | 0 | 0 | 0 |
| Layer 1 | pass | pass | pass (both attempts) |
| Layer 2 | PASS | PASS | **FAIL then PASS** |
| `failure_source` | — | — | **`evaluator`** |
| `failure_category` | — | — | **`BAD_PATCH`** |
| Steps / LLM / tools | 12 / 15 / 22 | 7 / 10 / 9 | 23 / 29 / 31 |
| File reads | 13 | 4 | 13 |
| Tokens | 99,532 | 35,964 | 245,310 |
| Latency | 157s | 73s | 361s |
| `recall_at_5` | — | **1.00** | **0.50** |

V2 retrieve stayed hashing + issue-only; retrieve LLM calls = 0. Live 009 hashing still missed gold file `app/orders.py` (same pattern as Phase 4). That is a RAG number, not the recovery result.

### Natural live recovery (issue-009 V2)

Run: `runs/issue-009-v2-20260816T104550Z.json`. Not scripted; not `--interactive-recovery`.

```text
verify   PASS  pytest=1 ruff=1 patch=1
evaluate FAIL  scope=appropriate risk=medium     ← Layer 2 mechanical fail
diagnose BAD_PATCH  (stock mutated before bin-slot check)
plan     retry  (new hypothesis: validate-then-mutate)
execute  retry 1
verify   PASS
evaluate PASS  scope=appropriate risk=low
success
gold     True
recovery_success True   human_retry_count 0
```

First attempt: Layer 1 passed (visible `tests/test_inventory_orders.py` + ruff + patch). Evaluator set `regression_risk=medium`, so `patch_evaluation_passed()` was false. Diagnose classified `BAD_PATCH`: `reserve()` ran before `bins['slots']` validation, so a 50-widget order could still 500 and leave `_STOCK` mutated. Replan moved validation before mutation. Second Layer 2 passed.

Final `changed_files`: `app/inventory.py`, `app/auth.py`, `app/main.py`. Evaluator called the auth/main edits unrelated but harmless. Gold still passed. Do not treat gold True as “patch was minimal.”

### Same-day earlier solves (pre-persist / in-progress graph)

These `issue-001` V1 files do **not** contain `recovery_success` / `attempt_history` (written before Step 7 persist). They still show gold ≠ workflow:

| Run | Gold | Workflow | Status | retry | Layer 1 | Notes |
|---|---|---|---|---|---|---|
| `…T061836Z` | True | False | `needs_human` | 2 | fail (`ruff_passed=False`) | Budget exhausted; hidden gold still passed |
| `…T064011Z` | True | False | `needs_human` | 2 | pass | Layer 1 pass, workflow fail, escalate (no stdin feedback) |
| `…T091814Z` | True | True | success | 0 | pass | First-try; no recovery fields on disk |

## Scripted recovery fixture

Live 009 is evaluator-reject recovery. Layer 1 fail → retry is pinned in tests, not manufactured with a live `solve`.

| Test | Trajectory |
|---|---|
| `tests/test_graph.py::test_retry_then_pass` | FAIL → diagnose → new hypothesis → retry → `deterministic_pass` → evaluator PASS → `workflow_passed`; `retry_count == 1`; `failure_source == "deterministic"` |
| `test_evaluator_reject_then_retry_pass` | Layer 1 pass, Layer 2 reject, then recover (same shape as live 009) |
| `test_fail_retries_then_needs_human` / `test_evaluator_reject_then_needs_human` | Budget exhausted → `needs_human` |
| `test_feedback_retry_then_pass` | One accepted hint → replan; second request does not call the provider |

V2: `tests/test_graph_v2.py::test_evaluator_reject_retries_without_re_retrieve`.

## Test evidence

`python -m pytest tests -q -m "not docker"` → **293 passed**, 2 skipped.  
`python -m pytest tests -q -m docker` → **10 passed** (daemon available).

| Suite | Role |
|---|---|
| `tests/test_state_contracts.py` | Diagnosis / evaluation / attempt schemas; no pass field on Layer 2 |
| `tests/test_verify.py` | Layer 1 AND; empty lint fail-closed |
| `tests/test_graph.py` | Routing, Layer 1 retry, evaluator reject, feedback |
| `tests/test_graph_v2.py` | V2 + no re-retrieve on evaluator reject |
| `tests/test_evaluate.py` / `test_diagnose.py` / `test_feedback.py` | Node contracts |
| `tests/test_runner.py` | Persist recovery fields; V0 omits them; gold independent of `workflow_passed` |
| `tests/test_report.py` | `recovery_rate` / `human_retries`; old JSON derived from `retry_count` + `workflow_passed` |

## Observations

1. **Gold is not recovery, and gold is not workflow.** Live 001 `…T061836Z` gold-passed after ruff-failed Layer 1 and `needs_human`. Live 009 recovered on Layer 2, then gold-passed. Report `success` / Resolve is still hidden gold only.
2. **Retry can fire on Layer 2 after Layer 1 already passed.** Phase 4 follow-up’s hard matrix never diagnosed because visible tests passed on the first execute. This 009 run’s visible tests also passed first; the evaluator still rejected (`risk=medium`).
3. **One natural live recovery exists.** Cite `runs/issue-009-v2-20260816T104550Z.json`. Do not pad the rate with `--interactive-recovery` or by editing issue/gold.
4. **First-try 001 is the common path.** `recovery_success=False` with `retry_count=0` is expected, not a harness miss.
5. **Human feedback was not used on these live runs.** `human_retry_count=0`. Same-process HITL is covered by tests + the CLI flag, not by this cohort.
6. **Recovery is expensive.** 009 V2 used 245k tokens / 6 minutes vs first-try 001 V2 at 36k / 73s. Diagnose + second execute dominate, not retrieve.

## Explicitly out of scope (Phase 5 / Day 5)

- Checkpoint / resume
- Approval HITL (review UI, durable pause)
- Observability beyond run JSON / `cli.py report`
- MCP
- Putting V2 or `--interactive-recovery` into `compare`
- Benchmark or gold edits

## How to reproduce

```powershell
cd issue-pilot
copy .env.example .env   # set DEEPSEEK_API_KEY
python cli.py sandbox doctor --require-image

python -m pytest tests -q -m "not docker"
python -m pytest tests -q -m docker

# Live (resets benchmark; writes runs/*.json)
python cli.py solve issue-001 --harness v1
python cli.py solve issue-001 --harness v2
python cli.py solve issue-009 --harness v2 --embedder hashing --query-mode issue

python cli.py report --split hard --base-commit 4b2258b1d16e802aa9b4a82bcb4a2b0f3911f84c
```

Optional, not used for the 009 recovery above:

```powershell
python cli.py solve issue-009 --harness v1 --interactive-recovery
```

Each solve: reset benchmark → one sandbox → harness → hidden gold → cleanup. Do not commit agent patches into `issue-pilot-benchmark`.
