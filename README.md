# IssuePilot

IssuePilot is an **evaluation-driven coding-agent harness** for measuring how
workflow design changes software-engineering outcomes. It runs the same model,
task, sandbox, and budget through three increasingly structured harnesses:

- **V0:** a minimal ReAct coding loop.
- **V1:** a LangGraph Plan-Execute workflow with verification and recovery.
- **V2:** V1 plus hybrid code retrieval and a semantic `search_code` tool.

The project includes a Docker sandbox, hidden-gold evaluation, retrieval
ablations, run telemetry, checkpointed human approval, and cohort reporting.
The buggy FastAPI application is deliberately kept in a **separate sibling
repository**, `../issue-pilot-benchmark`, so the agent cannot inspect the gold
tests or accidentally modify the harness.

> **Project status:** the seven-phase MVP is complete. The main finding is not
> that more orchestration always wins: V2 improved the small retrieval-focused
> ablation, while V1 did not beat the cheaper V0 baseline.

## Why this project exists

Coding agents are often evaluated as one opaque system: one model, one prompt,
and one pass/fail score. That makes it difficult to tell whether an improvement
came from the model, tool access, planning, retrieval, retry logic, or the test
suite.

IssuePilot separates those variables. Controlled runs keep the model,
temperature, sandbox image, executor step budget, benchmark revision, and
scoring rules fixed while changing the harness. This supports questions such
as:

- Does planning improve hidden-test resolve rate?
- Does retrieval help the agent find files that lexical search misses?
- Do deterministic verification and diagnosis produce useful recoveries?
- What do those features cost in tokens, tool calls, file reads, and latency?
- Is a failure caused by retrieval, planning, patching, evaluation, the
environment, or the benchmark itself?

## Features at a glance

- Three comparable harnesses: ReAct, Plan-Execute, and Plan-Execute + RAG.
- Structured plans, diagnoses, patch evaluations, and checkpoint-safe state.
- Seven coding tools with path jails, command allowlists, and bounded output.
- Network-disabled Docker execution with no host fallback for test commands.
- Deterministic verification followed by an LLM patch-quality gate.
- Automatic diagnose-and-replan recovery plus optional human feedback.
- Optional approval interrupts with SQLite checkpoints and resumable sessions.
- AST-aware hybrid retrieval using BM25, dense vectors, and reciprocal-rank
fusion.
- Visible task tests plus hidden gold tests that are unavailable to the agent.
- Reproducible benchmark matrices, retrieval ablations, telemetry, and reports.
- A small stdio MCP demo for `read_file`, `search_code`, and `git_diff`.

## Quick start

### Prerequisites

- Python 3.12
- Docker Desktop or Docker Engine
- A DeepSeek API key
- The benchmark repository at `../issue-pilot-benchmark`

The benchmark must be at the `base_commit` recorded in `eval/dataset.json`
(currently Benchmark v4, `f7dbd4000d94dc5aab3835698dc3cb3bbd3eabc7`).
A normal `solve` resets that worktree, so do not keep unrelated work there and
never commit an agent-generated patch into the benchmark.

### Install and verify

```powershell
cd issue-pilot
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env   # set DEEPSEEK_API_KEY

python cli.py sandbox doctor
python cli.py sandbox build
python -m pytest tests -q -m "not docker"
```

Optional environment overrides:

```dotenv
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

### Run the agent

```powershell
# V0 is the default
python cli.py solve issue-001

# Structured Plan-Execute
python cli.py solve issue-001 --harness v1

# Plan-Execute + hybrid retrieval
python cli.py solve issue-016 --harness v2

# Offline retrieval evaluation; no LLM call
python cli.py retrieve issue-015 --embedder hashing

# Controlled matrix and aggregate report
python cli.py bench --split ablation --harness v0,v1,v2 --n 1 --log
python cli.py report --split ablation --latest-per-cell
```

## Architecture

IssuePilot uses two repositories and three execution boundaries.


| Component      | Location                    | Responsibility                                                              |
| -------------- | --------------------------- | --------------------------------------------------------------------------- |
| Harness        | This repository             | Agent workflows, tools, policy, retrieval, evaluation, and reports          |
| Subject        | `../issue-pilot-benchmark`  | Buggy FastAPI app, visible tests, and the patch-producing worktree          |
| Sandbox        | Docker container            | Runs allowlisted test, lint, type-check, and read-only git commands         |
| Gold evaluator | `eval/gold/` in the harness | Stages hidden tests only after the agent run and removes them after scoring |


```mermaid
flowchart TB
    U[User or benchmark command] --> CLI[cli.py]
    CLI --> RUNNER[Evaluation runner]
    RUNNER --> DATA[eval/dataset.json]
    RUNNER --> RESET[Reset subject to base_commit]
    RESET --> SUBJECT[issue-pilot-benchmark]

    RUNNER --> HARNESS{Harness}
    HARNESS --> V0[V0 ReAct loop]
    HARNESS --> V1[V1 analyze → plan → execute → verify → evaluate]
    HARNESS --> V2[V2 analyze → retrieve → plan → execute → verify → evaluate]

    V0 --> TOOLS[Agent tools]
    V1 --> TOOLS
    V2 --> TOOLS
    TOOLS --> HOST[Path-jailed host file and retrieval operations]
    TOOLS --> DOCKER[Network-disabled Docker commands]
    HOST --> SUBJECT
    DOCKER --> SUBJECT

    RUNNER --> GOLD[Hidden-gold scoring]
    SUBJECT --> GOLD
    GOLD --> ARTIFACTS[runs/*.json and reports]
```



### Solve lifecycle

1. `eval/runner.py` loads a task from `eval/dataset.json`.
2. The trusted evaluator resets the benchmark to the task's `base_commit`.
3. `SandboxRunner` starts one hardened container with the benchmark mounted at
  `/workspace`.
4. The selected harness analyzes the issue, uses tools, and edits the subject
  worktree.
5. V1/V2 run their workflow verification and recovery gates.
6. After the harness finishes, the evaluator temporarily stages the hidden
  gold test under `tests/_gold/`, runs it, and removes it.
7. The runner writes a timestamped JSON artifact under `runs/`.

An approval-interrupted run pauses before gold scoring and writes a session
sidecar. Resuming does **not** reset the benchmark, because that would destroy
the patch under review.

## Harness versions

`harness_version` selects control flow; it does **not** select a different
sandbox. All versions use the same Docker runtime and benchmark scorer.


| Version | Control flow                                                                                  | Tool surface               | Purpose                                           |
| ------- | --------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------- |
| V0      | ReAct loop, up to 15 executor steps                                                           | Six base tools             | Minimal and inexpensive baseline                  |
| V1      | `analyze` → `plan` → `execute` → `verify` → `evaluate` → `diagnose` / `feedback`              | Same six tools             | Test structured planning, verification, and retry |
| V2      | `analyze` → `retrieve` → `plan` → `execute` → `verify` → `evaluate` → `diagnose` / `feedback` | Base tools + `search_code` | Test whether hybrid localization changes outcomes |


### V0: ReAct baseline

`agent/loop.py` alternates model responses and tool results until the model
submits an answer, reaches the 15-step limit, or fails. V0 has no separate
planner, verifier, evaluator, retrieval node, or workflow retry. Hidden-gold
scoring still happens after the loop, so V0 remains directly comparable with
V1 and V2.

### V1/V2: LangGraph workflow

`agent/graph.py` compiles a serializable state machine. Every V1/V2 run visits
the same node set except `retrieve`, which exists only on V2 (between
`analyze` and `plan`). Evaluator-reject does **not** re-run `retrieve`.

Happy path:

```text
START → analyze → [retrieve] → plan → execute → verify → evaluate
      → await_approval → mark_success → END
```

Recovery path:

```text
verify / evaluate fail → diagnose
  ├─ retry_count < 2  → plan → execute → …
  └─ else             → feedback
                          ├─ human hint → plan → execute → …
                          └─ declined   → mark_needs_human → END
```

```mermaid
flowchart TB
    START((START)) --> analyze
    analyze -->|V2| retrieve
    analyze -->|V1| plan
    retrieve --> plan
    plan --> execute
    execute --> verify

    verify -->|Layer 1 pass| evaluate
    verify -->|Layer 1 fail| diagnose
    evaluate -->|Layer 2 pass| await_approval
    evaluate -->|Layer 2 reject| diagnose

    await_approval -->|pass-through or approve| mark_success
    await_approval -->|feedback| diagnose
    await_approval -->|reject| mark_needs_human

    diagnose -->|retry_count less than MAX_RETRY| plan
    diagnose -->|budget exhausted| feedback
    feedback -->|feedback_retry| plan
    feedback -->|declined| mark_needs_human

    mark_success --> END((END))
    mark_needs_human --> END
```




| Node               | File                      | LLM? | Role                                                                                                                                                                             |
| ------------------ | ------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analyze`          | `agent/nodes/analyze.py`  | Yes  | Restate the issue and form an initial hypothesis. No tools, no edits.                                                                                                            |
| `retrieve`         | `agent/nodes/retrieve.py` | No   | V2 only. Hybrid search over `app/**/*.py`; writes `relevant_files` and `retrieved_context`.                                                                                      |
| `plan`             | `agent/nodes/plan.py`     | Yes  | Emit a validated `StructuredPlan` (`problem`, `hypothesis`, `files_to_inspect` ≤ 8, 3–5 `steps`). V2 grounds the planner in retrieved snippets instead of a directory inventory. |
| `execute`          | `agent/nodes/execute.py`  | Yes  | ReAct tool loop (same as V0) with the plan, diagnosis, and retrieved snippets in `workflow_context`.                                                                             |
| `verify`           | `agent/nodes/verify.py`   | No   | Layer 1: visible pytest + ruff + non-empty git diff.                                                                                                                             |
| `evaluate`         | `agent/nodes/evaluate.py` | Yes  | Layer 2 LLM-as-judge. Pass/fail is computed mechanically from `PatchEvaluation`.                                                                                                 |
| `await_approval`   | `agent/nodes/approve.py`  | No   | Pass-through unless `--require-approval`; then interrupt with a review bundle.                                                                                                   |
| `diagnose`         | `agent/nodes/diagnose.py` | Yes  | Structured failure analysis (`root_cause`, `failure_category`, `new_hypothesis`, `next_actions`) and increment `retry_count`.                                                    |
| `feedback`         | `agent/nodes/feedback.py` | No   | Opt-in human hint after the automatic retry budget (`--interactive-recovery`).                                                                                                   |
| `mark_success`     | `agent/graph.py`          | No   | Terminal status `success`.                                                                                                                                                       |
| `mark_needs_human` | `agent/graph.py`          | No   | Terminal status `needs_human`.                                                                                                                                                   |


Workflow state includes the issue, analysis, structured plan, retrieved files,
test result, patch evaluation, diagnosis, attempt history, retry counters,
human decisions, telemetry, and an append-only workflow trace. Runtime-only
objects such as the model client and sandbox stay outside the state so
checkpoints remain JSON-serializable.

### Verification and recovery

**Layer 1 is deterministic.** A patch passes only when:

1. The task's visible pytest command passes.
2. Ruff passes.
3. The git diff is non-empty.

Ruff autofix is restricted to Python files touched by the agent. Earlier
whole-package autofix behavior dirtied unrelated files and distorted patch
quality metrics.

**Layer 2 is an LLM patch evaluator with a mechanical decision rule.** The
model returns a structured `PatchEvaluation`; code computes pass only when:

- `issue_resolved` is true,
- `patch_scope` is `appropriate`,
- `regression_risk` is `low`, and
- `missing_tests` is false.

A Layer 1 failure or Layer 2 rejection enters structured diagnosis and
replanning. `MAX_RETRY=2` means one initial execution plus one automatic
replan after the first failed attempt. After that, `--interactive-recovery`
can accept one human hint before the workflow escalates to `needs_human`.
V2 does not re-run the retrieve node on evaluator rejection.

The hidden gold test is deliberately independent of both workflow layers. The
agent never receives gold output, and a gold-correct patch can still be
rejected by Layer 2.

## Agent tools and safety boundaries


| Tool          | Available in | Purpose                                 | Boundary          |
| ------------- | ------------ | --------------------------------------- | ----------------- |
| `list_files`  | V0/V1/V2     | List repository paths                   | Host, path-jailed |
| `read_file`   | V0/V1/V2     | Read bounded file content               | Host, path-jailed |
| `grep_code`   | V0/V1/V2     | Lexical search across the repository    | Host, path-jailed |
| `edit_file`   | V0/V1/V2     | Apply targeted text edits               | Host, path-jailed |
| `run_tests`   | V0/V1/V2     | Run the task's test command             | Docker            |
| `git_diff`    | V0/V1/V2     | Inspect the current patch               | Docker            |
| `search_code` | V2 only      | Hybrid semantic and lexical code search | Host, path-jailed |


The container runs as a non-root user with:

- no network,
- a read-only root filesystem,
- a temporary `/tmp`,
- all Linux capabilities dropped,
- `no-new-privileges`, and
- one read/write mount for the benchmark at `/workspace`.

Agent commands are parsed as argv rather than shell text. The allowlist covers
`pytest`, `ruff`, `mypy`, and `git status`/`git diff`; shell metacharacters are
rejected. Commands time out after 60 seconds, tool payloads are capped at
10,000 characters, and there is no host fallback for container commands.

Filesystem and retrieval tools currently run on the host against a strict
repository path jail. The Docker boundary therefore protects command
execution, while the path jail protects host-side file access.

## Hybrid code retrieval

V2 builds an in-memory index over `app/**/*.py` in the benchmark:

```text
Python source
  → AST chunks (module, class, function, method)
  → BM25 lexical ranking + dense cosine ranking
  → reciprocal-rank fusion (RRF k=60)
  → top 5 files and bounded snippets for the planner
```

- The retrieve node is deterministic and uses no LLM call.
- Live V2 defaults to the dependency-free hashing embedder and issue-only
queries.
- `python cli.py retrieve` defaults to FastEmbed and can compare `grep`,
`bm25`, `dense`, and `hybrid` without calling the model.
- The offline grep baseline is restricted to `app/**/*.py` to match the index
corpus; the agent's regular `grep_code` tool searches the whole repository.
- `search_code` is available during V2 execution when issue language does not
match identifiers.
- The index is rebuilt rather than persisted, avoiding stale data after edits
at the cost of repeated work.

Retrieval quality is measured with file-level `recall_at_5`. It is a
localization metric, not a substitute for hidden-gold resolve.

See `[docs/PHASE4-architecture.md](docs/PHASE4-architecture.md)` for the design
trade-offs.

## Human-in-the-loop and persistence

Human intervention is opt-in and serves two different purposes:

1. `--interactive-recovery` asks for one same-process hint after automatic
  recovery is exhausted.
2. `--require-approval` interrupts a V1/V2 run after both workflow layers pass
  and presents the issue, plan, patch, test result, evaluator result, and
   trace for review.

Approval uses LangGraph's SQLite checkpointer at
`runs/checkpoints.sqlite`. Paused session metadata lives under
`runs/sessions/`.

```powershell
python cli.py solve issue-001 --harness v1 --pause-on-approval
python cli.py runs
python cli.py review <run_id>
python cli.py resume <run_id> --approve
python cli.py resume <run_id> --reject
python cli.py resume <run_id> --feedback "Drop the unrelated edit"
```

`await_approval` is a pass-through node unless approval is required. `compare`
and `bench` do not enable approval or interactive recovery.

## Evaluation design

### Dataset

`eval/dataset.json` contains 17 tasks on one pinned Benchmark v4 commit.


| Split    | Tasks                   | Purpose                                                      |
| -------- | ----------------------- | ------------------------------------------------------------ |
| Smoke    | `issue-001`–`issue-007` | Easy and medium end-to-end sanity checks                     |
| Hard     | `issue-008`–`issue-014` | Cross-file storefront issues; gold often checks a second cut |
| Ablation | `issue-015`–`issue-017` | Two retrieval-sensitive tasks and one retry-sensitive task   |


Each task records its issue text, split, difficulty, subject path, base commit,
expected files, visible test command, lint command, and hidden-gold mapping.

### Visible tests and hidden gold

The two test layers answer different questions:


| Test layer | Location             | Visible to agent | Purpose                                           |
| ---------- | -------------------- | ---------------- | ------------------------------------------------- |
| Task tests | Benchmark repository | Yes              | Guide implementation and Layer 1 verification     |
| Gold tests | Harness `eval/gold/` | No               | Determine benchmark resolution after the workflow |


Gold files are copied into `tests/_gold/` only for scoring and are deleted
afterward. The path jail and pytest configuration exclude that directory from
normal agent access.

### Metric glossary

- `**success**` — hidden gold passed. This is the only benchmark resolve
metric.
- `**workflow_passed**` — Layer 1 and Layer 2 both passed. This is workflow
telemetry, not resolve.
- `**recovery_success**` — a retry occurred and `workflow_passed` is true. It
does not mean “gold passed after retry.”
- `**recall_at_5**` — expected-file coverage in the top five retrieval results.
- `**localization_precision**` — changed expected files divided by all changed
files.
- `**layer1_gate_rate**` — fraction of runs delivered with deterministic
verification passing.

Run cohorts are keyed by benchmark commit, model, temperature, sandbox image,
and `benchmark_spec_sha` (a hash of the dataset and gold specification). This
prevents results from different benchmark revisions from being silently
combined.

### Artifacts and reporting

- Solves: `runs/{task}-{v0|v1|v2}-{timestamp}.json`
- Retrieval evals: `runs/{task}-retrieve-{timestamp}.json`
- Matrices: `runs/matrix-{timestamp}.json` and optional logs
- Paused sessions: `runs/sessions/{run_id}.json`

Artifacts include the patch, trajectory, termination reason, tokens, latency,
tool calls, file reads, sandbox telemetry, workflow trace, stage-level usage,
verification results, diagnoses, retries, and retrieval fields where
applicable. `python cli.py report` aggregates compatible cohorts and can emit
human-readable or JSON output.

## Tech stack


| Area                  | Technology                                        |
| --------------------- | ------------------------------------------------- |
| Language/runtime      | Python 3.12                                       |
| Model provider        | DeepSeek through the OpenAI-compatible Python SDK |
| Agent orchestration   | Custom ReAct loop, LangGraph, LangChain Core      |
| Structured contracts  | Pydantic                                          |
| Checkpointing         | LangGraph SQLite checkpointer                     |
| Retrieval             | Python AST, `rank-bm25`, NumPy, FastEmbed, RRF    |
| Sandbox               | Docker, Python 3.12 slim, non-root execution      |
| Verification          | pytest, Ruff, mypy                                |
| CLI and output        | argparse, Rich                                    |
| Tool protocol demo    | Model Context Protocol (MCP), stdio transport     |
| Benchmark application | FastAPI, PyJWT, HTTPX                             |


The repository uses `requirements.txt` and direct `python cli.py` invocation;
it is not currently packaged as an installable CLI.

## Command reference


| Command                      | Purpose                                                  |
| ---------------------------- | -------------------------------------------------------- |
| `solve <task>`               | Run one task with V0, V1, or V2 and score hidden gold    |
| `compare <task>`             | Run V0 and then V1; it intentionally does not include V2 |
| `retrieve [task]`            | Evaluate grep/BM25/dense/hybrid Recall@K without an LLM  |
| `bench`                      | Run a split × harness × repetition matrix                |
| `report`                     | Aggregate compatible run artifacts                       |
| `sandbox doctor`             | Validate Docker and image availability                   |
| `sandbox build`              | Build `issue-pilot-sandbox:py312`                        |
| `runs` / `review` / `resume` | Inspect and continue approval sessions                   |
| `mcp serve` / `mcp demo`     | Run the three-tool stdio MCP demonstration               |


Useful examples:

```powershell
# Compare only the original baseline and Plan-Execute harness
python cli.py compare issue-001

# V2 configuration
python cli.py solve issue-016 --harness v2 --embedder hashing --query-mode issue

# One selected hard-task matrix
python cli.py bench --split hard --tasks issue-008,issue-011,issue-013,issue-014 `
  --harness v0,v1,v2 --n 1 --log

# Retrieval ablation for a complete split
python cli.py retrieve --split ablation --embedder hashing

# MCP is a demo; live V0/V1/V2 tools are dispatched directly
python cli.py mcp demo --repo ../issue-pilot-benchmark `
  --path app/auth.py --query decode_token
```

## Repository layout

```text
issue-pilot/
├── cli.py                  # Public command-line entry point
├── agent/
│   ├── client.py           # DeepSeek/OpenAI-compatible client
│   ├── loop.py             # V0 ReAct loop
│   ├── graph.py            # V1/V2 LangGraph workflows and routing
│   ├── state.py            # Serializable state and Pydantic contracts
│   ├── nodes/              # Analyze, retrieve, plan, execute, verify, etc.
│   └── tools/              # Tool schemas, dispatch, file/search/shell tools
├── eval/
│   ├── dataset.json        # Task definitions and benchmark provenance
│   ├── gold/               # Hidden evaluator tests
│   ├── runner.py           # Solve, resume, gold scoring, run records
│   ├── matrix.py           # Controlled benchmark matrices
│   ├── retrieval.py        # Offline retrieval evaluation
│   └── report.py           # Cohort aggregation
├── harness/
│   ├── limits.py           # Budgets, timeouts, and output bounds
│   ├── permissions.py      # Command policy
│   ├── checkpoint.py       # SQLite checkpointer
│   └── mcp_*.py            # MCP demonstration
├── retrieval/              # Chunking, lexical/dense search, fusion, indexing
├── sandbox/                # Dockerfile, image checks, and task runner
├── tests/                  # Harness unit and Docker integration tests
├── runs/                   # Generated run artifacts and sessions
└── docs/                   # Phase reports, architecture notes, and week plan
```

## Results so far

The primary controlled ablation used one run per cell on the three Benchmark
v4 ablation tasks (`runs/matrix-20260819T111828Z.json`).


| Task                    | V0       | V1       | V2       |
| ----------------------- | -------- | -------- | -------- |
| `issue-015` (retrieval) | Fail     | Fail     | Fail     |
| `issue-016` (retrieval) | Fail     | Fail     | **Pass** |
| `issue-017` (retry)     | **Pass** | **Pass** | **Pass** |
| **Gold resolve rate**   | **0.33** | **0.33** | **0.67** |


The supported conclusion is **V2 > V1 = V0 on this small ablation**, not a
general three-way ranking:

- V2's hybrid retrieval reached Recall@5 = 1.0 on issues 015 and 016 where the
scoped grep and BM25 baselines scored 0.
- Only issue 016 converted that localization gain into a gold pass.
- V0 solved issue 017 without a dedicated verify/retry graph, so it was not a
V1-specific recovery win.
- The seven-task hard split previously tied at 0.43 across all harnesses. A
four-task v4 rerun produced one additional V2 pass, but `n=1` is not enough
to call that an architectural effect.
- V0 remained much cheaper on the ablation: about 93k tokens / 63 seconds,
versus 112k / 183 seconds for V1 and 306k / 296 seconds for V2. V2 also read
more files.
- Scoping Ruff autofix improved V1/V2 localization precision from 0.26 to
1.00 on the selected hard tasks. That was a harness measurement fix, not a
model-quality improvement.

See `[docs/PHASE7.md](docs/PHASE7.md)` for full tables, failure attribution,
and reproduction notes.

## Current limitations

- The benchmark is small and the reported architecture comparison uses
`n=1`; results are evidence for specific cases, not a leaderboard.
- V1 has not demonstrated a hidden-gold resolve advantage over V0 in the
reported matrices.
- Structured plans reject more than eight `files_to_inspect`, which caused
`PlanValidationError` on decoy-heavy tasks.
- Layer 2 can reject a gold-correct patch, making `recovery_success` diverge
from actual benchmark resolution.
- Some hard gold tests assert a second behavioral cut that visible tests do
not expose, so the retry loop receives no deterministic failure signal.
- V2 rebuilds its in-memory index and often uses substantially more tokens,
latency, and file reads.
- Live V2 uses hashing embeddings by default; offline FastEmbed results should
not be presented as live-agent performance.
- `compare` covers V0 and V1 only.
- MCP is a demonstration layer; live harnesses call the local tools directly.
- Host filesystem operations are path-jailed but are not executed inside the
Docker container.
- The sandbox does not yet set explicit CPU, memory, or PID limits.
- The system currently targets one OpenAI-compatible provider and one
benchmark repository.

## Roadmap: ways to enhance the agent

The highest-value additions are the ones tied to observed failure modes and
measurable outcomes.


| Enhancement                                     | Why it matters                                                                    | How to evaluate it                                               |
| ----------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Repair or clip malformed structured plans       | Prevent `files_to_inspect` overflow from terminating a run before execution       | Plan-valid rate and gold resolve on decoy-heavy tasks            |
| Calibrate the Layer 2 evaluator                 | Reduce false rejection of gold-correct, low-risk patches                          | Evaluator precision/recall against hidden gold, measured offline |
| Generate targeted regression/property tests     | Give recovery a signal for behavioral “second cuts” without revealing gold tests  | New-test validity, Layer 1 catch rate, and held-out gold resolve |
| Add symbol and dependency-aware retrieval       | Improve localization beyond issue-token similarity                                | Recall@K, first expected read, search calls, and resolve         |
| Cache indexes with edit-aware invalidation      | Remove repeated indexing cost without serving stale chunks                        | Retrieval latency, freshness tests, and end-to-end cost          |
| Add reranking and query reformulation           | Improve ordering when BM25 or the hashing embedder retrieves distractors          | Fixed-corpus retrieval ablations before live solves              |
| Make retrieval and retry budgets adaptive       | Avoid paying V2's full cost on easy issues                                        | Resolve-versus-token/latency Pareto curves                       |
| Feed diagnoses back into retrieval              | Let retries search for evidence supporting a new hypothesis                       | Recovery rate and expected-file discovery after failure          |
| Strengthen patch controls                       | Add diff-size budgets, touched-file policies, and optional AST-aware edits        | Regression rate and localization precision                       |
| Add CPU, memory, and PID limits                 | Complete the sandbox resource boundary                                            | Adversarial sandbox integration tests                            |
| Broaden benchmark coverage                      | Add more harness-sensitive tasks and a small SWE-bench Verified integration smoke | Pre-registered multi-seed, cross-task confidence intervals       |
| Add provider/model adapters                     | Separate harness effects from one model's behavior                                | Identical matrices across multiple models                        |
| Package the CLI and add CI                      | Make setup and regression testing repeatable                                      | Clean-install tests and automated unit/Docker checks             |
| Route tools through MCP as a controlled variant | Measure interoperability overhead instead of assuming parity                      | Direct-tool versus MCP latency, failures, and resolve            |


Longer-term experiments could add call-graph context, patch ranking, or
specialized planner/critic roles, but they should be introduced one at a time
and ablated against the simpler harnesses.

## Development and tests

Run the fast harness suite:

```powershell
python -m pytest tests -q -m "not docker"
```

Run the live Docker integration module after building the image:

```powershell
python -m pytest tests/test_sandbox_docker.py -m docker
```

Reproduce the Phase 7 ablation:

```powershell
python cli.py retrieve issue-015 --embedder hashing
python cli.py retrieve issue-016 --embedder hashing

python cli.py bench --split ablation --harness v0,v1,v2 --n 1 --log
python cli.py report --split ablation `
  --base-commit f7dbd4000d94dc5aab3835698dc3cb3bbd3eabc7 `
  --latest-per-cell
```

## Project documentation

- `[docs/project-plan.md](docs/project-plan.md)` — original seven-day plan and
design goals.
- `[docs/PHASE1.md](docs/PHASE1.md)` — V0 ReAct harness and initial evaluator.
- `[docs/PHASE2.md](docs/PHASE2.md)` — V1 LangGraph Plan-Execute workflow.
- `[docs/PHASE3.md](docs/PHASE3.md)` — Docker sandbox and permissions.
- `[docs/PHASE3-enhance-tests.md](docs/PHASE3-enhance-tests.md)` — hidden-gold
benchmark integrity.
- `[docs/PHASE4.md](docs/PHASE4.md)` and
`[docs/PHASE4-architecture.md](docs/PHASE4-architecture.md)` — V2 retrieval.
- `[docs/PHASE4-followup.md](docs/PHASE4-followup.md)` — retrieval alignment,
retry wiring, and reports.
- `[docs/PHASE5.md](docs/PHASE5.md)` — dual-layer verification and recovery.
- `[docs/PHASE6.md](docs/PHASE6.md)` — checkpointing, approval, traces, and MCP.
- `[docs/PHASE7.md](docs/PHASE7.md)` — controlled ablation, costs, and failure
analysis.

Each normal solve resets the subject repository, runs one harness in Docker,
scores hidden gold, and writes a run artifact. Keep experimental patches in
the generated run data, not in `issue-pilot-benchmark`.