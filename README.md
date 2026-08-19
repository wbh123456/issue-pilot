# IssuePilot

A coding-agent **harness**: ReAct and LangGraph workflows, a Docker sandbox, hybrid code retrieval, and a hidden-gold eval. The buggy FastAPI app is a **separate git repo** (`../issue-pilot-benchmark`). Never commit an agent patch into the benchmark.

Day-by-day records: [`docs/PHASE1.md`](docs/PHASE1.md)–[`docs/PHASE7.md`](docs/PHASE7.md). Week plan: [`docs/project-plan.md`](docs/project-plan.md). Phase 7 (ablation + this README) is the close of the week MVP.

## Setup

```powershell
cd issue-pilot
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # set DEEPSEEK_API_KEY

python cli.py sandbox doctor
python cli.py sandbox build
python -m pytest tests -q -m "not docker"
```

Sibling worktree: `../issue-pilot-benchmark` at `eval/dataset.json` `base_commit` (`f7dbd4000d94…`, Benchmark v4). Docker image: `issue-pilot-sandbox:py312` (`--network none`, repo mounted at `/workspace`).

```powershell
python cli.py solve issue-001
python cli.py solve issue-001 --harness v1
python cli.py solve issue-016 --harness v2
python cli.py retrieve issue-015 --embedder hashing
python cli.py bench --split ablation --harness v0,v1,v2 --n 1 --log
python cli.py report --split ablation --latest-per-cell
```

`compare` still runs **v0 then v1 only**. Approval HITL is opt-in (`solve --require-approval`). MCP is a stdio demo (`mcp serve` / `mcp demo`), not the live tool path.

---

## 1. Problem

Issue-solving agents are usually scored as a single blob: one model, one prompt, one “it worked.” That hides whether **harness** choices (plan/execute, verify, retrieve) change outcomes, or whether the score is just the model and the test suite.

IssuePilot keeps those knobs separate:

- Same model, temperature, sandbox image, and task budget across v0 / v1 / v2.
- Visible tests in the benchmark; **hidden gold** only in `eval/gold/` (the only resolve metric is gold `success`).
- Offline retrieval Recall@5 is **not** gold pass/fail.

The question for Day 7: which harness increment actually moves gold, at what token/latency cost.

## 2. Architecture

Two worktrees. The harness never indexes itself.

```text
cli.py
  solve     → reset benchmark to base_commit → SandboxRunner
                ├─ v0  ReAct loop            6 tools
                ├─ v1  LangGraph             analyze → plan → execute → verify
                └─ v2  v1 + retrieve         + search_code
  compare   → v0 then v1 only
  retrieve  → grep / bm25 / dense / hybrid   no LLM, app/**/*.py only
  bench     → split × harness × n            no approval / interactive-recovery
  report    → runs/*.json cohorts
        ▼
runs/{task}-{v0|v1|v2|retrieve}-{stamp}.json
```

| Side | Repo | Role |
|---|---|---|
| Harness | this repo | Agent, tools, sandbox policy, retrieval, scoring |
| Subject | `../issue-pilot-benchmark` | Reset every solve. Visible tests = `dataset.test_command`. Gold lives only in the harness. |

## 3. Agent / workflow design

**v0 — ReAct.** `agent/loop.py`, 15 steps, six tools: `list_files`, `read_file`, `grep_code`, `edit_file`, `run_tests`, `git_diff`. No verify node, no retrieve, no `search_code`.

**v1 — Plan-Execute.** Structured plan → execute (same six tools) → Layer 1 (pytest + ruff + non-empty patch). Fail or Layer 2 reject → diagnose → **replan** while `retry_count < 2`. Optional `--interactive-recovery` after the automatic budget. Optional `--require-approval` after both layers pass.

**v2 — v1 + RAG.** A deterministic `retrieve` node (0 LLM) between analyze and plan. Execute may call `search_code` (hybrid over `app/**/*.py`). Evaluator-reject does **not** re-retrieve.

```text
verify ── Layer 1 pass ──► evaluate
     │                        ├─ Layer 2 pass → await_approval → mark_success
     │                        └─ Layer 2 fail ────────────────► diagnose
     └─ Layer 1 fail ─────────────────────────────────────────► diagnose
                                                                  │
                                            retry_count < 2  → plan
                                            else             → feedback → needs_human
```

Gold `success` is independent of approve/reject and of `workflow_passed`. `recovery_success` is extra: `retry_count > 0` and `workflow_passed`.

## 4. Harness engineering

| Limit | Value |
|---|---|
| Sandbox | Docker only; no host pytest/ruff/git fallback |
| Network | container `--network none` |
| Tools | allowlisted argv (`pytest`, `ruff`, `mypy`, `git status/diff`); path-jailed reads/edits |
| Output | truncated tool payloads |
| Ruff autofix | `--fix` only on agent-touched `.py` files (whole-package fix was poisoning v1/v2 diffs) |
| Checkpoint | opt-in sqlite at `runs/checkpoints.sqlite`; resume **never** `reset_repo` |

## 5. Code RAG

Corpus: `app/**/*.py` AST chunks (module / class / function / method). BM25 + dense (default **hashing** on live v2; FastEmbed opt-in) fused with RRF (`k=60`), top 5 files.

- Offline: `python cli.py retrieve` (grep vs bm25 vs dense vs hybrid). Grep ranks `app/**/*.py` by token hit count so it matches the indexer; agent `grep_code` still walks the whole repo.
- Live v2 query defaults to **issue text only**, same as `retrieve`.
- `search_code` is a starting locator when user words ≠ identifiers—not “only after grep misses.”
- Gold pass/fail is not the RAG metric; use `recall_at_5`.

Detail: [`docs/PHASE4-architecture.md`](docs/PHASE4-architecture.md).

## 6. Evaluation methodology

Dataset (`eval/dataset.json`), all `base_commit` = Benchmark v4:

| Split | Tasks | Role |
|---|---|---|
| smoke | issue-001–007 | Easy/medium sanity |
| hard | issue-008–014 | Cross-file / storefront; gold often a second cut |
| ablation | issue-015–017 | Layering: two retrieve-sensitive, one retry-sensitive |

Controls for `bench`: same model (`deepseek-v4-flash`), T=0, same image, `max_steps=15`, hashing + issue query on v2, n=1. `report` cohorts by `base_commit` / model / temperature / image / `benchmark_spec_sha`. `--latest-per-cell` keeps one solve per (task, harness).

Do **not** raise n to 3 to hunt a ranking. Token cost is already the trade-off.

## 7. Results

Ablation, n=1, 9 solves (`runs/matrix-20260819T111828Z.json`). Gold only.

| Task | v0 | v1 | v2 |
|---|---|---|---|
| issue-015 (retrieve) | fail | fail | fail |
| issue-016 (retrieve) | fail | fail | **pass** |
| issue-017 (retry) | **pass** | **pass** | **pass** |
| **resolve** | **0.33** | **0.33** | **0.67** |

That is **v2 > v1 = v0**, not a three-way ladder. Hard 008–014 stayed tied at **0.43** on Benchmark v3 (21 solves). A four-task v4 rerun (008/011/013/014) stayed a second-cut ceiling except one n=1 011-v2 pass.

Patch precision (changed ∩ expected / changed) after scoping ruff `--fix`: v1/v2 **0.26 → 1.00** on those four hard tasks. That is a harness bugfix, not an LLM jump.

Retrieval (hashing, issue query) on the ablation split:

| Task | grep | bm25 | dense | hybrid | v2 `search_code` |
|---|---|---|---|---|---|
| 015 | 0.00 | 0.00 | 0.00 | **1.00** | 0 |
| 016 | 0.00 | 0.00 | 1.00 | **1.00** | **1** |
| 017 | 1.00 | 0.50 | 0.50 | 0.50 | 0 |

**Cost:** v0 still wins tokens and latency (ablation ~93k / 63s vs v1 ~112k / 183s vs v2 ~306k / 296s). Full tables: [`docs/PHASE7.md`](docs/PHASE7.md).

## 8. Failure analysis

Do not dump every miss on “the model.”

| Failure | Where | Kind |
|---|---|---|
| 015 all harnesses | `rate_for` still falls back to shop tax for `remote` | **Patch / hypothesis** — retrieve can surface `tax.py`; the one-line fallback still was not applied. |
| 015/016/011 v1 | `PlanValidationError`: `files_to_inspect` > 8 | **Planning / schema** — decoy nouns in issue text (needed so hashing hybrid beats BM25@5) overflowed the structured plan cap. Empty tree, tokens=0, still a gold fail. |
| 016 v0 | Never landed the `hours_notice` / receipt skip | **Retrieval + patch** — grep/BM25 miss `notifications.py`; v0 has no hybrid retrieve. |
| 016 v2 pass | `search_code` once, then retry | Retrieval increment that also moved gold. |
| 017 all pass | Two-file digital shipping | **Not a v1-only win** — v0 ran tests and submitted both files. Verify did not uniquely save the cell. |
| 008 / 014 all harnesses | Gold asserts a second cut (idempotency payload scope / refunded rows) | **Benchmark ceiling** — retry only sees Layer 1/2, never gold. |
| `recovery_rate` = 0 | Gold-pass after retry still `workflow_passed` false | **Metric vs Layer 2** — `recovery_success` requires workflow pass; Layer 2 can reject a gold-correct patch. |

Environment (Docker / ruff EXE bits) was not the ablation miss. Pre-fix, whole-package ruff `--fix` *was* an environment-shaped self-harm on v1/v2 diffs.

## 9. Future work

- Relax or clip `files_to_inspect` so v1 does not crash on decoy-heavy issues (or stop stuffing decoys into issue text).
- Decide whether `recovery_success` should follow gold or Layer 2.
- Hard gold second cuts are a dataset problem if the goal is harness ranking, not “can the model finish the hidden assertion.”
- Stretch from the week plan (not done): 1–2 SWE-bench Verified tasks as an integration smoke, not a full leaderboard.
- MCP remains a demo; do not put live v1/v2 behind it without a separate ablation.

## Reproduce Phase 7 numbers

```powershell
python -m pytest tests -q -m "not docker"

python cli.py retrieve issue-015 --embedder hashing
python cli.py retrieve issue-016 --embedder hashing

python cli.py bench --split ablation --harness v0,v1,v2 --n 1 --log
python cli.py bench --split hard --tasks issue-008,issue-011,issue-013,issue-014 --harness v0,v1,v2 --n 1 --log

python cli.py report --split ablation --base-commit f7dbd4000d94dc5aab3835698dc3cb3bbd3eabc7 --latest-per-cell
```

Each **solve** resets the benchmark, runs one harness in Docker, scores hidden gold, then writes `runs/*.json`. Do not commit those patches into `issue-pilot-benchmark`.
