# Phase 4 follow-up — RAG alignment, sandbox git, retry wire, report

Date: 2026-08-15  
Harness: `issue-pilot`  
Benchmark (sibling repo): `issue-pilot-benchmark`  
`base_commit`: `4b2258b1d16e802aa9b4a82bcb4a2b0f3911f84c` (11 tasks)

Post-Phase 4 hygiene. This is **not** planned Phase 5 (Reliability + HITL). It wires Day 5 Tasks 1–2 (diagnose → retry) plus eval/RAG fixes. It is **not** HITL, checkpoint, MCP, reranker, or an LLM evaluator.

## Goal

1. Make live V2 retrieval comparable to offline `retrieve`
2. Make `git_diff` work inside the Docker sandbox
3. Wire diagnose → execute with `MAX_RETRY`
4. Stop mixing provenance when reading `runs/*.json`

## What we built

### RAG alignment

| Change | Detail |
|---|---|
| Grep ablation | `rank_files_by_grep` keeps `app/**/*.py` only (`INDEX_DIR`). Agent `grep_code` is unchanged (whole repo). |
| Query | Default search string is **issue-only**. Opt-in `--query-mode issue+analysis` for live V2 and `retrieve`. |
| Embedder | `solve --harness v2 --embedder hashing\|fastembed` (default **hashing**). `retrieve` still defaults to **fastembed**. |
| Persist | V2 run JSON stores `embedder_name`, `query_mode`, `retrieve_query`. |

### Docker git

Container start sets `GIT_OPTIONAL_LOCKS=0` and `safe.directory=*`. Image also writes those into the sandbox user gitconfig. Live test: host `git init` + commit + edit, then `git_diff` inside the container.

### Retry

`retry_count` starts at 0. Diagnose increments it. If `retry_count >= MAX_RETRY` (2) → `needs_human`, else re-execute. Effect: **1 initial execute + 1 retry**, then escalate. Diagnosis is injected into the executor `workflow_context`. Trajectories concatenate across attempts.

### Report and deps

- `python cli.py report` (`eval/report.py`) groups solve records by `base_commit`, `model`, `temperature`, `sandbox_image`. Retrieval artifacts stay a separate table.
- Solve records persist `temperature=0` and `harness_git_sha` (via `eval/repository.py:git_sha`).
- `requirements.txt` uses version ranges (not a lockfile).

## Explicitly out of scope

- Planned Phase 5 (Reliability + HITL)
- LLM-as-judge / failure classifier
- HITL, checkpoint/resume, MCP
- Putting V2 into `compare` (still V0 then V1)
- Reranker / FAISS / Chroma

## Hard-split matrix (issue-008–011)

One pass each of V0 / V1 / V2 under identical settings. `compare` is still V0 then V1 only; these were opt-in `solve` runs.

| Setting | Value |
|---|---|
| `base_commit` | `4b2258b1d16e802aa9b4a82bcb4a2b0f3911f84c` |
| Model | `deepseek-v4-flash` |
| Temperature | 0 |
| Sandbox | `issue-pilot-sandbox:py312` |
| Harness SHA | `94af8f97` (live progress commit) |
| V2 | `--embedder hashing --query-mode issue` |
| n | 4 tasks × 3 harnesses (n=1 per cell) |

Gold resolve: **4/4 on every harness**. No diagnose retry (`retry_count=0` on all V1/V2). This cohort cannot show a resolve-rate gap.

| Task | V0 | V1 | V2 | V2 Recall@5 |
|---|---|---|---|---|
| issue-008 | pass, 39k tok, 29s | pass, 73k, 61s | pass, 56k, 64s | 1.00 |
| issue-009 | pass, 42k, 36s | pass, 48k, 55s | pass, 59k, 61s | 0.50 |
| issue-010 | pass, 98k, 61s | pass, 61k, 62s | pass, 122k, 127s | 0.50 |
| issue-011 | pass, 91k, 55s | pass, 85k, 123s | pass, 105k, 144s | 0.67 |

Means (n=4):

| Harness | Resolve | Tokens | Reads | Tools | Latency s | Retries |
|---|---|---|---|---|---|---|
| v0 | 1.00 | 67k | 8.5 | 18.0 | 45 | — |
| v1 | 1.00 | 67k | 9.5 | 17.5 | 75 | 0 |
| v2 | 1.00 | 86k | 8.8 | 17.5 | 99 | 0 |

Run files: `runs/issue-00{8,9,10,11}-v{0,1,2}-20260815T13*.json` (008-v0 starts `20260815T125941Z`).

Observations:

1. **Resolve is saturated.** All 12 gold tests passed, so this split does not measure whether retry or RAG helps when the first patch is wrong.
2. **V1 is not cheaper.** Mean tokens match V0; wall clock is higher (analyze + plan + verify).
3. **V2 costs more and localizes unevenly.** Live hashing Recall@5 is 1.00 on 008, 0.50 on 009/010, 0.67 on 011. That is not the offline FastEmbed hard-split table (hybrid 1.00). Mixing those numbers would over-claim.
4. **Retry never fired.** Visible tests passed on the first execute, so Day 5 wiring was unused in this matrix.

```powershell
# Reproduce this cohort (V2 hashing + issue-only query)
foreach ($id in 008,009,010,011) {
  python cli.py solve "issue-$id" --harness v0
  python cli.py solve "issue-$id" --harness v1
  python cli.py solve "issue-$id" --harness v2 --embedder hashing --query-mode issue
}
python cli.py report --split hard --base-commit 4b2258b1d16e802aa9b4a82bcb4a2b0f3911f84c
```


## Test evidence

`python -m pytest tests -q -m "not docker"` → **194 passed**, 2 skipped (Windows symlink). Docker live suite skipped here (daemon not up); unit tests assert git env on `docker run`. Re-run `pytest tests/test_sandbox_docker.py -m docker` after `sandbox build` if the image predates the gitconfig lines.

## How to reproduce

```powershell
cd issue-pilot
python -m pytest tests -q -m "not docker"
python -m pytest tests/test_sandbox_docker.py -m docker -q
python cli.py report --split hard
python cli.py solve issue-009 --harness v2 --embedder hashing --query-mode issue
```
