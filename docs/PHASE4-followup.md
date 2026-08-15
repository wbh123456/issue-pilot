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
- Full V0/V1/V2 hard-split matrix
- Reranker / FAISS / Chroma

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
