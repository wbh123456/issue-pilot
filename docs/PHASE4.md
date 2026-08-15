# Phase 4 — Code RAG Completion Record

Date: 2026-08-15  
Harness: `issue-pilot`  
Benchmark (sibling repo): `issue-pilot-benchmark`  
LLM: DeepSeek (`deepseek-v4-flash`) via OpenAI-compatible SDK  
Harness versions: **V0** (ReAct), **V1** (LangGraph Plan-Execute), **V2** (V1 + hybrid code RAG)  
Sandbox: `issue-pilot-sandbox:py312` (`--network none`, benchmark mounted at `/workspace`)  
`base_commit`: `4b2258b1d16e802aa9b4a82bcb4a2b0f3911f84c`

## Goal

Add a **V2** harness that is V1 plus hybrid retrieval, without changing V0/V1 or `compare`:

1. AST chunker + BM25 + dense + RRF over `app/**/*.py`
2. Offline Recall@5 ablation (`grep` vs `bm25` vs `dense` vs `hybrid`) on the hard split
3. V2 graph: `analyze → retrieve → plan → execute → verify`
4. V2-only `search_code` tool; `python cli.py solve --harness v2`

Gold pass/fail is **not** the RAG result. The DoD is the Recall@5 table, especially **issue-009** (`app/inventory.py` + `app/orders.py`).

## Current agentic workflow architecture

```text
CLI  solve / compare / retrieve
        │
        ▼
eval/runner.solve_task          or     eval/retrieval.run_retrieval_eval
        │                                    │
        ├─ reset_repo                        ├─ reset once per (repo, base_commit)
        └─ SandboxRunner                     └─ index app/**/*.py; no LLM
                │
                ├─ V0  run_agent             ReAct, 6 tools
                ├─ V1  run_workflow          analyze → plan → execute → verify
                └─ V2  get_v2_graph()        analyze → retrieve → plan → execute → verify
                                             + search_code during execute
        ▼
runs/{task}-{v0|v1|v2|retrieve}-{stamp}.json
```

### V0 vs V1 vs V2 (control flow)

```text
V0 ReAct (unchanged)
  for step in 1..MAX_AGENT_STEPS:
      LLM(+6 tools) → execute_tool* → append result

V1 LangGraph (unchanged)
  START → analyze → plan → execute → verify
                                      ├─ PASS → END
                                      └─ FAIL → diagnose → END

V2 = V1 + retrieve + search_code
  START → analyze → retrieve → plan → execute → verify
                         │                 │
                         │                 └─ V2_TOOLS (6 + search_code)
                         └─ hybrid, 0 LLM, HashingEmbedder default
```

`compare` still loops `("v0", "v1")` only.

### Retrieval (host-side, same trust class as `grep_code`)

Path-jailed, `app/**/*.py` only. Skips `_gold` / `_app_bak`. No `subprocess`. No FAISS/Chroma.

| Piece | Location | Notes |
|---|---|---|
| Chunker | `retrieval/chunker.py` | AST `module` / `class` / `function` / `method` |
| Lexical | `retrieval/lexical.py` | BM25Okapi (`rank-bm25`) |
| Dense | `retrieval/dense.py` | numpy cosine; `HashingEmbedder` (tests + live V2) / `FastEmbedEmbedder` (eval CLI) |
| Hybrid | `retrieval/hybrid.py` | RRF `k=60`, then top-K |
| Index | `retrieval/indexer.py` | In-memory per worktree |
| Limits | `harness/context.py` | `RETRIEVE_K=5`, `MAX_CHUNK_CHARS=4000`, `MAX_PLANNER_CONTEXT_CHARS=8000` |
| Metric | `eval/metrics.py` | File-level `recall_at_k` after collapsing chunks |

Query for the **eval CLI** = raw `dataset.issue`. Query for the **V2 retrieve node** = `issue + analysis`.

`grep` in the ablation is **not** BM25: tokenize the issue, call existing `grep_code` per token, rank files by hit count.

## What we built

| Area | Location | Notes |
|---|---|---|
| Retrieve node | `agent/nodes/retrieve.py` | Hybrid; `stage_tokens.retrieve.llm_calls = 0` |
| Graph | `agent/graph.py` | `build_graph(include_retrieve=False)` is V1; `get_v2_graph()` is V2 |
| Planner | `agent/nodes/plan.py` | Uses retrieved snippets when present; still drops invented paths |
| Execute | `agent/nodes/execute.py` | Snippets in `workflow_context`; `V2_TOOLS` only if `enable_search_code` |
| `search_code` | `agent/tools/search.py` | Hybrid; gated in `execute_tool` (`unknown tool` unless enabled) |
| Tool schemas | `agent/tools/schema.py` | `TOOLS` stays 6; `V2_TOOLS = [*TOOLS, SEARCH_CODE_TOOL]` |
| Runner | `eval/runner.py` | `HarnessVersion` includes `v2`; persist `retrieval_mode`, `retrieval_calls`, `relevant_files`, `recall_at_5` |
| Retrieve CLI | `eval/retrieval.py`, `cli.py` | `python cli.py retrieve issue-009` / `--split hard` |
| Tests | `tests/test_retrieval.py`, `test_eval_retrieval.py`, `test_graph_v2.py`, plus runner/tool gating | Default pytest uses `HashingEmbedder` only |

## Eval results (DoD)

### 1. Hard-split Recall@5 (no LLM)

Command: `python cli.py retrieve --split hard`  
Embedder: `FastEmbedEmbedder` (`qdrant/bge-small-en-v1.5-onnx-q`)  
`k=5`. Artifacts: `runs/issue-00{8,9,10,11}-retrieve-20260815T06262{5,6}Z.json`.

| Task | Gold files | grep | bm25 | dense | hybrid |
|---|---|---|---|---|---|
| issue-008 | `app/orders.py` | 1.00 | 1.00 | 1.00 | 1.00 |
| **issue-009** | `app/inventory.py`, `app/orders.py` | **0.50** | **0.50** | **1.00** | **1.00** |
| issue-010 | `app/auth.py`, `app/users.py` | 1.00 | 1.00 | 1.00 | 1.00 |
| issue-011 | `app/orders.py`, `app/pricing.py`, `app/payments.py` | 0.67 | 1.00 | 1.00 | 1.00 |
| **mean** | | **0.79** | **0.88** | **1.00** | **1.00** |

Issue-009 is the interesting row: lexical modes find `inventory.py` and miss `orders.py`; dense/hybrid return both.

### 2. Live V2 solve vs V1 (issue-009)

Same model, `max_steps=15`, image `issue-pilot-sandbox:py312`, same `base_commit`.

| Metric | V1 (`…v1-20260813T145712Z.json`) | V2 (`…v2-20260815T112324Z.json`) |
|---|---|---|
| Gold success | True | True |
| Workflow passed | True | True |
| Steps | 10 | 10 |
| LLM calls | 12 | 12 |
| Tool calls | 21 | 12 |
| File reads | 12 | 6 |
| Tokens | 56,913 | 51,401 |
| Latency | ~55.3s | ~42.2s |
| `retrieval_mode` | — | hybrid |
| `retrieval_calls` | — | 1 (retrieve node only) |
| `relevant_files` | — | `app/payments.py`, `app/inventory.py` |
| `recall_at_5` | — | **0.50** |
| `search_code` used | n/a | **no** |
| Plan `files_to_inspect` | `orders.py`, `inventory.py`, `main.py`, `validators.py` | `inventory.py`, `payments.py` |

Gold True on both sides is **not** a RAG win. The RAG number on this live run is `recall_at_5 = 0.50`: hashing retrieve found `inventory.py` (the crash site) and a distractor (`payments.py`), and missed gold file `app/orders.py`.

That 0.50 is expected given the default live embedder. The Step 2 table used FastEmbed; the V2 retrieve node defaults to `HashingEmbedder` (no model download). Offline FastEmbed hybrid on the same issue was 1.00.

Executor tools on V2: `read_file`×6, `list_files`×3, `edit_file`×1, `run_tests`×1, `git_diff`×1. No `grep_code`, no `search_code`. The agent still opened `app/orders.py` and `app/main.py` after listing `app/`, so retrieval did not fully constrain exploration.

The V1 baseline’s `git_diff` failed with exit 129 (container had no git metadata yet); V2’s `git_diff` returned status. Some of V1’s extra reads may be recovery from that, not only missing RAG. A later V1 on the same task (`…v1-20260814T155317Z.json`, git_diff working) still used 11 reads / 87k tokens.

Retrieve added **0** LLM calls (`stage_tokens.retrieve.llm_calls = 0`). Planner tokens rose (snippets in the plan prompt: 1,064 vs V1 617 prompt tokens) while execute tokens fell.

## Test evidence

`python -m pytest tests -q -m "not docker"` → **180 passed**, 2 skipped (Windows symlink).

| Suite | Role |
|---|---|
| `tests/test_retrieval.py` | Chunk ranges, `_gold` not indexed, BM25/RRF, hashing-only |
| `tests/test_eval_retrieval.py` | Four modes; grep ≠ BM25 |
| `tests/test_graph_v2.py` | V2 topology; retrieve 0 LLM; `search_code` gated on execute |
| `tests/test_graph.py` | V1 still `analyze → plan`; V0 tools stay 6 |
| `tests/test_runner.py` | Accepts v2, rejects v3; `compare` still v0 then v1; persist `recall_at_5` |
| `tests/test_tools.py` | `search_code` unknown unless enabled; hybrid hit on `app/*.py` |

## Observations

1. **Hybrid vs grep is a localization result, not a resolve-rate result.** On 009, FastEmbed hybrid Recall@5 is 1.00 vs grep 0.50. Both V1 and V2 gold-passed by adding a 409 check in `allocate_bin`.
2. **Eval embedder ≠ live V2 embedder.** FastEmbed is the ablation; hashing is the default agent path. Mixing them would over-claim the live `recall_at_5`.
3. **Planner follows retrieved paths, including distractors.** Hashing top-K included `payments.py`; the structured plan inspected exactly those two files.
4. **`search_code` being available is not the same as being used.** The model never called it on this run.
5. **`compare` is still V0 then V1.** V2 is opt-in via `solve --harness v2`.

## Explicitly out of scope (Phase 4)

- Retry / replan (Day 5); `MAX_RETRY` unused; diagnose still ends the graph
- Changing V0/V1 behavior or putting V2 into `compare`
- Reranker, FAISS/Chroma, MCP, HITL, SWE-bench
- `search_symbol` (stretch after this write-up)
- Benchmark repo edits; gold files in the index

## How to reproduce

```powershell
cd issue-pilot
copy .env.example .env   # set DEEPSEEK_API_KEY
python -m pip install -r requirements.txt

# Docker Desktop must be running in Linux-container mode.
python cli.py sandbox doctor --require-image

python -m pytest tests -q -m "not docker"

# Offline Recall@5 (no LLM). First FastEmbed run may download the ONNX model.
python cli.py retrieve --split hard
python cli.py retrieve issue-009

# Live V2 (does not change compare)
python cli.py solve issue-009 --harness v2

# Ablation is still V0 then V1
python cli.py compare issue-009
```

Each solve: reset benchmark → one sandbox → harness → gold → cleanup. Retrieve eval resets once per `(repo, base_commit)` and does not start a container.
