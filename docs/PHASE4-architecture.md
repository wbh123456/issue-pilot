# Phase 4 — V2 hybrid code RAG architecture

Date: 2026-08-17  
Companion to the Phase 4 completion record (`docs/PHASE4.md`) and RAG alignment notes (`docs/PHASE4-followup.md`).  
This is **not** Phase 5 (retry / dual-layer eval / feedback). Those nodes sit around this stack; they do not replace it.

**V2 = unchanged V1 + host-side hybrid retrieve + execute-only `search_code`.**  
`compare` still runs V0 then V1. Gold pass/fail is not the RAG metric; use file-level `recall_at_5` / `python cli.py retrieve`.

## Snapshot vs later graph

Phase 4 shipped this V2 control flow:

```text
START → analyze → retrieve → plan → execute → verify
                                      ├─ pass → END
                                      └─ fail → diagnose → END
```

`diagnose` was a terminal node. `MAX_RETRY` was unused. Later `agent/graph.py` added `evaluate`, `await_approval`, `diagnose → plan`, and `feedback`. The RAG increment stayed the same: a deterministic `retrieve` node between analyze and plan, plus a gated `search_code` tool on execute. V2 evaluator-reject still does **not** re-retrieve (`docs/PHASE5.md`).

## System boundary

Two git worktrees. The harness never indexes itself. Retrieval and tools run on the host against the benchmark repo; Docker (`issue-pilot-sandbox:py312`, `--network none`) runs tests.

```text
cli.py
  solve     → eval/runner.solve_task → reset + SandboxRunner
                ├─ V0  run_agent              ReAct, 6 tools
                ├─ V1  run_workflow           analyze → plan → execute → verify
                └─ V2  get_v2_graph()         + retrieve + search_code
  compare   → same runner; ("v0", "v1") only
  retrieve  → eval/retrieval.run_retrieval_eval
                reset once per (repo, base_commit); index app/**/*.py; no LLM; no container
        ▼
runs/{task}-{v0|v1|v2|retrieve}-{stamp}.json
```

| Side | Repo | Role |
|---|---|---|
| Harness | `issue-pilot` | CLI, graph, tools, `retrieval/`, scoring |
| Subject | `../issue-pilot-benchmark` | Reset to `dataset.json` `base_commit`. Visible tests = `test_command`. Hidden gold lives in `eval/gold/` only. |

## Retrieval pipeline

Host-side, same trust class as `grep_code`: path-jailed, no `subprocess`, no FAISS/Chroma. Corpus is `app/**/*.py` (skip `_gold` / `_app_bak` / VCS). Index is an in-memory `CodeIndex` built by `retrieval/indexer.py:build_index()`.

```text
app/**/*.py
    → AST chunker          module / class / function / method
    → index_text           path + symbol + type + body
         ├─ BM25Okapi      lexical, rank-bm25
         └─ dense cosine   HashingEmbedder or FastEmbedEmbedder
              → RRF (k=60) → top-K (RETRIEVE_K=5)
                   ├─ V2 retrieve node     (once per solve, 0 LLM)
                   ├─ search_code          (once per tool call, if the model asks)
                   └─ retrieve CLI         (grep / bm25 / dense / hybrid ablation)
```

| Layer | Location | Job |
|---|---|---|
| Chunk | `retrieval/chunker.py` | Symbol spans. Syntax error → whole file as `module`. Truncate at `MAX_CHUNK_CHARS=4000`. |
| Lexical | `retrieval/lexical.py` | BM25 over chunks, not whole files. Shares `retrieval/tokenize.py` with hashing. |
| Dense | `retrieval/dense.py` | One vector per chunk; query is also one vector; numpy cosine. |
| Hybrid | `retrieval/hybrid.py` | `score(d) = Σ 1/(60 + rank_i(d))`, then cut to 5. |
| Query | `retrieval/query.py` | Default **issue-only** (aligned with offline eval in the follow-up). Opt-in `issue+analysis`. |

There is **no index cache**. `retrieve_context` and `search_code` each call `build_index()`. A V2 solve that never calls `search_code` builds one `CodeIndex`. Each `search_code` call builds another. That is redundant work on an unchanged tree, but after `edit_file` a reused retrieve-time index would be stale. Offline `cli.py retrieve` builds **once** per `(repo, base_commit)` and reuses that object for all four modes.

### grep ablation is not BM25

`python cli.py retrieve` compares `grep` vs `bm25` vs `dense` vs `hybrid`. Grep tokenizes the issue, calls existing `grep_code` per token, ranks files by hit count, then keeps `app/**/*.py` so the corpus matches the indexer. Agent `grep_code` still searches the whole repo. If grep called BM25, the table would be meaningless.

## How the three rankers score

All three consume the same chunks. They differ in how a query is compared to a chunk.

**BM25 (lexical).** Tokenize the query; for each token add an IDF × TF term for that chunk; sum. Shared identifiers (`widget`, `stock`, `500`) rank high. Near-paraphrase with no shared tokens ranks low.

**Dense.** Embed the **whole** chunk into one vector and the **whole** query into one vector, then cosine. Not per-token accumulation at search time (that is BM25). Two embedders implement `embed()`:

| | `HashingEmbedder` | `FastEmbedEmbedder` |
|---|---|---|
| What it is | Signed feature hashing into 256 dims | `BAAI/bge-small-en-v1.5` via FastEmbed ONNX |
| Similar when | Shared tokens (or hash collisions) | Similar meaning |
| Network | None | May download; extra dependency |
| Default | `solve --harness v2`, unit tests | `python cli.py retrieve` (DoD table) |

Hashing uses the same `tokenize()` as BM25, then blake2b-buckets each token into the vector. FastEmbed does **not** use that tokenizer; the model consumes the raw string. Downstream `DenseIndex.search` is identical: one query vector, cosine against the chunk matrix.

**Hybrid.** Not a third encoder. Run BM25 and dense over the full chunk pool, fuse ranks with RRF, take top-5. A chunk that is weak on one list can still enter the fused top-K if it is strong on the other.

### Worked example: issue-009

Issue: `Ordering 50 widgets when 3 are in stock crashes with 500`.  
Gold files: `app/inventory.py` (crash in `allocate_bin`) and `app/orders.py` (`create_order` calls it). `orders.py` does not contain `widgets` / `stock` / `500`.

`app/payments.py:fail_closed_message` is a planted red herring (`HTTP 500` in a docstring). Lexical rankers put it first.

Offline FastEmbed (`runs/issue-009-retrieve-20260815T062625Z.json`, k=5):

| Mode | Top files (abbrev.) | Recall@5 |
|---|---|---|
| grep / bm25 | inventory + **payments**; **no orders** | 0.50 |
| dense | **orders**, main, inventory, … | 1.00 |
| hybrid | inventory, **payments**, **orders**, … | 1.00 |

Hybrid keeps the lexical hit (`inventory`) and the semantic hit (`orders`). It does not drop a distractor that both lists still like (`payments`). Live V2 defaults to **hashing**, so the same issue’s solve-time `recall_at_5` was 0.50 (missed `orders.py`). Do not cite the FastEmbed table as the live agent number (`docs/PHASE4.md` observation 2).

## How snippets reach the agent

`agent/nodes/retrieve.py` does not call the LLM. It writes `relevant_files` (unique paths, retrieval order) and `retrieved_context` (symbol snippets, truncated to `MAX_PLANNER_CONTEXT_CHARS=8000`). Telemetry `stage_tokens.retrieve.llm_calls = 0`. `retrieval_calls` starts at 1 for that node.

**Planner grounding.** V1 injects a directory inventory (`list_files` on `.` / `app` / `tests`). V2, when snippets or files are present, **replaces** that inventory in the prompt with the retrieved list + bodies (`prefer these over a directory listing`). The model then emits JSON including `files_to_inspect`.

**Hallucinated paths.** After parse, `_filter_existing_files` keeps only paths that exist in the worktree (`resolve_in_repo(...).exists()`). Invented paths such as `src/main/java/JwtFilter.java` are dropped. Real-but-wrong files (the `payments.py` distractor) are **not** dropped. Executor tools can still `list_files` / `read_file` outside the retrieved set.

**Execute.** The same snippet blob is copied into `workflow_context`. Tool surface: `TOOLS` stays six (`list_files`, `read_file`, `grep_code`, `edit_file`, `run_tests`, `git_diff`). `V2_TOOLS = [*TOOLS, SEARCH_CODE_TOOL]`. If `enable_search_code` is false, a `search_code` call returns `unknown tool`.

`search_code` is hybrid over `app/**/*.py`, not grep and not BM25-only. Schema text tells the model to use it when literal grep misses. Availability ≠ use: all seven live V2 solve files under `runs/*-v2-*.json` have **zero** `search_code` trajectory events. `retrieval_calls=1` on those files is the retrieve node. `grep_code` appeared 11 times across those runs. Hybrid retrieval that actually ran was the retrieve node, not the extra tool.

## Limits

| Constraint | Value |
|---|---|
| Corpus | `app/**/*.py` only |
| `RETRIEVE_K` | 5 |
| `RRF_K` | 60 (smoothing, not top-K) |
| `MAX_CHUNK_CHARS` | 4000 |
| `MAX_PLANNER_CONTEXT_CHARS` | 8000 |
| Retrieve LLM | 0 |
| Vector DB | none (numpy in memory) |

## Module map

| Path | Role |
|---|---|
| `retrieval/chunker.py` | AST chunks; `index_text()` |
| `retrieval/tokenize.py` | Identifier tokenizer for BM25 + hashing |
| `retrieval/lexical.py` | BM25Okapi |
| `retrieval/dense.py` | Embedders + cosine |
| `retrieval/hybrid.py` | RRF |
| `retrieval/indexer.py` | `CodeIndex` / `build_index()` |
| `retrieval/query.py` | Issue vs issue+analysis |
| `agent/nodes/retrieve.py` | V2 retrieve node |
| `agent/graph.py` | `include_retrieve=True` → V2 |
| `agent/tools/search.py` | `grep_code` (whole repo) and `search_code` (hybrid) |
| `agent/tools/schema.py` | `TOOLS` vs `V2_TOOLS` |
| `agent/nodes/plan.py` | Snippet grounding; drop missing paths |
| `agent/nodes/execute.py` | `workflow_context`; gate `search_code` |
| `eval/retrieval.py` | Four-mode Recall@K CLI |
| `eval/metrics.py` | File-level `recall_at_k` |
| `harness/context.py` | K / RRF / truncation constants |

## Reproduce

```powershell
python cli.py retrieve --split hard                  # FastEmbed default; RAG DoD
python cli.py retrieve issue-009 --embedder hashing  # live-V2-like dense
python cli.py solve issue-009 --harness v2           # hashing + issue-only query
python cli.py compare issue-009                      # still V0 then V1
```

Eval numbers and observations: `docs/PHASE4.md`. Query/embedder CLI flags and live hashing matrix: `docs/PHASE4-followup.md`. Recovery graph around this stack: `docs/PHASE5.md`.
