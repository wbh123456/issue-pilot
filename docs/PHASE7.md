# Phase 7 — Benchmark, Ablation, README

Date: 2026-08-19  
Harness: `issue-pilot`  
Benchmark (sibling repo): `issue-pilot-benchmark` **v4** `f7dbd4000d94dc5aab3835698dc3cb3bbd3eabc7`  
LLM: DeepSeek (`deepseek-v4-flash`) via OpenAI-compatible SDK  
Harness versions: **V0** ReAct. **V1** Plan-Execute + verify + retry. **V2** V1 + hybrid retrieve + `search_code`.  
Sandbox: `issue-pilot-sandbox:py312` (`--network none`, benchmark mounted at `/workspace`)  
Spec (ablation + hard four-task rerun): `eb78bbfb5274…`  
Matrices: `runs/matrix-20260819T111828Z.json` (ablation, 9), `runs/matrix-20260819T115157Z.json` (hard 008/011/013/014, 12)

This is **Day 7** in `docs/project-plan.md`. Rule: **no major new product surface.** The day proves whether harness increments move gold, at what cost. Public write-up: [`README.md`](../README.md). Layering tables also sat in [`PHASE7-layering.md`](PHASE7-layering.md) while the 21 solves ran.

Gold `success` is the only resolve metric. It is not `workflow_passed` and not `recovery_success`.

## Goal

1. Run a controlled v0 / v1 / v2 matrix (same model, T=0, image, step budget).
2. Report resolve, tokens, tools, reads, retries, plus localization / Layer 1 / `search_code`.
3. Classify failures without “the LLM was not smart enough.”
4. Ship a README that can answer the Day 7 DoD questions.

Definition of Done (from the week plan):

```text
Which harness increment helped most?
Why?
What did it cost?
Did resolve rise?
Did tokens / file reads fall?
What is the trade-off?
```

## What we built (Day 7 only)

Not a new graph. Four eval-integrity fixes so the 21 solves were readable, then Benchmark v4 so retrieve/retry had a place to show up.

| Area | Location | Notes |
|---|---|---|
| Ruff `--fix` scope | `agent/nodes/verify.py` | Only agent-touched `.py` files. Whole-package `--fix app` had dirtied v1/v2 diffs (`auth.py` / `main.py` / `settings.py`) and fake-`BAD_PATCH`ed Layer 2. |
| `search_code` schema | `agent/tools/schema.py` | Starting locator when user words ≠ identifiers. |
| `bench --tasks`, split `ablation` | `eval/matrix.py`, `cli.py` | Keep the budget at 9 + 12, not 30. |
| Derived report columns | `eval/report.py` | `localization_precision`, `layer1_gate_rate`, `search_code_calls`, `first_expected_read_step` |
| Benchmark v4 | sibling repo | `issue-015`–`017`, `split: ablation`. Gold is one cut. Hashing prescreen: 015/016 grep=0, bm25=0, hybrid=1.0. |
| README | `README.md` | Problem → future work. |

`compare` is still v0 then v1. Approval HITL and MCP are unchanged from Phase 6.

## Results

### Gold resolve (ablation, n=1)

Expected v2 > v1 > v0. Got **v2 > v1 = v0**.

| Task | v0 | v1 | v2 | Note |
|---|---|---|---|---|
| issue-015 (retrieve) | fail | fail | fail | Missed `rate_for` fallback. v1: `PlanValidationError` (`files_to_inspect` > 8). |
| issue-016 (retrieve) | fail | fail | **pass** | v2 called `search_code` once, gold-passed after retry. v1: same plan-cap crash. |
| issue-017 (retry) | **pass** | **pass** | **pass** | v0 submitted both files. Not a v1-only win. |
| **resolve rate** | **0.33** | **0.33** | **0.67** | |

Hard 008/011/013/014 on v4 (same gold as v3, four-task subset):

| Harness | v3 (7-task hard) | v4 (008/011/013/014) |
|---|---|---|
| v0 | 0.43 | 0.25 |
| v1 | 0.43 | 0.25 |
| v2 | 0.43 | **0.50** (011 gold-passed this seed) |

008 and 014 still fail every harness (second-cut gold). That is a **narrow-patch ceiling**, not a harness ranking. Do not treat the 011-v2 pass as architecture proof; n=1.

### Patch precision and recovery (v3 vs v4)

`localization_precision` = changed files ∩ `expected_files` / changed files. Empty diffs omitted.

| Split / cohort | Harness | LocPrec | Recovery | Retries | L1 fail-at-delivery |
|---|---|---|---|---|---|
| hard v3 (7 tasks, ruff `--fix app`) | v0 | **1.00** | 0.00 | 0.00 | 0.00 |
| | v1 | **0.26** | 0.00 | 0.86 | 0.00 |
| | v2 | **0.26** | 0.00 | 0.86 | 0.00 |
| hard v4 (008/011/013/014, scoped `--fix`) | v0 | **1.00** | 0.00 | 0.00 | 0.00 |
| | v1 | **1.00** | 0.00 | 1.50 | 0.25 |
| | v2 | **1.00** | 0.00 | 2.00 | 0.00 |
| ablation v4 | v0 | **1.00** | 0.00 | 0.00 | 0.67 |
| | v1 | **1.00** | 0.00 | 0.67 | 0.67 |
| | v2 | **1.00** | 0.00 | 2.00 | 0.33 |

The 0.26 → 1.00 jump is the ruff-scope fix. `recovery_rate` stays 0: the field needs `retry_count > 0` **and** `workflow_passed`. Several gold passes after retry still have Layer 2 fail. v1 L1-fail 0.25 on hard v4 is issue-011’s empty tree after `PlanValidationError`.

### Retrieval (`search_code` is v2-only)

Hashing, `query_mode=issue` (live V2).

| Task | grep | bm25 | dense | hybrid | v2 `search_code` calls | first expected read (v0 / v1 / v2) |
|---|---|---|---|---|---|---|
| issue-015 | 0.00 | 0.00 | 0.00 | **1.00** | 0 | 3 / — / 1 |
| issue-016 | 0.00 | 0.00 | 1.00 | **1.00** | **1** | 4 / — / 1 |
| issue-017 | 1.00 | 0.50 | 0.50 | 0.50 | 0 | 3 / 2 / 1 |

v3 hard v2: `search_code_calls` = 0 on all 7 tasks. After the schema change, 016-v2 and 014-v2 each called it once.

### Cost

Token and latency still favor v0.

| Cohort | v0 tokens | v1 tokens | v2 tokens | v0 latency | v1 latency | v2 latency |
|---|---|---|---|---|---|---|
| ablation | 93k | 112k | 306k | 63s | 183s | 296s |
| hard v4 (4 tasks) | 57k | 179k | 278k | 32s | 253s | 423s |
| hard v3 (7 tasks) | 54k | 126k | 158k | 29s | 145s | 183s |

v1 ablation/hard means are pulled down by three `PlanValidationError` zeros (015-v1, 016-v1, 011-v1). Those runs spent almost no tokens and still count as gold failures. File reads on ablation **rose** on v2 (37 vs v0 13), not fell.

## Failure analysis

| Cell | Kind | What happened |
|---|---|---|
| 015 all | Patch / hypothesis | Hybrid can rank `app/tax.py`; nobody applied the `remote` / shop-rate fallback. |
| 015/016/011 v1 | Planning / schema | `files_to_inspect` max 8 vs decoy storefront nouns in the issue (needed for hashing hybrid to beat BM25@5). |
| 016 v0 | Retrieval + patch | No hybrid retrieve; grep/BM25 miss `notifications.py`. |
| 016 v2 | Retrieval win | `search_code` + retry; only ablation cell where RAG also moved gold. |
| 017 all | Not v1-specific | v0 ran visible tests and edited both files. |
| 008 / 014 all | Benchmark ceiling | Gold second cut; diagnose never sees gold. |
| Recovery column | Metric vs Layer 2 | Gold-correct retry still `workflow_passed` false. |

Not: Docker down, ruff EXE sandbox noise (ignored), or “raise n=3.”

## DoD answers

- **Largest increment:** v2 retrieve/`search_code` on the ablation split (016). Scoped ruff `--fix` is the largest **measurement** fix (loc 0.26 → 1.00), not a resolve lift.
- **Why:** hashing hybrid finds 015/016 when grep and BM25 do not; 016 is the cell that also gold-passed.
- **Cost:** v2 is ~3× tokens and ~5× latency vs v0 on ablation. File reads went **up**.
- **Resolve:** up on ablation for v2 only (0.67 vs 0.33). Hard stayed a tie except one n=1 cell.
- **Tokens / reads:** not down. v0 wins the cost axes.
- **Trade-off:** pay tokens for a retrieve-sensitive win; do not claim v1 > v0 on this benchmark.

Reports: `runs/phase7-layering-ablation.json`, `runs/phase7-layering-hard-v4.json`, `runs/phase7-layering-hard-v3.json`.

## Explicitly out of scope (Phase 7 / Day 7)

- n=3 (or any extra LLM budget to force a v1 > v0 gap)
- Putting v2 or `--require-approval` into `compare`
- Routing live tools through MCP
- SWE-bench Verified (week-plan stretch; not run)
- Changing live v2 from hashing to FastEmbed to inflate Recall@5

## How to reproduce

```powershell
cd issue-pilot
copy .env.example .env   # set DEEPSEEK_API_KEY
python cli.py sandbox doctor --require-image
python -m pytest tests -q -m "not docker"

python cli.py retrieve issue-015 --embedder hashing
python cli.py retrieve issue-016 --embedder hashing

python cli.py bench --split ablation --harness v0,v1,v2 --n 1 --log
python cli.py bench --split hard --tasks issue-008,issue-011,issue-013,issue-014 `
  --harness v0,v1,v2 --n 1 --log

python cli.py report --split ablation --base-commit f7dbd4000d94dc5aab3835698dc3cb3bbd3eabc7 --latest-per-cell
```

Keep `../issue-pilot-benchmark` at Benchmark v4. Each solve resets that worktree. Do not commit agent patches into it.
