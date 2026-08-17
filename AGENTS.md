# AGENTS.md

This repo is the **coding-agent harness** (agent, tools, sandbox, eval). The buggy FastAPI app is a **separate git repo**: `../issue-pilot-benchmark`. Keep that worktree at `eval/dataset.json` `base_commit`. Never commit an agent patch into the benchmark.

## Progress

Shipped work is recorded in `docs/PHASE*.md` (currently `PHASE1.md`–`PHASE5.md` plus `PHASE3-enhance-tests.md`, `PHASE4-followup.md`, and `PHASE4-architecture.md`). Read those before re-deriving status from code or chat. The week plan is `docs/project-plan.md`. Completed Phase 5 is Day 5 (retry / dual-layer eval / same-process feedback), not durable HITL. Checkpoint, approval HITL, and MCP remain Day 6.

## Easy to get wrong

- **Two test layers.** Visible tests are in the benchmark (`dataset.test_command`). Hidden gold is only in `eval/gold/`, copied to `tests/_gold/` for scoring, then deleted. Do not put gold, `GOLD:` comments, or the full spec into the benchmark or issue text. Gold `success` is the only benchmark resolve metric; it is not `workflow_passed` and not `recovery_success` (`retry_count > 0` and `workflow_passed`).
- **`harness_version` ≠ sandbox.** `v0` = ReAct, `v1` = LangGraph Plan-Execute, `v2` = V1 + hybrid retrieve + `search_code`. Docker (`issue-pilot-sandbox:py312`) is shared runtime. After verify, Layer 1 pass goes **evaluate**; Layer 1 fail or Layer 2 reject goes **diagnose → plan** while `retry_count < MAX_RETRY` (2 diagnose increments = 1 automatic retry). Then `feedback` (opt-in `python cli.py solve <id> --interactive-recovery`); no provider / blank / second request → `needs_human`. `compare` does not pass that flag.
- **Ablation.** `python cli.py compare <id>` still runs V0 then V1 only. V2 is opt-in: `python cli.py solve <id> --harness v2`. Prefer hard split `issue-008`–`011`. Smoke is `issue-001`–`007`. Gold pass/fail is not the RAG metric; use `recall_at_5` / `python cli.py retrieve`. Grep ablation ranks `app/**/*.py` only (same corpus as the indexer); agent `grep_code` still searches the whole repo. Live V2 defaults to hashing + issue-only query; `retrieve` defaults to fastembed. Aggregate with `python cli.py report` (cohorts by `base_commit` / model / temperature / sandbox image; `recovery_rate` / `human_retries` are extra columns).
