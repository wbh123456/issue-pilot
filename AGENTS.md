# AGENTS.md

This repo is the **coding-agent harness** (agent, tools, sandbox, eval). The buggy FastAPI app is a **separate git repo**: `../issue-pilot-benchmark`. Keep that worktree at `eval/dataset.json` `base_commit`. Never commit an agent patch into the benchmark.

## Progress

Shipped work is recorded in `docs/PHASE*.md` (currently `PHASE1.md`–`PHASE4.md` plus `PHASE3-enhance-tests.md`). Read those before re-deriving status from code or chat. The week plan is `docs/project-plan.md`.

## Easy to get wrong

- **Two test layers.** Visible tests are in the benchmark (`dataset.test_command`). Hidden gold is only in `eval/gold/`, copied to `tests/_gold/` for scoring, then deleted. Do not put gold, `GOLD:` comments, or the full spec into the benchmark or issue text.
- **`harness_version` ≠ sandbox.** `v0` = ReAct, `v1` = LangGraph Plan-Execute, `v2` = V1 + hybrid retrieve + `search_code`. Docker (`issue-pilot-sandbox:py312`) is shared runtime. `MAX_RETRY` is unused (diagnose does not re-execute).
- **Ablation.** `python cli.py compare <id>` still runs V0 then V1 only. V2 is opt-in: `python cli.py solve <id> --harness v2`. Prefer hard split `issue-008`–`011`. Smoke is `issue-001`–`007`. Gold pass/fail is not the RAG metric; use `recall_at_5` / `python cli.py retrieve`.
