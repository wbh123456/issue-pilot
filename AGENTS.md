# AGENTS.md

This repo is the **coding-agent harness** (agent, tools, sandbox, eval). The buggy FastAPI app is a **separate git repo**: `../issue-pilot-benchmark`. Keep that worktree at `eval/dataset.json` `base_commit`. Never commit an agent patch into the benchmark.

## Progress

Shipped work is recorded in `docs/PHASE*.md` (currently `PHASE1.md`–`PHASE3.md` plus `PHASE3-enhance-tests.md`). Read those before re-deriving status from code or chat. The week plan is `docs/project-plan.md`.

## Easy to get wrong

- **Two test layers.** Visible tests are in the benchmark (`dataset.test_command`). Hidden gold is only in `eval/gold/`, copied to `tests/_gold/` for scoring, then deleted. Do not put gold, `GOLD:` comments, or the full spec into the benchmark or issue text.
- **`harness_version` ≠ sandbox.** `v0` = ReAct, `v1` = LangGraph Plan-Execute. Docker (`issue-pilot-sandbox:py312`) is shared runtime. **V2 is reserved for RAG** and is not built. `MAX_RETRY` is unused (diagnose does not re-execute).
- **Ablation.** `python cli.py compare <id>` runs V0 then V1. Prefer hard split `issue-008`–`011`. Smoke is `issue-001`–`007`.
