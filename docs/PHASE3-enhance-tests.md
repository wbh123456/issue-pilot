# Phase 3 enhance tests — Benchmark v2 record

Date: 2026-08-13  
Harness: `issue-pilot`  
Benchmark (sibling repo): `issue-pilot-benchmark`  
LLM: DeepSeek (`deepseek-v4-flash`) via OpenAI-compatible SDK  
Harness versions: **V0** (ReAct) vs **V1** (LangGraph Plan-Execute), shared Docker sandbox  
This is a **benchmark / eval** change after Phase 3. Control flow is unchanged (no retry, no RAG).

Current buggy HEAD (`dataset.json` `base_commit`):

```text
4b2258b1d16e802aa9b4a82bcb4a2b0f3911f84c
```

---

## Goal

Phase 1–3 evals showed **V0 already resolving the original 8 tasks in one ReAct loop**. On that set, V1 Plan-Execute could not raise Resolve Rate; it only added analyze/plan/verify cost.

Project-plan Day 7 ablation needs tasks where:

```text
same model + different harness → different agent performance
```

This pass makes the benchmark usable for that question: strip answer leakage, hide gold tests from the agent, split smoke vs hard, and add four hard bugs that stress localization, incomplete first patches, and wrong hypotheses.

---

## Why the original 8 tasks were too easy

Phase 1 trajectories (`runs/issue-001|002|007|008-20260811*.json`) all succeeded with 6–13 ReAct steps. Root causes:

1. **Tiny repo** — five `app/*.py` files; first `grep` hit the target.
2. **Leaked specs** — README mapped issue → file; module docstrings stated the intended contract; gold tests were labeled `GOLD: issue-00X` and sometimes named the two functions to change.
3. **Issue text was a diagnosis** — e.g. “Expired JWT returns 500 instead of 401”.
4. **Neighbor leakage** — 001 and 006 shared `tests/test_auth.py`; running the module suite showed the other gold failure.
5. **One-line bugs** — catch the wrong JWT exception, `range(start, end)`, unused `_IDEM_KEYS` already declared.
6. **V1 diagnose → END** — `MAX_RETRY` unused. If the first patch worked, extra workflow nodes were pure overhead.

Day 1 DoD was “solve 1–2 easy bugs and save failure trajectories”. The dataset produced neither failures nor harness deltas.

---

## What we built

### Two-layer tests

```text
agent-visible (benchmark/tests/)          evaluator-only (issue-pilot/eval/gold/)
  symptom-level, one issue per file         complete spec + first-patch traps
  dataset.test_command                      runner.run_gold_test
```

Example (expired token):

- Visible: `assert status != 500`
- Hidden gold: `assert status == 401`

Gold modules live under `eval/gold/test_issue_00X.py`. Scoring copies one file into `tests/_gold/`, runs pytest, then deletes it. Agent `list_files` / `read_file` / `grep_code` skip `_gold` (path jail).

### Benchmark (`issue-pilot-benchmark`)

| Change | Detail |
|---|---|
| Leak strip | No README issue map, no `GOLD:` comments, no fix recipes in docstrings, no unused `_IDEM_KEYS` |
| One file per issue | e.g. `test_auth_expired.py` vs `test_auth_claims.py` |
| User-style issues | Symptom, not root cause |
| Distractors | `app/inventory.py`, `app/pricing.py`, `app/payments.py` |
| 001–007 | Kept as **smoke** (same bugs, weaker visible tests) |
| 008 | Upgraded: visible = same-user replay; gold = **per-user** key |
| 009–011 | New hard tasks |

Visible suite on buggy HEAD (after v2 seed): **12 failed, 8 passed** (happy-path only). Hidden gold: **11 failed** on buggy HEAD; **11 passed** against the intended fixes (verified, then restored).

### Dataset (`eval/dataset.json`)

11 tasks, fields `split`, `gold_file`, `gold_test`. `test_command` is the visible file only.

| ID | split | Visible test | Hidden gold extra |
|---|---|---|---|
| 001 | smoke | stale token not 500 | 401 |
| 002 | smoke | inclusive sum ≠ exclusive 10 | `== 15` and single-value 3 |
| 003 | smoke | empty email rejected | same |
| 004 | smoke | missing user not 500 | 404 |
| 005 | smoke | total ≠ sum of unit prices | qty-aware 35.0 |
| 006 | smoke | missing `user_id` not 500 | 401 |
| 007 | smoke | admin promote not 403 | JWT payload has `role=admin` |
| 008 | hard | Alice replay same key → same id | Bob + same key ≠ Alice’s order |
| 009 | hard | qty=50 not 500 | **409** and stock unchanged |
| 010 | hard | promoted Alice’s old token not 403 | `require_admin` uses live store role; Bob still 403 |
| 011 | hard | SAVE10 on $10 → 9.0 | coupon once **and** refund restores stock |

### Harness

| Area | Location | Notes |
|---|---|---|
| Gold staging | `eval/runner.py` | Copy → pytest `tests/_gold/…` → `finally` cleanup |
| Reset leftover | `eval/repository.py` | `rmtree tests/_gold` after `git reset` / `git clean` |
| Path jail | `agent/tools/_sandbox.py` | Skip `_gold` and `_app_bak` |
| Dataset tests | `tests/test_dataset.py` | Schema, no `GOLD:` / `issue-00X` in visible tests, staging cleanup |
| `git_diff` errors | `agent/tools/git.py` | One-line failure; do not dump git usage into the LLM context |

`cli.py compare` still runs V0 then V1 on one task; each side is a fresh container.

---

## Eval results

Same model, `max_steps=15`, Docker sandbox `issue-pilot-sandbox:py312`.

### After hidden gold, before remaining leak strip

`base_commit` `fcf311939df40bf715a35282c1575a0da5c71763`.

| Task | V0 gold | V1 gold | V0 tokens / reads / s | V1 tokens / reads / s |
|---|---|---|---|---|
| issue-001 | True | True | 53k / 9 / 33 | 105k / 14 / 88 |
| issue-009 | True | True | 73k / 12 / 46 | 74k / 12 / 58 |
| issue-008 | True | True | 75k / 14 / 37 | 153k / 14 / 113 |

Trajectories:

- `runs/issue-001-v0-20260813T143706Z.json` / `…v1-20260813T143837Z.json`
- `runs/issue-009-v0-20260813T144140Z.json` / `…v1-20260813T144241Z.json`
- `runs/issue-008-v0-20260813T144542Z.json` / `…v1-20260813T144737Z.json`

Resolve **6/6**. V1 did not win on gold. Notes from trajectories:

1. **009 still leaked in source** — `inventory.py` told the agent to call `reserve()` (409) instead of indexing `slots` (IndexError → 500). Both sides followed that comment.
2. **008 issue text leaked gold** — the coworker/docs sentence made both sides implement `(user_id, key)` on the first patch. Visible-green / gold-red never happened.
3. **001 V1 over-patched** — wrong hypothesis (user lookup) also edited `users.py` and `main.py`. Gold still passed; verify does not penalize patch size.
4. **`git_diff` in the container** returned exit 129 (`Not a git repository`) and originally dumped git’s usage text into the V1 008 context (main driver of 153k tokens).
5. **`_app_bak/`** showed up in `list_files` (leftover from local gold-pass verification).

### Leak-strip follow-up

Benchmark commit `4b2258b`: removed inventory fix-recipe comments. Dataset issue-008 shortened to “Retrying checkout with the same Idempotency-Key created a second order.” Deleted `_app_bak`. `git_diff` failures are one line.

Re-run **issue-009** on `4b2258b`:

| Metric | V0 | V1 |
|---|---:|---:|
| Gold | True | True |
| Steps | 8 | 10 |
| LLM calls | 8 | 12 |
| Tool calls | 12 | 21 |
| File reads | **7** | 12 |
| Tokens | **31,472** | 56,913 |
| Latency | **23.0s** | 55.3s |

Trajectories:

- `runs/issue-009-v0-20260813T145614Z.json`
- `runs/issue-009-v1-20260813T145712Z.json`

Without the comment, neither side was steered into `reserve()`. Both inlined a 409 check in `allocate_bin`. Gold only requires 409 + unchanged stock, so both scored True. **V0 used fewer reads and about half the tokens.** V1’s plan still hypothesized a DB constraint and listed `app/validators.py`.

CLI `compare` then crashed while printing the table (`UnicodeEncodeError` on `→` under Windows cp1252). Both run JSON files were already saved. `cli.py` uses Rich `force_terminal=True` but the final-answer dump still hits the console encoding.

---

## Observations

1. **Hidden gold works** — scoring command is `pytest tests/_gold/test_issue_00X.py::…`; agent `list_files tests/` does not list `_gold`.
2. **V1 is still a cost, not a resolve win** on 001 / 008 / 009 with `deepseek-v4-flash`. Diagnose still does not re-execute.
3. **Leakage is the main difficulty knob** — removing comments dropped 009 V0 from 73k→31k tokens; it did not make V1 better at localization.
4. **Issue text can leak gold as easily as tests** — 008’s coworker sentence defeated the two-layer design until it was removed. That compare has not been re-run yet.
5. **Wrong V1 plans are visible now** — 001 user-lookup, 009 validators/DB — but execute still succeeds, so the extra nodes do not change `success`.
6. **Sandbox git vs host git** — evaluator `git reset` is on the host; agent `git_diff` is `docker exec`. Container git did not see a repo (Windows bind-mount / `safe.directory`). Error truncation stops token blow-ups; the underlying git visibility is unfixed.
7. **Do not commit agent leftovers** — after 008 V1, `app/orders.py` still had the agent’s `_IDEMPOTENCY` patch until restored to HEAD.

---

## Explicitly out of scope

- V1 retry / replan (Day 5; `MAX_RETRY` still unused)
- Hybrid RAG (Day 4 / harness V2)
- Fixing container `git_diff` so it actually diffs
- CLI cp1252 crash on Unicode in final answers
- Re-running 008 / 010 / 011 after the issue-008 text change
- SWE-bench

---

## How to reproduce

```powershell
cd issue-pilot
copy .env.example .env   # DEEPSEEK_API_KEY
# Docker Desktop Linux containers; docker on PATH if needed:
# $env:Path = "C:\Users\octav\AppData\Local\Programs\DockerDesktop\resources\bin;" + $env:Path

python cli.py sandbox doctor --require-image
python -m pytest tests -q -m "not docker"

python cli.py compare issue-009
python cli.py compare issue-008
python cli.py compare issue-010
```

Each solve: `git reset --hard 4b2258b` → one sandbox → V0 or V1 → hidden gold → cleanup. Compare = two solves = two containers. Prefer hard split (008–011) for ablation; 001–007 are smoke.

---

## Next

| Priority | Item | Why |
|---|---|---|
| Eval | Re-compare **008** (no coworker sentence) and **010** (looks like 007) | First chance at visible-green / gold-red and wrong-hypothesis |
| Day 5 | Wire verify-fail → diagnose → execute (`MAX_RETRY=2`) | Otherwise V1 cannot beat V0 when the first patch is incomplete |
| Day 4 | RAG Recall@5 on **009** | Localization noise exists; V1 plan still lists distractors |
| Harness | `git_diff` inside Docker; CLI UTF-8 print | Observed failures, not model failures |

Later: Day 5 retry, Docker `git_diff`, and RAG alignment shipped in `docs/PHASE4-followup.md`. The table above is the post-enhance backlog as of that write-up.
