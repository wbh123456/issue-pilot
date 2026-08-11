# Phase 1 — Day 1 Completion Record

Date: 2026-08-11  
Harness: `issue-pilot`  
Benchmark (sibling repo): `issue-pilot-benchmark`  
LLM: DeepSeek (`deepseek-v4-flash`) via OpenAI-compatible SDK

## Goal

Ship a SWE-bench-style V0 coding-agent harness:

1. Independent buggy FastAPI benchmark (8 seeded bugs)
2. Minimal ReAct agent loop with 6 tools
3. Eval runner + CLI that resets to `base_commit`, scores gold tests, and saves trajectories

## What we built

### Benchmark (`../issue-pilot-benchmark`)

- FastAPI app modules: `auth`, `users`, `calculator`, `validators`, `orders`
- 8 intentional bugs (3 easy / 3 medium / 2 hard) with failing gold tests
- Buggy baseline commit (`base_commit`):
  - `573d8b1da2f587d509f4c17286eace4bac3715a4`
- Kept as a **separate git repo** so harness resets/diffs never touch agent code

### Harness (`issue-pilot`)

| Area | Location | Notes |
|---|---|---|
| Dataset | `eval/dataset.json` | 8 tasks with issue text, `base_commit`, `test_command`, `gold_test` |
| Tools | `agent/tools/` | `list_files`, `read_file`, `grep_code`, `edit_file`, `run_tests`, `git_diff` |
| Sandbox | `agent/tools/_sandbox.py` | Path jail + `MAX_TOOL_OUTPUT` truncation |
| Schemas / dispatch | `agent/tools/schema.py`, `execute.py` | OpenAI/DeepSeek tool format |
| ReAct loop | `agent/loop.py` | `max_steps=15`, trajectory + token/latency telemetry |
| Client | `agent/client.py` | DeepSeek via `openai` SDK + `.env` |
| Runner | `eval/runner.py` | reset → agent → gold test → `runs/*.json` |
| CLI | `cli.py` | `python cli.py solve <task_id>` |
| Tool unit tests | `tests/test_tools.py` | Isolated temp git repo; 23 tests |

## Eval results (DoD)

| Task | Difficulty | Success | Steps | Tool calls | Tokens | Latency |
|---|---|---|---|---|---|---|
| issue-002 | easy | True | 6 | 7 | 10,297 | ~9.5s |
| issue-001 | easy | True | 9 | 10 | 30,595 | ~20s |
| issue-007 | hard | True | 13 | 20 | 98,050 | ~98s |
| issue-008 | hard | True | 10 | 12 | 40,546 | ~22s |

Trajectories: `runs/issue-00{1,2,7,8}-*.json`

DoD (“auto-solve 1–2 Easy”) **met**.  
Stretch (“save a natural failure trajectory”) **not met** — DeepSeek also solved the hard tasks we tried.

## Observations for later phases

1. **Test leakage:** the agent reads `tests/` files; gold comments like `GOLD: issue-002` make tasks easier than issue-text-only.
2. **Reset timing:** runner resets **before** each task, not after; working tree may stay dirty until the next solve.
3. **Windows CLI:** Rich/cp1252 can crash on Unicode in final answers; mitigated in `cli.py`.
4. **No natural failures yet:** useful for Day 2 ablations (block test reads, lower `max_steps`, weaker model).

## Explicitly out of scope (Phase 1)

- LangGraph / Plan-Execute
- RAG / embeddings
- MCP
- Docker sandbox
- Human-in-the-loop

## How to reproduce

```powershell
cd issue-pilot
copy .env.example .env   # set DEEPSEEK_API_KEY
python -m pip install -r requirements.txt
python -m pytest tests/test_tools.py -q
python cli.py solve issue-002
```

Benchmark stays at the shared buggy baseline; the harness resets to `base_commit` at the start of every `solve`.
