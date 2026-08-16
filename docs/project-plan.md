# IssuePilot — Coding Agent Harness

> **1-Week Personal Project Plan**
>
> 目标时间：7 天 × 每天约 3 小时
> 技术方向：AI Agent / Agentic Workflow / RAG / Harness Engineering
> 主要用途：AI / Agent Engineer 求职项目 + Agentic Engineering 系统学习

---

# 1. 项目简介

## 1.1 项目名称

**IssuePilot — Evaluation-Driven Coding Agent Harness**

IssuePilot 是一个面向 GitHub Issue / Software Bug Fixing 场景的 Coding Agent。

输入：

```text
GitHub Issue
+
Source Code Repository
```

系统自动完成：

```text
Understand Issue
      ↓
Retrieve Code Context
      ↓
Plan
      ↓
Modify Code
      ↓
Run Tests
      ↓
Diagnose Failure
      ↓
Retry
      ↓
Evaluate Patch
      ↓
Human Approval
      ↓
Final Patch / Pull Request
```

项目重点不是简单实现一个「会修改代码的 LLM」，而是研究：

> **如何通过 Tool Design、Workflow、RAG、State、Sandbox、Verification、Retry 和 Guardrails，让同一个 LLM 更可靠地完成软件工程任务。**

---

# 2. 项目目的

本项目有三个主要目标。

## 2.1 学习 Agentic Workflow

理解现代 Agent 系统的核心运行机制：

* Agent Loop
* Function Calling
* Tool Calling
* ReAct
* Plan-Execute
* Workflow Orchestration
* Conditional Routing
* State Management
* Retry / Recovery
* Human-in-the-loop

重点理解：

```text
Agent
=
LLM 决定下一步行动

Workflow
=
工程系统决定流程结构
```

以及二者如何结合。

---

## 2.2 学习 Harness Engineering

理解：

```text
Model Capability
≠
Agent Capability
```

同一个模型在不同 Harness 下可能有完全不同的表现。

重点学习：

* Tool interface design
* Execution environment
* Sandbox
* Step budget
* Token / context budget
* Timeout
* Permission boundaries
* Output truncation
* Retry policy
* Deterministic verification
* Guardrails
* Submission gates
* Observability

核心原则：

> 不要只通过 Prompt 告诉 Agent 应该怎么做；能够由系统强制执行的规则，应尽量放入 Harness。

---

## 2.3 建立一个可以写进简历的 Agent 项目

项目最终不能只描述为：

```text
Used LangGraph + RAG + MCP to build a coding agent.
```

而应该能够描述为：

```text
Designed and evaluated a coding-agent harness that combines
stateful workflow orchestration, hybrid code retrieval,
sandboxed execution, deterministic verification and retry loops.
```

并通过 Benchmark 回答：

```text
哪一个 Harness 改动提高了 Resolve Rate？

哪一个改动减少了 File Reads？

RAG 是否降低 Token Usage？

Plan-Execute 是否减少无效 Tool Calls？

Retry 是否真正提高 Bug Resolution Rate？
```

---

# 3. 项目涵盖知识点

## Agent Fundamentals

* LLM Agent
* Agent Loop
* Function Calling
* Tool Calling
* ReAct
* Observation
* Tool Result
* Termination Condition
* Step Budget

---

## Agentic Workflow

* Agent vs Workflow
* ReAct vs Plan-Execute
* Planner / Executor
* Structured Output
* Conditional Routing
* State Machine
* Retry Loop
* Evaluator-Optimizer
* Human-in-the-loop

---

## LangGraph

* Graph
* Node
* Edge
* Conditional Edge
* Agent State
* Checkpoint
* Persistence
* Resume
* Interrupt / Human Approval

---

## RAG / Context Engineering

* Code RAG
* Embedding
* Dense Retrieval
* Lexical Retrieval
* BM25
* Hybrid Retrieval
* Reciprocal Rank Fusion
* Symbol-aware Chunking
* Metadata
* Recall@K
* Dynamic Retrieval
* Context Budget
* Context Pruning

---

## Harness Engineering

* Tool Design
* Agent-Computer Interface
* Sandbox
* Docker
* Timeout
* Output Truncation
* Tool Permission
* Command Allowlist
* Filesystem Boundary
* Network Isolation
* Retry Limit
* Step Limit
* Submission Gate

---

## Agent Reliability

* Deterministic Verification
* Test-based Evaluation
* Failure Diagnosis
* Retry
* Escalation
* Checkpoint
* Resume
* Human Approval

---

## Evaluation

* Benchmark Dataset
* Resolve Rate
* Relevant File Recall@K
* Tool Calls
* File Reads
* Token Usage
* Retry Count
* Latency
* Ablation Study

---

## Agent Infrastructure

* Docker
* Git
* MCP
* SQLite
* Logging
* Tracing
* Observability

---

# 4. 最终系统架构

```text
                    GitHub Issue
                         │
                         ▼
                ┌─────────────────┐
                │ Issue Analyzer  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Code Retrieval  │
                │ BM25 + Vector   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     Planner     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Agent Executor  │
                │                 │
                │ read_file       │
                │ grep_code       │
                │ search_code     │
                │ edit_file       │
                │ run_tests       │
                │ git_diff        │
                └────────┬────────┘
                         │
                         ▼
                  Docker Sandbox
                         │
                         ▼
                ┌─────────────────┐
                │   Verification  │
                │                 │
                │ pytest          │
                │ lint            │
                │ git diff        │
                └────────┬────────┘
                         │
              ┌──────────┴─────────┐
              │                    │
            FAIL                  PASS
              │                    │
              ▼                    ▼
          Diagnose              Evaluator
              │                    │
              └──── Retry          ▼
                              Human Approval
                                   │
                                   ▼
                              Final Patch
```

外围由 Agent Harness 控制：

```text
Agent Harness
│
├── State
├── Context Budget
├── Step Budget
├── Retry Policy
├── Tool Permissions
├── Sandbox
├── Timeout
├── Output Limits
├── Verification
└── Metrics / Tracing
```

---

# 5. 项目开发阶段

整个项目压缩为七个阶段。

```text
Phase 0    Benchmark
Phase 1    Minimal Agent
Phase 2    Workflow + State
Phase 3    Sandbox + Harness
Phase 4    Code RAG
Phase 5    Reliability + HITL
Phase 6    Evaluation + MCP
```

编号 `docs/PHASE*.md` 对应这里的 Day（Phase 1 = Day 1，以此类推）。已完成的 **Phase 5 文档是 Day 5**：自动恢复、双层评估、同进程一次反馈。它不是上表全称 “Reliability + HITL”。Durable HITL、checkpoint/resume、可观测性扩展、MCP 仍是 Day 6。

---

# Day 1 — Benchmark + Minimal Agent Loop

## 今日目标

理解 Agent 最基础的运行机制，并建立之后所有实验使用的 Benchmark。

预计时间：

```text
Benchmark       60 min
Agent Loop     100 min
Testing         20 min
```

---

## Task 1 — 建立 Benchmark Repository

创建：

```text
issue-pilot/

issue-pilot-benchmark/
```

Benchmark Repo 使用一个小型 Python FastAPI 项目。

例如：

```text
issue-pilot-benchmark/

├── app/
│   ├── auth.py
│   ├── users.py
│   ├── calculator.py
│   ├── orders.py
│   └── validators.py
│
└── tests/
```

创建：

```text
8 个 Bug
```

难度：

```text
Easy      3
Medium    3
Hard      2
```

每个任务记录：

```json
{
  "id": "issue-001",
  "issue": "Expired JWT returns 500 instead of 401",
  "base_commit": "...",
  "expected_files": ["app/auth.py"],
  "test_command": "pytest tests/test_auth.py",
  "gold_test": "test_expired_token_returns_401"
}
```

---

## Task 2 — 实现 Minimal Agent Loop

暂时：

```text
不用 LangGraph
不用 RAG
不用 MCP
```

自己实现：

```text
Issue
 ↓
LLM
 ↓
Tool Call
 ↓
Tool Result
 ↓
LLM
 ↓
...
 ↓
Finish
```

第一批 Tools：

```text
list_files(path)

read_file(path)

grep_code(query)

edit_file(path, patch)

run_tests()

git_diff()
```

Harness 管理：

```text
messages

tool calls

tool results

max_steps

termination condition

execution history
```

---

## 今天重点观察

Agent 是否出现：

```text
重复读取文件

无意义 grep

太早修改代码

测试失败不会恢复

Tool output 太长

无限循环

错误判断任务已经完成
```

这些 failure case 后面都将成为 Harness 改进依据。

---

## 今日学习

* Agent Loop
* Function Calling
* Tool Calling
* ReAct
* Observation
* Termination
* Tool Design

---

## Definition of Done

至少：

```text
Agent 可以自动解决 1–2 个 Easy Bug
```

不追求成功率。

重点保存失败 trajectory。

---

## Stretch Goal

增加：

```text
max_tokens

estimated_cost

tool_call_count
```

为每次 Agent Run 保存基础 telemetry。

---

# Day 2 — LangGraph + Plan-Execute Workflow

## 今日目标

把自由 Agent Loop 改造成可控的 Stateful Agent Workflow。

---

## Task 1 — 定义 Agent State

例如：

```python
class AgentState(TypedDict):
    issue: str

    analysis: str
    plan: list[str]

    relevant_files: list[str]
    changed_files: list[str]

    test_result: dict

    retry_count: int

    status: str
```

---

## Task 2 — 建立 LangGraph Workflow

```text
START
 ↓
analyze_issue
 ↓
plan
 ↓
execute
 ↓
verify
 ↓
END
```

随后加入：

```text
verify
 ├─ PASS → evaluate
 │
 └─ FAIL → diagnose
```

---

## Task 3 — Structured Planning

Planner 输出：

```json
{
  "problem": "...",

  "hypothesis": "...",

  "files_to_inspect": [],

  "steps": [
    "...",
    "...",
    "run tests"
  ]
}
```

---

## 核心设计原则

使用 LLM：

```text
理解 Issue

产生 hypothesis

决定需要检查什么

分析失败原因
```

使用 deterministic code：

```text
tests 是否通过

文件是否存在

process exit code

timeout

git diff 是否为空
```

例如：

```python
tests_passed = process.returncode == 0
```

而不是：

```text
LLM:
"The tests appear to have passed."
```

---

## 今日实验

比较：

```text
V0
Basic Agent Loop

vs

V1
Plan → Execute → Verify
```

记录：

```text
Resolve Rate

Tool Calls

File Reads

Token Usage
```

---

## 今日学习

* Agent vs Workflow
* ReAct vs Plan-Execute
* LangGraph
* State
* Node / Edge
* Conditional Routing
* Structured Output

---

## Definition of Done

同一个 Issue：

```text
可以分别通过 V0 / V1 执行
```

并保存两种 architecture 的 metrics。

---

## Stretch Goal

增加：

```text
Planner Replan
```

如果原始 hypothesis 被测试结果否定：

```text
重新制定 plan
```

而不是简单重复上一次操作。

---

# Day 3 — Docker Sandbox + Harness Engineering

## 今日目标

把 Agent 从「LLM Demo」变成具有真实执行边界的 Agent System。

---

## Task 1 — Docker Execution Environment

Agent 不允许：

```python
subprocess.run(...)
```

直接运行宿主机 command。

架构：

```text
Agent Runtime
     ↓
SandboxRunner
     ↓
Docker Container
     ↓
Repository
```

---

## Task 2 — Sandbox Policy

实现：

```text
command timeout

workspace isolation

network disabled

output truncation

container per task

container cleanup
```

Allowed commands：

```text
pytest

ruff

mypy

git
```

---

## Task 3 — Harness Limits

加入：

```python
MAX_AGENT_STEPS = 15

MAX_RETRY = 2

MAX_TOOL_OUTPUT = 10_000

COMMAND_TIMEOUT = 60
```

---

## Task 4 — Permission Guardrails

阻止：

```text
rm -rf

curl

wget

ssh

/etc

~/.ssh

filesystem escape
```

---

## 今日核心问题

思考：

```text
为什么 Prompt：

"Do not run dangerous commands"

不如：

Harness 根本不允许 dangerous command？
```

---

## 今日学习

* Harness Engineering
* Sandbox
* Security Boundary
* Tool Permissions
* Timeout
* Resource Limits
* Agent-Computer Interface
* Mechanical Guardrails

---

## Definition of Done

所有代码执行：

```text
Agent
→ Docker
→ Tests
→ Result
```

Agent 无法直接访问宿主机敏感目录。

---

## Stretch Goal

加入 Docker resource limit：

```text
CPU limit

memory limit
```

并记录 command execution latency。

---

# Day 4 — Code RAG + Context Engineering

## 今日目标

解决 Coding Agent 最常见的问题之一：

> 找不到正确代码或者读取大量无关上下文。

---

## Task 1 — Code Chunking

针对 Python Repository：

使用：

```text
Python AST
```

按照 symbol 切分：

```text
module

class

function

method
```

而不是固定：

```text
500 characters / chunk
```

Metadata：

```json
{
  "path": "app/auth.py",

  "symbol": "validate_token",

  "type": "function",

  "start_line": 20,

  "end_line": 55
}
```

---

## Task 2 — Dense Retrieval

实现：

```text
Issue / Query
 ↓
Embedding
 ↓
Vector Index
 ↓
Top-K Symbols
```

---

## Task 3 — Lexical Retrieval

使用：

```text
BM25
```

或 ripgrep-based lexical retrieval。

---

## Task 4 — Hybrid Retrieval

架构：

```text
              Query
                │
        ┌───────┴───────┐
        ▼               ▼
      Dense            BM25
        │               │
        └───────┬───────┘
                ▼
               RRF
                ▼
              Top-K
```

第一版使用：

```text
Reciprocal Rank Fusion
```

不用复杂 reranker。

---

## Task 5 — Agentic Retrieval

实现两类 Retrieval。

### Initial Retrieval

```text
Issue
 ↓
retrieve initial context
 ↓
Planner
```

### Dynamic Retrieval

执行过程中：

```text
new hypothesis
 ↓
Agent search
 ↓
new context
```

---

## Retrieval Evaluation

测：

```text
Relevant File Recall@5
```

比较：

```text
grep only

dense only

hybrid
```

同时记录：

```text
File Reads

Tokens

Resolve Rate
```

---

## 今日学习

* RAG
* Code RAG
* Embedding
* Chunking
* BM25
* Dense Retrieval
* Hybrid Retrieval
* RRF
* Agentic RAG
* Context Engineering
* Recall@K

---

## Definition of Done

能够回答：

```text
Hybrid Retrieval 是否比纯 grep 更容易找到 relevant files？
```

至少产生一组实验数据。

---

## Stretch Goal

增加：

```text
symbol search
```

例如：

```text
search_symbol("AuthService.validate")
```

---

# Day 5 — Retry + Failure Recovery + Evaluation

## 今日目标

让 Agent 从：

```text
一次失败 = Task Failed
```

进化为：

```text
失败
→ 分析
→ 产生新 hypothesis
→ Retry
```

---

## Task 1 — Failure Diagnosis

Workflow：

```text
execute
 ↓
test
 ↓
FAIL
 ↓
diagnose
 ↓
execute
 ↓
test
```

Diagnosis 输入：

```text
Issue

Current Plan

Git Diff

Test Failure

Previous Attempts
```

---

## Task 2 — Retry Policy

Harness 强制：

```python
if retry_count >= 2:
    status = "NEEDS_HUMAN"
```

而不是 Prompt：

```text
"Please stop after two attempts."
```

---

## Task 3 — Deterministic Evaluation

Layer 1：

```text
pytest

ruff

git diff
```

只有这些通过：

```text
deterministic_pass = True
```

---

## Task 4 — LLM Evaluator

Layer 2：

```json
{
  "issue_resolved": true,

  "patch_scope": "appropriate",

  "regression_risk": "low",

  "missing_tests": false,

  "feedback": ""
}
```

注意：

```text
LLM Evaluator
不能替代 Tests
```

---

## 今日学习

* Failure Recovery
* Retry
* Evaluator-Optimizer
* Deterministic Evaluation
* LLM-as-Judge
* Stopping Criteria
* Escalation

---

## Definition of Done

至少一个任务：

```text
First Attempt FAIL
        ↓
Diagnose
        ↓
Retry
        ↓
PASS
```

---

## Stretch Goal

增加：

```text
failure category
```

例如：

```text
RETRIEVAL_FAILURE

WRONG_HYPOTHESIS

BAD_PATCH

TEST_FAILURE

ENVIRONMENT_FAILURE
```

用于后续 Error Analysis。

---

# Day 6 — Persistence + HITL + Observability + MCP

## 今日目标

增加长期 Agent Workflow 所需要的系统能力。

---

# Part 1 — Persistence

增加 checkpoint：

```text
Issue analyzed

Plan generated

Patch generated

Tests executed

Waiting approval
```

实现：

```text
task_id
+
checkpoint
+
resume
```

测试：

```text
Kill process
 ↓
Restart
 ↓
Resume task
```

---

# Part 2 — Human-in-the-loop

Workflow：

```text
Tests PASS
 ↓
Evaluator PASS
 ↓
WAITING_FOR_APPROVAL
```

CLI 显示：

```text
Issue

Plan

Changed Files

Git Diff

Test Results

Evaluator Result
```

用户：

```text
Approve
Reject
Feedback
```

---

# Part 3 — Observability

保存：

```json
{
  "task_id": "...",

  "model": "...",

  "harness_version": "v2",

  "success": true,

  "tokens": 12000,

  "tool_calls": 16,

  "file_reads": 5,

  "retrieval_calls": 3,

  "retry_count": 1,

  "latency": 48.2
}
```

Trajectory：

```text
STATE

↓

LLM

↓

TOOL

↓

OBSERVATION

↓

STATE UPDATE
```

---

# Part 4 — Minimal MCP

只实现简单 MCP Server。

暴露：

```text
read_file

search_code

git_diff
```

目的：

理解：

```text
Agent
 ↓
MCP Client
 ↓
MCP Server
 ↓
Tool
```

而不是构建复杂 MCP Infrastructure。

---

## 今日学习

* Durable Execution
* Checkpoint
* Resume
* Human-in-the-loop
* Observability
* Tracing
* MCP
* Tool Protocol
* Capability Layer

---

## Definition of Done

完成：

```text
Agent Run
 ↓
Checkpoint
 ↓
Pause
 ↓
Human Approval
 ↓
Resume
 ↓
Finish
```

---

## Stretch Goal

允许：

```text
Human Feedback
```

重新进入：

```text
diagnose
 ↓
execute
```

形成完整 feedback loop。

---

# Day 7 — Benchmark + Ablation + README

## 今日目标

今天原则：

> **不增加主要新功能。**

今天的任务是证明：

```text
你设计的 Harness 是否真的有效。
```

---

# Task 1 — Harness Versioning

建立：

## V0

```text
Basic ReAct Agent

+

grep / read_file
```

---

## V1

```text
Plan-Execute

+

Sandbox

+

Verification

+

Retry
```

---

## V2

```text
V1

+

Hybrid RAG

+

Context Engineering
```

---

# Task 2 — Benchmark

固定：

```text
Same Model

Same Temperature

Same Benchmark

Same Agent Budget
```

运行全部：

```text
8 Tasks
```

---

# Task 3 — Metrics

生成：

| Harness | Resolve Rate | Tokens | Tool Calls | File Reads | Retries |
| ------- | -----------: | -----: | ---------: | ---------: | ------: |
| V0      |              |        |            |            |         |
| V1      |              |        |            |            |         |
| V2      |              |        |            |            |         |

Retrieval：

| Retrieval | Recall@5 |
| --------- | -------: |
| Grep      |          |
| Dense     |          |
| Hybrid    |          |

---

# Task 4 — Failure Analysis

对每一个失败任务分析：

```text
Model failure?

Tool failure?

Retrieval failure?

Context failure?

Planning failure?

Patch failure?

Verification failure?

Environment failure?
```

不要把所有失败简单归类成：

```text
LLM 不够聪明
```

---

# Task 5 — README

README 至少包括：

```text
1. Problem

2. Architecture

3. Agent / Workflow Design

4. Harness Engineering

5. Code RAG

6. Evaluation Methodology

7. Results

8. Failure Analysis

9. Future Work
```

---

## 今日学习

* Agent Evaluation
* Controlled Experiment
* Ablation Study
* Error Analysis
* Metrics
* Engineering Trade-offs

---

## Definition of Done

你必须能够回答：

```text
哪一个 Harness 改进提升最大？

为什么？

代价是什么？

Resolve Rate 是否提高？

Token 是否下降？

File Reads 是否下降？

有没有 trade-off？
```

---

## Stretch Goal

选择：

```text
1–2 个 SWE-bench Verified task
```

尝试运行 IssuePilot。

只验证 integration。

不要进行完整 SWE-bench。

---

# 6. 一周 MVP 范围

一周内必须完成：

```text
✅ Minimal Agent Loop

✅ Function Calling Tools

✅ LangGraph Workflow

✅ State Management

✅ Docker Sandbox

✅ Mechanical Guardrails

✅ Plan-Execute

✅ Retry

✅ Deterministic Verification

✅ Hybrid Code RAG

✅ Retrieval Evaluation

○ Human-in-the-loop

○ Checkpoint / Resume

✅ Observability

✅ Benchmark

○ Harness Ablation
```

---

# 7. 一周 Stretch Features

有额外时间再完成：

```text
○ MCP Server

○ Symbol Search

○ Failure Classification

○ Resource Limits

○ Human Feedback → Retry

○ SWE-bench Integration
```

优先级：

```text
Benchmark
    >
Reliability
    >
RAG
    >
MCP
    >
SWE-bench
```

---

# 8. 明确不做

本周暂时不做：

```text
❌ Multi-Agent

❌ Complex Frontend

❌ Kubernetes

❌ Long-term User Memory

❌ Cross-Encoder Reranker

❌ Full SWE-bench

❌ Distributed Agent Runtime
```

原因：

> 一周项目的目标是做出一个完整、可评测、可以深入解释的 Agent Harness，而不是尽可能多地堆 Agent 相关技术。

---

# 9. 推荐技术栈

```text
Language
Python 3.12

LLM
OpenAI / Anthropic API

Agent
Raw Function Calling

Workflow
LangGraph

Code Search
ripgrep

Code Parsing
Python AST

Dense Retrieval
FAISS / Chroma

Lexical Retrieval
BM25

Sandbox
Docker

Persistence
SQLite / LangGraph Checkpoint

Testing
pytest
ruff
mypy

CLI
Typer
Rich

Observability
JSONL
SQLite

MCP
Official MCP Python SDK
```

---

# 10. 推荐项目目录

```text
issue-pilot/

├── agent/
│   ├── loop.py
│   ├── graph.py
│   ├── state.py
│   │
│   ├── nodes/
│   │   ├── analyze.py
│   │   ├── retrieve.py
│   │   ├── plan.py
│   │   ├── execute.py
│   │   ├── verify.py
│   │   ├── diagnose.py
│   │   └── evaluate.py
│   │
│   └── tools/
│       ├── filesystem.py
│       ├── search.py
│       ├── shell.py
│       └── git.py
│
├── retrieval/
│   ├── indexer.py
│   ├── dense.py
│   ├── lexical.py
│   └── hybrid.py
│
├── sandbox/
│   ├── Dockerfile
│   └── runner.py
│
├── harness/
│   ├── limits.py
│   ├── permissions.py
│   └── context.py
│
├── eval/
│   ├── dataset.json
│   ├── runner.py
│   ├── metrics.py
│   └── report.py
│
├── mcp/
│   └── server.py
│
├── cli.py
│
└── tests/
```

---

# 11. 最终项目 Demo

最终应该可以执行：

```bash
issue-pilot solve issue-003
```

显示：

```text
Issue
────────────────────
Expired JWT returns 500 instead of 401


Retrieving repository context...

Relevant files:
- app/auth.py
- tests/test_auth.py


Plan
────────────────────
1. Inspect token expiration validation
2. Inspect error handling
3. Update exception mapping
4. Run auth tests


Executing...


Tests
────────────────────
FAILED

test_expired_token_returns_401


Diagnosing...

Hypothesis:
ExpiredSignatureError is not handled separately.


Retrying...


Tests
────────────────────
PASSED


Patch Evaluation
────────────────────
Issue resolved: YES
Regression risk: LOW
Patch scope: APPROPRIATE


Waiting for Human Approval

[A] Approve
[R] Reject
[F] Feedback
```

---

# 12. 项目结束后应该掌握的能力

如果项目真正完成，而不是完全依赖 Cursor 生成代码，你应该能够解释：

### Agent

```text
Agent Loop 是如何工作的？

Function Calling 如何完成一次 Tool Call？

ReAct 为什么会出现无限循环？

Stopping Condition 应该由谁控制？
```

### Workflow

```text
Agent 和 Workflow 有什么区别？

什么时候使用 Plan-Execute？

为什么 LangGraph 需要显式 State？
```

### RAG

```text
为什么代码不能简单按照固定 Token Chunk？

为什么 Hybrid Retrieval 通常优于单一 Vector Search？

Recall@K 怎么计算？

Agentic Retrieval 和普通 RAG 有什么区别？
```

### Harness Engineering

```text
为什么 Sandbox 属于 Agent Harness？

为什么某些约束应该由代码实现而不是 Prompt？

Step Budget / Token Budget 有什么作用？

Tool API 如何影响 Agent 能力？
```

### Reliability

```text
测试失败以后怎么处理？

Retry 为什么需要 hard limit？

LLM Evaluator 为什么不能替代 pytest？

什么时候应该升级到 Human？
```

### Evaluation

```text
如何评价一个 Coding Agent？

Resolve Rate 为什么比简单 LLM Judge 更重要？

如何设计 Harness Ablation？

怎样判断 RAG 真正提高了 Agent 能力？
```

---

# 13. 项目最终成功标准

项目成功不以：

```text
实现了多少 Framework
```

衡量。

而以你是否能够使用实际数据解释：

```text
同一个 Model

+

不同 Harness

↓

为什么产生不同 Agent Performance
```

最终理想成果：

```text
IssuePilot

8-task benchmark

V0 / V1 / V2 harness comparison

Resolve Rate

Recall@5

Token Usage

Tool Calls

File Reads

Retry Count

Failure Analysis
```

最终你应该能够在面试中说：

> IssuePilot 不是单纯调用 LLM API 的 Coding Agent。我把 Agent 的代码检索、规划、执行、测试和失败恢复拆成一个 stateful workflow，并设计了 Docker sandbox、tool permissions、retry limits 和 deterministic verification 等 Harness 机制。同时通过自建 benchmark 对不同 Harness 版本进行 ablation，在固定模型条件下测量 Resolve Rate、Token Usage、Tool Calls 和 File Reads，从而分析哪些系统设计真正提升了 Agent 的可靠性和效率。
