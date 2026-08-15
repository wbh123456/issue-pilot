"""Plan node: request structured JSON and reject malformed plans."""

from __future__ import annotations

import re

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState, PlanValidationError, parse_structured_plan
from agent.tools._sandbox import resolve_in_repo
from agent.tools.filesystem import list_files
from harness.limits import AGENT_TEMPERATURE
from harness.progress import summarize_files

from ._runtime import get_reporter, merge_telemetry, require_config, stage_usage

PLAN_SYSTEM = """
You are the planner for a coding-agent workflow.

Return ONLY a single JSON object (no markdown, no prose) with exactly these keys:
{
  "problem": string,
  "hypothesis": string,
  "files_to_inspect": string[],
  "steps": string[]
}

Rules:
- problem and hypothesis must be non-empty strings
- files_to_inspect must use only repo-relative paths that appear in the inventory
- do not invent frameworks, languages, or paths absent from the inventory
- steps must be 3 to 5 concise checklist items for the executor
- do not instruct modifying tests
- do not call tools
""".strip()

_INVENTORY_DIRS = (".", "app", "tests")
_INVENTORY_CHAR_LIMIT = 4000


def _strip_code_fence(text: str) -> str:
    """Allow ```json ... ``` wrappers; still reject non-JSON prose."""
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*\n?(.*?)\n?```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return stripped


def _repo_inventory(repo_path: str) -> str:
    """Deterministic bounded listing for grounding the planner."""
    parts: list[str] = []
    for path in _INVENTORY_DIRS:
        listing = list_files(repo_path, path)
        if listing.startswith("Error:"):
            continue
        parts.append(f"## {path}\n{listing}")
    text = "\n\n".join(parts) if parts else "(empty)"
    if len(text) > _INVENTORY_CHAR_LIMIT:
        return text[:_INVENTORY_CHAR_LIMIT] + "\n\n...[inventory truncated]"
    return text


def _filter_existing_files(repo_path: str, paths: list[str]) -> list[str]:
    kept: list[str] = []
    for path in paths:
        try:
            if resolve_in_repo(repo_path, path).exists():
                kept.append(path)
        except (OSError, PermissionError, FileNotFoundError, ValueError):
            continue
    return kept


def _planner_grounding(state: AgentState, repo_path: str) -> tuple[str, str]:
    """V2 uses retrieved snippets; V1 falls back to a directory inventory."""
    snippets = (state.get("retrieved_context") or "").strip()
    files = [p for p in (state.get("relevant_files") or []) if p]
    if snippets or files:
        listing = "\n".join(f"- {path}" for path in files) or "(none)"
        body = f"Relevant files:\n{listing}\n\nSnippets:\n{snippets or '(none)'}"
        return "Retrieved code (prefer these over a directory listing)", body
    return "Repository inventory (authoritative)", _repo_inventory(repo_path)


def structured_plan(state: AgentState, config: RunnableConfig) -> dict:
    cfg = require_config(config, "client", "model", "repo_path")
    client = cfg["client"]
    model = cfg["model"]
    repo_path = str(cfg["repo_path"])
    analysis = state.get("analysis") or ""
    heading, grounding = _planner_grounding(state, repo_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PLAN_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Issue:\n{state['issue']}\n\n"
                    f"Analysis:\n{analysis}\n\n"
                    f"{heading}:\n{grounding}\n\n"
                    "Produce the structured plan JSON now."
                ),
            },
        ],
        temperature=AGENT_TEMPERATURE,
    )
    raw = (response.choices[0].message.content or "").strip()
    telemetry = merge_telemetry(state, **stage_usage("plan", response))

    plan = parse_structured_plan(_strip_code_fence(raw))
    plan_data = plan.model_dump()
    plan_data["files_to_inspect"] = _filter_existing_files(
        repo_path,
        plan_data.get("files_to_inspect") or [],
    )

    reporter = get_reporter(config)
    files = summarize_files(plan_data.get("files_to_inspect") or [])
    n_steps = len(plan_data.get("steps") or [])
    reporter.stage("plan", f"files={files}  steps={n_steps}")

    return {
        "plan": plan_data,
        "status": "planned",
        "telemetry": telemetry,
    }
