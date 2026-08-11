import json
import time

from .tools import TOOLS, execute_tool

SYSTEM_PROMPT = """
You are a coding agent fixing one issue in a sandboxed repository.

Inspect relevant files before editing.
Make the smallest correct change.
Use run_tests after editing.
Use git_diff to inspect your final changes.
Do not modify tests.
When the issue is fixed and tests pass, respond with a concise final answer
and make no further tool calls.
"""

def run_agent(
    *,
    client,
    issue: str,
    repo_path: str,
    test_command: str,
    model: str = "deepseek-v4-flash",
    max_steps: int = 15,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Fix this issue:\n\n{issue}"},
    ]

    trajectory = []
    tool_call_count = 0
    file_reads = 0
    prompt_tokens = 0
    completion_tokens = 0
    final_answer = ""
    termination = "max_steps"
    started_at = time.perf_counter()

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # Convert the SDK object into a plain dict
        assistant_message = {
            "role": "assistant",
            "content": message.content,
        }

        if tool_calls:
            assistant_message["tool_calls"] = [
                call.model_dump() for call in tool_calls
            ]

        messages.append(assistant_message)

        if response.usage:
            prompt_tokens += response.usage.prompt_tokens or 0
            completion_tokens += response.usage.completion_tokens or 0

        # No tool call means the model has finished.
        if not tool_calls:
            final_answer = message.content or ""
            termination = "completed"
            steps = step + 1
            break

        for call in tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                result = execute_tool(
                    name,
                    args,
                    repo_path=repo_path,
                    test_command=test_command,
                )
            except json.JSONDecodeError as exc:
                args = {}
                result = f"Error: invalid tool arguments: {exc}"
            except Exception as exc:
                result = (
                    f"Error executing {name}: "
                    f"{type(exc).__name__}: {exc}"
                )

            tool_call_count += 1

            if name == "read_file":
                file_reads += 1

            trajectory.append({
                "step": step + 1,
                "tool_call_id": call.id,
                "tool": name,
                "arguments": args,
                "result": result,
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    else:
        steps = max_steps

    return {
        "final_answer": final_answer,
        "termination": termination,
        "steps": steps,
        "tool_call_count": tool_call_count,
        "file_reads": file_reads,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens": prompt_tokens + completion_tokens,
        "latency": time.perf_counter() - started_at,
        "trajectory": trajectory,
        "messages": messages,
    }