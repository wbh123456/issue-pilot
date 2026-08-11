"""IssuePilot coding-agent package."""

__all__ = ["create_client", "default_model", "run_agent"]


def __getattr__(name: str):
    if name in {"create_client", "default_model"}:
        from .client import create_client, default_model

        return {"create_client": create_client, "default_model": default_model}[name]
    if name == "run_agent":
        from .loop import run_agent

        return run_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
