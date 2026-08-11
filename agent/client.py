"""LLM client factory (DeepSeek via OpenAI-compatible SDK)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def create_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """Return an OpenAI SDK client pointed at DeepSeek.

    Reads ``DEEPSEEK_API_KEY`` (and optional ``DEEPSEEK_BASE_URL``) from the
    environment / ``.env`` unless explicit values are passed.
    """
    load_dotenv()

    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return OpenAI(
        api_key=key,
        base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
    )


def default_model() -> str:
    """Model id used by the harness unless overridden."""
    load_dotenv()
    return os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
