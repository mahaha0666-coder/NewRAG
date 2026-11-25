"""Provider configuration shared by the MechRAG notebooks.

Secrets are deliberately read at runtime and are never stored in a notebook.
"""

from __future__ import annotations

import os


DEEPSEEK_OPENAI_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_MODEL = "deepseek-flash"


def require_deepseek_api_key(api_key: str | None = None) -> str:
    """Return a configured DeepSeek key or fail with a safe setup message."""

    value = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    if not value:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Set it in the terminal that starts "
            "Jupyter; do not paste the key into the notebook or commit it to Git."
        )
    return value


def create_deepseek_anthropic_client(*, api_key: str | None = None):
    """Create an Anthropic-SDK client pointed at DeepSeek's compatible API."""

    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("Install the notebook dependencies with pip install -e '.[notebooks]'") from exc
    return anthropic.Anthropic(
        api_key=require_deepseek_api_key(api_key),
        base_url=DEEPSEEK_ANTHROPIC_BASE_URL,
    )


def create_deepseek_openai_client(*, api_key: str | None = None):
    """Create an OpenAI-SDK client for DeepSeek multimodal requests."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Install the notebook dependencies with pip install -e '.[notebooks]'"
        ) from exc
    return OpenAI(
        api_key=require_deepseek_api_key(api_key),
        base_url=DEEPSEEK_OPENAI_BASE_URL,
    )


def create_deepseek_chat_model(
    *,
    api_key: str | None = None,
    model: str = DEEPSEEK_MODEL,
    temperature: float = 0,
):
    """Create a LangChain chat model for DeepSeek text and image messages."""

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("Install the notebook dependencies with pip install -e '.[notebooks]'") from exc
    return ChatOpenAI(
        api_key=require_deepseek_api_key(api_key),
        base_url=DEEPSEEK_OPENAI_BASE_URL,
        model=model,
        temperature=temperature,
    )
