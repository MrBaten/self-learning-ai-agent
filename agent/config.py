"""Configuration helpers: project paths, data directory and LLM provider selection.

Environment variables are loaded from ``PROJECT_ROOT/.env`` (if present) with
python-dotenv, never overriding variables that are already set. API keys are
only ever read from the environment; they are never stored or printed here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SEED_DATA_DIR: Path = PROJECT_ROOT / "data"

VALID_PROVIDERS: tuple[str, ...] = ("openai", "groq", "mock")

_env_loaded: bool = False


def load_env() -> None:
    """Load ``PROJECT_ROOT/.env`` into ``os.environ`` once (override=False).

    Safe to call many times: the file is only read on the first call, and
    variables already present in the environment always win.
    """
    global _env_loaded
    if _env_loaded:
        return
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)
    _env_loaded = True


def data_dir() -> Path:
    """Return the runtime data directory, creating it if needed.

    Uses ``AGENT_DATA_DIR`` when set (re-read on every call so tests and the
    eval harness can switch directories at runtime), else ``SEED_DATA_DIR``.
    """
    raw = os.environ.get("AGENT_DATA_DIR", "").strip()
    path = Path(raw).expanduser() if raw else SEED_DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def workspace_dir() -> Path:
    """Return ``data_dir() / "workspace"`` (the file-tool sandbox), creating it if needed."""
    path = data_dir() / "workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_provider() -> str:
    """Pick the LLM provider name.

    Order: explicit ``LLM_PROVIDER`` (openai/groq/mock, case-insensitive);
    otherwise ``"openai"`` if ``OPENAI_API_KEY`` is set, ``"groq"`` if
    ``GROQ_API_KEY`` is set, else ``"mock"``.

    Raises:
        ValueError: if ``LLM_PROVIDER`` is set to an unsupported value.
    """
    load_env()
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid LLM_PROVIDER '{explicit}'. Expected one of: {', '.join(VALID_PROVIDERS)}."
            )
        return explicit
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    return "mock"
