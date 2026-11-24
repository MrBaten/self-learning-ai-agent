"""Short-term and long-term memory for the self-learning agent.

Long-term memory is a small JSON document (``memory.json`` inside the active data
directory) holding the user's name, preferences and facts. Short-term memory is the
rolling window of chat messages sent to the LLM on each step.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from agent.config import SEED_DATA_DIR, data_dir

MEMORY_FILENAME = "memory.json"
SEED_MEMORY_FILENAME = "memory.example.json"
BACKUP_SUFFIX = ".bak"
DEFAULT_USER_NAME = "User"


class LongTermMemory(BaseModel):
    """Persistent knowledge about the user: name, preferences and facts."""

    user_name: str = DEFAULT_USER_NAME
    preferences: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)

    @field_validator("user_name", mode="before")
    @classmethod
    def _normalise_name(cls, value: Any) -> str:
        """Strip the user name and fall back to the default when it is missing or blank."""
        if value is None:
            return DEFAULT_USER_NAME
        text = str(value).strip()
        return text or DEFAULT_USER_NAME

    @field_validator("preferences", "facts", mode="before")
    @classmethod
    def _normalise_items(cls, value: Any) -> Any:
        """Treat ``None`` as an empty list and strip / drop blank string entries."""
        if value is None:
            return []
        if isinstance(value, list):
            cleaned: list[Any] = []
            for item in value:
                if isinstance(item, str):
                    item = item.strip()
                    if not item:
                        continue
                cleaned.append(item)
            return cleaned
        return value


def memory_path() -> Path:
    """Return the path of ``memory.json`` in the current data directory.

    ``config.data_dir()`` is re-evaluated on every call so a runtime change of
    ``AGENT_DATA_DIR`` is honoured.
    """
    return data_dir() / MEMORY_FILENAME


def _seed_memory_path() -> Path:
    """Return the path of the repository's seed memory file."""
    return SEED_DATA_DIR / SEED_MEMORY_FILENAME


def _read_seed_memory(target: Path) -> LongTermMemory:
    """Load the seed memory for initialising ``target``; defaults if absent or invalid."""
    seed = _seed_memory_path()
    try:
        if not seed.is_file() or seed.resolve() == target.resolve():
            return LongTermMemory()
        data = json.loads(seed.read_text(encoding="utf-8"))
        return LongTermMemory.model_validate(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError):
        return LongTermMemory()


def load_memory() -> LongTermMemory:
    """Load long-term memory from disk, creating or repairing the file as needed.

    - Missing file: initialised from ``SEED_DATA_DIR/memory.example.json`` when that
      exists (otherwise defaults) and written to disk.
    - Corrupt file (invalid JSON or wrong shape): the original is backed up to
      ``memory.json.bak`` and defaults are returned and saved.
    """
    path = memory_path()
    if not path.exists():
        mem = _read_seed_memory(path)
        save_memory(mem)
        return mem
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LongTermMemory.model_validate(data)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError):
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        try:
            shutil.copy2(path, backup)
        except OSError:
            pass
        mem = LongTermMemory()
        save_memory(mem)
        return mem


def save_memory(mem: LongTermMemory) -> None:
    """Atomically write ``mem`` to ``memory.json`` (temp file + ``os.replace``)."""
    path = memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(mem.model_dump(), indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".memory.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _add_item(field: Literal["preferences", "facts"], text: str) -> bool:
    """Append ``text`` to the given memory list unless blank or already present (case-insensitive)."""
    item = (text or "").strip()
    if not item:
        return False
    mem = load_memory()
    items: list[str] = getattr(mem, field)
    key = item.casefold()
    if any(existing.strip().casefold() == key for existing in items):
        return False
    items.append(item)
    save_memory(mem)
    return True


def add_fact(fact: str) -> bool:
    """Add a fact to long-term memory; return True if it was newly added."""
    return _add_item("facts", fact)


def add_preference(pref: str) -> bool:
    """Add a preference to long-term memory; return True if it was newly added."""
    return _add_item("preferences", pref)


def format_memory_for_prompt(mem: LongTermMemory) -> str:
    """Render memory in the exact text block used in the system prompt.

    Format::

        User name: <name>
        Preferences (defaults only; an explicit instruction in the task always wins):
        - <preference>          (or "- (none)")
        Facts:
        - <fact>                (or "- (none)")
    """
    lines = [f"User name: {mem.user_name}", "Preferences (defaults only; an explicit instruction in the task always wins):"]
    lines.extend(_bullet_lines(mem.preferences))
    lines.append("Facts:")
    lines.extend(_bullet_lines(mem.facts))
    return "\n".join(lines)


def _bullet_lines(items: list[str]) -> list[str]:
    """Return ``"- <item>"`` lines for ``items``, or ``["- (none)"]`` when empty."""
    if not items:
        return ["- (none)"]
    return [f"- {item}" for item in items]


def _drop_leading_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove ``role == "tool"`` messages from the start of ``messages`` (orphan tool results)."""
    start = 0
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1
    return messages[start:]


class ShortTermMemory:
    """Rolling window of recent chat messages sent to the LLM.

    Keeps at most ``max_messages`` messages, never starts with an orphan tool result,
    and pins the first user message (the task) so it is never trimmed away.
    """

    def __init__(self, max_messages: int = 20) -> None:
        """Create an empty window holding at most ``max_messages`` messages (>= 1)."""
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages
        self._messages: list[dict[str, Any]] = []
        self._pinned: dict[str, Any] | None = None

    def add(self, message: dict[str, Any]) -> None:
        """Append a copy of ``message`` and trim the window."""
        stored = copy.deepcopy(message)
        if self._pinned is None and stored.get("role") == "user":
            self._pinned = stored
        self._messages.append(stored)
        self._trim()

    def get(self) -> list[dict[str, Any]]:
        """Return a deep copy of the messages currently in the window."""
        return copy.deepcopy(self._messages)

    def clear(self) -> None:
        """Remove all messages and forget the pinned task message."""
        self._messages = []
        self._pinned = None

    def __len__(self) -> int:
        """Return the number of messages currently in the window."""
        return len(self._messages)

    def _pinned_index(self) -> int | None:
        """Return the index of the pinned message in the window, or None if absent."""
        if self._pinned is None:
            return None
        for index, message in enumerate(self._messages):
            if message is self._pinned:
                return index
        return None

    def _trim(self) -> None:
        """Apply the window rules: size limit, no leading tool result, pinned task message."""
        messages = self._messages
        limit = self.max_messages
        start = max(len(messages) - limit, 0)
        pinned_index = self._pinned_index()
        if pinned_index is None or pinned_index >= start:
            self._messages = _drop_leading_tool_messages(messages[start:])
            return
        tail_start = len(messages) - (limit - 1)
        tail = _drop_leading_tool_messages(messages[tail_start:]) if limit > 1 else []
        self._messages = [messages[pinned_index], *tail]
