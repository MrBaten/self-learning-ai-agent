"""Provider-switchable LLM clients.

* ``OpenAICompatibleClient`` talks to OpenAI or Groq (OpenAI-compatible API)
  through the official ``openai`` SDK, with retries on transient errors.
* ``MockLLM`` is a deterministic, rule-based offline stand-in that speaks the
  same interface (tool calls and JSON reflection). It is a simulation used for
  tests and demos when no API key / network is available. It is stateless:
  every call re-derives the next action purely from the ``messages`` list.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent.config import get_provider, load_env


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single function/tool call requested by the model."""

    id: str
    name: str
    arguments: str  # JSON-encoded object string, exactly as OpenAI returns it


class LLMResponse(BaseModel):
    """A model reply: either plain ``content`` (final answer) or ``tool_calls``."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLMError(Exception):
    """Raised on unrecoverable provider errors (missing key, auth, retries exhausted)."""


@runtime_checkable
class LLMClient(Protocol):
    """Interface shared by every LLM client."""

    provider: str
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send ``messages`` (and optional tool schemas) and return the model's reply."""
        ...


# ---------------------------------------------------------------------------
# Real providers (OpenAI / Groq)
# ---------------------------------------------------------------------------

_PROVIDER_SETTINGS: dict[str, dict[str, str | None]] = {
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": None,
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "llama-3.3-70b-versatile",
        "base_url_env": "GROQ_BASE_URL",
        "default_base_url": "https://api.groq.com/openai/v1",
    },
}

_BACKOFF_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)


class OpenAICompatibleClient:
    """Chat client for OpenAI-compatible APIs (provider ``"openai"`` or ``"groq"``).

    The API key is read from the environment (or passed in) and kept private;
    it never appears in ``repr`` or in raised error messages.
    """

    def __init__(
        self,
        provider: str,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        """Create the client.

        Args:
            provider: ``"openai"`` or ``"groq"``.
            api_key: API key; defaults to ``OPENAI_API_KEY`` / ``GROQ_API_KEY``.
            model: model name; defaults to ``OPENAI_MODEL``/``GROQ_MODEL`` or a built-in default.
            base_url: API base URL; defaults to ``OPENAI_BASE_URL`` / ``GROQ_BASE_URL``.
            temperature: sampling temperature.
            max_retries: number of retries for transient errors (backoff 0.5s, 1s, 2s, ...).

        Raises:
            ValueError: unknown provider.
            LLMError: API key missing or the SDK could not be initialised.
        """
        load_env()
        provider = (provider or "").strip().lower()
        if provider not in _PROVIDER_SETTINGS:
            raise ValueError(
                f"Unsupported provider '{provider}' for OpenAICompatibleClient. Use 'openai' or 'groq'."
            )
        settings = _PROVIDER_SETTINGS[provider]
        key_env = str(settings["key_env"])
        key = (api_key or os.environ.get(key_env, "")).strip()
        if not key:
            raise LLMError(
                f"{key_env} is not set. Put it in your environment or .env file, "
                f"or use LLM_PROVIDER=mock for the offline simulation."
            )
        self.provider: str = provider
        self.model: str = (
            model
            or os.environ.get(str(settings["model_env"]), "").strip()
            or str(settings["default_model"])
        )
        env_base = os.environ.get(str(settings["base_url_env"]), "").strip()
        self.base_url: str | None = base_url or env_base or settings["default_base_url"]
        self.temperature: float = temperature
        self.max_retries: int = max(0, int(max_retries))
        self._api_key: str = key
        self._sleep: Callable[[float], None] = time.sleep
        try:
            from openai import OpenAI

            # SDK-level retries disabled: this class implements its own backoff.
            self._client = OpenAI(api_key=key, base_url=self.base_url, max_retries=0, timeout=60.0)
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError(f"Could not initialise the {provider} client: {self._redact(str(exc))}") from None

    def __repr__(self) -> str:
        """Return a representation that never includes the API key."""
        return f"OpenAICompatibleClient(provider={self.provider!r}, model={self.model!r}, base_url={self.base_url!r})"

    def _redact(self, text: str) -> str:
        """Remove the API key (and anything that looks like a secret key) from ``text``."""
        key = getattr(self, "_api_key", "")
        if key:
            text = text.replace(key, "***")
        return re.sub(r"\b(sk|gsk)[-_][A-Za-z0-9_\-*]{6,}", r"\1-***", text)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Call ``chat.completions.create`` and convert the reply to an ``LLMResponse``.

        Transient errors (connection, timeout, rate limit, 5xx) are retried with
        exponential backoff; other errors raise ``LLMError`` immediately.
        """
        import openai

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        transient = (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )
        attempt = 0
        while True:
            try:
                completion = self._client.chat.completions.create(**kwargs)
                break
            except transient as exc:
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"{self.provider} request failed after {attempt + 1} attempt(s): "
                        f"{type(exc).__name__}: {self._redact(str(exc))}"
                    ) from None
                delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)] * (
                    2 ** max(0, attempt - len(_BACKOFF_SECONDS) + 1)
                )
                self._sleep(delay)
                attempt += 1
            except openai.OpenAIError as exc:
                raise LLMError(
                    f"{self.provider} request failed: {type(exc).__name__}: {self._redact(str(exc))}"
                ) from None
            except Exception as exc:
                raise LLMError(
                    f"{self.provider} request failed: {type(exc).__name__}: {self._redact(str(exc))}"
                ) from None

        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMError(f"{self.provider} returned no choices.")
        message = choices[0].message
        tool_calls: list[ToolCall] = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(call, "function", None)
            if function is None:
                continue
            tool_calls.append(
                ToolCall(
                    id=getattr(call, "id", None) or f"call_{index + 1}",
                    name=getattr(function, "name", "") or "",
                    arguments=getattr(function, "arguments", None) or "{}",
                )
            )
        return LLMResponse(content=getattr(message, "content", None), tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Mock provider: helpers
# ---------------------------------------------------------------------------

_WEEKDAYS: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_WEEKDAY_ALT = "|".join(_WEEKDAYS)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
_FILENAME_RE = re.compile(r"[\w\-./]+\.(?:txt|md|json|csv|py)\b", re.IGNORECASE)
_AGENT_NOTE_RE = re.compile(r"\s*\[agent note\].*\Z", re.DOTALL)
_PREF_VERBS: dict[str, str] = {
    "like": "likes",
    "love": "loves",
    "prefer": "prefers",
    "enjoy": "enjoys",
    "hate": "hates",
    "dislike": "dislikes",
}
_MAX_TOOL_CALLS = 6

FALLBACK_ANSWER = (
    "I can help with email, calendar events, files, web search and your memory. "
    "Could you rephrase your request?"
)
LESSON_ISO_TEMPLATE = (
    "Calendar times must be ISO 8601 with the user's UTC offset (YYYY-MM-DDTHH:MM{offset}); "
    "convert relative times before calling create_calendar_event."
)
LESSON_ISO = LESSON_ISO_TEMPLATE.format(offset="+HH:MM")
EMAIL_FOOTER_LINE = "-- Sent via AI assistant"
LESSON_FOOTER = f"End every email body with the exact line '{EMAIL_FOOTER_LINE}' (outbox policy)."
_OFFSET_RE = re.compile(r"[+-]\d{2}:\d{2}")
_FOOTER_ERR_RE = re.compile(r"exact line '([^']+)'")


def lesson_iso(error_text: str = "") -> str:
    """Return the calendar-time lesson, naming the UTC offset found in ``error_text`` if any."""
    match = re.search(r"UTC offset is ([+-]\d{2}:\d{2})", error_text)
    return LESSON_ISO_TEMPLATE.format(offset=match.group(1) if match else "+HH:MM")
LESSON_EMAIL = (
    "When only a contact name is given, call get_user_preference first to look up their email "
    "address before calling send_email."
)
LESSON_FILE = "If a file is not found, use the exact filename from the available workspace files listed in the error."
LESSON_CANCELLED = "The user declined an action; do not retry it without new instructions."
LESSON_GENERIC_ERROR = (
    "When a tool returns an error, read the error message and change the arguments or approach "
    "instead of repeating the same call."
)


def parse_natural_time(phrase: str, now: datetime) -> str | None:
    """Convert a natural-language time phrase to ``YYYY-MM-DDTHH:MM``.

    Supported date parts: ``today``, ``tomorrow``, weekday names (next
    occurrence strictly after today), ``YYYY-MM-DD``. Supported time parts:
    ``at H``, ``at H:MM``, optional ``am``/``pm`` (``H am`` also without ``at``),
    ``HH:MM``, ``noon``, ``midnight``. A date without a time defaults to 10:00;
    a time without a date means today if still ahead of ``now``, else tomorrow.

    Returns:
        The ISO string, or ``None`` if nothing recognisable was found or the
        time is invalid.
    """
    try:
        text = (phrase or "").strip().lower()
        if not text:
            return None

        iso = re.search(r"\b(\d{4}-\d{2}-\d{2})[t ](\d{1,2}):(\d{2})(?::\d{2})?\b", text)
        if iso:
            parsed = datetime.fromisoformat(f"{iso.group(1)}T{int(iso.group(2)):02d}:{iso.group(3)}")
            return parsed.strftime("%Y-%m-%dT%H:%M")

        day: date | None = None
        explicit_date = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if explicit_date:
            day = date.fromisoformat(explicit_date.group(1))
        elif re.search(r"\btoday\b|\btonight\b", text):
            day = now.date()
        elif re.search(r"\btomorrow\b", text):
            day = now.date() + timedelta(days=1)
        else:
            weekday = re.search(rf"\b({_WEEKDAY_ALT})\b", text)
            if weekday:
                target = _WEEKDAYS.index(weekday.group(1))
                delta = (target - now.weekday()) % 7 or 7
                day = now.date() + timedelta(days=delta)

        hour: int | None = None
        minute = 0
        if re.search(r"\bnoon\b|\bmidday\b", text):
            hour = 12
        elif re.search(r"\bmidnight\b", text):
            hour = 0
        else:
            ampm = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", text)
            if ampm:
                h = int(ampm.group(1))
                minute = int(ampm.group(2) or 0)
                if not 1 <= h <= 12:
                    return None
                is_pm = ampm.group(3).startswith("p")
                hour = (h % 12) + (12 if is_pm else 0)
            else:
                at_time = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\b", text) or re.search(
                    r"(?<![\d-])(\d{1,2}):(\d{2})\b", text
                )
                if at_time:
                    hour = int(at_time.group(1))
                    minute = int(at_time.group(2) or 0)
        if hour is not None and not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None

        if day is None and hour is None:
            return None
        if hour is None:
            hour, minute = 10, 0
        if day is None:
            candidate = datetime.combine(now.date(), datetime.min.time()).replace(hour=hour, minute=minute)
            day = now.date() if candidate > now.replace(second=0, microsecond=0) else now.date() + timedelta(days=1)
        return datetime(day.year, day.month, day.day, hour, minute).strftime("%Y-%m-%dT%H:%M")
    except (ValueError, TypeError, AttributeError):
        return None


def _text_of(content: Any) -> str:
    """Return message content as plain text (handles ``None`` and content-part lists)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _strip_agent_note(text: str) -> str:
    """Remove a trailing ``[agent note] ...`` appended by the agent loop."""
    return _AGENT_NOTE_RE.sub("", text).strip()


def _status_of(observation: str) -> str:
    """Classify a tool observation as ``"error"``, ``"cancelled"`` or ``"ok"``."""
    head = observation.lstrip()
    if head.startswith("ERROR"):
        return "error"
    if head.startswith("CANCELLED"):
        return "cancelled"
    return "ok"


class _Step(BaseModel):
    """One executed tool call reconstructed from the message history."""

    name: str
    args: dict[str, Any]
    observation: str
    status: str


class _Context(BaseModel):
    """Everything the mock policy derives from the incoming messages."""

    now: datetime
    user_name: str = "User"
    learnings: list[str] = Field(default_factory=list)
    task: str = ""
    history: list[_Step] = Field(default_factory=list)
    tool_calls_so_far: int = 0


class _Decision(BaseModel):
    """Policy output: a tool call (``tool`` set) or a final answer (``final`` set)."""

    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    thought: str | None = None
    final: str | None = None


def _final(text: str) -> _Decision:
    """Build a final-answer decision."""
    return _Decision(final=text)


def _call(tool: str, args: dict[str, Any], thought: str | None = None) -> _Decision:
    """Build a tool-call decision."""
    return _Decision(tool=tool, args=args, thought=thought)


def _parse_args(raw: Any) -> dict[str, Any]:
    """Decode tool-call arguments that may be a JSON string or already a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _build_context(messages: list[dict[str, Any]]) -> _Context:
    """Parse system prompt, task and tool history out of ``messages``."""
    system = ""
    task = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system" and not system:
            system = _text_of(msg.get("content"))
        if msg.get("role") == "user" and not task:
            task = _text_of(msg.get("content")).strip()

    now = datetime.now().replace(second=0, microsecond=0)
    match = re.search(r"Current date/time:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})", system)
    if match:
        try:
            now = datetime.fromisoformat(match.group(1))
        except ValueError:
            pass

    user_name = "User"
    name_match = re.search(r"^\s*User name:\s*(.+?)\s*$", system, re.MULTILINE)
    if name_match:
        user_name = name_match.group(1)

    learnings: list[str] = []
    section = re.search(r"^##\s*Past Learnings\s*$(.*?)(?=^##\s|\Z)", system, re.MULTILINE | re.DOTALL)
    if section:
        for line in section.group(1).splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")):
                lesson = stripped[2:].strip()
                if lesson:
                    learnings.append(lesson)

    results: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            results[str(msg.get("tool_call_id", ""))] = _strip_agent_note(_text_of(msg.get("content")))

    history: list[_Step] = []
    count = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            count += 1
            function = call.get("function") or {}
            name = str(function.get("name") or call.get("name") or "")
            call_id = str(call.get("id", ""))
            if call_id in results:
                observation = results[call_id]
                history.append(
                    _Step(
                        name=name,
                        args=_parse_args(function.get("arguments", call.get("arguments"))),
                        observation=observation,
                        status=_status_of(observation),
                    )
                )
    return _Context(
        now=now,
        user_name=user_name,
        learnings=learnings,
        task=task,
        history=history,
        tool_calls_so_far=count,
    )


def _has_lesson(ctx: _Context, *needles: str) -> bool:
    """True if any Past Learning contains all ``needles`` (case-insensitive)."""
    return any(all(n.lower() in lesson.lower() for n in needles) for lesson in ctx.learnings)


def _last(ctx: _Context) -> _Step | None:
    """Return the most recent executed tool call, if any."""
    return ctx.history[-1] if ctx.history else None


def _error_text(observation: str) -> str:
    """Return an error observation without its ``ERROR:`` prefix."""
    return re.sub(r"^\s*ERROR:\s*", "", observation).strip()


def _could_not_complete(step: _Step) -> _Decision:
    """Generic final answer for an error the current intent does not handle."""
    return _final(f"I could not complete the task: {_error_text(step.observation)}")


def _clean_fragment(text: str) -> str:
    """Strip whitespace, surrounding quotes and trailing punctuation from a fragment."""
    text = text.strip().strip("\"'“”‘’").strip()
    return re.sub(r"[.!?]+$", "", text).strip()


# ---------------------------------------------------------------------------
# Mock provider: intent policies
# ---------------------------------------------------------------------------


def _policy_write_file(ctx: _Context, path: str, content: str) -> _Decision:
    """Write ``content`` to ``path`` with write_file, then report."""
    last = _last(ctx)
    if last is None:
        return _call("write_file", {"path": path, "content": content}, f"Writing the file {path}.")
    if last.name == "write_file" and last.status == "ok":
        return _final(f"Done: {last.observation.rstrip('.')}.")
    if last.status == "cancelled":
        return _final("Okay — I did not write the file.")
    return _could_not_complete(last) if last.status == "error" else _final(FALLBACK_ANSWER)


def _policy_read_file(ctx: _Context, path: str) -> _Decision:
    """Read ``path``; on File-not-found retry once with the closest available name."""
    last = _last(ctx)
    if last is None:
        return _call("read_file", {"path": path}, f"Reading {path}.")
    if last.name != "read_file":
        return _could_not_complete(last) if last.status == "error" else _final(FALLBACK_ANSWER)
    used_path = str(last.args.get("path", path))
    if last.status == "ok":
        lines = [line.strip() for line in last.observation.splitlines() if line.strip()]
        summary = " | ".join(lines[:3]) if lines else "(the file is empty)"
        return _final(f"Summary of {used_path}: {summary}")
    if "File not found" in last.observation:
        listing = re.search(r"Available files in workspace:\s*(.*)", last.observation)
        files = [f.strip() for f in listing.group(1).split(",")] if listing else []
        files = [f for f in files if f and f != "(none)"]
        reads = [s for s in ctx.history if s.name == "read_file"]
        tried = {str(s.args.get("path", "")) for s in reads}
        if len(reads) < 2:
            close = difflib.get_close_matches(used_path, files, n=1, cutoff=0.3)
            if close and close[0] not in tried:
                return _call(
                    "read_file",
                    {"path": close[0]},
                    f"'{used_path}' was not found; the closest workspace file is '{close[0]}'.",
                )
        available = ", ".join(files) if files else "(none)"
        return _final(f"I could not find the file '{path}'. Available files in workspace: {available}")
    return _could_not_complete(last)


def _email_subject_and_body(task: str, greeting_name: str, user_name: str) -> tuple[str, str]:
    """Draft a short subject and body for an email request."""
    topic_raw = ""
    about = re.search(r"\babout\s+(.+)$", task, re.IGNORECASE | re.DOTALL)
    saying = re.search(r"\bsaying\s+(.+)$", task, re.IGNORECASE | re.DOTALL)
    if about:
        topic_raw = _clean_fragment(about.group(1))
        subject_src = re.sub(
            rf"\s+\b(today|tonight|tomorrow|next\s+\w+|this\s+\w+|on\s+\w+|at\s+.*|{_WEEKDAY_ALT})\b.*$",
            "",
            topic_raw,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip() or topic_raw
        words = subject_src.split()[:6]
        subject = " ".join(w[:1].upper() + w[1:] for w in words) or "Quick note"
        sentence = f"I wanted to reach out about {topic_raw}."
    elif saying:
        topic_raw = _clean_fragment(saying.group(1))
        words = topic_raw.split()[:6]
        subject = " ".join(w[:1].upper() + w[1:] for w in words) or "Quick note"
        sentence = (topic_raw[:1].upper() + topic_raw[1:] + ".") if topic_raw else "Just a quick note."
    else:
        subject = "Quick note"
        sentence = "Just a quick note to get in touch."
    body = f"Hi {greeting_name},\n\n{sentence}\n\nBest,\n{user_name}"
    return subject, body


def _email_recipient_name(task: str) -> str | None:
    """Return the capitalised contact name after ``to`` / ``email`` in the task, if any."""
    stop = {"me", "i", "the", "my", "a", "an", "him", "her", "them", "us"}
    # Lookahead so overlapping matches work ("email to John": "email"->"to", then "to"->"John").
    for match in re.finditer(r"(?i:\b(?:to|email))\s+(?=([A-Za-z][A-Za-z'\-]*))", task):
        word = re.sub(r"'s$", "", match.group(1))
        if word[:1].isupper() and word.lower() not in stop:
            return word
    return None


def _find_contact_email(name: str, observation: str) -> str | None:
    """Look up ``name``'s email address in a get_user_preference JSON observation."""
    entries: list[str] = []
    try:
        data = json.loads(observation)
        if isinstance(data, dict):
            for key in ("facts", "preferences"):
                value = data.get(key)
                if isinstance(value, list):
                    entries.extend(str(v) for v in value)
    except (json.JSONDecodeError, TypeError):
        entries = observation.splitlines()
    pattern = re.compile(rf"\b{re.escape(name)}\b.*?({_EMAIL_RE.pattern})", re.IGNORECASE)
    for entry in entries:
        found = pattern.search(entry)
        if found:
            return found.group(1)
    return None


def _policy_email(ctx: _Context) -> _Decision:
    """Send an email, looking up contact addresses in memory when only a name is given."""
    task = ctx.task
    address_match = _EMAIL_RE.search(task)
    address = address_match.group(0) if address_match else None
    name = None if address else _email_recipient_name(task)
    subject, body = _email_subject_and_body(task, name or "there", ctx.user_name)

    footer_known = _has_lesson(ctx, "Sent via AI assistant") or any(
        s.name == "send_email" and "outbox policy" in s.observation for s in ctx.history
    )
    if footer_known:
        body = f"{body}\n\n{EMAIL_FOOTER_LINE}"

    def send(to: str, thought: str) -> _Decision:
        """Build the send_email call for ``to``."""
        return _call("send_email", {"to": to, "subject": subject, "body": body}, thought)

    last = _last(ctx)
    if last is None:
        if address:
            return send(address, f"Sending the email to {address}.")
        if name:
            if _has_lesson(ctx, "email", "get_user_preference"):
                return _call(
                    "get_user_preference",
                    {},
                    f"Past Learnings say to look up a contact's email first, so I'll find {name}'s "
                    "address with get_user_preference.",
                )
            return send(name, f"Sending the email to {name}.")
        return _final("Who should I send the email to? Please give me a name or an email address.")

    if last.name == "send_email":
        to = str(last.args.get("to", ""))
        if last.status == "ok":
            return _final(f"Email sent to {to} with subject '{last.args.get('subject', subject)}'.")
        if last.status == "cancelled":
            return _final("Okay — I did not send the email.")
        if "outbox policy" in last.observation:
            footer = _FOOTER_ERR_RE.search(last.observation)
            already = sum(1 for s in ctx.history if s.name == "send_email" and "outbox policy" in s.observation)
            if footer and already == 1 and EMAIL_FOOTER_LINE in footer.group(1):
                return send(
                    to,
                    f"The outbox requires the body to end with '{footer.group(1)}'. Adding it and resending.",
                )
        looked_up = any(s.name == "get_user_preference" for s in ctx.history)
        if "is not a valid email address" in last.observation and not looked_up and (name or to):
            return _call(
                "get_user_preference",
                {},
                f"'{to}' is not an email address. Contacts are stored in memory, so I'll look up "
                f"{name or to}'s email with get_user_preference.",
            )
        return _could_not_complete(last)

    if last.name == "get_user_preference":
        if last.status != "ok":
            return _could_not_complete(last)
        who = name or ""
        found = _find_contact_email(who, last.observation) if who else None
        if found:
            already = any(s.name == "send_email" and s.args.get("to") == found for s in ctx.history)
            if not already:
                return send(found, f"Found {who}'s address in memory: {found}. Sending the email.")
        return _final(
            f"I couldn't find an email address for {who or 'that contact'} in your memory. "
            "What is their email address?"
        )

    return _could_not_complete(last) if last.status == "error" else _final(FALLBACK_ANSWER)


def _calendar_title_and_phrase(task: str) -> tuple[str, str]:
    """Extract the event title and the raw natural-language time phrase from the task."""
    stop_words = rf"at|on|for|tomorrow|today|tonight|next|{_WEEKDAY_ALT}"
    title = "Meeting"
    search_from = 0
    title_match = re.search(
        rf"\b(?:called|titled|named)\s+(.+?)(?=\s+(?:{stop_words})\b|[.?!,]?\s*$)",
        task,
        re.IGNORECASE | re.DOTALL,
    )
    if title_match:
        candidate = _clean_fragment(title_match.group(1))
        if candidate:
            title = candidate
        search_from = title_match.end()
    time_re = re.compile(rf"\b(today|tonight|tomorrow|{_WEEKDAY_ALT}|on|at)\b", re.IGNORECASE)
    time_match = time_re.search(task, search_from) or time_re.search(task)
    phrase = _clean_fragment(task[time_match.start():]) if time_match else ""
    return title, phrase or "tomorrow at 10 AM"


def _policy_calendar(ctx: _Context) -> _Decision:
    """Create a calendar event, converting the time to ISO 8601 (up front if a lesson says so)."""
    title, phrase = _calendar_title_and_phrase(ctx.task)
    base_iso = parse_natural_time(phrase, ctx.now)
    lesson_offset = next(
        (m.group(0) for line in ctx.learnings if "ISO 8601" in line for m in [_OFFSET_RE.search(line)] if m),
        None,
    )
    error_offset = next(
        (
            m.group(1)
            for st in ctx.history
            if st.name == "create_calendar_event"
            for m in [re.search(r"UTC offset is ([+-]\d{2}:\d{2})", st.observation)]
            if m
        ),
        None,
    )
    offset = error_offset or lesson_offset
    iso = f"{base_iso}{offset}" if base_iso and offset else base_iso
    last = _last(ctx)
    if last is None:
        if _has_lesson(ctx, "ISO 8601") and iso:
            return _call(
                "create_calendar_event",
                {"title": title, "time": iso},
                f"Past Learnings say calendar times need ISO 8601 with the UTC offset: '{phrase}' -> {iso}.",
            )
        return _call(
            "create_calendar_event",
            {"title": title, "time": phrase},
            f"Creating the event '{title}' for '{phrase}'.",
        )
    if last.name == "create_calendar_event":
        if last.status == "ok":
            return _final(f"Created calendar event '{last.args.get('title', title)}' at {last.args.get('time', iso)}.")
        if last.status == "cancelled":
            return _final("Okay — I did not create the event.")
        tried = {str(s.args.get("time", "")) for s in ctx.history if s.name == "create_calendar_event"}
        if "ISO 8601" in last.observation and iso and iso not in tried:
            return _call(
                "create_calendar_event",
                {"title": title, "time": iso},
                f"The tool needs ISO 8601 with the UTC offset. Converting '{phrase}' to {iso} and retrying.",
            )
    return _could_not_complete(last) if last.status == "error" else _final(FALLBACK_ANSWER)


def _memory_content(task: str) -> tuple[str, str]:
    """Normalise a 'save/remember' request into (content, kind)."""
    original = task.strip()
    verb_re = r"(?:save|remember|note)(?:\s+down)?(?:\s+(?:that|this))?(?:\s+in\s+memory)?[:,]?\s+"
    text = re.sub(rf"^\s*(?:please\s+)?{verb_re}", "", original, count=1, flags=re.IGNORECASE)
    if text == original and not re.match(r"^\s*i\s+(?:like|love|prefer|enjoy|hate|dislike)\b", text, re.I):
        # e.g. "Can you remember that ..." -> text after the first save/remember/note verb.
        match = re.search(rf"\b{verb_re}(.+)$", original, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1)
    text = re.sub(r"^\s*that\s+", "", text, flags=re.IGNORECASE)
    pref = re.match(r"^\s*i\s+(like|love|prefer|enjoy|hate|dislike)\s+(.+)$", text, re.IGNORECASE | re.DOTALL)
    if pref:
        content = f"{_PREF_VERBS[pref.group(1).lower()]} {_clean_fragment(pref.group(2))}"
        return content, "preference"
    content = _clean_fragment(text)
    kind = (
        "preference"
        if re.search(r"\b(like|likes|love|loves|prefer|prefers|enjoy|enjoys|hate|hates|dislike|dislikes)\b", content, re.I)
        else "fact"
    )
    return content, kind


def _policy_save_memory(ctx: _Context) -> _Decision:
    """Save a preference or fact with save_to_memory."""
    content, kind = _memory_content(ctx.task)
    last = _last(ctx)
    if last is None:
        if not content:
            return _final("What would you like me to remember?")
        return _call("save_to_memory", {"content": content, "kind": kind}, f"Saving this {kind} to long-term memory.")
    if last.name == "save_to_memory" and last.status == "ok":
        saved = str(last.args.get("content", content))
        if last.observation.startswith("Already in memory"):
            return _final(f"Saved to memory: {saved} (it was already there).")
        return _final(f"Saved to memory: {saved}")
    return _could_not_complete(last) if last.status == "error" else _final(FALLBACK_ANSWER)


def _policy_memory_question(ctx: _Context) -> _Decision:
    """Answer questions about the user (name, preferences) from get_user_preference."""
    last = _last(ctx)
    if last is None:
        return _call("get_user_preference", {}, "Checking what I know about you in long-term memory.")
    if last.name != "get_user_preference" or last.status != "ok":
        return _could_not_complete(last) if last.status == "error" else _final(FALLBACK_ANSWER)
    try:
        data = json.loads(last.observation)
        if not isinstance(data, dict):
            data = {}
    except json.JSONDecodeError:
        data = {}
    user_name = str(data.get("user_name") or ctx.user_name)
    prefs = [str(p) for p in data.get("preferences") or []]
    facts = [str(f) for f in data.get("facts") or []]
    low = ctx.task.lower()
    pref_text = "; ".join(prefs) if prefs else "none saved yet"
    if "about me" in low:
        fact_text = "; ".join(facts) if facts else "none saved yet"
        return _final(f"Your name is {user_name}. Preferences: {pref_text}. Facts: {fact_text}.")
    if "my preferences" in low or "what do i like" in low:
        if not prefs:
            return _final("I don't have any preferences saved for you yet.")
        return _final(f"Your preferences: {pref_text}.")
    return _final(f"Your name is {user_name}.")


def _policy_search(ctx: _Context) -> _Decision:
    """Run search_web and summarise the first two result lines."""
    last = _last(ctx)
    if last is None:
        query = re.sub(
            r"^\s*(?:please\s+)?(?:search\s+(?:the\s+web|the\s+internet|online|google)?\s*(?:for\s+)?|look\s+up\s+|find\s+(?:me\s+)?)",
            "",
            ctx.task,
            flags=re.IGNORECASE,
        )
        query = _clean_fragment(query) or ctx.task.strip()
        return _call("search_web", {"query": query}, f"Searching the web for '{query}'.")
    if last.name == "search_web" and last.status == "ok":
        lines = [line.strip() for line in last.observation.splitlines() if line.strip()]
        numbered = [line for line in lines if re.match(r"^\d+[.)]\s", line)]
        picked = numbered[:2] or [line for line in lines if not line.startswith("[")][:2] or lines[:2]
        return _final(f"Here is what I found: {' | '.join(picked)}")
    return _could_not_complete(last) if last.status == "error" else _final(FALLBACK_ANSWER)


def _decide(ctx: _Context) -> _Decision:
    """Pick the next action for the task (intent order per PLAN §4.2)."""
    task = ctx.task
    low = task.lower()
    last = _last(ctx)

    filename_match = _FILENAME_RE.search(task)
    filename = filename_match.group(0).rstrip(".") if filename_match else None

    # 1. file write
    if filename and re.search(r"\b(write|create|save)\b", low):
        marker = re.search(
            r"\b(?:with\s+the\s+content|with\s+content|with\s+the\s+text|with\s+text|containing|saying)\b[:\s]*",
            task,
            re.IGNORECASE,
        )
        if marker:
            content = _clean_fragment(task[marker.end():])
            if content:
                return _policy_write_file(ctx, filename, content)
        if last is None and not re.search(r"\b(read|open|show|summari[sz]e)\b", low):
            return _final(f"What content should I write to {filename}?")

    # 2. file read
    if filename and re.search(r"\b(read|open|show|summari[sz]e)\b", low):
        return _policy_read_file(ctx, filename)

    # Explicit memory statements ("Save that ...", "Remember that ...", "I like ...") are
    # saved even if they mention email/meetings (e.g. "Remember that Bob's email is ...").
    explicit_memory = re.match(
        r"^\s*(?:please\s+)?(?:(?:save|remember|note)(?:\s+down)?\s+that\b|i\s+(?:like|love|prefer|enjoy|hate|dislike)\b)",
        low,
    )
    if explicit_memory:
        return _policy_save_memory(ctx)

    # 3. email
    if "email" in low or "mail " in low:
        return _policy_email(ctx)

    # 4. calendar
    if re.search(r"calendar|event|meeting|schedule|appointment", low):
        return _policy_calendar(ctx)

    # 5. save memory
    if re.search(r"\b(save|remember|note)\b", low):
        return _policy_save_memory(ctx)

    # 6. name / memory question
    if re.search(r"my name|who am i|my preferences|what do i like|about me", low):
        return _policy_memory_question(ctx)

    # 7. web search
    if re.search(r"search|look up|find|latest|news", low):
        return _policy_search(ctx)

    # 8. fallback
    if last is not None and last.status == "error":
        return _could_not_complete(last)
    return _final(FALLBACK_ANSWER)


# ---------------------------------------------------------------------------
# Mock provider: reflection (json_mode)
# ---------------------------------------------------------------------------


def _reflection_json(messages: list[dict[str, Any]]) -> str:
    """Build a Reflection JSON string from the reflection user prompt (PLAN §4.6 format)."""
    prompt = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            prompt = _text_of(msg.get("content"))

    task_match = re.search(r"Task:\s*(.*?)\s*(?:\nFinal answer:|\Z)", prompt, re.DOTALL)
    task = task_match.group(1).strip() if task_match else ""
    transcript_match = re.search(r"\nTranscript:\s*\n(.*)\Z", prompt, re.DOTALL)
    transcript = transcript_match.group(1) if transcript_match else prompt

    tools: list[str] = []
    errors: list[str] = []
    cancelled: list[str] = []
    step_re = re.compile(r"^-\s*step\s+\d+:\s*(.*)$", re.IGNORECASE)
    call_re = re.compile(r"^([\w\-]+)\((.*?)\)\s*->\s*(OK|ERROR|CANCELLED)\b:?\s?(.*)$", re.IGNORECASE | re.DOTALL)
    structured = False
    for line in transcript.splitlines():
        step = step_re.match(line.strip())
        if not step:
            continue
        structured = True
        call = call_re.match(step.group(1).strip())
        if not call:
            continue
        tools.append(call.group(1))
        status = call.group(3).upper()
        observation = call.group(4)
        if status == "ERROR":
            errors.append(observation)
        elif status == "CANCELLED":
            cancelled.append(observation)
    if not structured:
        errors = [line for line in transcript.splitlines() if "ERROR" in line]
        cancelled = [line for line in transcript.splitlines() if "CANCELLED" in line]

    count_match = re.search(r"Tool errors:\s*(\d+)", transcript)
    error_count = max(len(errors), int(count_match.group(1)) if count_match else 0)

    lessons: list[str] = []

    def add(lesson: str) -> None:
        """Append ``lesson`` once, keeping at most 3."""
        if lesson not in lessons and len(lessons) < 3:
            lessons.append(lesson)

    for err in errors:
        if "ISO 8601" in err:
            add(lesson_iso(err))
        if "outbox policy" in err:
            add(LESSON_FOOTER)
        if "is not a valid email address" in err:
            add(LESSON_EMAIL)
        if "File not found" in err:
            add(LESSON_FILE)
    if cancelled:
        add(LESSON_CANCELLED)
    if error_count and not lessons:
        add(LESSON_GENERIC_ERROR)
    if not lessons:
        chain: list[str] = []
        for name in tools:
            if not chain or chain[-1] != name:
                chain.append(name)
        snippet = task.strip()[:40].rstrip() or "this request"
        if chain:
            add(f"For tasks like '{snippet}', calling {' -> '.join(chain)} worked without errors.")
        else:
            add(f"For tasks like '{snippet}', answering directly worked without errors.")

    new_preferences: list[str] = []
    pref = re.search(r"\bi\s+(like|love|prefer|enjoy)\s+(.+)$", task, re.IGNORECASE | re.DOTALL)
    if pref and "save_to_memory" not in tools:
        thing = _clean_fragment(pref.group(2))
        if thing:
            new_preferences.append(f"{_PREF_VERBS[pref.group(1).lower()]} {thing}")

    n_calls = len(tools)
    if n_calls:
        went_well = f"Finished the task using {n_calls} tool call(s)."
    else:
        went_well = "Answered the task directly without tools."
    if error_count and cancelled:
        went_wrong = f"{error_count} tool call(s) failed and the user declined {len(cancelled)} action(s)."
    elif error_count:
        went_wrong = f"{error_count} tool call(s) returned errors before the agent adjusted its approach."
    elif cancelled:
        went_wrong = f"The user declined {len(cancelled)} action(s)."
    else:
        went_wrong = "Nothing went wrong."

    return json.dumps(
        {
            "went_well": went_well,
            "went_wrong": went_wrong,
            "lessons": lessons,
            "new_preferences": new_preferences,
            "new_facts": [],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Mock provider: client
# ---------------------------------------------------------------------------


class MockLLM:
    """Deterministic, rule-based offline LLM (a simulation, not a real model).

    Stateless: each ``chat`` call re-derives the next action from ``messages``
    (system prompt date/name/Past Learnings, the first user message as the task,
    and prior assistant tool calls paired with their tool results).
    """

    def __init__(self) -> None:
        """Initialise the mock with its fixed provider/model names."""
        self.provider: str = "mock"
        self.model: str = "mock-rule-based-v1"

    def __repr__(self) -> str:
        """Return a short representation."""
        return f"MockLLM(model={self.model!r})"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return the next simulated reply. Never raises.

        ``json_mode=True`` returns a Reflection JSON object; otherwise a tool
        call (only for tools offered in ``tools``) or a final answer.
        """
        try:
            safe_messages = [m for m in (messages or []) if isinstance(m, dict)]
            if json_mode:
                return LLMResponse(content=_reflection_json(safe_messages))
            ctx = _build_context(safe_messages)
            decision = _decide(ctx)
            if decision.tool is None:
                return LLMResponse(content=decision.final or FALLBACK_ANSWER)

            offered = {
                str((t.get("function") or {}).get("name", t.get("name", "")))
                for t in (tools or [])
                if isinstance(t, dict)
            }
            if decision.tool not in offered:
                return LLMResponse(
                    content=f"I would need the {decision.tool} tool for this, but it is not available right now."
                )
            if ctx.tool_calls_so_far >= _MAX_TOOL_CALLS:
                last = _last(ctx)
                detail = f" Last result: {last.observation}" if last else ""
                return LLMResponse(
                    content=f"I could not complete the task within {_MAX_TOOL_CALLS} tool calls.{detail}"
                )
            call = ToolCall(
                id=f"call_{ctx.tool_calls_so_far + 1}",
                name=decision.tool,
                arguments=json.dumps(decision.args, ensure_ascii=False),
            )
            return LLMResponse(content=decision.thought, tool_calls=[call])
        except Exception:
            if json_mode:
                return LLMResponse(
                    content=json.dumps(
                        {
                            "went_well": "",
                            "went_wrong": "",
                            "lessons": ["Review the task outcome and keep the approach that worked."],
                            "new_preferences": [],
                            "new_facts": [],
                        }
                    )
                )
            return LLMResponse(content="Sorry, I could not work out how to help with that. Could you rephrase your request?")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm(provider: str | None = None) -> LLMClient:
    """Return an LLM client for ``provider`` (``None`` -> ``config.get_provider()``).

    Raises:
        ValueError: unknown provider name.
        LLMError: the chosen real provider has no API key.
    """
    name = (provider or get_provider()).strip().lower()
    if name == "mock":
        return MockLLM()
    if name in _PROVIDER_SETTINGS:
        return OpenAICompatibleClient(name)
    raise ValueError(f"Unknown LLM provider '{name}'. Expected one of: openai, groq, mock.")
