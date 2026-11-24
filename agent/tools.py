"""Tool layer for the self-learning agent.

Every tool has a Pydantic argument model (used both for validation and for the
OpenAI function schema), a plain Python function that returns a string, and an
optional ``requires_confirmation`` flag. :func:`execute_tool` is the single
entry point used by the agent loop: it parses and validates arguments, runs
pre-checks, enforces the user-confirmation gate for side-effecting tools and
converts every failure into a :class:`ToolResult` -- it never raises.

The e-mail and calendar tools are mocks: they only append records to JSON files
in ``data_dir()`` and never contact any external service.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent.config import data_dir, workspace_dir
from agent.memory import add_fact, add_preference, load_memory

ConfirmFn = Callable[[str, dict[str, Any]], bool]
"""Signature of a confirmation callback: ``(tool_name, validated_args) -> approved?``."""

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ISO_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2}|Z)$")
# Outbox policy: every e-mail body must end with this exact line. The tool schema
# deliberately does not mention it - the agent discovers it from the error and learns it.
EMAIL_FOOTER = "-- Sent via AI assistant"
MAX_READ_CHARS = 20_000
MAX_LISTED_FILES = 50
SEARCH_TIMEOUT_SECONDS = 5
DUCKDUCKGO_URL = "https://api.duckduckgo.com/"


# --------------------------------------------------------------------------- #
# Result / tool models
# --------------------------------------------------------------------------- #
class ToolResult(BaseModel):
    """Outcome of one tool execution as seen by the agent loop."""

    tool: str
    status: Literal["ok", "error", "cancelled"]
    output: str

    @property
    def ok(self) -> bool:
        """Return True when the tool ran successfully (status ``"ok"``)."""
        return self.status == "ok"


class Tool(BaseModel):
    """A callable tool: metadata, argument model, implementation and safety flags.

    ``precheck`` (optional) receives the validated argument model and returns an
    error string (starting with ``"ERROR: "``) or None. :func:`execute_tool`
    runs it BEFORE asking for confirmation, so invalid requests are rejected
    without bothering the user.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., str]
    requires_confirmation: bool = False
    precheck: Callable[[Any], str | None] | None = None


# --------------------------------------------------------------------------- #
# Argument models
# --------------------------------------------------------------------------- #
class SearchWebArgs(BaseModel):
    """Arguments for search_web."""

    query: str = Field(min_length=1, description="The search query")


class ReadFileArgs(BaseModel):
    """Arguments for read_file."""

    path: str = Field(description="Path of the file to read, relative to the workspace (e.g. notes.txt)")


class WriteFileArgs(BaseModel):
    """Arguments for write_file."""

    path: str = Field(description="Path of the file to write, relative to the workspace (e.g. ideas.txt)")
    content: str = Field(description="Full text content to write to the file")


class SendEmailArgs(BaseModel):
    """Arguments for send_email."""

    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Plain-text email body")

    @field_validator("to")
    @classmethod
    def _strip_to(cls, value: str) -> str:
        """Strip surrounding whitespace from the recipient."""
        return value.strip()


class CreateCalendarEventArgs(BaseModel):
    """Arguments for create_calendar_event."""

    title: str = Field(description="Event title")
    time: str = Field(description="Event start time")

    @field_validator("time")
    @classmethod
    def _strip_time(cls, value: str) -> str:
        """Strip surrounding whitespace from the time string."""
        return value.strip()


class GetUserPreferenceArgs(BaseModel):
    """Arguments for get_user_preference (none)."""


class SaveToMemoryArgs(BaseModel):
    """Arguments for save_to_memory."""

    content: str = Field(
        min_length=1,
        description="The preference or fact to remember, e.g. 'likes green tea' or \"Tom's email is tom@example.com\"",
    )
    kind: Literal["preference", "fact"] = Field(
        default="preference",
        description="'preference' for likes/dislikes/habits, 'fact' for other information such as contacts",
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_env_name(name: str) -> bool:
    """Return True for secret env-file names (``.env`` and ``.env.*``)."""
    return name == ".env" or name.startswith(".env.")


def _safe_path(path: str) -> Path:
    """Resolve ``path`` to an absolute path guaranteed to lie inside ``workspace_dir()``.

    Relative paths resolve against the workspace; a leading ``workspace/`` or
    ``data/workspace/`` prefix is stripped first. Absolute paths are accepted only
    if they are inside the workspace. Any ``.env`` path component is refused.

    Raises:
        PermissionError: with a message starting ``"Access denied: "``.
    """
    workspace = workspace_dir().resolve()
    raw = str(path).strip()
    cleaned = raw
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    for prefix in ("data/workspace/", "workspace/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    candidate = Path(cleaned)
    if any(_is_env_name(part) for part in candidate.parts):
        raise PermissionError(f"Access denied: '{raw}' refers to an environment/secrets file (.env).")

    target = candidate if candidate.is_absolute() else workspace / candidate
    resolved = target.resolve()
    if not resolved.is_relative_to(workspace):
        raise PermissionError(
            f"Access denied: '{raw}' is outside the workspace. Only files inside the workspace can be accessed."
        )
    if any(_is_env_name(part) for part in resolved.relative_to(workspace).parts):
        raise PermissionError(f"Access denied: '{raw}' refers to an environment/secrets file (.env).")
    return resolved


def _relative_to_workspace(path: Path) -> str:
    """Return ``path`` relative to the workspace as a POSIX string."""
    return path.relative_to(workspace_dir().resolve()).as_posix()


def _list_workspace_files() -> list[str]:
    """List workspace files (recursive, POSIX relative paths, sorted), excluding .env files."""
    workspace = workspace_dir().resolve()
    files: list[str] = []
    for item in workspace.rglob("*"):
        try:
            rel = item.relative_to(workspace)
        except ValueError:
            continue
        if any(_is_env_name(part) for part in rel.parts):
            continue
        if item.is_file():
            files.append(rel.as_posix())
    return sorted(files)


def _available_files_text() -> str:
    """Return the comma-separated workspace file list, or ``"(none)"``."""
    files = _list_workspace_files()
    if not files:
        return "(none)"
    shown = files[:MAX_LISTED_FILES]
    text = ", ".join(shown)
    if len(files) > MAX_LISTED_FILES:
        text += f", ... ({len(files) - MAX_LISTED_FILES} more)"
    return text


def _append_json_list(path: Path, item: dict[str, Any]) -> None:
    """Append ``item`` to the JSON list stored at ``path`` (atomic write).

    A missing file is created; a corrupt file or one not containing a JSON list
    is replaced by a new list.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    items: list[Any] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                items = loaded
        except (OSError, ValueError):
            items = []
    items.append(item)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(items, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _now_iso() -> str:
    """Return the current local time as an ISO 8601 string (seconds precision)."""
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Validation (used both as pre-checks and inside the tool functions)
# --------------------------------------------------------------------------- #
def _email_error(to: str) -> str | None:
    """Return the invalid-address error message for ``to``, or None if it is valid."""
    if EMAIL_RE.match(to.strip()):
        return None
    return (
        f"ERROR: '{to}' is not a valid email address. If you only have a contact name, "
        "look up their email with get_user_preference (contacts are stored in facts)."
    )


def _local_utc_offset() -> str:
    """Return this machine's current UTC offset formatted as ``+HH:MM``."""
    raw = datetime.now().astimezone().strftime("%z") or "+0000"
    return f"{raw[:3]}:{raw[3:5]}"


def _time_error(time: str) -> str | None:
    """Return the invalid-time error for ``time``, or None if it is ISO 8601 with a UTC offset."""
    value = time.strip()
    offset = _local_utc_offset()
    if ISO_TIME_RE.match(value):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            given = parsed.strftime("%z")
            # The event is in the user's own timezone: "3 PM" means 3 PM local, never UTC.
            if f"{given[:3]}:{given[3:5]}" == offset:
                return None
        except ValueError:
            pass
    return (
        f"ERROR: Invalid time format '{time}'. create_calendar_event requires ISO 8601 with "
        f"the user's UTC offset: YYYY-MM-DDTHH:MM+HH:MM (e.g. 2026-11-25T11:00{offset}; the "
        f"user's UTC offset is {offset}). Convert relative times like 'tomorrow at 11 AM' "
        "using the current date."
    )


def _footer_error(body: str) -> str | None:
    """Return the outbox-policy error if ``body`` does not end with ``EMAIL_FOOTER``."""
    if body.rstrip().endswith(EMAIL_FOOTER):
        return None
    return (
        "ERROR: Email rejected by outbox policy: the body must end with the exact line "
        f"'{EMAIL_FOOTER}'."
    )


def _precheck_send_email(args: SendEmailArgs) -> str | None:
    """Pre-confirmation check for send_email: recipient address, then outbox policy."""
    return _email_error(args.to) or _footer_error(args.body)


def _precheck_calendar(args: CreateCalendarEventArgs) -> str | None:
    """Pre-confirmation check for create_calendar_event: validate the time format."""
    return _time_error(args.time)


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def _mock_search(query: str) -> str:
    """Return deterministic simulated search results that mention the query words."""
    q = " ".join(query.split())
    return "\n".join(
        [
            f"[mock search results for '{q}']",
            f"1. {q} - Overview - A summary of {q}: key points, background and common questions.",
            f"2. Latest news on {q} - Recent articles, announcements and updates about {q}.",
            f"3. {q} explained - A practical guide to {q} with examples and further reading.",
        ]
    )


def _live_search(query: str) -> str | None:
    """Query the DuckDuckGo Instant Answer API; return formatted results or None if nothing usable."""
    params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
    request = urllib.request.Request(
        f"{DUCKDUCKGO_URL}?{params}",
        headers={"User-Agent": "self-learning-ai-agent/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=SEARCH_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        return None

    results: list[str] = []
    abstract = str(payload.get("AbstractText") or "").strip()
    if abstract:
        heading = str(payload.get("Heading") or query).strip()
        url = str(payload.get("AbstractURL") or "").strip()
        results.append(f"{heading} - {abstract}" + (f" ({url})" if url else ""))
    answer = str(payload.get("Answer") or "").strip()
    if answer:
        results.append(f"Answer - {answer}")

    def collect(topics: Any) -> None:
        """Flatten (possibly nested) RelatedTopics into result lines."""
        if not isinstance(topics, list):
            return
        for topic in topics:
            if len(results) >= 5:
                return
            if not isinstance(topic, dict):
                continue
            if "Topics" in topic:
                collect(topic.get("Topics"))
                continue
            text = str(topic.get("Text") or "").strip()
            if text:
                url = str(topic.get("FirstURL") or "").strip()
                results.append(text + (f" ({url})" if url else ""))

    collect(payload.get("RelatedTopics"))
    if not results:
        return None
    lines = [f"[live search results for '{query}' via DuckDuckGo]"]
    lines.extend(f"{i}. {line}" for i, line in enumerate(results[:5], start=1))
    return "\n".join(lines)


def search_web(query: str) -> str:
    """Search the web: mock results by default, live DuckDuckGo when ``WEB_SEARCH_MODE=live``.

    Live mode falls back to the mock on any failure (network, timeout, parse,
    or no usable results).
    """
    if os.environ.get("WEB_SEARCH_MODE", "").strip().lower() == "live":
        try:
            live = _live_search(query)
            if live:
                return live
        except Exception:  # noqa: BLE001 - any live failure falls back to the mock
            pass
    return _mock_search(query)


def read_file(path: str) -> str:
    """Read a UTF-8 text file from the workspace (truncated beyond 20 000 characters)."""
    target = _safe_path(path)
    if not target.exists():
        return f"ERROR: File not found: '{path}'. Available files in workspace: {_available_files_text()}"
    if not target.is_file():
        return f"ERROR: '{path}' is not a file. Available files in workspace: {_available_files_text()}"
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        return (
            text[:MAX_READ_CHARS]
            + f"\n\n[truncated: showing the first {MAX_READ_CHARS} of {len(text)} characters]"
        )
    return text


def write_file(path: str, content: str) -> str:
    """Write ``content`` (UTF-8) to a workspace file, creating parent directories."""
    target = _safe_path(path)
    if target == workspace_dir().resolve() or target.is_dir():
        return f"ERROR: '{path}' is a directory. Provide a file name such as ideas.txt."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {_relative_to_workspace(target)}"


def send_email(to: str, subject: str, body: str) -> str:
    """Mock e-mail: validate the address and append the message to ``sent_emails.json``.

    Nothing is actually sent.
    """
    error = _email_error(to) or _footer_error(body)
    if error:
        return error
    recipient = to.strip()
    _append_json_list(
        data_dir() / "sent_emails.json",
        {"to": recipient, "subject": subject, "body": body, "sent_at": _now_iso()},
    )
    return (
        "Email sent (simulated: recorded in sent_emails.json, no real email is delivered): "
        f"to={recipient}, subject={subject}"
    )


def create_calendar_event(title: str, time: str) -> str:
    """Mock calendar: validate the ISO 8601 time (with UTC offset) and append to ``calendar.json``."""
    error = _time_error(time)
    if error:
        return error
    when = time.strip()
    _append_json_list(
        data_dir() / "calendar.json",
        {"title": title, "time": when, "created_at": _now_iso()},
    )
    return f"Calendar event created (simulated, recorded in calendar.json): '{title}' at {when}"


def get_user_preference() -> str:
    """Return the user's long-term memory (name, preferences, facts) as JSON."""
    return json.dumps(load_memory().model_dump(), ensure_ascii=False)


def save_to_memory(content: str, kind: Literal["preference", "fact"] = "preference") -> str:
    """Save a preference or fact to long-term memory (case-insensitive de-duplication)."""
    text = content.strip()
    if not text:
        return "ERROR: content must not be empty. Provide the preference or fact to save."
    added = add_fact(text) if kind == "fact" else add_preference(text)
    return f"Saved {kind}: {text}" if added else f"Already in memory: {text}"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool(
            name="search_web",
            description="Search the web and return the top results as text lines.",
            args_model=SearchWebArgs,
            func=search_web,
        ),
        Tool(
            name="read_file",
            description="Read a text file from the user's workspace folder.",
            args_model=ReadFileArgs,
            func=read_file,
        ),
        Tool(
            name="write_file",
            description="Create or overwrite a text file in the user's workspace folder.",
            args_model=WriteFileArgs,
            func=write_file,
        ),
        Tool(
            name="send_email",
            description=(
                "Send an email. The system shows the user the exact details and asks for "
                "approval before anything is sent."
            ),
            args_model=SendEmailArgs,
            func=send_email,
            requires_confirmation=True,
            precheck=_precheck_send_email,
        ),
        Tool(
            name="create_calendar_event",
            description=(
                "Create an event in the user's calendar. The system shows the user the exact "
                "details and asks for approval before the event is created."
            ),
            args_model=CreateCalendarEventArgs,
            func=create_calendar_event,
            requires_confirmation=True,
            precheck=_precheck_calendar,
        ),
        Tool(
            name="get_user_preference",
            description=(
                "Return the user's long-term memory as JSON: user_name, preferences and facts "
                "(facts include contacts such as email addresses)."
            ),
            args_model=GetUserPreferenceArgs,
            func=get_user_preference,
        ),
        Tool(
            name="save_to_memory",
            description="Save a user preference or fact to long-term memory so it is remembered in future tasks.",
            args_model=SaveToMemoryArgs,
            func=save_to_memory,
        ),
    )
}


def _strip_titles(node: Any, in_properties: bool = False) -> Any:
    """Recursively remove JSON-schema ``title`` keywords.

    Keys of a ``properties``/``$defs`` mapping are field names (a field may be
    called ``title``), so they are preserved; only schema-level ``title``
    keywords are dropped.
    """
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if in_properties:
                cleaned[key] = _strip_titles(value, in_properties=False)
                continue
            if key == "title":
                continue
            cleaned[key] = _strip_titles(
                value, in_properties=key in ("properties", "$defs", "definitions", "patternProperties")
            )
        return cleaned
    if isinstance(node, list):
        return [_strip_titles(item, in_properties=False) for item in node]
    return node


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI function-calling schemas for all tools (``title`` keys stripped)."""
    schemas: list[dict[str, Any]] = []
    for tool in TOOLS.values():
        parameters = _strip_titles(tool.args_model.model_json_schema())
        parameters.pop("description", None)  # class docstring; the function description is used instead
        parameters["type"] = "object"
        parameters.setdefault("properties", {})
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )
    return schemas


def _format_validation_error(exc: ValidationError) -> str:
    """Condense a pydantic ValidationError into a short single-line message."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ())) or "arguments"
        parts.append(f"{loc}: {err.get('msg', 'invalid value')}")
    return "; ".join(parts)


def _cancelled(name: str) -> ToolResult:
    """Build the standard 'user declined' result for ``name``."""
    return ToolResult(
        tool=name,
        status="cancelled",
        output=f"CANCELLED: The user declined {name}. Do not retry this action unless asked.",
    )


def execute_tool(
    name: str,
    arguments: str | dict[str, Any] | None,
    confirm: ConfirmFn | None = None,
) -> ToolResult:
    """Validate and run a tool, enforcing confirmation for side-effecting tools.

    Never raises: unknown tools, malformed JSON, validation errors and any
    exception become ``status="error"`` results whose output starts with
    ``"ERROR: "``. For tools with ``requires_confirmation``, ``confirm`` is
    called with the validated arguments before any side effect; if it is None,
    returns anything other than True, or raises, the result is ``"cancelled"``.
    """
    tool_name = str(name)
    try:
        tool = TOOLS.get(tool_name)
        if tool is None:
            return ToolResult(
                tool=tool_name,
                status="error",
                output=f"ERROR: Unknown tool '{tool_name}'. Available tools: {', '.join(TOOLS)}",
            )

        if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
            raw_args: Any = {}
        elif isinstance(arguments, str):
            try:
                raw_args = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return ToolResult(
                    tool=tool_name,
                    status="error",
                    output=f"ERROR: Invalid JSON arguments for {tool_name}: {exc.msg}. "
                    "Arguments must be a JSON object.",
                )
            if raw_args is None:
                raw_args = {}
        else:
            raw_args = arguments
        if not isinstance(raw_args, dict):
            return ToolResult(
                tool=tool_name,
                status="error",
                output=f"ERROR: Invalid arguments for {tool_name}: expected a JSON object, "
                f"got {type(raw_args).__name__}.",
            )

        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult(
                tool=tool_name,
                status="error",
                output=f"ERROR: Invalid arguments for {tool_name}: {_format_validation_error(exc)}",
            )

        if tool.precheck is not None:
            problem = tool.precheck(args)
            if problem:
                return ToolResult(tool=tool_name, status="error", output=problem)

        kwargs = args.model_dump()
        if tool.requires_confirmation:
            if confirm is None:
                return _cancelled(tool_name)
            try:
                approved = confirm(tool_name, dict(kwargs))
            except Exception:  # noqa: BLE001 - a failing confirmation never approves
                approved = False
            if approved is not True:
                return _cancelled(tool_name)

        output = tool.func(**kwargs)
        output = "" if output is None else str(output)
        status: Literal["ok", "error"] = "error" if output.startswith("ERROR:") else "ok"
        return ToolResult(tool=tool_name, status=status, output=output)
    except PermissionError as exc:
        message = str(exc)
        if not message.startswith("Access denied"):
            message = f"Access denied: {message}"
        return ToolResult(tool=tool_name, status="error", output=f"ERROR: {message}")
    except Exception as exc:  # noqa: BLE001 - execute_tool must never raise
        return ToolResult(tool=tool_name, status="error", output=f"ERROR: {type(exc).__name__}: {exc}")
