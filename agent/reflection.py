"""Post-task self-reflection: the core self-learning part of the agent.

After every task the agent reflects on what happened (via the LLM in JSON mode,
with a deterministic local heuristic as fallback), appends reusable lessons to
``learnings.md`` and stores newly stated user preferences / facts in long-term
memory. The next task loads those lessons into the system prompt's
"Past Learnings" section.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

from agent.config import data_dir
from agent.llm import LLMClient, LLMError
from agent.memory import add_fact, add_preference, format_memory_for_prompt, load_memory
from agent.prompts import REFLECTION_SYSTEM_PROMPT, build_reflection_user_prompt

if TYPE_CHECKING:
    from agent.agent import AgentResult

logger = logging.getLogger(__name__)

LEARNINGS_HEADER = "# Agent Learnings\n"
LESSON_PREFIX = "Lesson: "
MAX_LESSONS_PER_REFLECTION = 3
MAX_TITLE_CHARS = 60
MAX_OBSERVATION_CHARS = 300
DEFAULT_LESSON = "Review the task outcome and keep the approach that worked."

LESSON_ISO_8601 = (
    "Convert natural-language times to ISO 8601 (YYYY-MM-DDTHH:MM) before calling create_calendar_event."
)
LESSON_EMAIL_LOOKUP = (
    "When only a contact name is given, call get_user_preference first to look up their "
    "email address before calling send_email."
)
LESSON_FILE_NOT_FOUND = (
    "If a file is not found, use the exact filename from the available workspace files listed in the error."
)
LESSON_CANCELLED = "The user declined an action; do not retry it without new instructions."

_STATUS_LABELS: dict[str, str] = {"ok": "OK", "error": "ERROR", "cancelled": "CANCELLED"}
_STEP_TOOL_RE = re.compile(r"^- step \d+: (\w+)\(")
_ERROR_LINE_RE = re.compile(r"^- step \d+: .*\) -> ERROR: ")
_CANCELLED_LINE_RE = re.compile(r"^- step \d+: .*\) -> CANCELLED: ")
_OK_LINE_RE = re.compile(r"^- step \d+: .*\) -> OK: ")
_PREFERENCE_RE = re.compile(r"\bi\s+(like|love|prefer|enjoy)\s+(.+)", re.IGNORECASE | re.DOTALL)


class Reflection(BaseModel):
    """Structured result of reflecting on one finished task."""

    went_well: str = ""
    went_wrong: str = ""
    lessons: list[str] = []
    new_preferences: list[str] = []
    new_facts: list[str] = []

    @field_validator("went_well", "went_wrong", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> Any:
        """Accept None or a list of sentences from lenient LLM output."""
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(str(item) for item in value if item is not None)
        return value

    @field_validator("lessons", "new_preferences", "new_facts", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> Any:
        """Accept None, a single string, or a list with non-string items."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        return value


class ReflectionReport(BaseModel):
    """What a reflection produced and what was actually persisted."""

    reflection: Reflection
    lessons_added: list[str]
    preferences_added: list[str]
    facts_added: list[str]


# --------------------------------------------------------------------------- #
# learnings.md storage
# --------------------------------------------------------------------------- #


def learnings_path() -> Path:
    """Return the path of ``learnings.md`` in the current data dir (re-resolved on every call)."""
    return data_dir() / "learnings.md"


def load_lessons() -> list[str]:
    """Return the text of every ``Lesson: `` line in learnings.md, in file order ([] if missing)."""
    path = learnings_path()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError):
        return []
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return []
    lessons: list[str] = []
    for line in text.splitlines():
        if line.startswith(LESSON_PREFIX):
            lesson = line[len(LESSON_PREFIX):].strip()
            if lesson:
                lessons.append(lesson)
    return lessons


def load_learnings(max_lessons: int = 30) -> str:
    """Return the most recent unique lessons as ``- <lesson>`` lines joined by newlines.

    Duplicates (case-insensitive) are collapsed onto their LAST occurrence, so a
    lesson that was re-learned recently counts as recent. Returns "" if none.
    """
    if max_lessons <= 0:
        return ""
    unique: dict[str, str] = {}
    for lesson in load_lessons():
        key = lesson.casefold()
        unique.pop(key, None)
        unique[key] = lesson
    recent = list(unique.values())[-max_lessons:]
    return "\n".join(f"- {lesson}" for lesson in recent)


def _single_line(text: str) -> str:
    """Collapse all whitespace (including newlines) in ``text`` to single spaces."""
    return " ".join(text.split())


def _task_title(task: str) -> str:
    """Return a single-line task title of at most 60 chars, with "…" appended if cut."""
    title = _single_line(task)
    if not title:
        return "(untitled task)"
    if len(title) > MAX_TITLE_CHARS:
        return title[:MAX_TITLE_CHARS].rstrip() + "…"
    return title


def append_learnings(task: str, lessons: list[str], when: date | None = None) -> list[str]:
    """Append new lessons for ``task`` to learnings.md and return those actually appended.

    Lessons already present in the file (case-insensitive) or repeated within
    ``lessons`` are skipped. Creates the file with the ``# Agent Learnings``
    header if it is missing or empty.
    """
    existing = {lesson.casefold() for lesson in load_lessons()}
    to_add: list[str] = []
    for raw in lessons:
        lesson = _single_line(str(raw))
        if lesson.startswith(LESSON_PREFIX.strip()):
            lesson = lesson[len(LESSON_PREFIX.strip()):].strip()
        key = lesson.casefold()
        if not lesson or key in existing:
            continue
        existing.add(key)
        to_add.append(lesson)
    if not to_add:
        return []

    path = learnings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    prefix = ""
    if not current.strip():
        path.write_text(LEARNINGS_HEADER, encoding="utf-8")
    elif not current.endswith("\n"):
        prefix = "\n"

    day = (when or date.today()).isoformat()
    block = prefix + f"\n### {day} - {_task_title(task)}\n"
    block += "".join(f"{LESSON_PREFIX}{lesson}\n" for lesson in to_add)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)
    return to_add


# --------------------------------------------------------------------------- #
# Transcript + reflection
# --------------------------------------------------------------------------- #


def format_transcript(result: "AgentResult") -> str:
    """Render the agent's steps as the line-based transcript used in the reflection prompt.

    One line per step: ``- step <n>: <tool>(<args json>) -> <STATUS>: <observation>``
    (observation truncated to 300 chars, newlines replaced by spaces) or
    ``- step <n>: final answer``; followed by ``Tool errors: <count>``.
    """
    lines: list[str] = []
    for step in result.steps:
        if step.status == "final" or not step.tool_name:
            lines.append(f"- step {step.step}: final answer")
            continue
        args_json = json.dumps(step.tool_args or {}, ensure_ascii=False)
        status = _STATUS_LABELS.get(step.status, str(step.status).upper())
        observation = (step.observation or "").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        observation = observation[:MAX_OBSERVATION_CHARS]
        lines.append(f"- step {step.step}: {step.tool_name}({args_json}) -> {status}: {observation}")
    if not lines:
        lines.append("- (no steps)")
    lines.append(f"Tool errors: {result.tool_errors}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in ``text`` (tolerates code fences and prose), else None."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    return None


def _heuristic_reflection(task: str, transcript: str) -> Reflection:
    """Rule-based reflection used when the LLM fails; mirrors the MockLLM reflection rules."""
    lines = transcript.splitlines()
    error_lines = [line for line in lines if _ERROR_LINE_RE.match(line)]
    cancelled_lines = [line for line in lines if _CANCELLED_LINE_RE.match(line)]
    ok_count = sum(1 for line in lines if _OK_LINE_RE.match(line))
    tools = [m.group(1) for line in lines if (m := _STEP_TOOL_RE.match(line))]

    lessons: list[str] = []
    if any("ISO 8601" in line for line in error_lines):
        lessons.append(LESSON_ISO_8601)
    if any("is not a valid email address" in line for line in error_lines):
        lessons.append(LESSON_EMAIL_LOOKUP)
    if any("File not found" in line for line in error_lines):
        lessons.append(LESSON_FILE_NOT_FOUND)
    if cancelled_lines:
        lessons.append(LESSON_CANCELLED)
    if not error_lines and not cancelled_lines:
        short_task = task.strip()[:40]
        approach = f"calling {' -> '.join(tools)}" if tools else "answering directly"
        lessons.append(f"For tasks like '{short_task}', {approach} worked without errors.")

    new_preferences: list[str] = []
    match = _PREFERENCE_RE.search(task)
    if match and "save_to_memory" not in tools:
        thing = _single_line(match.group(2)).rstrip(".!?,; ")
        if thing:
            new_preferences.append(f"{match.group(1).lower()}s {thing}")

    went_well = f"Completed {ok_count} successful tool call(s) out of {len(tools)}."
    if error_lines or cancelled_lines:
        went_wrong = f"{len(error_lines)} tool error(s) and {len(cancelled_lines)} cancelled action(s)."
    else:
        went_wrong = "No tool errors."
    return Reflection(
        went_well=went_well,
        went_wrong=went_wrong,
        lessons=lessons[:MAX_LESSONS_PER_REFLECTION],
        new_preferences=new_preferences,
        new_facts=[],
    )


def _dedupe_clean(items: list[str]) -> list[str]:
    """Strip/collapse whitespace, drop empties and case-insensitive duplicates, keep order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in items:
        item = _single_line(str(raw))
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            cleaned.append(item)
    return cleaned


def _normalise(reflection: Reflection) -> Reflection:
    """Normalise a reflection: 1-3 clean unique lessons and clean preference/fact lists."""
    lessons = []
    for lesson in reflection.lessons:
        text = _single_line(str(lesson))
        if text.startswith(LESSON_PREFIX.strip()):
            text = text[len(LESSON_PREFIX.strip()):].strip()
        lessons.append(text)
    lessons = _dedupe_clean(lessons)[:MAX_LESSONS_PER_REFLECTION] or [DEFAULT_LESSON]
    return Reflection(
        went_well=_single_line(reflection.went_well),
        went_wrong=_single_line(reflection.went_wrong),
        lessons=lessons,
        new_preferences=_dedupe_clean(reflection.new_preferences),
        new_facts=_dedupe_clean(reflection.new_facts),
    )


def reflect(result: "AgentResult", llm: LLMClient) -> Reflection:
    """Ask the LLM (JSON mode) to reflect on ``result``; fall back to local heuristics on any failure."""
    transcript = format_transcript(result)
    memory_text = format_memory_for_prompt(load_memory())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_reflection_user_prompt(
                task=result.task,
                transcript=transcript,
                final_answer=result.final_answer,
                memory_text=memory_text,
            ),
        },
    ]
    reflection: Reflection | None = None
    try:
        response = llm.chat(messages, tools=None, json_mode=True)
        data = _extract_json_object(response.content or "")
        if data is None:
            logger.info("Reflection reply contained no JSON object; using heuristic fallback.")
        else:
            reflection = Reflection.model_validate(data)
    except LLMError as exc:
        logger.info("Reflection LLM call failed (%s); using heuristic fallback.", exc)
    except Exception as exc:  # noqa: BLE001 - reflection must never break the task run
        logger.info("Reflection failed (%s: %s); using heuristic fallback.", type(exc).__name__, exc)
    if reflection is None:
        reflection = _heuristic_reflection(result.task, transcript)
    return _normalise(reflection)


def run_reflection(result: "AgentResult", llm: LLMClient) -> ReflectionReport:
    """Reflect on a finished task, persist new lessons and memory updates, and report what changed."""
    reflection = reflect(result, llm)
    lessons_added = append_learnings(result.task, reflection.lessons)
    preferences_added = [pref for pref in reflection.new_preferences if add_preference(pref)]
    facts_added = [fact for fact in reflection.new_facts if add_fact(fact)]
    return ReflectionReport(
        reflection=reflection,
        lessons_added=lessons_added,
        preferences_added=preferences_added,
        facts_added=facts_added,
    )
