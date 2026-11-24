"""Prompt templates for the self-learning agent.

This module builds two prompts:

* the agent's system prompt (current date/time, long-term user memory,
  lessons learned from past tasks and the operating rules), and
* the reflection prompts used after every task to extract reusable lessons
  and newly stated user preferences/facts as strict JSON.

Formatting safety: user-controlled text (memory, learnings, transcripts) may
contain braces or anything else. It is NEVER passed through ``str.format``.
Placeholders are substituted in a single regex pass, so text inserted for one
placeholder can never be re-interpreted as another placeholder.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

NO_LEARNINGS = "(none yet)"
"""Shown under ``## Past Learnings`` when there are no stored lessons."""

NO_MEMORY = "(no memory saved yet)"
"""Shown under ``## User Memory`` when the memory text is empty."""

# NOTE: the structural lines below ("Current date/time: ...", "## User Memory",
# "## Past Learnings", "## Rules") are parsed by the mock LLM. Keep them exact
# and in this order. Rules are numbered (not "- " bullets) so they can never be
# mistaken for Past Learnings bullet lines.
SYSTEM_PROMPT_TEMPLATE = """You are a helpful self-learning AI agent. You complete the user's task by calling tools, observing the results and then giving a short final answer. After every task you reflect, and the lessons you learn are shown to you below under "Past Learnings".

Current date/time: {now}
Upcoming dates (use this list to resolve "tomorrow", "Friday" etc.; do not compute weekdays yourself):
{dates}

## User Memory
{memory}
## Past Learnings
{learnings}
## Rules
1. ALWAYS read the Past Learnings above before acting and follow them. They are lessons from your own earlier mistakes; they override your default habits.
2. Use tools instead of guessing. Never invent file contents, search results, email addresses or memory facts.
3. Sending an email (send_email) and creating a calendar event (create_calendar_event) ALWAYS require the user's confirmation, and this confirmation is enforced by the system: when you call the tool, the user is shown the exact details and asked to approve. Therefore do NOT ask for permission in text. Call the tool directly with complete, correct details.
4. If a tool result says the user declined or CANCELLED the action, do not retry it. Tell the user briefly that it was not done.
5. If a tool fails, read the error message carefully, then fix the arguments as the error says or try a different approach. Never repeat the same failing call with the same arguments.
6. If the user gives only a contact's name (e.g. "John"), the contact's details may be stored in the user's memory facts. Look them up there first (in User Memory above, or with the tool that returns the saved preferences and facts) before using them. Never pass a bare name where an email address is required.
7. Draft short, sensible defaults for missing non-critical details (for example an email subject or a brief polite body signed with the user's name). Ask the user only if a critical detail (such as a recipient address) cannot be found.
8. Use the user's preferences from User Memory only to fill in details the task leaves open. Anything the task states explicitly (a time, date, recipient or wording) ALWAYS overrides a stored preference: do exactly what the task says, and if it conflicts with a preference, mention that in one short sentence instead of changing it.
9. When the user explicitly asks you to remember or save something, store it with save_to_memory.
10. One tool call at a time is fine. Stop calling tools as soon as the task is done.
11. Never claim or promise an action you have not performed. If the task needs a tool, call it now instead of saying you "will" do it. Your final answer may only report what tool results confirm.
12. Keep the final answer short and factual: say what you did (or why you could not do it)."""

REFLECTION_SYSTEM_PROMPT = """You are the reflection module of a self-learning AI agent. You review one finished task (the task, the final answer, the user's current memory and the step-by-step transcript of tool calls) and answer three questions:
1. What went well?
2. What went wrong?
3. What rule should I follow next time?

Reply with ONLY a JSON object (no markdown, no code fences, no extra text) with exactly these keys:
- "went_well": string, one short sentence.
- "went_wrong": string, one short sentence ("Nothing" if there were no errors).
- "lessons": list of 1-3 short, general, reusable rules written as imperatives (e.g. "Look up ... before calling ..."). Base them on errors, retries or declined actions in the transcript; if everything worked, state the approach that worked. Do not mention one-off details such as specific names, dates or message text. If a tool error revealed a requirement (a format, a policy, a required value), state that requirement itself in the lesson, including the exact required value. Explicit instructions in a task always outrank stored preferences: never write a lesson that tells the agent to change something the user explicitly asked for.
- "new_preferences": list of preferences the user EXPLICITLY stated in the task (e.g. "likes short emails") that are NOT already in Current memory. Use [] if none. Never guess.
- "new_facts": list of durable facts the user explicitly stated (e.g. a contact's email address) that are NOT already in Current memory. Use [] if none.

Example of a valid reply:
{"went_well": "The file was read and summarized.", "went_wrong": "The first call used a wrong filename.", "lessons": ["Use the exact filename listed by the tool when a file is missing."], "new_preferences": [], "new_facts": []}"""

_PLACEHOLDER_RE = re.compile(r"\{(now|dates|memory|learnings)\}")


def _fill(template: str, values: dict[str, str]) -> str:
    """Substitute ``{name}`` placeholders in one regex pass.

    Only the placeholder names present in ``values`` are replaced; every other
    brace in the template is left untouched. Because substitution happens in a
    single pass over the template, inserted values (which may themselves
    contain braces or placeholder-like text) are never re-scanned.

    Args:
        template: Template text containing ``{now}``, ``{memory}`` and/or
            ``{learnings}`` placeholders.
        values: Mapping from placeholder name to replacement text.

    Returns:
        The template with placeholders replaced.
    """

    def _replace(match: re.Match[str]) -> str:
        """Return the replacement text for a single placeholder match."""
        return values.get(match.group(1), match.group(0))

    return _PLACEHOLDER_RE.sub(_replace, template)


def format_upcoming_dates(now: datetime, days: int = 7) -> str:
    """Return a weekday -> date list for today and the next ``days`` days.

    LLMs are unreliable at weekday arithmetic ("Friday" -> which date?), so the
    prompt gives them a lookup table instead.

    Args:
        now: The current local date/time.
        days: How many days after today to list.

    Returns:
        One line per day, e.g. ``  Wednesday = 2026-11-25 (tomorrow)``.
    """
    lines = []
    for offset in range(days + 1):
        day = now + timedelta(days=offset)
        label = " (today)" if offset == 0 else " (tomorrow)" if offset == 1 else ""
        lines.append(f"  {day.strftime('%A')} = {day.strftime('%Y-%m-%d')}{label}")
    return "\n".join(lines)


def build_system_prompt(memory_text: str, learnings_text: str, now: datetime) -> str:
    """Build the agent's system prompt.

    The result contains, in this order, the exact structural lines
    ``Current date/time: %Y-%m-%dT%H:%M (%A)``, ``## User Memory``,
    ``## Past Learnings`` and ``## Rules``.

    Args:
        memory_text: Long-term memory formatted by
            ``memory.format_memory_for_prompt`` (may contain any characters).
        learnings_text: Past lessons as ``- <lesson>`` lines; empty or
            whitespace-only means there are no lessons yet.
        now: Current local date/time used for resolving relative dates.

    Returns:
        The complete system prompt string.
    """
    memory = (memory_text or "").strip("\n").rstrip() or NO_MEMORY
    learnings = (learnings_text or "").strip("\n").rstrip()
    if not learnings.strip():
        learnings = NO_LEARNINGS
    return _fill(
        SYSTEM_PROMPT_TEMPLATE,
        {
            "now": now.strftime("%Y-%m-%dT%H:%M (%A)"),
            "dates": format_upcoming_dates(now),
            "memory": memory,
            "learnings": learnings,
        },
    )


def build_reflection_user_prompt(
    task: str, transcript: str, final_answer: str, memory_text: str
) -> str:
    """Build the user message for the reflection LLM call.

    Format (parsed by the mock LLM and the heuristic fallback)::

        Task: <task>
        Final answer: <final_answer>
        Current memory:
        <memory_text>
        Transcript:
        <transcript>

    Args:
        task: The original user task.
        transcript: Step lines produced by ``reflection.format_transcript``.
        final_answer: The agent's final answer for the task.
        memory_text: Long-term memory formatted for prompts.

    Returns:
        The reflection user prompt string (plain concatenation, no
        ``str.format``, so braces in any argument are safe).
    """
    parts = [
        "Task: " + (task or "").strip(),
        "Final answer: " + (final_answer or "").strip(),
        "Current memory:",
        (memory_text or "").strip("\n").rstrip() or NO_MEMORY,
        "Transcript:",
        (transcript or "").strip("\n").rstrip() or "(no steps)",
    ]
    return "\n".join(parts)
