"""ReAct agent loop: Think -> Act (tool call) -> Observe -> repeat.

The :class:`Agent` loads long-term memory and past learnings at the start of
every task, builds the system prompt, and then alternates between LLM calls
and tool executions until the model returns a plain-content final answer, the
step budget is exhausted, or the LLM provider fails.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from agent.llm import LLMClient, LLMError, LLMResponse, ToolCall, get_llm
from agent.memory import ShortTermMemory, format_memory_for_prompt, load_memory, save_memory
from agent.prompts import build_system_prompt
from agent.reflection import load_learnings
from agent.tools import ConfirmFn, ToolResult, execute_tool, get_tool_schemas

ERROR_RECOVERY_NOTE = (
    "\n[agent note] This call failed. Re-plan: fix the arguments or use a different tool. "
    "Do not repeat the same call."
)
REPEATED_CALL_TEMPLATE = (
    "ERROR: Repeated failing call blocked: {name} with the same arguments already failed. "
    "Try a different approach."
)
MAX_STEPS_TEMPLATE = "Stopped: reached the maximum of {n} steps without a final answer."
NO_ANSWER = "(no answer)"
SHORT_TERM_MAX_MESSAGES = 20
_EVENT_PREVIEW_CHARS = 200


class AgentStep(BaseModel):
    """One recorded step of the loop: a single tool call or the final answer."""

    step: int
    thought: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    observation: str | None = None
    status: Literal["ok", "error", "cancelled", "final"] = "ok"


class AgentResult(BaseModel):
    """Outcome of :meth:`Agent.run`, consumed by the CLI, eval and reflection."""

    task: str
    final_answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    tool_calls: int = 0
    tool_errors: int = 0
    tools_used: list[str] = Field(default_factory=list)
    success: bool = False
    stopped_reason: Literal["final_answer", "max_steps", "llm_error"]
    learnings_used: int = 0
    provider: str
    messages: list[dict[str, Any]] = Field(default_factory=list)


def _parse_arguments(arguments: str | dict[str, Any] | None) -> Any:
    """Decode tool-call arguments into a Python value.

    Returns the decoded JSON value, ``{}`` for empty/None input, or the raw
    string unchanged when it is not valid JSON.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    text = arguments.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return arguments


def _canonical_call_key(name: str, parsed_args: Any) -> tuple[str, str]:
    """Build a hashable key identifying a call by name and canonical-JSON arguments."""
    try:
        canonical = json.dumps(parsed_args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        canonical = repr(parsed_args)
    return name, canonical


def _preview(text: str, limit: int = _EVENT_PREVIEW_CHARS) -> str:
    """Return a single-line preview of ``text`` truncated to ``limit`` characters."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _count_learning_lines(learnings_text: str) -> int:
    """Count the non-empty lines of the learnings text injected into the prompt."""
    return sum(1 for line in learnings_text.splitlines() if line.strip())


class Agent:
    """ReAct-style tool-using agent with memory and learnings injection."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        max_steps: int = 10,
        confirm: ConfirmFn | None = None,
        use_learnings: bool = True,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        """Create an agent.

        Args:
            llm: LLM client; ``None`` selects one via ``get_llm()``.
            max_steps: maximum number of LLM calls per task (must be >= 1).
            confirm: confirmation callback passed to side-effecting tools.
            use_learnings: inject ``learnings.md`` lessons into the system prompt.
            on_event: optional callback receiving short progress strings.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.llm: LLMClient = llm if llm is not None else get_llm()
        self.max_steps = max_steps
        self.confirm = confirm
        self.use_learnings = use_learnings
        self.on_event = on_event

    def _emit(self, message: str) -> None:
        """Forward a progress message to ``on_event`` if a callback is set."""
        if self.on_event is not None:
            self.on_event(message)

    def _build_system_message(self) -> tuple[dict[str, Any], int]:
        """Load memory and learnings and build the system message.

        Returns:
            The system message dict and the number of lesson lines injected.
        """
        memory_text = format_memory_for_prompt(load_memory())
        learnings_text = load_learnings() if self.use_learnings else ""
        system_prompt = build_system_prompt(memory_text, learnings_text, datetime.now())
        return {"role": "system", "content": system_prompt}, _count_learning_lines(learnings_text)

    def _run_tool_call(
        self,
        call: ToolCall,
        step_number: int,
        thought: str | None,
        failed_calls: set[tuple[str, str]],
    ) -> tuple[AgentStep, str]:
        """Execute (or block) one tool call.

        Args:
            call: the tool call requested by the LLM.
            step_number: current LLM step number.
            thought: assistant content accompanying the tool calls, if any.
            failed_calls: keys of calls that already failed in this run (updated in place).

        Returns:
            The recorded step and the content for the ``role="tool"`` message.
        """
        parsed_args = _parse_arguments(call.arguments)
        tool_args = parsed_args if isinstance(parsed_args, dict) else None
        key = _canonical_call_key(call.name, parsed_args)
        self._emit(f"→ {call.name} {_preview(key[1])}")

        if key in failed_calls:
            result = ToolResult(
                tool=call.name,
                status="error",
                output=REPEATED_CALL_TEMPLATE.format(name=call.name),
            )
        else:
            result = execute_tool(call.name, call.arguments, confirm=self.confirm)
            if result.status == "error":
                failed_calls.add(key)

        self._emit(f"← {result.status}: {_preview(result.output)}")
        content = result.output + ERROR_RECOVERY_NOTE if result.status == "error" else result.output
        step = AgentStep(
            step=step_number,
            thought=thought,
            tool_name=call.name,
            tool_args=tool_args,
            observation=result.output,
            status=result.status,
        )
        return step, content

    def run(self, task: str) -> AgentResult:
        """Run the ReAct loop for ``task`` and return the full result.

        Args:
            task: the user's natural-language request.

        Returns:
            An :class:`AgentResult` describing the steps, outcome and transcript.
        """
        provider = str(getattr(self.llm, "provider", "unknown"))
        system_message, learnings_used = self._build_system_message()
        tool_schemas = get_tool_schemas()

        short_term = ShortTermMemory(max_messages=SHORT_TERM_MAX_MESSAGES)
        transcript: list[dict[str, Any]] = [system_message]
        steps: list[AgentStep] = []
        tools_used: list[str] = []
        failed_calls: set[tuple[str, str]] = set()

        def record(message: dict[str, Any]) -> None:
            """Append a message to both the short-term window and the full transcript."""
            short_term.add(message)
            transcript.append(message)

        record({"role": "user", "content": task})
        self._emit(f"Task: {_preview(task)}")
        if learnings_used:
            self._emit(f"Using {learnings_used} past learning(s)")

        final_answer: str | None = None
        stopped_reason: Literal["final_answer", "max_steps", "llm_error"] = "max_steps"

        for step_number in range(1, self.max_steps + 1):
            self._emit(f"Step {step_number}: thinking…")
            try:
                response: LLMResponse = self.llm.chat(
                    [system_message] + short_term.get(), tools=tool_schemas
                )
            except LLMError as exc:
                final_answer = f"LLM error: {exc}"
                stopped_reason = "llm_error"
                self._emit(final_answer)
                break

            if not response.tool_calls:
                answer = (response.content or "").strip() or NO_ANSWER
                record({"role": "assistant", "content": answer})
                steps.append(AgentStep(step=step_number, observation=answer, status="final"))
                final_answer = answer
                stopped_reason = "final_answer"
                self._emit(f"Final answer: {_preview(answer)}")
                break

            thought = response.content.strip() if response.content and response.content.strip() else None
            if thought:
                self._emit(f"Thought: {_preview(thought)}")
            record(
                {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in response.tool_calls
                    ],
                }
            )
            for call in response.tool_calls:
                tools_used.append(call.name)
                step, content = self._run_tool_call(call, step_number, thought, failed_calls)
                steps.append(step)
                record({"role": "tool", "tool_call_id": call.id, "content": content})

        if final_answer is None:
            final_answer = MAX_STEPS_TEMPLATE.format(n=self.max_steps)
            self._emit(final_answer)

        save_memory(load_memory())

        return AgentResult(
            task=task,
            final_answer=final_answer,
            steps=steps,
            tool_calls=len(tools_used),
            tool_errors=sum(1 for s in steps if s.status == "error"),
            tools_used=tools_used,
            success=stopped_reason == "final_answer",
            stopped_reason=stopped_reason,
            learnings_used=learnings_used,
            provider=provider,
            messages=transcript,
        )
