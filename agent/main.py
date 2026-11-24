"""Command-line entry point for the self-learning agent.

Usage::

    python agent/main.py --task "..." [--yes] [--json] [--no-learnings]
                         [--provider openai|groq|mock] [--max-steps N] [--quiet]
    python agent/main.py --eval [--json] [--provider ...] [--max-steps N] [--quiet]

Task mode runs one task through the ReAct agent loop, then reflects on it
(lessons are appended to ``learnings.md``). Eval mode runs the 10 tasks in
``data/eval_tasks.json`` twice in throwaway data directories - once without
and once with the lessons learned in the first run - and prints both scores.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


MIN_PYTHON = (3, 9)


def _check_python_version() -> None:
    """Exit with a clear, actionable message when the interpreter can't run the agent.

    Python 3.11+ is recommended. Python 3.9 (the macOS system Python) works only
    with the ``eval_type_backport`` package, which lets pydantic evaluate the
    ``X | None`` annotations used throughout the code base.
    """
    found = ".".join(str(part) for part in sys.version_info[:3])
    fix = "fix: pip install -r requirements.txt  (or use Python 3.11+: brew install python@3.12)\n"
    if sys.version_info < MIN_PYTHON:
        sys.stderr.write(
            f"error: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required (3.11+ recommended), "
            f"but this is Python {found} ({sys.executable}).\n" + fix
        )
        raise SystemExit(2)
    if sys.version_info < (3, 10):
        try:
            import eval_type_backport  # noqa: F401
        except ImportError:
            sys.stderr.write(
                f"error: on Python {found} the 'eval_type_backport' package is required.\n" + fix
            )
            raise SystemExit(2)


_check_python_version()


def _fix_sys_path() -> Path:
    """Make ``python agent/main.py`` import the ``agent`` package correctly.

    When run as a script, Python puts the script's own directory (``agent/``)
    first on ``sys.path``; ``agent/agent.py`` would then shadow the ``agent``
    package. Remove that directory and put the project root first instead.

    Returns:
        The project root directory.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    cleaned: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except (OSError, RuntimeError):
            cleaned.append(entry)
            continue
        if resolved in (script_dir, project_root):
            continue
        cleaned.append(entry)
    sys.path[:] = [str(project_root), *cleaned]
    return project_root


_fix_sys_path()

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from typing import Any, TextIO  # noqa: E402

from agent import config  # noqa: E402
from agent.agent import Agent, AgentResult  # noqa: E402
from agent.llm import LLMClient, LLMError, get_llm  # noqa: E402
from agent.memory import load_memory  # noqa: E402
from agent.reflection import ReflectionReport, load_lessons, run_reflection  # noqa: E402
from agent.tools import ConfirmFn  # noqa: E402

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

LEARNINGS_HEADER = "# Agent Learnings\n"
DATA_DIR_ENV = "AGENT_DATA_DIR"
KNOWN_CHECKS = frozenset(
    {
        "answer_contains",
        "memory_contains",
        "email_sent_to",
        "calendar_title",
        "calendar_time_contains",
        "file_contains",
        "tools_used_include",
        "max_tool_errors",
        "max_steps",
    }
)


# --------------------------------------------------------------------------- #
# Small output helpers
# --------------------------------------------------------------------------- #
def _err(message: str = "") -> None:
    """Print a line to stderr (prompts, progress and diagnostics go here)."""
    print(message, file=sys.stderr, flush=True)


def _stderr_event(message: str) -> None:
    """Progress callback for the agent: print the event indented to stderr."""
    _err(f"  {message}")


def _configure_streams() -> None:
    """Avoid UnicodeEncodeError on consoles that cannot encode every character."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(errors="backslashreplace")


def count_llm_steps(result: AgentResult) -> int:
    """Return the number of LLM steps taken (distinct step numbers in ``result.steps``)."""
    return len({step.step for step in result.steps})


def _format_improvement(delta: int) -> str:
    """Format a score difference as ``+N``, ``0`` or ``-N``."""
    return f"+{delta}" if delta > 0 else str(delta)


# --------------------------------------------------------------------------- #
# Confirmation
# --------------------------------------------------------------------------- #
def _format_arg_value(value: Any) -> str:
    """Render an argument value for the confirmation prompt (multi-line values indented)."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text.replace("\n", "\n    ")


def interactive_confirm(tool_name: str, args: dict[str, Any]) -> bool:
    """Ask the user on stderr/stdin whether a side-effecting tool call may run.

    Prints ``CONFIRMATION REQUIRED: <tool_name>`` followed by one ``  <k>: <v>``
    line per argument and the prompt ``Approve? [y/N]: `` to stderr, then reads
    one line from stdin.

    Args:
        tool_name: Name of the tool that wants to run.
        args: The validated tool arguments.

    Returns:
        True only if the user answered ``y`` or ``yes``; False on anything
        else, including EOF or an unreadable stdin.
    """
    _err()
    _err(f"CONFIRMATION REQUIRED: {tool_name}")
    for key, value in args.items():
        _err(f"  {key}: {_format_arg_value(value)}")
    sys.stderr.write("Approve? [y/N]: ")
    sys.stderr.flush()
    if sys.stdin is None:
        _err()
        return False
    try:
        line = sys.stdin.readline()
    except (OSError, ValueError):
        line = ""
    if not line:
        _err()
        _err("(no input - declined)")
        return False
    with contextlib.suppress(Exception):
        if not sys.stdin.isatty():
            _err(line.strip())  # echo piped answers so the transcript stays readable
    return line.strip().lower() in {"y", "yes"}


def auto_confirm(tool_name: str, args: dict[str, Any]) -> bool:
    """Approve every confirmation request (used by ``--yes`` and eval mode)."""
    return True


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _positive_int(value: str) -> int:
    """argparse type: a strictly positive integer."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent/main.py",
        description="Self-learning AI agent: runs a task with tools, then reflects and learns.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", help="the task to run, e.g. \"Send email to John about lunch\"")
    mode.add_argument(
        "--eval",
        action="store_true",
        help="run the 10 eval tasks twice (without, then with learnings) and print both scores",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="auto-approve emails and calendar events")
    parser.add_argument("--json", action="store_true", help="print one JSON object to stdout (everything else to stderr)")
    parser.add_argument("--no-learnings", action="store_true", help="do not inject past learnings into the prompt")
    parser.add_argument("--provider", choices=["openai", "groq", "mock"], default=None,
                        help="LLM provider (default: LLM_PROVIDER env, else auto-detect from API keys, else mock)")
    parser.add_argument("--max-steps", type=_positive_int, default=10, help="maximum LLM steps per task (default 10)")
    parser.add_argument("--quiet", "-q", action="store_true", help="hide step-by-step progress")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments (exits with code 2 on usage errors)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.task is not None and not args.task.strip():
        parser.error("--task must not be empty")
    if args.eval and args.no_learnings:
        parser.error("--no-learnings cannot be combined with --eval (eval compares runs without and with learnings)")
    return args


# --------------------------------------------------------------------------- #
# Task mode
# --------------------------------------------------------------------------- #
def _safe_reflection(result: AgentResult, llm: LLMClient) -> ReflectionReport | None:
    """Run reflection, returning None (with a warning on stderr) if it fails unexpectedly."""
    try:
        return run_reflection(result, llm)
    except Exception as exc:  # reflection must never take the CLI down
        _err(f"warning: reflection failed: {type(exc).__name__}: {exc}")
        return None


def _total_lessons() -> int:
    """Return the number of lessons currently stored in learnings.md (0 on read errors)."""
    try:
        return len(load_lessons())
    except Exception:
        return 0


def task_result_to_dict(result: AgentResult, report: ReflectionReport | None) -> dict[str, Any]:
    """Build the ``--json`` output object for task mode."""
    reflection = report.reflection if report is not None else None
    return {
        "task": result.task,
        "final_answer": result.final_answer,
        "success": result.success,
        "stopped_reason": result.stopped_reason,
        "steps": count_llm_steps(result),
        "tool_calls": result.tool_calls,
        "tool_errors": result.tool_errors,
        "tools_used": list(result.tools_used),
        "learnings_used": result.learnings_used,
        "provider": result.provider,
        "reflection": {
            "went_well": reflection.went_well if reflection else "",
            "went_wrong": reflection.went_wrong if reflection else "",
            "lessons": list(reflection.lessons) if reflection else [],
        },
        "lessons_added": list(report.lessons_added) if report else [],
        "preferences_added": list(report.preferences_added) if report else [],
        "facts_added": list(report.facts_added) if report else [],
    }


def print_task_report(
    result: AgentResult,
    report: ReflectionReport | None,
    total_lessons: int,
    model: str,
    out: TextIO,
) -> None:
    """Print the human-readable task report with the three section headers."""
    status = "success" if result.success else f"stopped ({result.stopped_reason})"
    tools = " -> ".join(result.tools_used) if result.tools_used else "(none)"
    print("=== Result ===", file=out)
    print(f"Task: {result.task}", file=out)
    print(f"Answer: {result.final_answer}", file=out)
    print(f"Status: {status}", file=out)
    print(
        f"Steps: {count_llm_steps(result)} | Tool calls: {result.tool_calls} | "
        f"Tool errors: {result.tool_errors}",
        file=out,
    )
    print(f"Tools used: {tools}", file=out)
    print(f"Past learnings used: {result.learnings_used}", file=out)
    print(f"Provider: {result.provider} ({model})", file=out)
    print(file=out)

    print("=== Reflection ===", file=out)
    if report is None:
        print("(reflection unavailable)", file=out)
    else:
        print(f"What went well: {report.reflection.went_well or '-'}", file=out)
        print(f"What went wrong: {report.reflection.went_wrong or '-'}", file=out)
        print("Rules for next time:", file=out)
        for lesson in report.reflection.lessons or ["(none)"]:
            print(f"- {lesson}", file=out)
    print(file=out)

    print("=== Learnings ===", file=out)
    added = report.lessons_added if report else []
    print(f"New lessons added to learnings.md: {len(added)}", file=out)
    for lesson in added:
        print(f"- {lesson}", file=out)
    print(f"Total lessons in learnings.md: {total_lessons}", file=out)
    updates = []
    if report is not None:
        updates = [f"preference: {p}" for p in report.preferences_added]
        updates += [f"fact: {f}" for f in report.facts_added]
    if updates:
        print("Memory updates (memory.json):", file=out)
        for update in updates:
            print(f"- {update}", file=out)
    else:
        print("Memory updates (memory.json): none", file=out)


def run_task_mode(args: argparse.Namespace, llm: LLMClient) -> int:
    """Run a single task, reflect on it and print the report.

    Returns:
        0 on completion; 1 if the run stopped because of an LLM error
        (typically a provider/configuration problem such as an invalid key).
    """
    confirm: ConfirmFn = auto_confirm if args.yes else interactive_confirm
    on_event = None if args.quiet else _stderr_event
    if not args.quiet:
        _err(f"Running task with provider '{llm.provider}' (model {getattr(llm, 'model', '?')})...")
    agent = Agent(
        llm=llm,
        max_steps=args.max_steps,
        confirm=confirm,
        use_learnings=not args.no_learnings,
        on_event=on_event,
    )
    result = agent.run(args.task)
    report = _safe_reflection(result, llm)
    total = _total_lessons()

    if args.json:
        payload = task_result_to_dict(result, report)
        _write_json(payload)
    else:
        print_task_report(result, report, total, getattr(llm, "model", "?"), sys.stdout)

    if result.stopped_reason == "llm_error":
        _err(f"error: {result.final_answer} (check your provider settings and API key in .env)")
        return EXIT_RUNTIME_ERROR
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _as_str_list(value: Any) -> list[str]:
    """Normalise a check value that may be a string or a list of strings into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _read_json(path: Path) -> tuple[Any, str | None]:
    """Read a JSON file, returning ``(data, None)`` or ``(None, reason)`` on failure."""
    if not path.is_file():
        return None, f"{path.name} does not exist"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, f"{path.name} could not be read: {exc}"


def _read_json_list(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Read a JSON list of objects (sent_emails.json / calendar.json)."""
    data, problem = _read_json(path)
    if problem is not None:
        return [], problem
    if not isinstance(data, list):
        return [], f"{path.name} is not a JSON list"
    return [item for item in data if isinstance(item, dict)], None


def evaluate_task(task_spec: dict[str, Any], result: AgentResult, data_dir: Path) -> tuple[bool, list[str]]:
    """Check an agent result (and the files it produced) against an eval task's checks.

    A task passes only if ``result.success`` is true AND every check holds.
    Supported checks: ``answer_contains``, ``memory_contains``, ``email_sent_to``,
    ``calendar_title``, ``calendar_time_contains``, ``file_contains``,
    ``tools_used_include``, ``max_tool_errors``, ``max_steps``. String matching is
    case-insensitive. An unknown check name fails the task (to catch typos).

    Args:
        task_spec: One entry of eval_tasks.json (``{"id", "task", "checks"}``).
        result: The agent's result for that task.
        data_dir: The data directory the task ran against.

    Returns:
        ``(passed, reasons)`` where ``reasons`` lists every failed check.
    """
    checks: dict[str, Any] = task_spec.get("checks") or {}
    reasons: list[str] = []

    if not result.success:
        reasons.append(f"agent did not finish (stopped_reason={result.stopped_reason})")

    for name in checks:
        if name not in KNOWN_CHECKS:
            reasons.append(f"unknown check '{name}'")

    answer = (result.final_answer or "").lower()
    for needle in _as_str_list(checks.get("answer_contains")):
        if needle.lower() not in answer:
            reasons.append(f"final answer does not contain '{needle}'")

    memory_needles = _as_str_list(checks.get("memory_contains"))
    if memory_needles:
        data, problem = _read_json(data_dir / "memory.json")
        if problem is not None or not isinstance(data, dict):
            reasons.append(f"memory check failed: {problem or 'memory.json is not an object'}")
        else:
            entries = [str(x).lower() for x in (data.get("preferences") or []) + (data.get("facts") or [])]
            for needle in memory_needles:
                if not any(needle.lower() in entry for entry in entries):
                    reasons.append(f"memory.json has no preference/fact containing '{needle}'")

    if "email_sent_to" in checks:
        expected = str(checks["email_sent_to"]).strip().lower()
        emails, problem = _read_json_list(data_dir / "sent_emails.json")
        if problem is not None:
            reasons.append(f"no email sent to '{expected}' ({problem})")
        elif not any(str(e.get("to", "")).strip().lower() == expected for e in emails):
            sent = ", ".join(str(e.get("to", "")) for e in emails) or "none"
            reasons.append(f"no email sent to '{expected}' (sent to: {sent})")

    if "calendar_title" in checks or "calendar_time_contains" in checks:
        events, problem = _read_json_list(data_dir / "calendar.json")
        title = str(checks.get("calendar_title", "")).lower()
        matching = [e for e in events if title in str(e.get("title", "")).lower()]
        if problem is not None:
            reasons.append(f"calendar check failed: {problem}")
        elif not matching:
            titles = ", ".join(repr(str(e.get("title", ""))) for e in events) or "none"
            reasons.append(f"no calendar event titled like '{checks.get('calendar_title')}' (events: {titles})")
        elif "calendar_time_contains" in checks:
            wanted = str(checks["calendar_time_contains"]).lower()
            if not any(wanted in str(e.get("time", "")).lower() for e in matching):
                times = ", ".join(str(e.get("time", "")) for e in matching)
                reasons.append(f"calendar event time does not contain '{checks['calendar_time_contains']}' (got: {times})")

    if "file_contains" in checks:
        spec = checks["file_contains"] or {}
        rel_path = str(spec.get("path", ""))
        text = str(spec.get("text", ""))
        file_path = data_dir / "workspace" / rel_path
        if not rel_path or not file_path.is_file():
            reasons.append(f"workspace file '{rel_path}' does not exist")
        else:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                reasons.append(f"workspace file '{rel_path}' could not be read: {exc}")
            else:
                if text.lower() not in content.lower():
                    reasons.append(f"workspace file '{rel_path}' does not contain '{text}'")

    for tool in _as_str_list(checks.get("tools_used_include")):
        if tool not in result.tools_used:
            reasons.append(f"tool '{tool}' was not used (used: {', '.join(result.tools_used) or 'none'})")

    if "max_tool_errors" in checks and result.tool_errors > int(checks["max_tool_errors"]):
        reasons.append(f"tool_errors {result.tool_errors} > max_tool_errors {checks['max_tool_errors']}")

    if "max_steps" in checks:
        steps = count_llm_steps(result)
        if steps > int(checks["max_steps"]):
            reasons.append(f"steps {steps} > max_steps {checks['max_steps']}")

    return (not reasons), reasons


def load_eval_tasks(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the eval task list (default ``SEED_DATA_DIR/eval_tasks.json``).

    Raises:
        ValueError: If the file is missing, not valid JSON or malformed.
    """
    path = path or (config.SEED_DATA_DIR / "eval_tasks.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"eval task file not found: {path}") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read eval task file {path}: {exc}") from exc
    if not isinstance(data, list) or not all(
        isinstance(t, dict) and isinstance(t.get("task"), str) for t in data
    ):
        raise ValueError(f"{path} must be a JSON list of objects with a 'task' string")
    return data


def seed_data_dir(target: Path, learnings_text: str | None = None) -> None:
    """Populate a fresh data dir with the seed memory, workspace files and learnings.md.

    Args:
        target: The (empty) data directory to fill.
        learnings_text: Content for learnings.md; None writes the header only.
    """
    seed = config.SEED_DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    example = seed / "memory.example.json"
    if example.is_file():
        shutil.copyfile(example, target / "memory.json")
    workspace = target / "workspace"
    workspace.mkdir(exist_ok=True)
    seed_workspace = seed / "workspace"
    if seed_workspace.is_dir():
        for item in seed_workspace.iterdir():
            if item.is_dir():
                shutil.copytree(item, workspace / item.name, dirs_exist_ok=True)
            elif item.is_file():
                shutil.copyfile(item, workspace / item.name)
    (target / "learnings.md").write_text(learnings_text or LEARNINGS_HEADER, encoding="utf-8")


@contextlib.contextmanager
def temporary_data_dir(learnings_text: str | None = None) -> Iterator[Path]:
    """Create a seeded temp data dir, point ``AGENT_DATA_DIR`` at it, restore and delete afterwards."""
    tmp = Path(tempfile.mkdtemp(prefix="agent-eval-"))
    previous = os.environ.get(DATA_DIR_ENV)
    try:
        seed_data_dir(tmp, learnings_text)
        os.environ[DATA_DIR_ENV] = str(tmp)
        yield tmp
    finally:
        if previous is None:
            os.environ.pop(DATA_DIR_ENV, None)
        else:
            os.environ[DATA_DIR_ENV] = previous
        shutil.rmtree(tmp, ignore_errors=True)


def run_eval_pass(
    tasks: list[dict[str, Any]],
    llm: LLMClient,
    use_learnings: bool,
    learnings_text: str | None,
    max_steps: int,
    on_event: Callable[[str], None] | None,
    log: TextIO,
    label: str,
) -> tuple[list[dict[str, Any]], str]:
    """Run every eval task once in a fresh temp data dir, reflecting after each task.

    Per-task exceptions never abort the pass: the task is marked FAIL with the reason.

    Returns:
        ``(results, learnings_md_text)`` where ``results`` holds one dict per task
        and ``learnings_md_text`` is the final content of learnings.md.
    """
    results: list[dict[str, Any]] = []
    final_learnings = learnings_text or LEARNINGS_HEADER
    print(f"=== {label} ===", file=log, flush=True)
    with temporary_data_dir(learnings_text) as data_dir:
        for index, spec in enumerate(tasks, start=1):
            task_id = spec.get("id", index)
            task_text = str(spec.get("task", ""))
            entry: dict[str, Any] = {
                "id": task_id,
                "task": task_text,
                "passed": False,
                "reasons": [],
                "final_answer": "",
                "success": False,
                "stopped_reason": None,
                "steps": 0,
                "tool_calls": 0,
                "tool_errors": 0,
                "tools_used": [],
                "learnings_used": 0,
            }
            try:
                agent = Agent(
                    llm=llm,
                    max_steps=max_steps,
                    confirm=auto_confirm,
                    use_learnings=use_learnings,
                    on_event=on_event,
                )
                result = agent.run(task_text)
                entry.update(
                    final_answer=result.final_answer,
                    success=result.success,
                    stopped_reason=result.stopped_reason,
                    steps=count_llm_steps(result),
                    tool_calls=result.tool_calls,
                    tool_errors=result.tool_errors,
                    tools_used=list(result.tools_used),
                    learnings_used=result.learnings_used,
                )
                try:
                    run_reflection(result, llm)
                except Exception as exc:
                    _err(f"warning: reflection failed for task {task_id}: {type(exc).__name__}: {exc}")
                passed, reasons = evaluate_task(spec, result, data_dir)
                entry.update(passed=passed, reasons=reasons)
            except Exception as exc:  # never crash the eval
                entry.update(passed=False, reasons=[f"exception: {type(exc).__name__}: {exc}"])
            results.append(entry)
            verdict = "PASS" if entry["passed"] else "FAIL"
            print(
                f"[{task_id}] {verdict}  {task_text}  "
                f"(steps={entry['steps']}, tool_errors={entry['tool_errors']}, "
                f"learnings_used={entry['learnings_used']})",
                file=log,
                flush=True,
            )
            for reason in entry["reasons"]:
                print(f"      - {reason}", file=log, flush=True)
        learnings_file = data_dir / "learnings.md"
        with contextlib.suppress(OSError):
            final_learnings = learnings_file.read_text(encoding="utf-8")
    print(file=log, flush=True)
    return results, final_learnings


def run_eval_mode(args: argparse.Namespace, llm: LLMClient) -> int:
    """Run the two-pass evaluation and print the scores (always exit 0 once it completes)."""
    try:
        tasks = load_eval_tasks()
    except ValueError as exc:
        _err(f"error: {exc}")
        return EXIT_RUNTIME_ERROR
    log: TextIO = sys.stderr if args.json else sys.stdout
    on_event = None if args.quiet else _stderr_event
    total = len(tasks)
    _err(f"Evaluating {total} tasks with provider '{llm.provider}' (model {getattr(llm, 'model', '?')})...")

    run1, learned = run_eval_pass(
        tasks, llm, use_learnings=False, learnings_text=None, max_steps=args.max_steps,
        on_event=on_event, log=log, label="Run 1: without learnings (reflection still records lessons)",
    )
    run2, _ = run_eval_pass(
        tasks, llm, use_learnings=True, learnings_text=learned, max_steps=args.max_steps,
        on_event=on_event, log=log, label="Run 2: with learnings from run 1",
    )
    score1 = sum(1 for r in run1 if r["passed"])
    score2 = sum(1 for r in run2 if r["passed"])
    improvement = score2 - score1

    if args.json:
        _write_json(
            {
                "run1": {"score": score1, "total": total, "results": run1},
                "run2": {"score": score2, "total": total, "results": run2},
                "improvement": improvement,
            }
        )
    else:
        print(f"Run 1 (without learnings): {score1}/{total} passed")
        print(f"Run 2 (with learnings): {score2}/{total} passed")
        print(f"Improvement: {_format_improvement(improvement)}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
_REAL_STDOUT: TextIO = sys.stdout


def _write_json(payload: dict[str, Any]) -> None:
    """Write the single JSON result object to the real stdout."""
    _REAL_STDOUT.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _REAL_STDOUT.flush()


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, create the LLM client and dispatch to task or eval mode.

    Returns:
        Process exit code: 0 on success, 2 on usage or LLM configuration errors,
        1 on runtime failures, 130 when interrupted.
    """
    global _REAL_STDOUT
    _configure_streams()
    args = parse_args(argv)
    config.load_env()
    _REAL_STDOUT = sys.stdout
    # In --json mode, anything printed to stdout by any module goes to stderr instead,
    # so stdout carries exactly one JSON object.
    redirect = contextlib.redirect_stdout(sys.stderr) if args.json else contextlib.nullcontext()
    try:
        with redirect:
            try:
                llm = get_llm(args.provider)
            except (LLMError, ValueError) as exc:
                _err(f"error: LLM configuration problem: {exc}")
                _err("hint: set OPENAI_API_KEY or GROQ_API_KEY in .env, or use --provider mock for the offline simulation")
                return EXIT_USAGE
            if args.eval:
                return run_eval_mode(args, llm)
            # Ensure memory.json exists (seeded from the example) before the run.
            load_memory()
            return run_task_mode(args, llm)
    except KeyboardInterrupt:
        _err("\ninterrupted")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
