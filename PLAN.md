# PLAN.md — self-learning-ai-agent

This is the single source of truth for every subagent. **Code against these
contracts exactly** (names, signatures, return types, exact strings). Each
subagent writes ONLY the files it owns (see §3). Use absolute imports
(`from agent.tools import ...`). Python 3.11+, type hints and a docstring on
every function/class/method. Allowed third-party packages: `openai`,
`pydantic` (v2), `python-dotenv`. Everything else must be stdlib. No TODOs.

---

## 1. Architecture

```
User (CLI: agent/main.py)
   │  --task "..."   |   --eval
   ▼
Agent loop (agent/agent.py)  ── ReAct: Think → Act (tool call) → Observe → repeat (max 10)
   │      ▲
   │      │ LLMResponse (content | tool_calls)
   │   LLM client (agent/llm.py)  ── openai | groq | mock (offline, deterministic)
   │
   ├── Tools (agent/tools.py)            Pydantic-validated, confirmation-gated side effects
   ├── Memory (agent/memory.py)          short-term (last 20 msgs) + long-term memory.json
   ├── Prompts (agent/prompts.py)        system prompt w/ Memory + Past Learnings + date
   └── Reflection (agent/reflection.py)  after EVERY task: lessons → learnings.md, prefs → memory.json
                                         next task loads learnings.md into "Past Learnings"
```

Per task flow (`Agent.run`):
1. `load_memory()` and (if `use_learnings`) `load_learnings()`.
2. Build system prompt via `prompts.build_system_prompt(...)`.
3. Loop ≤ `max_steps`: call LLM with tool schemas. Tool calls → `execute_tool` →
   feed observations back. Plain content → final answer, stop.
4. Re-load + `save_memory()` at the end.
5. Caller (main.py) runs `reflection.run_reflection(result, llm)`.

### Why a `mock` provider
The build/test machine has no API key and no network to OpenAI/Groq. `mock`
is a deterministic, rule-based stand-in that speaks the same `LLMClient`
interface (including tool calls and JSON reflection). It is clearly labelled
as a simulation. Real providers share all other code.

### Deviations from the original brief (deliberate, documented in README)
- Extra files: `agent/__init__.py`, `agent/config.py` (paths/env), `agent/llm.py`
  (provider-switchable LLM client incl. mock), `data/memory.example.json`,
  `data/workspace/*` (sandbox for read/write_file), `tests/fake_openai_server.py`.
- Extra 7th tool `save_to_memory` so "Save that I like tea" is an explicit,
  reliable action (reflection ALSO updates memory with newly found preferences).

---

## 2. File structure

```
self-learning-ai-agent/
  agent/
    __init__.py          (main agent)
    config.py            (S7)
    llm.py               (S7)
    agent.py             (S1)
    tools.py             (S2)
    memory.py            (S3)
    reflection.py        (S4)
    prompts.py           (S5)
    main.py              (S6)
  data/
    memory.json          (main agent; gitignored local copy of example)
    memory.example.json  (main agent)
    learnings.md         (main agent; starts with header only)
    eval_tasks.json      (S6)
    workspace/notes.txt  (main agent)
    workspace/todo.md    (main agent)
  tests/fake_openai_server.py (main agent)
  .env.example  .gitignore  requirements.txt  README.md  run.sh  test.sh  PLAN.md  (main agent)
```

Runtime-created (gitignored): `data/sent_emails.json`, `data/calendar.json`.

---

## 3. File ownership (no two subagents touch the same file)

| Subagent | Owns (writes) | Must NOT write |
|---|---|---|
| S1 Agent loop | `agent/agent.py` | anything else |
| S2 Tools | `agent/tools.py` | anything else |
| S3 Memory | `agent/memory.py` | anything else |
| S4 Reflection | `agent/reflection.py` | anything else |
| S5 Prompts | `agent/prompts.py` | anything else |
| S6 CLI + Eval | `agent/main.py`, `data/eval_tasks.json` | anything else |
| S7 LLM + Config | `agent/llm.py`, `agent/config.py` | anything else |
| Main agent | everything else + integration fixes | — |

Subagents may create throwaway test scripts ONLY under `/tmp/<subagent>/`.
Other modules may not exist yet while you work: stub them in `/tmp` for your
own unit tests, never in the project.

---

## 4. Interface contracts

### 4.1 `agent/config.py` (S7)
```python
PROJECT_ROOT: Path            # Path(__file__).resolve().parent.parent
SEED_DATA_DIR: Path           # PROJECT_ROOT / "data"   (repo copy of seeds)
def load_env() -> None        # python-dotenv: load PROJECT_ROOT/.env, override=False. Idempotent.
def data_dir() -> Path        # env AGENT_DATA_DIR if set (re-read on EVERY call) else SEED_DATA_DIR; mkdir(parents, exist_ok)
def workspace_dir() -> Path   # data_dir() / "workspace"; mkdir
def get_provider() -> str     # env LLM_PROVIDER (lowercased) if set in {"openai","groq","mock"};
                              # else "openai" if OPENAI_API_KEY, else "groq" if GROQ_API_KEY, else "mock".
                              # Calls load_env() first. Invalid value -> ValueError with clear message.
```

### 4.2 `agent/llm.py` (S7)
```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str            # JSON-encoded object string, exactly as OpenAI returns it

class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = []

class LLMError(Exception): ...   # raised on unrecoverable provider errors

class LLMClient(Protocol):
    provider: str
    model: str
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             json_mode: bool = False) -> LLMResponse: ...

class OpenAICompatibleClient:     # provider "openai" or "groq"
    def __init__(self, provider: str, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, temperature: float = 0.2, max_retries: int = 3) -> None
    def chat(...) -> LLMResponse
class MockLLM:                    # provider "mock", model "mock-rule-based-v1"
    def chat(...) -> LLMResponse
def get_llm(provider: str | None = None) -> LLMClient   # None -> config.get_provider()
```
- openai: key `OPENAI_API_KEY`, model env `OPENAI_MODEL` default `gpt-4o-mini`,
  base_url env `OPENAI_BASE_URL` (optional; used by tests' fake server).
- groq: key `GROQ_API_KEY`, model env `GROQ_MODEL` default `llama-3.3-70b-versatile`,
  base_url `https://api.groq.com/openai/v1` (env `GROQ_BASE_URL` overrides).
- Missing key for chosen real provider → `LLMError("OPENAI_API_KEY is not set ...")`.
- `chat`: `client.chat.completions.create(model, messages, temperature, tools=tools, tool_choice="auto")`
  (omit tools/tool_choice when `tools` is None/empty). `json_mode=True` → `response_format={"type":"json_object"}`.
  Retry transient errors (`openai.APIConnectionError`, `RateLimitError`, `APITimeoutError`,
  `InternalServerError`) with exponential backoff (0.5s, 1s, 2s); then raise `LLMError`.
  Convert `message.tool_calls` → `ToolCall` list (function.name, function.arguments).
- Never log or print API keys.

**MockLLM policy (deterministic, stateless — re-derives next action from `messages`):**
- Reads the system message (first message, role "system"): parses
  `Current date/time: YYYY-MM-DDTHH:MM (Weekday)` (§4.6), `User name: X`, and the
  bullet lines under `## Past Learnings` (until next `## ` header).
- Task = content of the FIRST `role=="user"` message.
- History = assistant `tool_calls` paired with following `role=="tool"` messages (by `tool_call_id`).
- `json_mode=True` (reflection) → returns `LLMResponse(content=<JSON str>)` matching the
  `Reflection` schema in §4.5, derived from the reflection user prompt (§4.6 format), rules:
  - an error line containing `ISO 8601` → lesson
    `"Convert natural-language times to ISO 8601 (YYYY-MM-DDTHH:MM) before calling create_calendar_event."`
  - containing `is not a valid email address` → lesson
    `"When only a contact name is given, call get_user_preference first to look up their email address before calling send_email."`
  - containing `File not found` → lesson
    `"If a file is not found, use the exact filename from the available workspace files listed in the error."`
  - a CANCELLED line → lesson `"The user declined an action; do not retry it without new instructions."`
  - no errors → one lesson: `"For tasks like '<first 40 chars of task>', calling <tool1 -> tool2> worked without errors."`
    (or `"... answering directly worked without errors."` if no tools). Max 3 lessons.
  - `new_preferences`: if task matches `i (like|love|prefer|enjoy) X` AND no `save_to_memory` call appears
    in the transcript → `["likes X"]` (use the verb's 3rd person: likes/loves/prefers/enjoys). Else `[]`.
  - `new_facts`: `[]`. went_well/went_wrong: short sentences built from the counts.
- Tool mode — intent by task (lower-cased), checked in THIS order:
  1. **file write**: contains a filename (`[\w\-./]+\.(txt|md|json|csv|py)`) and one of `write|create|save` and
     content after `with the content|containing|saying|with text` → `write_file(path=filename, content=...)`.
  2. **file read**: filename + one of `read|open|show|summarize|summarise` → `read_file(path)`. On
     `ERROR: File not found` → pick closest name from the `Available files in workspace:` list
     (difflib) and retry once; else final answer explaining. On success final answer:
     `"Summary of <path>: <first 3 non-empty lines joined with ' | '>"`.
  3. **email**: contains `email` or `mail ` → recipient = first email regex match, else the word after
     `to ` / `email ` (case-insensitive) that is capitalized in the original task (e.g. John, Sarah).
     subject = topic after `about ` (strip trailing time phrase), Title-cased, ≤ 6 words; else after
     `saying ` ; default `"Quick note"`. body = `"Hi <Name or there>,\n\n<One short sentence about topic>.\n\nBest,\n<User name>"`.
     - address known → `send_email`.
     - name only: if a Past Learning contains both `email` and `get_user_preference` → call
       `get_user_preference` FIRST, find a fact matching `<name>.*?<email>` → `send_email`.
       Without that lesson → `send_email(to=<name>)` (fails) → then `get_user_preference` → `send_email`.
       Not found in memory → final answer asking for the address.
     - After success final answer: `"Email sent to <to> with subject '<subject>'."`;
       after CANCELLED → `"Okay — I did not send the email."`
  4. **calendar**: contains `calendar|event|meeting|schedule|appointment` (and not email intent) →
     title from `called|titled|named <Title>` (stop at ` at|on|for|tomorrow|today|next` or end),
     else `"Meeting"`. time phrase = text from first of `today|tomorrow|monday..sunday|on |at ` to end;
     if none → `"tomorrow at 10 AM"`. If a Past Learning contains `ISO 8601` → convert first and call
     `create_calendar_event(title, time=<ISO>)`. Otherwise call with the raw phrase (fails) then
     convert and retry. Converter `parse_natural_time(phrase, now) -> str | None`: today/tomorrow/
     weekday (next occurrence, strictly after today)/`YYYY-MM-DD`; `at H(:MM)? (am|pm)?`, `noon`;
     default hour 10:00. Final answer `"Created calendar event '<title>' at <ISO>."`; CANCELLED →
     `"Okay — I did not create the event."`
  5. **save memory**: `save|remember|note` or starts with `i (like|love|prefer|enjoy)` →
     content normalised (`"that I like tea"` → `"likes tea"`; otherwise the text after the verb),
     kind `"preference"` if like/love/prefer/enjoy/hate/dislike else `"fact"` →
     `save_to_memory(content, kind)` → final `"Saved to memory: <content>"`.
  6. **name / memory question**: `my name|who am i|my preferences|what do i like|about me` →
     `get_user_preference` → final `"Your name is <user_name>."` (name) or lists preferences.
  7. **web search**: `search|look up|find|latest|news` → `search_web(query=<task minus "search the web for">)`
     → final `"Here is what I found: <first 2 result lines>"`.
  8. otherwise: final answer `"I can help with email, calendar events, files, web search and your memory. Could you rephrase your request?"`
- Generic rules: after any tool result starting with `ERROR:` that the intent does not handle →
  final answer `"I could not complete the task: <error>"`. If ≥ 6 tool calls in history → final answer.
  Tool call ids: `"call_<n>"` where n = number of tool calls so far + 1. Arguments via `json.dumps`.

### 4.3 `agent/memory.py` (S3)
```python
class LongTermMemory(BaseModel):
    user_name: str = "User"
    preferences: list[str] = []
    facts: list[str] = []
def memory_path() -> Path                  # config.data_dir() / "memory.json"
def load_memory() -> LongTermMemory        # if missing: copy SEED_DATA_DIR/"memory.example.json" if it exists
                                           # (and differs from target dir) else defaults; write it. Corrupt JSON ->
                                           # back up to memory.json.bak, return defaults (and save).
def save_memory(mem: LongTermMemory) -> None   # atomic write (tmp + os.replace), indent=2, ensure_ascii=False
def add_fact(fact: str) -> bool            # strip; ignore empty; case-insensitive dedupe; load→append→save; True if added
def add_preference(pref: str) -> bool      # same semantics
def format_memory_for_prompt(mem: LongTermMemory) -> str
    # EXACT format:
    # User name: Jewel
    # Preferences:
    # - prefers meetings after 10 AM
    # Facts:
    # - John's email is john@example.com
    # (empty list -> line "- (none)")

class ShortTermMemory:
    def __init__(self, max_messages: int = 20) -> None
    def add(self, message: dict[str, Any]) -> None   # append then trim
    def get(self) -> list[dict[str, Any]]            # copy
    def clear(self) -> None
    def __len__(self) -> int
    # trim rule: keep the last max_messages; then, while the first kept message has role "tool",
    # drop it (never start with an orphan tool result). The FIRST user message (the task) is pinned:
    # if it would be trimmed, keep it at index 0 and trim the rest to fit max_messages.
```

### 4.4 `agent/tools.py` (S2)
```python
ConfirmFn = Callable[[str, dict[str, Any]], bool]   # (tool_name, validated_args) -> approved?

class ToolResult(BaseModel):
    tool: str
    status: Literal["ok", "error", "cancelled"]
    output: str
    @property
    def ok(self) -> bool                    # status == "ok"

class Tool(BaseModel):  # model_config arbitrary_types_allowed
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., str]
    requires_confirmation: bool = False

TOOLS: dict[str, Tool]
def get_tool_schemas() -> list[dict[str, Any]]
    # OpenAI format: {"type":"function","function":{"name","description","parameters": args_model.model_json_schema()}}
    # strip "title" keys from parameters for cleanliness.
def execute_tool(name: str, arguments: str | dict[str, Any] | None, confirm: ConfirmFn | None = None) -> ToolResult
    # NEVER raises. Unknown tool / bad JSON / pydantic ValidationError / exceptions -> status "error",
    # output starting with "ERROR: ". For requires_confirmation tools: call confirm(name, args.model_dump())
    # BEFORE the side effect; confirm None or returns False -> status "cancelled",
    # output "CANCELLED: The user declined <tool_name>. Do not retry this action unless asked."
```
Tools (arg models with `Field(description=...)`):

| name | args | behaviour |
|---|---|---|
| `search_web` | `query: str` (min_length 1) | If env `WEB_SEARCH_MODE=live`: DuckDuckGo Instant Answer API via `urllib.request` (`https://api.duckduckgo.com/?q=...&format=json&no_html=1`), timeout 5s; any failure → mock. Mock (default): deterministic text, first line `"[mock search results for '<query>']"`, then 3 numbered lines `"1. <title> - <snippet>"` derived from the query (include the query words). Live output also starts with a header line. |
| `read_file` | `path: str` | Resolve with `_safe_path` (below). Missing → `"ERROR: File not found: '<path>'. Available files in workspace: <comma list or (none)>"`. Truncate > 20 000 chars with note. |
| `write_file` | `path: str`, `content: str` | `_safe_path`, mkdir parents, write utf-8 → `"Wrote <n> characters to <relative path>"`. |
| `send_email` | `to: str`, `subject: str`, `body: str` | **requires_confirmation**. Validate `to` with regex `^[^@\s]+@[^@\s]+\.[^@\s]+$` (in the function, so it returns an error, not a schema error) → else `"ERROR: '<to>' is not a valid email address. If you only have a contact name, look up their email with get_user_preference (contacts are stored in facts)."` Validation happens BEFORE asking confirmation. Append `{"to","subject","body","sent_at": ISO now}` to `data_dir()/sent_emails.json` (JSON list; create if missing) → `"Email saved to outbox (mock, not actually sent): to=<to>, subject=<subject>"`. |
| `create_calendar_event` | `title: str`, `time: str` | **requires_confirmation**. Validate time `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$` + `datetime.fromisoformat` BEFORE confirmation, else `"ERROR: Invalid time format '<time>'. create_calendar_event requires ISO 8601 format YYYY-MM-DDTHH:MM (e.g. 2026-11-25T11:00). Convert relative times like 'tomorrow at 11 AM' using the current date."` Append `{"title","time","created_at"}` to `data_dir()/calendar.json` → `"Calendar event created (mock): '<title>' at <time>"`. Field description for `time`: `"Event start time"` (intentionally does not reveal the format; the agent learns it). |
| `get_user_preference` | (no args) | Returns `json.dumps(load_memory().model_dump(), ensure_ascii=False)`. |
| `save_to_memory` | `content: str` (min_length 1), `kind: Literal["preference","fact"] = "preference"` | `add_preference`/`add_fact` → `"Saved <kind>: <content>"` or `"Already in memory: <content>"`. |

Confirmation is enforced HERE (tool layer), not by trusting the LLM.
`_safe_path(path) -> Path`: relative paths resolve against `workspace_dir()`; a leading
`workspace/` or `data/workspace/` prefix is stripped first; absolute paths allowed only if inside
`workspace_dir()`; result must stay inside `workspace_dir()` (after `.resolve()`), and any path
component named `.env` is refused → `"ERROR: Access denied: ..."` (raise `PermissionError` internally,
converted by execute_tool). JSON list files: helper `_append_json_list(path, item)` (corrupt → start new list).

### 4.5 `agent/reflection.py` (S4)
```python
class Reflection(BaseModel):
    went_well: str = ""
    went_wrong: str = ""
    lessons: list[str] = []          # 1–3 after normalisation
    new_preferences: list[str] = []
    new_facts: list[str] = []

class ReflectionReport(BaseModel):
    reflection: Reflection
    lessons_added: list[str]         # actually appended (after dedupe)
    preferences_added: list[str]
    facts_added: list[str]

def learnings_path() -> Path                       # config.data_dir() / "learnings.md"
def load_lessons() -> list[str]                    # all lines starting with "Lesson: " (text after prefix), file order; [] if missing
def load_learnings(max_lessons: int = 30) -> str   # unique lessons (case-insens, keep LAST occurrence order),
                                                   # most recent max_lessons, as "- <lesson>" lines joined "\n"; "" if none
def format_transcript(result: "AgentResult") -> str  # see §4.6 reflection prompt input
def reflect(result: "AgentResult", llm: LLMClient) -> Reflection
    # messages = [system REFLECTION_SYSTEM_PROMPT, user build_reflection_user_prompt(...)]; llm.chat(..., tools=None, json_mode=True)
    # parse first {...} JSON block -> Reflection. On LLMError/parse/validation failure -> heuristic fallback
    # (same rules as the mock in §4.2, implemented locally). Normalise: strip, drop empty, dedupe, cap 3;
    # if 0 lessons -> ["Review the task outcome and keep the approach that worked."]
def append_learnings(task: str, lessons: list[str], when: date | None = None) -> list[str]
    # skip lessons already in load_lessons() (case-insens). If any remain, append block:
    # "\n### YYYY-MM-DD - <task title>\nLesson: <l1>\nLesson: <l2>\n"
    # task title = task stripped, single line, max 60 chars (+ "…" if cut). Create file with header
    # "# Agent Learnings\n" if missing. Returns lessons appended.
def run_reflection(result: "AgentResult", llm: LLMClient) -> ReflectionReport
    # reflect -> append_learnings -> memory.add_preference / add_fact for new_* -> report
```
Import `AgentResult` only under `typing.TYPE_CHECKING` (avoid circular import); use attributes
listed in §4.7.

### 4.6 `agent/prompts.py` (S5)
```python
NO_LEARNINGS = "(none yet)"
SYSTEM_PROMPT_TEMPLATE: str
def build_system_prompt(memory_text: str, learnings_text: str, now: datetime) -> str
REFLECTION_SYSTEM_PROMPT: str
def build_reflection_user_prompt(task: str, transcript: str, final_answer: str, memory_text: str) -> str
```
`build_system_prompt` output MUST contain these exact structural lines (mock parses them):
```
Current date/time: 2026-11-24T11:33 (Tuesday)        <- now.strftime("%Y-%m-%dT%H:%M (%A)")
## User Memory
<memory_text>
## Past Learnings
<learnings_text or NO_LEARNINGS>
## Rules
...
```
SYSTEM prompt content must state: you are a helpful self-learning agent; ALWAYS check Past
Learnings before acting and follow them; always confirm before sending email / creating a calendar
event — explain that confirmation is enforced by the system: calling `send_email` /
`create_calendar_event` shows the user the exact details and waits for approval, so call the tool
with complete, correct details instead of asking in text; if the user declines, don't retry; if a
tool fails, read the error, change the arguments or try a different approach — never repeat the
same failing call; use tools instead of guessing; when only a contact name is given contacts may be
in memory facts; draft short sensible defaults for missing non-critical details (subject/body);
respect user preferences; keep the final answer short; one tool call at a time is fine.

`REFLECTION_SYSTEM_PROMPT`: asks "What went well? What went wrong? What rule should I follow next
time?"; reply ONLY with a JSON object with keys `went_well`, `went_wrong`, `lessons` (1-3 short,
general, reusable imperative rules), `new_preferences` (only preferences the user explicitly stated
and that are NOT already saved in memory), `new_facts`. Must contain the word "JSON".

`build_reflection_user_prompt` format (the mock + heuristic parse it):
```
Task: <task>
Final answer: <final_answer>
Current memory:
<memory_text>
Transcript:
<transcript>
```
`format_transcript` (S4) produces lines, one per step, exactly:
`- step <n>: <tool>(<args json>) -> <STATUS>: <observation first 300 chars, newlines→space>`
with STATUS in OK / ERROR / CANCELLED (so error lines contain the tool's "ERROR: ..." text),
or `- step <n>: final answer` for the final step. Followed by `Tool errors: <count>`.

### 4.7 `agent/agent.py` (S1)
```python
class AgentStep(BaseModel):
    step: int
    thought: str | None = None            # assistant content accompanying tool calls, if any
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    observation: str | None = None
    status: Literal["ok", "error", "cancelled", "final"] = "ok"

class AgentResult(BaseModel):
    task: str
    final_answer: str
    steps: list[AgentStep]
    tool_calls: int
    tool_errors: int                      # count of status "error" (cancelled is NOT an error)
    tools_used: list[str]                 # tool names in call order
    success: bool                         # stopped_reason == "final_answer"
    stopped_reason: Literal["final_answer", "max_steps", "llm_error"]
    learnings_used: int                   # number of lesson lines injected into the prompt
    provider: str
    messages: list[dict[str, Any]]        # full transcript incl. system prompt

class Agent:
    def __init__(self, llm: LLMClient | None = None, max_steps: int = 10,
                 confirm: ConfirmFn | None = None, use_learnings: bool = True,
                 on_event: Callable[[str], None] | None = None) -> None   # llm None -> get_llm()
    def run(self, task: str) -> AgentResult
```
Loop details:
- `messages` sent each LLM call = `[system] + ShortTermMemory(max 20).get()`; full transcript kept
  separately for `AgentResult.messages`.
- An LLM "step" = one LLM call. Up to `max_steps` (default 10) LLM calls.
- Assistant tool-call message format: `{"role":"assistant","content": content or None,
  "tool_calls":[{"id","type":"function","function":{"name","arguments"}}]}`; tool result
  message: `{"role":"tool","tool_call_id": id, "content": output}`. Execute ALL tool calls of a
  response, each recorded as its own AgentStep (same step number).
- Error recovery: when a tool returns status "error", append to that tool message content:
  `"\n[agent note] This call failed. Re-plan: fix the arguments or use a different tool. Do not repeat the same call."`
  If the identical (name, canonical-json args) call already failed earlier in this run, do NOT
  execute it again; return observation `"ERROR: Repeated failing call blocked: <name> with the same arguments already failed. Try a different approach."` (counts as error).
- Final answer = response without tool_calls; `content` (empty → "(no answer)"). Record AgentStep status "final".
- max steps reached → final_answer `"Stopped: reached the maximum of <n> steps without a final answer."`
- `LLMError` → stopped_reason "llm_error", final_answer `"LLM error: <msg>"`.
- `learnings_used` = number of non-empty lines in the learnings text injected.
- At end: `save_memory(load_memory())`. `on_event` (if given) receives short human-readable progress
  strings (e.g. `"→ send_email {...}"`, `"← ok: ..."`).

### 4.8 `agent/main.py` (S6)
Top of file: make `python agent/main.py` work — remove the script's own directory from
`sys.path` (it would shadow the `agent` package with `agent/agent.py`) and insert `PROJECT_ROOT`.
Call `config.load_env()`.

CLI (`argparse`):
```
python agent/main.py --task "..." [--yes] [--json] [--no-learnings] [--provider openai|groq|mock] [--max-steps N] [--quiet]
python agent/main.py --eval [--json] [--provider ...]
```
- `--task`: build Agent (confirm = interactive unless `--yes`), run, then `run_reflection`.
  Human output sections (exact headers): `=== Result ===`, `=== Reflection ===`, `=== Learnings ===`.
  Learnings section lists lessons added this run and total lessons in learnings.md, plus any
  memory updates. `--json`: print ONE JSON object to stdout (all prompts/progress → stderr) with keys
  `task, final_answer, success, stopped_reason, steps (int), tool_calls, tool_errors, tools_used,
  learnings_used, provider, reflection {went_well, went_wrong, lessons}, lessons_added,
  preferences_added, facts_added`.
- Interactive confirm: print to stderr a block `CONFIRMATION REQUIRED: <tool_name>` + each arg
  `  <k>: <v>` and prompt `Approve? [y/N]: ` (to stderr), read one line from stdin; `y`/`yes` → True;
  EOF/anything else → False.
- `--eval`: load tasks from `SEED_DATA_DIR/"eval_tasks.json"`. Two runs, each in a fresh temp data
  dir (`tempfile.mkdtemp`) seeded by copying `SEED_DATA_DIR/memory.example.json` → `memory.json`
  and `SEED_DATA_DIR/workspace/*` → `workspace/`; set `os.environ["AGENT_DATA_DIR"]` for the run
  and restore afterwards.
  Run 1: `use_learnings=False`, empty learnings.md; reflection still runs after every task (this is
  where it learns). Run 2: fresh seed state, but `learnings.md` copied from run 1;
  `use_learnings=True`. Auto-confirm (`confirm=lambda n, a: True`). Same llm instance for both.
  Print per task `PASS`/`FAIL` with failed-check reasons, then exactly:
  `Run 1 (without learnings): X/10 passed` and `Run 2 (with learnings): Y/10 passed` and
  `Improvement: +N` (or `Improvement: 0`/negative). `--json` → `{"run1": {"score", "total", "results": [...]}, "run2": {...}, "improvement": int}`.
  Exit code 0 if eval completed (regardless of score). Temp dirs removed afterwards.
- Check evaluation (`evaluate_task(task_spec, result, data_dir) -> tuple[bool, list[str]]`):
  pass requires `result.success` AND all checks:
  - `answer_contains: list[str]` — each substring in final_answer (case-insensitive)
  - `memory_contains: list[str]` — each substring in some preference or fact of memory.json (ci)
  - `email_sent_to: str` — an entry in sent_emails.json with `to` equal (ci)
  - `calendar_title: str` — an event whose title contains it (ci)
  - `calendar_time_contains: str` — that same-title event's time contains it (e.g. "T09:00")
  - `file_contains: {"path": str, "text": str}` — workspace file exists and contains text (ci)
  - `tools_used_include: list[str]`
  - `max_tool_errors: int` — result.tool_errors ≤ value
  - `max_steps: int` — number of LLM steps (distinct step numbers) ≤ value

### 4.9 `data/eval_tasks.json` (S6)
JSON list of exactly 10 objects `{"id","task","checks"}`. Seed memory (main agent):
`{"user_name":"Jewel","preferences":["prefers meetings after 10 AM","likes short emails"],
"facts":["John's email is john@example.com","Sarah's email is sarah@example.com"]}`.
Workspace seeds: `notes.txt` (first line `Project kickoff notes`, mentions "Q1 launch"), `todo.md`.
Tasks (all with `max_tool_errors: 0`):
1. "What is my name?" → answer_contains ["Jewel"]
2. "Save that I like green tea" → memory_contains ["green tea"]
3. "Send an email to test@test.com saying hello" → email_sent_to "test@test.com"
4. "Send email to John about meeting tomorrow at 11 AM" → email_sent_to "john@example.com"
5. "Email Sarah about the project update" → email_sent_to "sarah@example.com"
6. "Create a calendar event called Standup tomorrow at 9 AM" → calendar_title "Standup", calendar_time_contains "T09:00"
7. "Schedule a meeting titled Design Review on Friday at 3 PM" → calendar_title "Design Review", calendar_time_contains "T15:00"
8. "Read the file notes.txt and summarize it" → answer_contains ["kickoff"]
9. "Write a file called ideas.txt with the content Build a self-learning agent" → file_contains {"path":"ideas.txt","text":"self-learning"}
10. "Search the web for Python 3.11 new features" → tools_used_include ["search_web"], answer_contains ["Python"]

Expected with mock: run 1 = 6/10 (tasks 4–7 recover from errors but break `max_tool_errors: 0`),
run 2 = 10/10 because Past Learnings make the agent avoid those errors.

---

## 5. Testing (main agent, `test.sh`)
All tests run with `LLM_PROVIDER=mock` in a temp `AGENT_DATA_DIR`:
1. `--task "What is my name?"` → output contains "Jewel".
2. `--task "Save that I like tea"` → memory.json contains "tea".
3. `--task "Send email to test@test.com"` with stdin `n` → "CONFIRMATION REQUIRED" shown, nothing
   saved; with stdin `y` → saved to sent_emails.json.
4. `--task "Create calendar event"` (stdin `y`) → calendar.json has an event.
5. Same calendar task twice in a fresh dir → run 1 `tool_errors ≥ 1`, run 2 `tool_errors == 0` and
   `learnings_used > 0`.
6. `--eval` → prints both scores, no crash, run 2 > run 1.
7. No API keys in code (regex scan), keys read only via env.
8. Bonus: real `openai` SDK path against `tests/fake_openai_server.py` (local OpenAI-compatible
   server) via `LLM_PROVIDER=openai OPENAI_BASE_URL=http://127.0.0.1:<port>/v1`.
