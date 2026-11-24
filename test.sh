#!/usr/bin/env bash
# End-to-end tests for the self-learning agent.
# Runs fully offline with the deterministic mock LLM in throwaway data dirs,
# so your real data/ folder is never touched.
set -uo pipefail
cd "$(dirname "$0")"

# Pick the interpreter: $PYTHON if set, else the project .venv, else python3.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/bin/python ]; then PY=".venv/bin/python"
else PY="python3"; fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
  echo "error: $PY is Python $("$PY" -c 'import platform; print(platform.python_version())' 2>/dev/null || echo '?'); Python 3.9+ is required (3.11+ recommended)." >&2
  echo "fix (macOS): brew install python@3.12 && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 2
fi
export LLM_PROVIDER=mock
export WEB_SEARCH_MODE=mock
unset OPENAI_BASE_URL

PASS=0
FAIL=0
TMP_ROOT="$(mktemp -d)"
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null; fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

ok()   { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

# Create a fresh seeded data dir and export it as AGENT_DATA_DIR.
fresh_data_dir() {
  local d="$TMP_ROOT/$1"
  mkdir -p "$d/workspace"
  cp data/memory.example.json "$d/memory.json"
  cp data/workspace/* "$d/workspace/"
  printf '# Agent Learnings\n' > "$d/learnings.md"
  export AGENT_DATA_DIR="$d"
}

# json_get <json> <python expression on obj `j`>
json_get() { "$PY" -c 'import json,sys; j=json.loads(sys.argv[1]); print(eval(sys.argv[2]))' "$1" "$2"; }

echo "== 1. Name from memory.json =="
fresh_data_dir t1
OUT="$($PY agent/main.py --task "What is my name?" 2>&1)"
if grep -q "Jewel" <<< "$OUT"; then ok "answer contains name 'Jewel' from memory.json"; else bad "name not returned"; echo "$OUT"; fi
for h in "=== Result ===" "=== Reflection ===" "=== Learnings ==="; do
  if grep -qF "$h" <<< "$OUT"; then ok "output has section '$h'"; else bad "missing section '$h'"; fi
done

echo "== 2. Save a preference =="
fresh_data_dir t2
$PY agent/main.py --task "Save that I like tea" >/dev/null 2>&1
if grep -qi "tea" "$AGENT_DATA_DIR/memory.json"; then ok "memory.json updated with 'tea'"; else bad "memory.json not updated"; cat "$AGENT_DATA_DIR/memory.json"; fi

echo "== 3. Email asks for confirmation first =="
fresh_data_dir t3
OUT="$(echo "n" | $PY agent/main.py --task "Send email to test@test.com" 2>&1)"
if grep -q "CONFIRMATION REQUIRED: send_email" <<< "$OUT"; then ok "confirmation prompt shown"; else bad "no confirmation prompt"; echo "$OUT"; fi
if [ ! -s "$AGENT_DATA_DIR/sent_emails.json" ] || [ "$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$AGENT_DATA_DIR/sent_emails.json")" = "0" ]; then
  ok "declined -> nothing saved to sent_emails.json"
else bad "email saved despite decline"; fi
OUT="$(echo "y" | $PY agent/main.py --task "Send email to test@test.com" 2>&1)"
if grep -q "CONFIRMATION REQUIRED: send_email" <<< "$OUT" && \
   $PY -c 'import json,sys; e=json.load(open(sys.argv[1])); assert any(x["to"]=="test@test.com" for x in e)' "$AGENT_DATA_DIR/sent_emails.json" 2>/dev/null; then
  ok "approved -> saved to sent_emails.json"
else bad "approved email not saved"; echo "$OUT"; fi

echo "== 4. Calendar event =="
fresh_data_dir t4
echo "y" | $PY agent/main.py --task "Create calendar event" >/dev/null 2>&1
if $PY -c 'import json,sys; e=json.load(open(sys.argv[1])); assert len(e)>=1 and e[0]["title"] and e[0]["time"]' "$AGENT_DATA_DIR/calendar.json" 2>/dev/null; then
  ok "event saved to calendar.json"
else bad "calendar.json missing event"; fi

echo "== 5. Same task twice: second run uses learning from first =="
fresh_data_dir t5
TASK="Create a calendar event called Standup tomorrow at 9 AM"
R1="$($PY agent/main.py --task "$TASK" --yes --json 2>/dev/null)"
R2="$($PY agent/main.py --task "$TASK" --yes --json 2>/dev/null)"
E1="$(json_get "$R1" 'j["tool_errors"]')"; E2="$(json_get "$R2" 'j["tool_errors"]')"
L1="$(json_get "$R1" 'j["learnings_used"]')"; L2="$(json_get "$R2" 'j["learnings_used"]')"
echo "     run1: tool_errors=$E1 learnings_used=$L1 | run2: tool_errors=$E2 learnings_used=$L2"
if [ "$E1" -ge 1 ] && [ "$E2" -eq 0 ] && [ "$L1" -eq 0 ] && [ "$L2" -gt 0 ]; then
  ok "run 2 loaded learnings and avoided run 1's error"
else bad "no learning effect between runs"; fi
if grep -q "ISO 8601" "$AGENT_DATA_DIR/learnings.md"; then ok "lesson written to learnings.md"; else bad "lesson not in learnings.md"; fi

echo "== 6. Eval mode =="
EVAL_OUT="$($PY agent/main.py --eval 2>&1)"; EVAL_RC=$?
S1="$(sed -n 's/^Run 1 (without learnings): \([0-9]*\)\/10 passed.*/\1/p' <<< "$EVAL_OUT")"
S2="$(sed -n 's/^Run 2 (with learnings): \([0-9]*\)\/10 passed.*/\1/p' <<< "$EVAL_OUT")"
echo "     run1=$S1/10 run2=$S2/10 exit=$EVAL_RC"
if [ "$EVAL_RC" -eq 0 ] && [ -n "$S1" ] && [ -n "$S2" ]; then ok "eval ran 10 tasks twice and printed scores"; else bad "eval crashed or no score"; echo "$EVAL_OUT" | tail -20; fi
if [ -n "$S1" ] && [ -n "$S2" ] && [ "$S2" -gt "$S1" ]; then ok "run 2 score ($S2) > run 1 score ($S1)"; else bad "run 2 did not improve"; fi

echo "== 7. No API keys in code =="
if grep -rEn "sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{20,}" agent/ tests/ README.md .env.example run.sh 2>/dev/null; then
  bad "possible hard-coded API key found"
else ok "no hard-coded keys"; fi
if grep -q 'OPENAI_API_KEY' agent/llm.py && grep -qE 'os\.(getenv|environ)' agent/llm.py; then ok "keys read from environment (.env)"; else bad "keys not read from env"; fi
if grep -qx "OPENAI_API_KEY=your_key_here" .env.example && grep -qx ".env" .gitignore; then ok ".env.example placeholder + .env gitignored"; else bad ".env.example/.gitignore wrong"; fi

echo "== 8. Real openai SDK path (local OpenAI-compatible server) =="
fresh_data_dir t8
PORT_FILE="$TMP_ROOT/port"; REQ_LOG="$TMP_ROOT/requests.jsonl"
$PY tests/fake_openai_server.py "$PORT_FILE" "$REQ_LOG" &
SERVER_PID=$!
for _ in $(seq 1 50); do [ -s "$PORT_FILE" ] && break; sleep 0.1; done
PORT="$(cat "$PORT_FILE" 2>/dev/null)"
R="$(LLM_PROVIDER=openai OPENAI_API_KEY=test-key-not-real OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" \
     $PY agent/main.py --task "Send email to John about meeting tomorrow at 11 AM" --yes --json 2>/dev/null)"
if [ -n "$R" ] && [ "$(json_get "$R" 'j["provider"]')" = "openai" ] && [ "$(json_get "$R" 'j["success"]')" = "True" ] && \
   $PY -c 'import json,sys; e=json.load(open(sys.argv[1])); assert e[-1]["to"]=="john@example.com"' "$AGENT_DATA_DIR/sent_emails.json" 2>/dev/null; then
  ok "openai SDK tool-calling loop works end-to-end"
else bad "openai SDK path failed"; echo "$R"; fi
if $PY -c '
import json,sys
reqs=[json.loads(l) for l in open(sys.argv[1])]
assert all(r["auth_header_present"] for r in reqs)
assert any(r["has_tools"] and r["tool_choice"]=="auto" for r in reqs)
assert any((r["response_format"] or {}).get("type")=="json_object" and not r["has_tools"] for r in reqs)
' "$REQ_LOG" 2>/dev/null; then ok "SDK sent tools+tool_choice for the agent loop and json_object for reflection"; else bad "unexpected SDK requests"; fi
kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""

echo "== 9. Guards found by the real-LLM run =="
fresh_data_dir t9
if $PY - <<'PYEOF' 2>&1; then ok "false-success nudge, weekday table, offset + footer rules, explicit-beats-preference rule"; else bad "guard checks failed"; fi
import json, sys
from datetime import datetime
sys.path.insert(0, ".")
from agent.agent import Agent
from agent.llm import LLMResponse, ToolCall
from agent.prompts import build_system_prompt
from agent.tools import execute_tool

class Scripted:
    """Promises an action first, then actually calls the tool after the nudge."""
    provider, model = "scripted", "scripted"
    def __init__(self): self.calls = 0
    def chat(self, messages, tools=None, json_mode=False):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="I will proceed to create the calendar event.")
        if self.calls == 2:
            assert "did not call any tool" in messages[-1]["content"]
            return LLMResponse(tool_calls=[ToolCall(id="c1", name="get_user_preference", arguments="{}")])
        return LLMResponse(content="Done.")

r = Agent(llm=Scripted(), confirm=lambda n, a: True, use_learnings=False).run("Create an event")
assert r.tools_used == ["get_user_preference"] and r.final_answer == "Done.", r

prompt = build_system_prompt("User name: X", "", datetime(2026, 11, 24, 13, 0))
assert "Friday = 2026-11-27" in prompt and "ALWAYS overrides a stored preference" in prompt

bad_time = execute_tool("create_calendar_event", {"title": "T", "time": "2026-11-25T09:00"}, lambda n, a: True)
assert bad_time.status == "error" and "UTC offset" in bad_time.output, bad_time
good_time = execute_tool("create_calendar_event", {"title": "T", "time": "2026-11-25T09:00+06:00"}, lambda n, a: True)
assert good_time.status == "ok", good_time
no_footer = execute_tool("send_email", {"to": "a@b.co", "subject": "s", "body": "hi"}, lambda n, a: True)
assert no_footer.status == "error" and "outbox policy" in no_footer.output, no_footer
footer = execute_tool("send_email", {"to": "a@b.co", "subject": "s", "body": "hi\n\n-- Sent via AI assistant"}, lambda n, a: True)
assert footer.status == "ok" and "simulated" in footer.output, footer
PYEOF

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
