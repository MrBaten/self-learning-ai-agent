#!/usr/bin/env bash
# Run the agent. Examples:
#   ./run.sh --task "Send email to John about meeting tomorrow at 11 AM"
#   ./run.sh --eval
set -euo pipefail
cd "$(dirname "$0")"
# Pick the interpreter: $PYTHON if set, else the project .venv, else python3.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/bin/python ]; then PY=".venv/bin/python"
else PY="python3"; fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "error: $PY is Python $("$PY" -c 'import platform; print(platform.python_version())' 2>/dev/null || echo '?'); Python 3.10+ is required (3.11+ recommended)." >&2
  echo "fix (macOS): brew install python@3.12 && rm -rf .venv && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 2
fi
if [ ! -f .env ] && [ -f .env.example ]; then
  echo "note: no .env found - copy .env.example to .env and add a key (falling back to the offline mock LLM)" >&2
fi
exec "$PY" agent/main.py "$@"
