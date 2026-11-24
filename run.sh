#!/usr/bin/env bash
# Run the agent. Examples:
#   ./run.sh --task "Send email to John about meeting tomorrow at 11 AM"
#   ./run.sh --eval
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ -d .venv ]; then PY=".venv/bin/python"; fi
if [ ! -f .env ] && [ -f .env.example ]; then
  echo "note: no .env found - copy .env.example to .env and add a key (falling back to the offline mock LLM)" >&2
fi
exec "$PY" agent/main.py "$@"
