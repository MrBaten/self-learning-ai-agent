"""Local OpenAI-compatible server used to test the real `openai` SDK code path offline.

It implements ``POST /v1/chat/completions`` and answers with the deterministic
:class:`agent.llm.MockLLM` policy, returning responses in the exact OpenAI wire
format (including ``tool_calls``). Every request is appended as one JSON line to
the log file so tests can assert what the SDK actually sent.

Usage::

    python tests/fake_openai_server.py <port_file> <request_log>

The chosen port is written to ``port_file`` once the server is listening.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm import MockLLM  # noqa: E402

_MOCK = MockLLM()


def build_completion(body: dict[str, Any]) -> dict[str, Any]:
    """Run the mock policy on an OpenAI request body and wrap it as a chat completion."""
    json_mode = (body.get("response_format") or {}).get("type") == "json_object"
    response = _MOCK.chat(body.get("messages", []), tools=body.get("tools"), json_mode=json_mode)
    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    finish_reason = "stop"
    if response.tool_calls:
        message["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in response.tool_calls
        ]
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-fake-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "fake-model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    """HTTP handler for the fake chat completions endpoint."""

    log_path: Path

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        """Handle ``POST /v1/chat/completions``."""
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        record = {
            "auth_header_present": bool(self.headers.get("Authorization")),
            "model": body.get("model"),
            "has_tools": bool(body.get("tools")),
            "tool_choice": body.get("tool_choice"),
            "response_format": body.get("response_format"),
            "n_messages": len(body.get("messages", [])),
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        self._send(200, build_completion(body))

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        """Write a JSON response."""
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence default request logging."""


def main() -> None:
    """Start the server on a free port and write the port to the given file."""
    port_file, log_file = Path(sys.argv[1]), Path(sys.argv[2])
    Handler.log_path = log_file
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port_file.write_text(str(server.server_address[1]), encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
