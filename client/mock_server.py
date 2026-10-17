"""Mock of the vLLM OpenAI-compatible server for local client development (no GPU).

Mimics the parts the client relies on: /health, /v1/models, /version, /metrics and
streaming /v1/completions + /v1/chat/completions with vLLM's chunk shape (chat starts with a
role-only chunk; usage arrives in a final chunk with empty choices; then [DONE]).
Latency is synthetic: fixed TTFT, then one token every ITL seconds. Not a performance model.

    uv run python -m client.mock_server --port 8000 --ttft 0.2 --itl 0.03
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock/model"


def make_handler(ttft: float, itl: float, model: str = MODEL) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep test output clean
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b"")
            elif self.path == "/version":
                self._send(200, b'{"version": "mock"}')
            elif self.path == "/v1/models":
                body = {"object": "list", "data": [{"id": model, "max_model_len": 8192}]}
                self._send(200, json.dumps(body).encode())
            elif self.path == "/metrics":
                self._send(200, b'vllm:num_requests_running{model_name="mock"} 0.0\n', "text/plain")
            else:
                self._send(404, b"{}")

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chat = self.path == "/v1/chat/completions"
            if not chat and self.path != "/v1/completions":
                self._send(404, b"{}")
                return
            if req.get("model") != model:
                self._send(
                    404, json.dumps({"error": f"model {req.get('model')} not found"}).encode()
                )
                return
            text = req["messages"][-1]["content"] if chat else req["prompt"]
            prompt_tokens = len(text.split())
            n = int(req.get("max_tokens", 16))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(obj) -> None:
                data = obj if isinstance(obj, str) else json.dumps(obj)
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()

            def choice(tok: str, finish: str | None) -> dict:
                if chat:
                    return {"index": 0, "delta": {"content": tok}, "finish_reason": finish}
                return {"index": 0, "text": tok, "finish_reason": finish}

            if chat:
                emit({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
            time.sleep(ttft)
            for i in range(n):
                if i:
                    time.sleep(itl)
                emit({"choices": [choice(f" t{i}", "length" if i == n - 1 else None)]})
            if (req.get("stream_options") or {}).get("include_usage"):
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": n,
                    "total_tokens": prompt_tokens + n,
                }
                emit({"choices": [], "usage": usage})
            emit("[DONE]")
            self.close_connection = True

    return Handler


def start(port: int = 0, ttft: float = 0.2, itl: float = 0.03) -> ThreadingHTTPServer:
    """Start in a daemon thread; port 0 picks a free port (see server.server_address)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(ttft, itl))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ttft", type=float, default=0.2, help="seconds before the first token")
    p.add_argument("--itl", type=float, default=0.03, help="seconds between tokens")
    a = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.ttft, a.itl))
    print(f"mock vLLM on http://127.0.0.1:{a.port} model={MODEL} ttft={a.ttft} itl={a.itl}")
    server.serve_forever()


if __name__ == "__main__":
    main()
