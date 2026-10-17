"""Mock of the vLLM OpenAI-compatible server for local client development (no GPU).

Mimics the parts the client relies on: /health, /v1/models, /version, /metrics,
/tokenize, /detokenize and streaming /v1/completions + /v1/chat/completions with vLLM's chunk
shape (chat starts with a role-only chunk; usage arrives in a final chunk with empty choices;
then [DONE]). Completions accept a text prompt or a list of token ids.

Toy latency model, NOT a performance model of vLLM:
  - at most `max_num_seqs` requests run at once, the rest wait in a FIFO queue;
  - TTFT = ttft + queue wait;
  - each next token takes itl * (1 + batch_slowdown * (running - 1)).
It exists so the load generator, metrics scraper and analysis can be exercised end to end.
Fake tokenizer: a word "w<id>" is token <id>; any other word is token 0.

    uv run python -m client.mock_server --port 8000 --ttft 0.2 --itl 0.03 --max-num-seqs 8
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock/model"


@dataclass
class MockConfig:
    ttft: float = 0.2
    itl: float = 0.03
    max_num_seqs: int = 256
    batch_slowdown: float = 0.0
    fail_first: int = 0  # answer the first N generation requests with 503 (retry tests)
    model: str = MODEL


class MockState:
    def __init__(self, cfg: MockConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.slots = threading.Semaphore(cfg.max_num_seqs)
        self.running = 0
        self.waiting = 0
        self.max_running_seen = 0
        self.failures_left = cfg.fail_first
        self.counters = {
            "prompt_tokens_total": 0.0,
            "generation_tokens_total": 0.0,
            "request_success_total": 0.0,
            "time_to_first_token_seconds_sum": 0.0,
            "time_to_first_token_seconds_count": 0.0,
            "request_queue_time_seconds_sum": 0.0,
            "request_queue_time_seconds_count": 0.0,
        }

    def add(self, **kw: float) -> None:
        with self.lock:
            for k, v in kw.items():
                self.counters[k] += v

    def metrics_text(self) -> str:
        lbl = f'{{engine="0",model_name="{self.cfg.model}"}}'
        with self.lock:
            lines = [
                f"vllm:num_requests_running{lbl} {float(self.running)}",
                f"vllm:num_requests_waiting{lbl} {float(self.waiting)}",
                f'vllm:num_requests_waiting_by_reason{lbl[:-1]},reason="capacity"}} '
                f"{float(self.waiting)}",
                f'vllm:num_requests_waiting_by_reason{lbl[:-1]},reason="deferred"}} 0.0',
                f"vllm:kv_cache_usage_perc{lbl} {self.running / self.cfg.max_num_seqs}",
                f"vllm:num_preemptions_total{lbl} 0.0",
            ]
            lines += [f"vllm:{k}{lbl} {v}" for k, v in self.counters.items()]
        return "\n".join(lines) + "\n"


def fake_tokenize(text: str) -> list[int]:
    return [int(w[1:]) if w[:1] == "w" and w[1:].isdigit() else 0 for w in text.split()]


def make_handler(state: MockState) -> type[BaseHTTPRequestHandler]:
    cfg = state.cfg

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

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b"")
            elif self.path == "/version":
                self._json({"version": "mock"})
            elif self.path == "/v1/models":
                self._json({"object": "list", "data": [{"id": cfg.model, "max_model_len": 8192}]})
            elif self.path == "/metrics":
                self._send(200, state.metrics_text().encode(), "text/plain")
            else:
                self._send(404, b"{}")

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                toks = fake_tokenize(req["prompt"])
                self._json({"tokens": toks, "count": len(toks), "max_model_len": 8192})
                return
            if self.path == "/detokenize":
                self._json({"prompt": " ".join(f"w{t}" for t in req["tokens"])})
                return
            chat = self.path == "/v1/chat/completions"
            if not chat and self.path != "/v1/completions":
                self._send(404, b"{}")
                return
            if req.get("model") != cfg.model:
                self._json({"error": f"model {req.get('model')} not found"}, 404)
                return
            with state.lock:
                fail = state.failures_left > 0
                state.failures_left -= fail
            if fail:
                self._json({"error": "mock overload"}, 503)
                return
            self._generate(req, chat)

        def _generate(self, req: dict, chat: bool) -> None:
            prompt = req["messages"][-1]["content"] if chat else req["prompt"]
            prompt_tokens = len(prompt) if isinstance(prompt, list) else len(prompt.split())
            n = int(req.get("max_tokens", 16))
            arrived = time.perf_counter()

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
            with state.lock:
                state.waiting += 1
            state.slots.acquire()
            queued = time.perf_counter() - arrived
            with state.lock:
                state.waiting -= 1
                state.running += 1
                state.max_running_seen = max(state.max_running_seen, state.running)
            try:
                time.sleep(cfg.ttft)
                state.add(
                    time_to_first_token_seconds_sum=time.perf_counter() - arrived,
                    time_to_first_token_seconds_count=1,
                    request_queue_time_seconds_sum=queued,
                    request_queue_time_seconds_count=1,
                )
                for i in range(n):
                    if i:
                        time.sleep(cfg.itl * (1 + cfg.batch_slowdown * (state.running - 1)))
                    emit({"choices": [choice(f" t{i}", "length" if i == n - 1 else None)]})
            finally:
                with state.lock:
                    state.running -= 1
                state.slots.release()
            state.add(prompt_tokens_total=prompt_tokens, generation_tokens_total=n,
                      request_success_total=1)  # fmt: skip
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


class MockServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024  # listen backlog: open-loop tests open many connections at once

    def __init__(self, port: int, cfg: MockConfig):
        self.state = MockState(cfg)
        super().__init__(("127.0.0.1", port), make_handler(self.state))


def start(port: int = 0, ttft: float = 0.2, itl: float = 0.03, **kw) -> MockServer:
    """Start in a daemon thread; port 0 picks a free port (see server.server_address)."""
    server = MockServer(port, MockConfig(ttft=ttft, itl=itl, **kw))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ttft", type=float, default=0.2, help="seconds before the first token")
    p.add_argument("--itl", type=float, default=0.03, help="seconds between tokens at batch 1")
    p.add_argument("--max-num-seqs", type=int, default=256, help="running slots; rest queue")
    p.add_argument("--batch-slowdown", type=float, default=0.0,
                   help="relative ITL increase per extra running request")  # fmt: skip
    a = p.parse_args()
    cfg = MockConfig(a.ttft, a.itl, a.max_num_seqs, a.batch_slowdown)
    server = MockServer(a.port, cfg)
    print(f"mock vLLM on http://127.0.0.1:{a.port} {cfg}")
    server.serve_forever()


if __name__ == "__main__":
    main()
