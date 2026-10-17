#!/usr/bin/env bash
# M1 smoke test of the standalone vLLM Pod through `kubectl port-forward`.
#
# Checks: /health, /v1/models, chat completion (non-streaming and streaming), /metrics.
# Client-side timings printed here are DIAGNOSTIC ONLY: the path Mac -> port-forward ->
# Kubernetes API -> Pod is not a workload path and must not be used as benchmark data.
#
#   scripts/smoke.sh                 # starts its own port-forward on localhost:8000
#   PORT=8001 scripts/smoke.sh
#   NO_PORT_FORWARD=1 scripts/smoke.sh   # reuse an existing port-forward
set -euo pipefail

NS="${NS:-vllm}"
POD="${POD:-vllm}"
PORT="${PORT:-8000}"
BASE="http://localhost:${PORT}"
OUT="${OUT:-results/raw/smoke/$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { log "FAIL: $*"; exit 1; }

PF_PID=""
cleanup() { [ -n "$PF_PID" ] && kill "$PF_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ "${NO_PORT_FORWARD:-0}" != 1 ]; then
  kubectl -n "$NS" port-forward "pod/$POD" "${PORT}:8000" >"$OUT/port-forward.log" 2>&1 &
  PF_PID=$!
  for _ in $(seq 1 20); do
    curl -s -o /dev/null "$BASE/health" && break
    sleep 0.5
  done
fi

# 1. Health
code="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")"
[ "$code" = 200 ] || fail "/health returned $code"
log "OK  /health 200"

# 2. Models
curl -sf "$BASE/v1/models" | tee "$OUT/models.json" | jq -c '.data[] | {id, max_model_len}' \
  || fail "/v1/models"
MODEL="$(jq -r '.data[0].id' "$OUT/models.json")"
log "OK  /v1/models -> $MODEL"

REQ="$(jq -n --arg m "$MODEL" '{
  model: $m,
  messages: [{role: "user", content: "Explain in two sentences what a KV cache is in LLM inference."}],
  max_tokens: 64,
  temperature: 0
}')"

# 3. Chat completion, non-streaming
curl -sf "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" \
  -w '\n%{time_total}\n' > "$OUT/chat.raw" || fail "chat completion"
head -n 1 "$OUT/chat.raw" > "$OUT/chat.json"
jq -r '.choices[0].message.content' "$OUT/chat.json"
jq -c '.usage' "$OUT/chat.json"
log "OK  chat completion (diagnostic client time: $(tail -n 1 "$OUT/chat.raw") s)"

# 4. Chat completion, streaming (SSE)
echo "$REQ" | jq '. + {stream: true, stream_options: {include_usage: true}}' > "$OUT/stream-request.json"
curl -sfN "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d @"$OUT/stream-request.json" > "$OUT/stream.sse" || fail "streaming chat completion"
chunks="$(grep -c '^data: {' "$OUT/stream.sse" || true)"
grep -q '^data: \[DONE\]' "$OUT/stream.sse" || fail "stream did not end with [DONE]"
grep '^data: {' "$OUT/stream.sse" | tail -n 1 | sed 's/^data: //' | jq -c '.usage'
log "OK  streaming: $chunks data chunks, terminated with [DONE]"

# 5. Metrics: save the full exposition (used to verify metric names for this vLLM version)
curl -sf "$BASE/metrics" > "$OUT/metrics.txt" || fail "/metrics"
grep -oE '^vllm:[a-zA-Z0-9_:]+' "$OUT/metrics.txt" | sed -E 's/_(bucket|sum|count|total|created)$//' \
  | sort -u > "$OUT/metric-names.txt"
log "OK  /metrics: $(wc -l < "$OUT/metric-names.txt" | tr -d ' ') vllm:* metric families"
grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total|prompt_tokens_total|generation_tokens_total|time_to_first_token_seconds_count)[{ ]' \
  "$OUT/metrics.txt" || true

log "saved to $OUT"
