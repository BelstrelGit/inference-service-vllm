#!/usr/bin/env bash
# Show the KV-related config of the running vLLM Deployment before a run: extra args from the
# Exp4 overlays (k8s/exp4/), pool size and dtype from /metrics (cache_config_info), and the
# KV / attention-backend lines of the startup log.
#
#   scripts/check-server.sh            # print only
#   scripts/check-server.sh 53696      # also fail unless kv_cache_size_tokens == 53696
set -euo pipefail

NS="${NS:-vllm}"
want="${1:-}"

echo "KV args on the Deployment:"
kubectl -n "$NS" get deploy vllm -o jsonpath='{.spec.template.spec.containers[0].args}' \
  | tr ',' '\n' | grep -E 'num-gpu-blocks-override|kv-cache-dtype' | sed 's/^/  /' \
  || echo "  none (baseline k8s/serving/vllm.yaml)"

info="$(kubectl -n "$NS" exec deploy/vllm -- python3 -c \
  "import urllib.request as u; print(u.urlopen('http://localhost:8000/metrics').read().decode())" \
  | grep '^vllm:cache_config_info' || true)"
echo "cache_config_info:"
tr ',{}' '\n\n\n' <<<"$info" | grep -E 'kv_cache_size_tokens|cache_dtype|num_gpu_blocks' \
  | sed 's/^/  /' || echo "  (not found)"

echo "startup log:"
kubectl -n "$NS" logs deploy/vllm \
  | grep -i -E 'num_gpu_blocks|kv cache size|attention backend' | tail -5 | sed 's/^/  /' || true

if [ -n "$want" ]; then
  got="$(grep -oE 'kv_cache_size_tokens="[0-9]+"' <<<"$info" | grep -oE '[0-9]+' || true)"
  if [ "$got" != "$want" ]; then
    echo "FAIL: kv_cache_size_tokens=${got:-?}, expected $want" >&2
    exit 1
  fi
  echo "OK: kv_cache_size_tokens=$got"
fi
