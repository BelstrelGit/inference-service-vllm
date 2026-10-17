#!/usr/bin/env bash
# Capture the effective environment of the running vLLM Pod for docs/ENVIRONMENT.md:
# versions, image digest, node, GPU, startup timeline (events + Pod conditions) and the
# startup log lines that describe memory, KV cache and engine config.
#
# Run soon after the Pod is Ready: Kubernetes events are kept for ~1 h only.
#
#   scripts/collect-env.sh
set -euo pipefail

NS="${NS:-vllm}"
POD="${POD:-vllm}"
OUT="${OUT:-results/raw/env/$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

section() { printf '\n## %s\n' "$*"; }
kx() { kubectl -n "$NS" exec "$POD" -- "$@"; }

{
  section "Pod / node"
  kubectl -n "$NS" get pod "$POD" -o jsonpath='image:   {.spec.containers[0].image}{"\n"}imageID: {.status.containerStatuses[0].imageID}{"\n"}node:    {.spec.nodeName}{"\n"}'
  node="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.spec.nodeName}')"
  kubectl get node "$node" -L topology.kubernetes.io/zone,node.kubernetes.io/instance-type,cloud.google.com/gke-accelerator --no-headers

  section "Versions (inside container)"
  kx python3 -c 'import platform, torch, vllm; print("python", platform.python_version()); print("torch", torch.__version__, "cuda", torch.version.cuda); print("vllm", vllm.__version__)'
  kx nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv

  section "Explicit server args"
  kubectl -n "$NS" get pod "$POD" -o jsonpath='{range .spec.containers[0].args[*]}{@}{"\n"}{end}'

  section "Startup timeline (events)"
  kubectl -n "$NS" get events --field-selector "involvedObject.name=$POD" --sort-by=.lastTimestamp \
    -o custom-columns=TIME:.lastTimestamp,REASON:.reason,MESSAGE:.message --no-headers | cut -c1-200

  section "Startup timeline (Pod conditions)"
  kubectl -n "$NS" get pod "$POD" -o jsonpath='{range .status.conditions[*]}{.lastTransitionTime}{"  "}{.type}={.status}{"\n"}{end}' | sort
  kubectl -n "$NS" get pod "$POD" -o jsonpath='container started: {.status.containerStatuses[0].state.running.startedAt}{"\n"}'

  section "Startup log: memory / KV cache / engine config"
  kubectl -n "$NS" logs "$POD" > "$OUT/vllm.log"
  grep -iE 'non-default args|engine config|initializing a v1|model weights|loading weights took|model loading took|kv cache|maximum concurrency|available kv|block_size|chunked prefill|prefix caching|max_num_seqs|max_num_batched_tokens|cuda graph|torch\.compile|compil|memory profiling|init engine|starting vllm api server|application startup complete' \
    "$OUT/vllm.log" | cut -c1-400 || true
} 2>&1 | tee "$OUT/env.txt"

echo
echo "saved to $OUT (full log: $OUT/vllm.log)"
