#!/usr/bin/env bash
# Run client.run inside the cluster (k8s/bench/client-pod.yaml) and copy results to results/raw/.
#
#   scripts/run-bench.sh --series l4-qwen7b-bf16-m1 -n 20 --warmup 3 --max-tokens 256
#
# All arguments go to `python -m client.run`; --base-url defaults to the vLLM Service.
# The current local client/ code is shipped as a ConfigMap, so uncommitted changes are
# what runs: the meta records the commit and a dirty flag.
set -euo pipefail

NS="${NS:-vllm}"
POD=bench-client
BASE_URL="${BASE_URL:-http://vllm.${NS}.svc.cluster.local:8000}"
ARGS="--base-url $BASE_URL $*"

series="m1-dev"
prev=""
for a in "$@"; do
  [ "$prev" = "--series" ] && series="$a"
  prev="$a"
done

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

dirty=0
[ -n "$(git status --porcelain -- client)" ] && dirty=1
[ "$dirty" = 1 ] && log "WARNING: client/ has uncommitted changes (recorded as dirty)"

kubectl -n "$NS" delete pod "$POD" --ignore-not-found --wait=true >/dev/null
kubectl -n "$NS" create configmap bench-client-code --from-file=client/ \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl -n "$NS" create configmap bench-run-config \
  --from-literal=args="$ARGS" \
  --from-literal=git_commit="$(git rev-parse HEAD)" \
  --from-literal=git_dirty="$dirty" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl apply -f k8s/bench/client-pod.yaml >/dev/null
log "client args: $ARGS"

kubectl -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=300s >/dev/null
# Stream the client's output until it writes its exit code.
kubectl -n "$NS" logs -f "$POD" &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
until kubectl -n "$NS" exec "$POD" -- test -f /out/.exit_code 2>/dev/null; do sleep 2; done
sleep 1
kill $LOGS_PID 2>/dev/null || true
rc="$(kubectl -n "$NS" exec "$POD" -- cat /out/.exit_code)"

mkdir -p "results/raw/$series"
kubectl -n "$NS" exec "$POD" -- tar -C "/out/$series" -cf - . | tar -C "results/raw/$series" -xf -
log "copied to results/raw/$series/:"
ls -1t "results/raw/$series" | head -4

kubectl -n "$NS" delete pod "$POD" --wait=false >/dev/null
log "client exit code $rc; pod $POD deleted"
exit "$rc"
