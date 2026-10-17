#!/usr/bin/env bash
# Run the load generator inside the cluster (k8s/bench/client-pod.yaml) and copy the results
# to results/raw/.
#
#   scripts/run-bench.sh load configs/scenarios/exp1_concurrency.yaml [--set key=value ...]
#   scripts/run-bench.sh run --series l4-qwen7b-bf16-m1 -n 20 --warmup 3 --max-tokens 256
#   scripts/run-bench.sh --collect     # copy results of a still-running/finished pod, delete it
#
# First argument: client module (`load` = client.load, `run` = client.run); the rest is passed
# through. --base-url defaults to the vLLM Service. The local client/ code and
# configs/scenarios/ are shipped as ConfigMaps, so uncommitted changes are what runs:
# the meta records the commit and a dirty flag.
set -euo pipefail

NS="${NS:-vllm}"
POD=bench-client
BASE_URL="${BASE_URL:-http://vllm.${NS}.svc.cluster.local:8000}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

collect() {
  until kubectl -n "$NS" exec "$POD" -- test -f /out/.exit_code 2>/dev/null; do sleep 5; done
  local rc
  rc="$(kubectl -n "$NS" exec "$POD" -- cat /out/.exit_code)"
  mkdir -p results/raw
  # Pack inside the Pod, then stream the archive with retries and verify its checksum before
  # unpacking: a dropped `kubectl exec` stream must not leave truncated files or lose the
  # results (the Pod is deleted only after a verified copy).
  kubectl -n "$NS" exec "$POD" -- tar -C /out --exclude=.exit_code -czf /tmp/results.tgz .
  local want got tmp attempt
  want="$(kubectl -n "$NS" exec "$POD" -- sha256sum /tmp/results.tgz | cut -d' ' -f1)"
  tmp="$(mktemp)"
  for attempt in 1 2 3 4 5; do
    kubectl -n "$NS" exec "$POD" -- cat /tmp/results.tgz >"$tmp" || true
    got="$(shasum -a 256 "$tmp" | cut -d' ' -f1)"
    [ "$got" = "$want" ] && break
    log "copy attempt $attempt failed (checksum mismatch), retrying"
    sleep 5
  done
  if [ "$got" != "$want" ]; then
    log "ERROR: could not copy results; Pod $POD kept, retry with: $0 --collect"
    rm -f "$tmp"
    return 1
  fi
  tar -C results/raw -xzf "$tmp"
  rm -f "$tmp"
  log "copied to results/raw/ (sha256 verified):"
  kubectl -n "$NS" exec "$POD" -- sh -c 'cd /out && find . -name "*.meta.json" | sort' | sed 's|^\./|  |'
  kubectl -n "$NS" delete pod "$POD" --wait=false >/dev/null
  log "client exit code $rc; pod $POD deleted"
  return "$rc"
}

if [ "${1:-}" = "--collect" ]; then
  collect
  exit $?
fi

case "${1:-}" in
  load | run) MODULE="client.$1" ;;
  *) echo "usage: $0 load|run ARGS... | --collect" >&2; exit 2 ;;
esac
shift
ARGS="$* --base-url $BASE_URL"

dirty=0
[ -n "$(git status --porcelain -- client configs)" ] && dirty=1
[ "$dirty" = 1 ] && log "WARNING: client/ or configs/ has uncommitted changes (recorded as dirty)"

kubectl -n "$NS" delete pod "$POD" --ignore-not-found --wait=true >/dev/null
kubectl -n "$NS" create configmap bench-client-code --from-file=client/ \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl -n "$NS" create configmap bench-scenarios --from-file=configs/scenarios/ \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl -n "$NS" create configmap bench-run-config \
  --from-literal=module="$MODULE" \
  --from-literal=args="$ARGS" \
  --from-literal=git_commit="$(git rev-parse HEAD)" \
  --from-literal=git_dirty="$dirty" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null
kubectl apply -f k8s/bench/client-pod.yaml >/dev/null
log "python -m $MODULE $ARGS"

kubectl -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=300s >/dev/null
# Stream the client's output; if this terminal disconnects, the run continues in the
# cluster and `scripts/run-bench.sh --collect` fetches the results later.
kubectl -n "$NS" logs -f "$POD" &
LOGS_PID=$!
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
until kubectl -n "$NS" exec "$POD" -- test -f /out/.exit_code 2>/dev/null; do sleep 5; done
sleep 1
kill $LOGS_PID 2>/dev/null || true
wait $LOGS_PID 2>/dev/null || true # reap it, so bash does not print "Terminated"
collect
