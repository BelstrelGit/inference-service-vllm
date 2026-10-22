#!/usr/bin/env bash
# Exp5 (docs/experiments/EXP_006_RESTART_UNDER_LOAD.md): inject a fault into the vLLM Pod
# during a load run and capture what Kubernetes and vLLM did, on one UTC time axis with the
# client records (analysis/restart.py reads the files).
#
#   scripts/fault.sh delete  --save results/raw/<series> [--after 150]  # R1: delete the Pod
#   scripts/fault.sh kill    --save results/raw/<series> [--after 150]  # R2: SIGTERM to PID 1
#   scripts/fault.sh observe --save results/raw/<series>                # R0: no fault, record
#                                                                       #     a cold start
#
# delete / kill: wait until the bench client has started its run (a run .jsonl exists in the
# bench-client Pod), then --after seconds (warm-up + pre-fault baseline), then inject the
# fault and poll the Pods every 2 s until the vLLM Pod is Ready again (--timeout, default
# 1800 s). observe: start right after `kubectl apply -f k8s/serving/vllm.yaml` and poll until
# Ready. Files: <save>/<UTC stamp>_fault_<mode>.{json,poll.txt,pod_before.json,pod_after.json,
# deploy.json,events.json,node.json,log_before.txt,log_after.txt}; vLLM logs keep the kubelet
# timestamps (--timestamps).
#
# Clocks: the fault time is taken on this machine, Kubernetes times on the control plane,
# client records on the cluster node; all NTP-synced, ~1 s is enough on a minutes scale.
set -euo pipefail

NS="${NS:-vllm}"
SEL=app=vllm
CLIENT=bench-client
mode="${1:-}"
case "$mode" in
  delete | kill | observe) shift ;;
  *) echo "usage: $0 delete|kill|observe --save DIR [--after S] [--timeout S]" >&2; exit 2 ;;
esac
after=150
timeout=1800
save=""
while [ $# -gt 0 ]; do
  case "$1" in
    --save) save="${2:?--save needs a directory}"; shift 2 ;;
    --after) after="${2:?}"; shift 2 ;;
    --timeout) timeout="${2:?}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$save" ] || { echo "--save DIR is required" >&2; exit 2; }
[ "$mode" = observe ] && after=""  # no fault, nothing to wait for

now() { perl -MTime::HiRes=time -e 'printf "%.3f\n", time'; }  # epoch s, ms (macOS + Linux)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
kc() { kubectl -n "$NS" "$@"; }

mkdir -p "$save"
prefix="$save/$(date -u +%Y%m%d-%H%M%S)_fault_${mode}"
name="$(basename "$prefix")"

# one line per Pod: name|deletionTimestamp|phase|ready|restartCount
pods() {
  kc get pod -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name}|{.metadata.deletionTimestamp}|{.status.phase}|{.status.containerStatuses[0].ready}|{.status.containerStatuses[0].restartCount}{"\n"}{end}'
}

pod_before=""
restarts_before=0
pid1=""
t_client=""
t_fault=""
follow_pid=""
if [ "$mode" != observe ]; then
  pod_before="$(pods | awk -F'|' '$2 == "" && $4 == "true" {print $1}')"
  [ "$(echo "$pod_before" | grep -c .)" = 1 ] \
    || { echo "expected exactly one Ready vLLM Pod, got: '$pod_before'" >&2; exit 1; }
  restarts_before="$(pods | awk -F'|' -v p="$pod_before" '$1 == p {print $5}')"
  kc get pod "$pod_before" -o json >"$prefix.pod_before.json"
  if [ "$mode" = kill ]; then
    # R2 is only meaningful if PID 1 is the vLLM server itself (then SIGTERM stops it and
    # kubelet restarts the container in the same Pod, emptyDir kept)
    pid1="$(kc exec "$pod_before" -c vllm -- cat /proc/1/cmdline | tr '\0' ' ' | tr -d '"\\')"
    log "PID 1 in $pod_before: $pid1"
    case "$pid1" in
      *vllm*) ;;
      *) echo "PID 1 is not vllm, R2 would not restart the server: $pid1" >&2; exit 1 ;;
    esac
  fi
  log "waiting for the bench client run to start"
  until kc exec "$CLIENT" -- sh -c 'ls /out/*/*.jsonl 2>/dev/null | grep -qv "\.server\.jsonl$"' \
    2>/dev/null; do
    sleep 2
  done
  t_client="$(now)"
  log "client run started; fault in ${after} s"
  sleep "$after"
  # follow the old process's log to its end: after a delete it is gone with the Pod
  kc logs -f "$pod_before" -c vllm --timestamps >"$prefix.log_before.txt" 2>&1 &
  follow_pid=$!
  sleep 1
  t_fault="$(now)"
  if [ "$mode" = delete ]; then
    kc delete pod "$pod_before" --wait=false
  else
    kc exec "$pod_before" -c vllm -- sh -c 'kill -TERM 1'
  fi
  log "fault injected: $mode $pod_before at $t_fault"
fi

# poll until the vLLM Pod is serving again
t_start="$(now)"
t_old_gone=""
pod_after=""
timed_out=false
: >"$prefix.poll.txt"
while :; do
  t="$(now)"
  state="$(pods || true)"
  echo "$state" | sed "s/^/$t|/" >>"$prefix.poll.txt"
  if [ "$mode" = delete ] && [ -z "$t_old_gone" ] && ! echo "$state" | grep -q "^$pod_before|"; then
    t_old_gone="$t"
    log "old Pod $pod_before gone"
  fi
  case "$mode" in
    delete) pod_after="$(echo "$state" | awk -F'|' -v p="$pod_before" '$1 != p && $2 == "" && $4 == "true" {print $1}')" ;;
    kill) pod_after="$(echo "$state" | awk -F'|' -v p="$pod_before" -v r="$restarts_before" '$1 == p && $4 == "true" && $5 > r {print $1}')" ;;
    observe) pod_after="$(echo "$state" | awk -F'|' '$2 == "" && $4 == "true" {print $1}')" ;;
  esac
  [ -n "$pod_after" ] && break
  if [ "$(perl -e "print $t - $t_start > $timeout ? 1 : 0")" = 1 ]; then
    timed_out=true
    case "$mode" in  # capture the Pod that should have become Ready
      kill) pod_after="$pod_before" ;;
      *) pod_after="$(echo "$state" | awk -F'|' -v p="$pod_before" '$1 != p {print $1; exit}')" ;;
    esac
    log "timeout after ${timeout} s; capturing anyway"
    break
  fi
  sleep 2
done
t_ready="$(now)"
log "vLLM Pod ready: $pod_after"

[ -n "$follow_pid" ] && { sleep 2; kill "$follow_pid" 2>/dev/null || true; }
if [ -n "$pod_after" ]; then
  kc get pod "$pod_after" -o json >"$prefix.pod_after.json"
  kc logs "$pod_after" -c vllm --timestamps >"$prefix.log_after.txt" 2>&1 || true
  node="$(kc get pod "$pod_after" -o jsonpath='{.spec.nodeName}')"
  kubectl get node "$node" -o json >"$prefix.node.json" 2>/dev/null || true
fi
kc get deploy vllm -o json >"$prefix.deploy.json"
kc get events -o json >"$prefix.events.json"  # events expire after ~1 h: capture now

j() { [ -n "$1" ] && echo "$1" || echo null; }   # number or null
s() { [ -n "$1" ] && echo "\"$1\"" || echo null; }  # string or null
f() { [ -f "$prefix.$1" ] && echo "\"$name.$1\"" || echo null; }
cat >"$prefix.json" <<EOF
{
  "mode": "$mode",
  "namespace": "$NS",
  "pod_before": $(s "$pod_before"),
  "pod_after": $(s "$pod_after"),
  "restarts_before": $restarts_before,
  "pid1_cmdline": $(s "$pid1"),
  "after_s": $(j "$after"),
  "t_client_run_seen_unix": $(j "$t_client"),
  "t_fault_unix": $(j "$t_fault"),
  "t_old_pod_gone_unix": $(j "$t_old_gone"),
  "t_ready_seen_unix": $(j "$t_ready"),
  "timed_out": $timed_out,
  "files": {
    "poll": $(f poll.txt),
    "pod_before": $(f pod_before.json),
    "pod_after": $(f pod_after.json),
    "deploy": $(f deploy.json),
    "events": $(f events.json),
    "node": $(f node.json),
    "log_before": $(f log_before.txt),
    "log_after": $(f log_after.txt)
  }
}
EOF
log "saved $prefix.json (+ poll, pods, events, logs)"
