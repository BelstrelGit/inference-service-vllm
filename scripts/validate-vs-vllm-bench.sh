#!/usr/bin/env bash
# Validate client.load against `vllm bench serve` (docs/EXPERIMENT_PROTOCOL.md, section 8).
#
#   scripts/validate-vs-vllm-bench.sh
#
# Both tools run inside the vLLM Pod against http://localhost:8000, one after the other,
# with the same workload (configs/scenarios/validation_vllm_bench.yaml):
# closed loop c=8, 200 requests + 10 warm-ups, 512 -> 128 tokens, ignore_eos, temperature 0.
# Same path for both (no Service hop), so differences come from the tools, not the network.
# Caveat: both clients share the Pod's CPUs with the vLLM API server.
set -euo pipefail

NS="${NS:-vllm}"
SCENARIO=configs/scenarios/validation_vllm_bench.yaml
SERIES=l4-qwen7b-bf16-v1-validation
POD="$(kubectl -n "$NS" get pod -l app=vllm -o jsonpath='{.items[0].metadata.name}')"
MODEL="$(kubectl -n "$NS" exec "$POD" -- python3 -c 'import json, urllib.request as u; print(json.load(u.urlopen("http://localhost:8000/v1/models"))["data"][0]["id"])')"
LAB=/tmp/lab

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
kx() { kubectl -n "$NS" exec "$POD" -- "$@"; }

dirty=0
[ -n "$(git status --porcelain -- client configs)" ] && dirty=1

log "pod $POD, model $MODEL"
kx rm -rf "$LAB"
kx mkdir -p "$LAB"
COPYFILE_DISABLE=1 tar --no-xattrs -cf - client/*.py configs/scenarios | kubectl -n "$NS" exec -i "$POD" -- tar -C "$LAB" -xf -
kx python3 -c 'import httpx, yaml' || { echo "vLLM image lacks httpx/pyyaml" >&2; exit 1; }

log "1/2 client.load"
kx env CLIENT_GIT_COMMIT="$(git rev-parse HEAD)" CLIENT_GIT_DIRTY="$dirty" \
  sh -c "cd $LAB && python3 -m client.load $SCENARIO --base-url http://localhost:8000 --out-dir $LAB/out"

log "2/2 vllm bench serve"
kx vllm bench serve \
  --backend openai --endpoint /v1/completions \
  --base-url http://localhost:8000 --model "$MODEL" \
  --dataset-name random --random-input-len 512 --random-output-len 128 --random-range-ratio 0 \
  --num-prompts 200 --max-concurrency 8 --request-rate inf --num-warmups 10 \
  --ignore-eos --temperature 0 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99 \
  --save-result --result-dir "$LAB/vb" --result-filename vb.json

meta="$(kx sh -c "ls $LAB/out/$SERIES/*.meta.json" | tail -n 1)"
run_id="$(basename "$meta" .meta.json)"
mkdir -p "results/raw/$SERIES"
kx tar -C "$LAB/out/$SERIES" -cf - . | tar -C "results/raw/$SERIES" -xf -
kx cat "$LAB/vb/vb.json" > "results/raw/$SERIES/$run_id.vllm-bench.json"
log "saved results/raw/$SERIES/$run_id.{jsonl,meta.json,server.jsonl,vllm-bench.json}"

uv run python -m analysis.compare_vllm_bench "results/raw/$SERIES/$run_id.meta.json" \
  "results/raw/$SERIES/$run_id.vllm-bench.json"
