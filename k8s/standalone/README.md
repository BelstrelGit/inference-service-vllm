# Standalone vLLM (M1)

One bare Pod with vLLM on one L4. Goal: first verified response, effective config and startup timeline captured.
No Deployment, Service, PVC or monitoring yet.

| | |
|---|---|
| Image | `vllm/vllm-openai:v0.30.0` pinned by digest, CUDA 13.0.2 (host driver 580.173.02 supports CUDA 13.0) |
| Model | `Qwen/Qwen2.5-7B-Instruct` @ `a09a35458c702b33eeacc393d103063234e8bc28` (model + tokenizer) |
| Explicit args | `--dtype=bfloat16 --max-model-len=8192 --gpu-memory-utilization=0.90 --seed=0` |
| Other engine params | v0.30.0 defaults, captured from the startup log; set explicitly before Exp0 |
| Model cache | `emptyDir` — lost when the Pod is deleted (cached vs uncached startup is a later experiment) |

KV budget: estimated 4–5 GiB (~75–90k tokens) before the first run; **actual 2.87 GiB → 53,696 tokens** (device 22.03 GiB × 0.90 = 19.83 − weights + non-torch 15.53 − peak activation 1.43). The estimate underrated non-weight overhead (~2.7 GiB, not ~1).

## Run

Prerequisite: cluster up (`k8s/cluster/README.md`), `default-pool` at 1 node, no other workload holding the project's single GPU.

```bash
# 1. Create namespace and Pod; the autoscaler starts the L4 node
kubectl apply -f k8s/standalone/namespace.yaml -f k8s/standalone/vllm-pod.yaml

# 2. Follow startup: node scale-up → image pull → model download → load → Ready
kubectl -n vllm get pod vllm -w                          # Ctrl-C once READY is 1/1
kubectl -n vllm get events --sort-by=.lastTimestamp      # scheduling / pull events
kubectl -n vllm logs -f vllm                             # vLLM startup log

# 3. Smoke test through port-forward (diagnostic, not a benchmark)
scripts/smoke.sh

# 4. Capture effective environment + startup timeline (within ~1 h: events expire)
scripts/collect-env.sh

# 5. Delete the Pod; the L4 node scales down ~10 min later
kubectl delete -f k8s/standalone/vllm-pod.yaml
```

Outputs go to `results/raw/smoke/<ts>/` and `results/raw/env/<ts>/`; values from `env.txt` are copied into `docs/ENVIRONMENT.md`.

## Startup timeline to record

```text
T0  TriggeredScaleUp        (event, Pod)
T1  Scheduled               (event / PodScheduled condition)
T2  Pulling image           (event)
T3  Pulled / Started        (events; container startedAt)
T4  model download done     (vLLM log)
T5  weights loaded          (vLLM log)
T6  Ready (/health = 200)   (Ready condition)
```

Single observation, not an experiment: the cold-start decomposition is measured properly in M4.

## Troubleshooting

| Symptom | Where to look | Likely cause |
|---|---|---|
| Pending, `FailedScaleUp … out of resources` | `kubectl -n vllm describe pod vllm` | no L4 capacity in the region right now |
| Pending, `Insufficient memory/cpu` | same | requests too large for `g2-standard-8` allocatable → lower `cpu`/`memory` requests |
| Restarted / killed during startup | `kubectl -n vllm describe pod vllm` | startupProbe budget (20 min) exceeded |
| CUDA / driver error in log | `kubectl -n vllm logs vllm` | image vs driver mismatch |
| `OOMKilled` | `describe pod` | host RAM; GPU OOM shows up in the log instead |
