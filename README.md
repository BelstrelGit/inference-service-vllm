# inference-service-vllm

Hands-on research lab for **LLM inference infrastructure**: how concurrency, context length, output length and KV-cache pressure affect latency, throughput and GPU utilization of a vLLM server.

> Status: **v0 — scope defined, no experiments run yet.** No numbers in this repository are claimed until the corresponding experiment report exists in `docs/experiments/`.

## Research question

How do latency (TTFT, TPOT, p95), throughput and GPU/VRAM usage change as the shape of the inference workload changes?

## Architecture (MVP)

```text
Python async load generator ──HTTP (OpenAI-compatible)──▶ vLLM server ──▶ 1 × NVIDIA GPU
          │                                                  │
          └── raw JSONL per request                          └── /metrics (Prometheus)
```

Phase 1 runs vLLM standalone (bare Pod on GKE or Colab). Phase 2 moves to a Kubernetes Deployment + Service with Prometheus/Grafana.

## Planned experiments

| ID | Question | Status |
|---|---|---|
| Exp0 | Baseline at concurrency = 1, run-to-run variance | planned |
| Exp1 | Concurrency scaling → saturation point | planned |
| Exp2 | Context length → TTFT / KV pressure | planned |
| Exp3 | Output length → prefill-heavy vs decode-heavy | planned |
| Exp4 | KV-cache pressure → preemption | planned |

## Repository layout

```text
client/        load generator, request model, metrics collection
experiments/   experiment runners
analysis/      aggregation, percentiles, plots
configs/       scenario YAML
docker/ k8s/   container and Kubernetes manifests
monitoring/    Prometheus / Grafana
results/       raw (JSONL) → aggregated → figures
docs/          concepts, protocol, environment, experiment reports
```

## Docs

- [Execution plan](docs/ROADMAP.md)
- [Inference concepts](docs/INFERENCE_CONCEPTS.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Environment](docs/ENVIRONMENT.md)

## Local setup

```bash
uv sync --all-extras   # client + analysis + dev tools
uv run pytest
```

How to run vLLM and the first benchmark will be documented in M1.
