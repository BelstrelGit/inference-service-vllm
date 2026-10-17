# vLLM Inference Service — High-Level Execution Plan

Ориентир: ~10 недель, 10+ часов в неделю.

Цель проекта — построить воспроизводимый GPU-backed inference service на базе vLLM, исследовать его поведение под различными типами нагрузки и добавить необходимую эксплуатационную инфраструктуру: Kubernetes, observability, reproducibility и failure testing.

## M0 — Foundation

**Неделя:** 1

- создать репозиторий и Python project (`uv`, `pyproject.toml`);
- подготовить README v0;
- описать базовые inference concepts:
  - prefill;
  - decode;
  - TTFT;
  - TPOT;
  - continuous batching;
  - KV cache;
- исследовать доступный GPU environment:
  - GPU model;
  - VRAM;
  - driver;
  - CUDA;
  - GPU Operator;
  - persistent storage;
  - Hugging Face access;
  - model cache;
- выбрать модель в зависимости от доступной VRAM;
- зафиксировать experiment protocol.

**Результат:**

- `docs/INFERENCE_CONCEPTS.md`
- `docs/ENVIRONMENT.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- выбранная модель и зафиксированная configuration baseline.

---

## M1 — First Measured Inference

**Недели:** 1–2

- запустить минимальный vLLM server на одной GPU;
- загрузить модель;
- проверить OpenAI-compatible API;
- написать Python client;
- добавить streaming response;
- измерять:
  - TTFT;
  - TPOT;
  - total latency;
  - prompt/output token count;
- сохранять результаты запросов в JSONL;
- после standalone baseline добавить минимальный Kubernetes Deployment и Service;
- для локальной разработки использовать mock server.

**Результат:**

первый воспроизводимый измеренный inference request.

---

## M2 — Load Generator and Analysis

**Недели:** 2–3

Разработать собственный асинхронный load generator.

Поддержать:

- `asyncio`;
- configurable concurrency;
- заданный request rate;
- request timeout;
- retry;
- exponential backoff;
- YAML scenarios;
- raw JSONL output.

Собирать request-level metrics:

- TTFT;
- TPOT;
- end-to-end latency;
- tokens/sec;
- errors;
- retries.

Собирать server metrics из vLLM:

- running requests;
- waiting requests;
- queue time;
- KV-cache utilization;
- tokens/sec.

Добавить:

- aggregation;
- p50 / p90 / p95 / p99;
- plots;
- unit tests;
- comparison with `vllm bench serve`.

**Результат:**

воспроизводимый инструмент для controlled load experiments.

---

## M3 — Core Experiments

**Недели:** 3–5

### Exp 0 — Baseline

Зафиксированная модель, GPU, prompt length, output length и concurrency=1.

Измерить run-to-run variance.

### Exp 1 — Concurrency Scaling

Изменять concurrency и исследовать переход:

`underutilized → efficient → saturated → queue growth → tail latency degradation`

Основные метрики:

- throughput;
- TTFT p50/p95;
- latency p50/p95;
- queue time;
- GPU utilization;
- KV-cache utilization.

### Exp 2 — Context Length

Изменять prompt/context length.

Исследовать влияние на:

- prefill;
- TTFT;
- throughput;
- VRAM;
- KV cache.

### Exp 3 — Output Length

Фиксировать prompt, менять количество генерируемых токенов.

Исследовать:

- TPOT;
- total latency;
- tokens/sec;
- decode-heavy workload behavior.

### Exp 4 — KV-cache Pressure

Создать контролируемое давление на KV cache через workload и vLLM scheduler/memory configuration.

Исследовать:

- KV-cache utilization;
- waiting requests;
- queue growth;
- TTFT;
- throughput;
- preemption/recomputation;
- memory-related failures.

Для каждого эксперимента создавать отдельный report:

`docs/experiments/EXP_00X_*.md`

со структурой:

- Question
- Hypothesis
- Environment
- Setup
- Controlled variables
- Results
- Graphs
- Interpretation
- Limitations
- Next question

**Результат:**

характеристика single-GPU vLLM serving behavior и saturation point.

---

## M4 — Kubernetes and Operations

**Недели:** 5–7

После стабильного standalone setup:

- собрать Docker image;
- pin dependencies;
- pin model revision;
- pin tokenizer revision;
- pin container digest;
- создать raw Kubernetes manifests:
  - Deployment;
  - Service;
  - GPU request;
  - resources;
  - probes;
  - metrics endpoint;
- после стабилизации manifests собрать минимальный Helm chart;
- добавить Prometheus;
- добавить Grafana;
- добавить DCGM Exporter, если environment позволяет.

### Cold-start experiment

Разложить startup на фазы:

- pod scheduling;
- container start;
- model artifact availability/download;
- model loading;
- GPU initialization;
- readiness.

Сравнить warm и cold restart.

### Failure / overload experiment

Во время load проверить:

- pod restart;
- request failures;
- retries;
- recovery time;
- timeout behavior;
- retry amplification;
- queue growth.

Добавить нормализованные cost metrics:

- GPU-hours;
- cost/request;
- cost/1M input tokens;
- cost/1M output tokens;
- idle GPU capacity.

**Результат:**

воспроизводимый Kubernetes deployment с observability и описанным failure behavior.

---

## M5 — System Design Notes

**Недели:** 7–9

Подготовить технические design notes.

### Autoscaling

Исследовать потенциальные scaling signals:

- queue length;
- waiting requests;
- active sequences;
- TTFT;
- tokens/sec;
- GPU utilization.

Описать влияние:

- cold start;
- minimum warm capacity;
- GPU scarcity;
- model loading time.

### Routing and Admission Control

Описать:

- routing между replicas;
- bounded queues;
- backpressure;
- overload protection;
- request fairness.

### Model Lifecycle

Описать:

- model revision;
- tokenizer revision;
- server config;
- container digest;
- rollback target;
- blue/green deployment;
- canary deployment.

### Serving Runtime Comparison

Сделать архитектурное сравнение:

- vLLM;
- Triton Inference Server;
- TGI.

Без необходимости разворачивать все runtimes.

### CI

Минимальный pipeline:

- unit tests;
- container build validation;
- `helm template`;
- manifest validation.

**Результат:**

набор технических документов, дополняющих реализованный сервис.

---

## M6 — Finalization

**Недели:** 9–10

- привести README к финальному состоянию;
- завершить `FINDINGS.md`;
- проверить воспроизводимость;
- привести experiment artifacts к единому формату;
- добавить architecture diagram;
- описать limitations;
- проверить clean setup from scratch.

После core версии решить, оправданы ли дополнительные эксперименты:

- prefix caching;
- second replica;
- horizontal scaling;
- second GPU;
- tensor parallelism.

Они добавляются только при наличии отдельного технического вопроса.

## Основная последовательность

```text
environment discovery
        ↓
first standalone vLLM inference
        ↓
Python client
        ↓
async load generator
        ↓
baseline
        ↓
concurrency / saturation
        ↓
context / output / KV-cache experiments
        ↓
containerization
        ↓
Kubernetes
        ↓
observability
        ↓
restart / overload experiments
        ↓
Helm + CI
        ↓
design notes
        ↓
final findings
```

## Stop Condition

Core project завершён, когда:

- vLLM serving воспроизводим;
- load generator работает;
- environment полностью зафиксирован;
- baseline и минимум три содержательных эксперимента завершены;
- saturation point исследован;
- KV-cache behavior исследован;
- request-level и GPU-level metrics собираются;
- Kubernetes deployment работает;
- Prometheus/Grafana работают;
- проведён хотя бы один failure/restart experiment;
- результаты и ограничения задокументированы;
- проект можно воспроизвести из README.

После этого новые компоненты добавляются только тогда, когда они отвечают на новый технический вопрос.
