# vLLM Inference Service — High-Level Execution Plan

Ориентир: ~10 недель, 10+ часов в неделю.

Цель проекта — построить воспроизводимый GPU-backed inference service на базе vLLM, исследовать его поведение под различными типами нагрузки и добавить необходимую эксплуатационную инфраструктуру: Kubernetes, observability, reproducibility и failure testing.

## Текущее состояние и core path (2026-10-21)

Исходный план ниже сохранён как был, с пометками. Фактический core path короче:

```text
core experiments (Exp0–Exp3)      done
        ↓
KV intervention (Exp4A, bf16)     done
        ↓
monitoring (Prometheus, Grafana, DCGM)   done
        ↓
restart / recovery (Exp5)         дизайн и инструменты готовы, GPU-сессия впереди
        ↓
final README / architecture / findings
        ↓
project freeze
```

- Новые performance-эксперименты после Exp4 заморожены.
- После Exp5 core project считается практически завершённым: остаётся финализация (M6 в сокращённом виде).
- M5 (design notes) и расширения из M4 и M6 — Helm, CI сверх unit tests, autoscaling, сравнение runtimes, вторая реплика и т. п. — не обязательны и перенесены в [Future work](#future-work). Они добавляются, только если отвечают на отдельный технический вопрос.

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

- собрать Docker image (используется upstream `vllm/vllm-openai`, закреплённый по digest; свой образ — Future work);
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
- ~~после стабилизации manifests собрать минимальный Helm chart~~ → Future work;
- добавить Prometheus;
- добавить Grafana;
- добавить DCGM Exporter, если environment позволяет.

### Cold-start experiment

Входит в Exp5 компактно: фазы старта R0 / R1 / R2 ([EXP_006](experiments/EXP_006_RESTART_UNDER_LOAD.md)).

Разложить startup на фазы:

- pod scheduling;
- container start;
- model artifact availability/download;
- model loading;
- GPU initialization;
- readiness.

Сравнить warm и cold restart.

### Failure / overload experiment

Exp5: рестарт пода под нагрузкой ([EXP_006](experiments/EXP_006_RESTART_UNDER_LOAD.md)). Cost metrics — Future work.

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

**Статус:** не входит в core, перенесено в [Future work](#future-work).

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

**Core:** README, findings, architecture diagram, limitations, проверка воспроизводимости. Дополнительные эксперименты из списка ниже — в [Future work](#future-work).

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

Исходная последовательность; фактический core path — в начале документа.

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

## Future work

Сознательно не входит в core. Каждый пункт — отдельный вопрос, а не обязательный шаг.

**Performance (заморожено после Exp4):**
- Exp4B / 4C, FP8 KV cache ([EXP_005](experiments/EXP_005_KV_PRESSURE.md)): переносится ли модель приёма (K − P) / F на пул K ≈ 2× базового (в bf16 больше базового пула не получить); как FP8 KV вместе со сменой attention backend на L4 (FlashAttention 2 не поддерживает FP8 KV на sm89) влияет на TPOT; как FP8 KV влияет на качество ответов (нужен клиентский режим с реальным текстом).
- Повтор проверки модели приёма при P=4096 на урезанных пулах (запасной тест из EXP_005; при P=2048 модели уже различились).
- Prefix caching: эффект общих префиксов (`shared_prefix_len` в клиенте уже есть).
- Отказ при старте, если пул меньше `max-model-len` (`num-gpu-blocks-override=500`), и другие memory-related failures.
- Дополнительные sweeps планировщика (`max-num-batched-tokens`, ступенька TTFT между c=1 и c=2, TPOT при c=96 против c=128).

**Масштабирование и надёжность:**
- Вторая реплика, multi-GPU, tensor parallelism: квота проекта — одна GPU.
- Рестарт без полной недоступности: ≥ 2 реплики, PodDisruptionBudget, `RollingUpdate`, `preStop`; кэш модели вне `emptyDir` (PVC, образ с весами), чтобы сократить cold start.
- Autoscaling: эксперименты с сигналами (waiting, running, TTFT, GPU), minimum warm capacity.

**Инфраструктура и design notes (бывший M5):**
- Helm chart, CI сверх unit tests (сборка образа, `helm template`, валидация манифестов), свой Docker image.
- Design notes: routing и admission control, model lifecycle (rollback, blue/green, canary), сравнение vLLM / Triton / TGI.
- Нормализованные cost metrics (GPU-hours, cost per request, cost per 1M tokens, idle capacity).
