# Experiment Protocol

Правила, по которым проводится и документируется каждый benchmark run.
Цель — чтобы любой результат можно было воспроизвести, сравнить и честно интерпретировать.

---

## 1. Серии и главное правило

> **Не смешивать результаты разных GPU / configurations в одну серию.**

**Серия (series)** — набор runs, в которых зафиксирована вся конфигурация (раздел 2), а меняется только одна independent variable эксперимента.

Новая серия начинается при **любом** изменении:

- GPU model / count;
- driver / CUDA;
- vLLM version или container image digest;
- model или model revision;
- tokenizer revision;
- precision / quantization;
- любого server argument (`max_model_len`, `gpu_memory_utilization`, `max_num_seqs`, …);
- environment (Colab ↔ GKE ↔ VM).

Сравнивать между сериями можно, но только явно, с указанием, что именно изменилось.

`series_id` — короткое осмысленное имя: `l4-qwen3b-bf16-v1`, `l4-qwen3b-bf16-kvlimit-v1`.

---

## 2. Fixed configuration record

Снимается **для каждого run** и сохраняется в `<run_id>.meta.json`.

| Группа | Поле |
|---|---|
| Hardware | GPU model, GPU count, VRAM |
| Drivers | NVIDIA driver version, CUDA version |
| Runtime | vLLM version, container image digest (`sha256:…`), PyTorch version, Python version |
| Model | model id, model revision (commit hash), tokenizer revision |
| Precision | dtype, quantization (если есть) |
| Server args | `max_model_len`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, `enable_prefix_caching`, chunked prefill (on/off), seed, прочие нестандартные аргументы |
| Environment | environment (A/B/C), node/machine type, zone |
| Workload | scenario name, load mode, concurrency / rate, prompt tokens, max output tokens, duration / request count, warm-up count |
| Client | git commit load generator, client timeout, retry policy |

Значения по умолчанию тоже записываются явно: default может поменяться между версиями vLLM.

---

## 3. Run procedure

Для каждого эксперимента:

1. **Question** — что именно хотим узнать.
2. **Hypothesis** — что ожидаем увидеть и почему.
3. **Environment** — серия и конфигурация (раздел 2).
4. **Controlled variables** — что зафиксировано.
5. **Independent variable** — что меняется и с какими уровнями.
6. **Metrics** — какие метрики считаются (раздел 5).
7. **Warm-up** — первые **N запросов** (по умолчанию N = 10 или 30 с, что больше) отправляются, но **отбрасываются** из анализа. Помечаются в raw как `phase = warmup`.
8. **Run** — основной прогон.
9. **Raw results** — сохранить JSONL + meta (раздел 6).
10. **Aggregation** — посчитать агрегаты.
11. **Graph / table** — построить.
12. **Findings** — что показали данные.
13. **Limitations** — что ограничивает выводы.

Отчёт пишется по `docs/experiments/TEMPLATE.md` → `docs/experiments/EXP_00X_<NAME>.md`.

---

## 4. Workload definition

### Load modes

| Mode | Параметр | Как работает | Когда использовать |
|---|---|---|---|
| **closed-loop** | concurrency N | N клиентов; каждый отправляет следующий запрос сразу после завершения предыдущего | Exp0, Exp1 — поиск saturation, базовая кривая throughput vs latency |
| **open-loop** | rate R req/s | запросы приходят по Poisson process (exponential inter-arrival) независимо от ответов сервера | рост очереди, перегрузка, поведение при arrival rate > capacity |

Closed-loop сам себя «тормозит» (coordinated omission): при насыщении клиенты просто ждут, и очередь не растёт неограниченно. Open-loop это не скрывает. Это два разных эксперимента, их результаты не смешиваются.

### Prompts

- **По умолчанию — synthetic prompts** с точным числом input tokens, измеренным **токенайзером самой модели** (а не по символам/словам).
- Для каждого запроса — уникальный prompt (например, случайный префикс-идентификатор), чтобы prefix caching не искажал результаты, если он не является предметом эксперимента.
- Dataset prompts (ShareGPT и т.п.) — только как отдельный сценарий «realistic distribution», не для controlled experiments.

### Output length

- Фиксированный `max_tokens` + `ignore_eos=true` → каждый запрос генерирует ровно заданное число токенов.
- Фактическое `output_tokens` всё равно записывается и проверяется.

### Длительность

- **duration-based** (например, 120 с после warm-up) — для closed-loop и open-loop, когда важна steady state;
- **request-count-based** (например, 200 запросов) — для baseline и малой concurrency, где duration дал бы слишком мало точек.
- Выбранный способ фиксируется в meta.

### Повторы и variance

- Каждая точка — **минимум 3 повтора**.
- Отчитываемся медианой по повторам; разброс показываем как min–max или стандартное отклонение (error bars на графиках).
- Если разброс между повторами сравним с эффектом — вывод не делается, только отмечается.

---

## 5. Metrics

### Request-level (client-side)

| Метрика | Определение |
|---|---|
| **TTFT** | `first_token_at − started_at` |
| **TPOT** | `(completed_at − first_token_at) / (output_tokens − 1)` — первый токен **исключён** |
| **ITL** | интервалы между соседними токенами в стриме (распределение, если пишем token timestamps) |
| **E2E latency** | `completed_at − started_at` |

### Run-level

| Метрика | Определение |
|---|---|
| request throughput | успешные запросы / длительность измерения, req/s |
| output token throughput | сумма `output_tokens` / длительность, tok/s |
| total token throughput | сумма `prompt_tokens + output_tokens` / длительность, tok/s |
| error rate | неуспешные / все запросы (warm-up исключён) |

Для latency-метрик: **mean, p50, p90, p95, p99**. Перцентили считаются только по успешным запросам; ошибки — отдельно.

### Server-side

Во время run периодически (например, раз в 1 с) снимается vLLM `/metrics`: running / waiting requests, KV-cache usage, preemptions, queue time, server-side TTFT / ITL histograms. GPU utilization и VRAM — `nvidia-smi` или DCGM.

> **Правило:** client-side и server-side измерения записываются **оба** и сравниваются. Расхождение (например, client TTFT ≫ server TTFT) — это тоже результат (сеть, port-forward, клиент), его нужно объяснить.

---

## 6. Data layout

```text
results/
  raw/<series_id>/<run_id>.jsonl          # request records, по одному на строку
  raw/<series_id>/<run_id>.meta.json      # config snapshot (раздел 2)
  raw/<series_id>/<run_id>.server.jsonl   # snapshots vLLM /metrics (GPU-метрики: DCGM, позже)
  aggregated/<series_id>/...              # CSV / Parquet
  figures/<series_id>/...                 # PNG / SVG
```

- `run_id` = `<YYYYMMDD-HHMMSS>_<scenario>_<level>_r<rep>`, например `20261020-143005_concurrency_c16_r2`.
- **Raw results никогда не редактируются.** Исправления — только через повторный run или в коде агрегации.
- Агрегаты и графики должны полностью пересоздаваться из raw одной командой.

---

## 7. Request record schema

| Поле | Тип | Описание |
|---|---|---|
| `request_id` | str | уникальный id запроса |
| `run_id` | str | id run |
| `scenario` | str | имя сценария |
| `phase` | str | `warmup` / `measure` |
| `endpoint` | str | `completions` / `chat` |
| `max_tokens` | int | запрошенное число output tokens |
| `started_at` | float | время отправки первой попытки (perf_counter, s); ретраи входят в latency |
| `scheduled_at` | float \| null | open-loop: когда запрос должен был уйти по расписанию; `started_at − scheduled_at` = отставание генератора |
| `first_chunk_at` | float \| null | первый SSE chunk любого вида (в chat — chunk только с `role`) |
| `first_token_at` | float \| null | первый chunk с непустым текстом — точка TTFT |
| `completed_at` | float \| null | получен `data: [DONE]` |
| `started_at_unix` | float | wall clock момента `started_at` (для привязки к server metrics) |
| `finish_reason` | str \| null | `length` / `stop` |
| `prompt_tokens` | int | input tokens (из usage ответа сервера) |
| `output_tokens` | int | output tokens (из usage ответа сервера) |
| `status` | str | `ok` / `error` / `timeout` |
| `http_status` | int \| null | HTTP status code |
| `error` | str \| null | текст ошибки |
| `retry_count` | int | число ретраев (только до первого токена: соединение не установлено — отказ или `ConnectTimeout`, обрыв до первого токена, HTTP 429/502/503/504; экспоненциальный backoff с jitter, опционально с потолком `backoff_max_s`) |
| `token_timestamps` | list[float] | время каждого chunk с текстом, для ITL-распределения |
| `attempts` | list[dict] | журнал попыток: `start` / `end` (perf_counter), `outcome` (`ok`, `http_error`, `connect_error`, `connect_timeout`, `disconnect`, `timeout`, `incomplete`, `error`), `http_status`, `mid_stream` (после первого токена), `error`, `retry_in_s` (запланированная пауза перед следующей попыткой; `null` у последней). В записях до Exp5 пуст |

Реализация: `client/records.py` (схема и формулы), `client/stream.py` (точки замера).
Все времена для вычисления интервалов — `time.monotonic()`/`perf_counter()`; wall-clock (UTC) пишется отдельно только для привязки к server metrics.

---

## 8. Validation

- Перед Exp1 один сценарий прогоняется **и своим load generator, и `vllm bench serve`** с одинаковыми параметрами (тот же prompt/output length, concurrency, число запросов).
- Сравниваются TTFT p50/p95, TPOT p50, output tok/s.
- Ожидаемое расхождение — **в пределах ~5–10%**. Больше — искать причину до начала основных экспериментов (ошибка в измерении TTFT, токенизация, overhead клиента).
- Результат валидации фиксируется в `docs/experiments/`.

---

## 9. Honesty rules

- Никаких утверждений без данных: вывод в отчёте должен ссылаться на конкретный run/series.
- Секция **Limitations обязательна** в каждом отчёте.
- Результаты из **Colab помечаются как exploratory** и не попадают в основные benchmark-серии.
- Гипотетические цены (cost metrics) явно помечаются как гипотетические.
- Неподтвердившаяся гипотеза — тоже результат, её не удаляют.
