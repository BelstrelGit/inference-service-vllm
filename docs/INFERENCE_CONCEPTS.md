# Inference Concepts

> Defaults и имена метрик vLLM меняются между версиями. Здесь описана концептуальная модель; effective config и фактические имена метрик фиксируются для выбранной версии в `ENVIRONMENT.md`.

Цель документа: минимальная ментальная модель LLM inference, достаточная, чтобы планировать эксперименты из `ROADMAP.md` и интерпретировать их результаты. Это не учебник.

---

## 1. Inference vs training

| | Training / fine-tuning | Online inference |
|---|---|---|
| Единица работы | job (часы) | request (секунды) |
| Проход | forward + backward + optimizer step | только forward |
| Память | weights + gradients + optimizer states + activations | weights + **KV cache** + небольшие activations |
| Цель | throughput / JCT / стоимость job | latency (TTFT, TPOT) **при** приемлемом throughput |
| Batch | фиксированный, заранее известный | меняется каждую итерацию, запросы приходят и уходят |
| Scheduler | кому и когда выдать GPU (job-level) | какие токены каких запросов считать на следующем шаге (iteration-level) |

Основное отличие: в inference ресурсом, который распределяется, становится в первую очередь **память под KV cache**, а не только время GPU.

---

## 2. Token, tokenizer, context window

- **Token**: единица текста для модели (часть слова). В среднем ~3–4 символа английского текста; для русского токенов на слово больше. Точное число зависит от tokenizer.
- **Tokenizer** превращает текст в token ids. Он версионируется вместе с моделью, поэтому в эксперименте фиксируются model revision **и** tokenizer revision.
- **Context window / `max_model_len`**: максимум prompt + output tokens на один request. В vLLM это ограничение сервера, а не только модели: от него зависит, сколько KV cache может понадобиться одному запросу.
- Все длины в экспериментах задаём **в токенах**, а не в символах, и проверяем по `usage` в ответе сервера.

---

## 3. Prefill vs decode

Генерация одного request идёт в две фазы:

1. **Prefill**: модель обрабатывает все токены prompt, строит для них KV cache и выдаёт первый токен. Концептуально это первая фаза request; физически vLLM может выполнить её за одну или несколько scheduler iterations (chunked prefill, §6).
   - Матричные умножения большие (`N_prompt × hidden`), поэтому prefill **обычно более compute-bound**, особенно при достаточно длинном prompt.
   - Время растёт с длиной prompt (линейно по MLP; attention квадратично, но на средних длинах доминирует линейная часть).
   - Prefill — основная вычислительная составляющая **TTFT** после ожидания в очереди (`TTFT ≈ queue + prefill + overhead`).
2. **Decode**: генерация по одному токену за шаг; каждый шаг читает **все веса модели** и KV cache всех предыдущих токенов.
   - На один токен приходится мало вычислений и много чтения памяти, поэтому decode при небольшом batch **обычно memory-bandwidth-bound**. При больших batch/context и других kernels граница может сдвигаться.
   - Decode определяет **TPOT / ITL**.

### Интуиция arithmetic intensity

Arithmetic intensity = FLOPs / bytes, прочитанные из памяти.

- Decode при batch = 1: на каждый параметр (2 bytes в bf16) ~2 FLOPs, то есть **~1 FLOP/byte**.
- У L4: ~121 TFLOPS bf16 (dense) и ~300 GB/s памяти (спецификация NVIDIA, без sparsity), поэтому порог (ridge point) ≈ 121e12 / 300e9 ≈ **~400 FLOP/byte**.
- Вывод: decode с одним запросом использует ничтожную долю compute. Идеализированная bandwidth lower bound времени шага ≈ `weights_bytes / bandwidth`; для 8B модели в bf16 это ≈ 16 GB / 300 GB/s ≈ **~53 ms/token (~19 tok/s)** на L4, без учёта KV reads, kernels и других overhead. Это нижняя граница, а не performance model: реальный TPOT будет выше.
- **Batching** decode: веса читаются один раз за шаг, а используются для B запросов, поэтому intensity растёт ~B раз. Отсюда почти бесплатный рост throughput с concurrency, пока decode не упрётся в compute или в KV cache.

---

## 4. KV cache

### Что хранится

Для каждого токена, каждого слоя, каждой KV-головы хранятся векторы **K** и **V**, чтобы на следующих шагах не пересчитывать attention по всему префиксу.

```text
KV bytes per token = 2 (K и V) × num_layers × num_kv_heads × head_dim × bytes_per_elem
```

**GQA** (grouped-query attention): `num_kv_heads < num_attention_heads`, поэтому KV cache в разы меньше, чем при обычном MHA.

### Примеры (bf16, 2 bytes)

| Модель | layers | kv_heads | head_dim | KV / token | Weights (bf16) |
|---|---|---|---|---|---|
| Llama-3.1-8B | 32 | 8 | 128 | 2·32·8·128·2 = **128 KiB** | ~16.1 GB |
| Qwen2.5-7B | 28 | 4 | 128 | 2·28·4·128·2 = **56 KiB** | ~15.2 GB |
| Qwen2.5-3B | 36 | 2 | 128 | 2·36·2·128·2 = **36 KiB** | ~6.2 GB |

*(Перед использованием сверить с `config.json` выбранной revision на HF.)*

### Сколько помещается в L4 (24 GB)

Грубый бюджет для Llama-3.1-8B, для иллюстрации при `gpu_memory_utilization=0.9` (фактический default зависит от версии vLLM):

```text
≈ 22.5 GiB usable × 0.9          ≈ 20.2 GiB  (бюджет vLLM)
− weights                         ≈ 15.0 GiB
− activations / CUDA graphs / etc ≈ 1–2 GiB   (оценка, зависит от версии и настроек)
= KV cache                        ≈ 3–4 GiB
→ 3.5 GiB / 128 KiB               ≈ ~28k tokens на ВСЕ одновременные запросы
```

То есть, например, ~7 запросов по 4k токенов (prompt + output) **одновременно**. Для Qwen2.5-7B (56 KiB/token, KV ≈ 4–5 GiB) выходит ~80k tokens. Для 3B модели KV cache помещается заметно больше, и упереться в него труднее.

Реальное число vLLM пишет в лог при старте (что-то вроде `GPU KV cache size: N tokens` / `Maximum concurrency for X tokens per request`). **Это число — первое, что записываем в `ENVIRONMENT.md`.**

Вывод для выбора модели: 7–8B на L4 удобна, чтобы изучать KV pressure (Exp4); 3B даёт широкий диапазон concurrency (Exp1).

---

## 5. PagedAttention

- Раньше KV cache выделялся **непрерывным куском под `max_model_len`** на запрос, из-за чего возникала огромная internal fragmentation (выделено 4k, использовано 300).
- **PagedAttention** (vLLM) хранит KV cache **блоками** фиксированного для данного запуска размера (`block_size` — effective config, фиксируем в `ENVIRONMENT.md`). Логические блоки запроса отображаются на физические через block table, как страницы виртуальной памяти.
- Последствия:
  - потеря памяти ≤ 1 неполный блок на запрос, поэтому в ту же VRAM помещается больше одновременных запросов;
  - блоки можно **шарить** между запросами с общим prefix (prefix caching, §10);
  - память выделяется по мере генерации, поэтому её может **не хватить посреди генерации** (preemption, §7).

---

## 6. Batching: static → dynamic → continuous

| Тип | Как работает | Проблема |
|---|---|---|
| **Static** | собрали batch, генерируем, пока **все** не закончат | короткие запросы ждут самый длинный; новые ждут весь batch |
| **Dynamic** (напр. Triton dynamic batcher) | собираем batch по таймауту / размеру на уровне **request** | всё ещё request-level: batch живёт до конца самого длинного |
| **Continuous** (iteration-level scheduling, Orca / vLLM) | решение принимается **каждый forward step**: закончившиеся запросы уходят, новые сразу добавляются | нужен scheduler + управление KV cache |

**Почему batching повышает throughput:** decode memory-bound, и веса читаются один раз на весь batch (§3).

**Почему batching может ухудшать latency:**
- каждый шаг с большим batch медленнее, поэтому растёт TPOT у всех;
- prefill нового запроса, вставленный в шаг, задерживает decode остальных (всплески ITL);
- когда KV cache или `max_num_seqs` исчерпаны, новые запросы стоят в очереди, и растёт TTFT.

### Chunked prefill

Длинный prompt режется на куски, и каждый шаг совмещает decode-токены текущих запросов + кусок prefill в пределах **token budget** (`max_num_batched_tokens`). Это сглаживает ITL ценой немного большего TTFT для длинных prompt. В vLLM V1 chunked prefill включается по умолчанию для поддерживаемых моделей, а scheduler не разделяет «prefill-фазу» и «decode-фазу», а распределяет token budget между запросами. Effective setting фиксируем для выбранной версии.

---

## 7. Queue, running/waiting, preemption

Состояния запроса в scheduler vLLM:

```text
arrive → WAITING (очередь) → RUNNING (в batch, есть KV блоки) → FINISHED
                ↑                    │
                └── PREEMPTED ←──────┘  (не хватило KV блоков)
```

- В RUNNING запрос не попадает, если: достигнут `max_num_seqs`, не хватает KV блоков или исчерпан token budget шага.
- **Preemption**: когда растущим запросам не хватает блоков, scheduler вытесняет часть RUNNING-запросов (обычно самых новых), освобождая их KV. В стандартном V1 preemption выполняется через **recompute**: request возвращается в WAITING, и его KV state позже пересчитывается. Старый SWAP mode (выгрузка KV на CPU) не является стандартным механизмом V1. Цена: потраченный впустую compute, выше latency у вытесненных.
- Признаки KV pressure: `kv_cache_usage_perc` → ~100%, растёт `num_requests_waiting`, растёт `num_preemptions`, TTFT растёт, при этом GPU не «загружен» на 100% compute.

Настройки, которые влияют (подбираются экспериментально в Exp4): `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`.

---

## 8. Метрики

### Request-level (меряем клиентом)

| Метрика | Определение | Что отражает |
|---|---|---|
| **TTFT** | от отправки до первого токена (первый SSE chunk с текстом) | queue time + prefill |
| **TPOT** | `(E2E − TTFT) / (output_tokens − 1)` | средняя скорость decode для запроса |
| **ITL** | интервалы между соседними токенами | распределение, видны всплески из-за prefill других запросов |
| **E2E latency** | от отправки до последнего токена | ≈ TTFT + TPOT × (N_out − 1) |

TPOT — среднее по запросу, ITL — распределение. Один запрос может иметь нормальный TPOT, но плохой p99 ITL.

### System-level

- **Throughput**: requests/s, **output tokens/s** (основная для LLM), total tokens/s.
- **Goodput**: requests/s, удовлетворившие заданным **per-request SLO** (например, TTFT < 1 s и TPOT < 100 ms). Для серии в целом дополнительно контролируем TTFT p95, TPOT p95, error rate. Максимальный throughput часто достигается в точке, где SLO уже нарушен.
- Error rate, timeouts, retries.

### Aggregation

mean скрывает хвосты, поэтому всегда p50 / p90 / p95 / p99. Число запросов в серии должно быть достаточным для p99 (≥ несколько сотен).

### Соответствие метрикам vLLM `/metrics`

> Имена ниже соответствуют текущей документации vLLM V1, но **меняются между версиями** (например, `gpu_cache_usage_perc` → `kv_cache_usage_perc`, `time_per_output_token_seconds` → `inter_token_latency_seconds`). После выбора версии сверяем автоматически через `curl :8000/metrics`.
>
> Counters в документации указаны базовым именем; в выводе `/metrics` и в PromQL Prometheus client добавляет суффикс `_total` (например, `vllm:num_preemptions` → `vllm:num_preemptions_total`).

| Концепция | Метрика vLLM |
|---|---|
| Running / waiting | `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| KV cache заполнение | `vllm:kv_cache_usage_perc` |
| Preemption | `vllm:num_preemptions` (counter), `vllm:request_num_preemptions` (histogram, на request) |
| TTFT | `vllm:time_to_first_token_seconds` (histogram) |
| ITL / TPOT | `vllm:inter_token_latency_seconds` (ранее `vllm:time_per_output_token_seconds`) |
| Queue time | `vllm:request_queue_time_seconds` |
| Prefill / decode time | `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` |
| E2E | `vllm:e2e_request_latency_seconds` |
| Tokens | `vllm:prompt_tokens`, `vllm:generation_tokens` (counters) |
| Prefix cache | `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits` (counters) |
| GPU util / VRAM | не vLLM: `nvidia-smi` / DCGM (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`) |

Клиентские и серверные TTFT должны быть близки. Разница = сеть + HTTP + tokenization/detokenization. Сравнить их — хорошая проверка корректности своего load generator.

---

## 9. Queueing intuition

- **Little's law**: `L = λ × W`, где L — среднее число запросов в системе, λ — arrival rate, W — среднее время в системе. Пример: 4 req/s × 5 s = ~20 одновременных запросов. Полезно для sanity check результатов.
- **Utilization → latency**: в простых моделях очереди время ожидания растёт как ~`1 / (1 − ρ)`. При ρ → 1 latency растёт не линейно, а взрывается. Отсюда «колено» на графике latency vs load.
- **Saturation point**: дальше рост нагрузки почти не увеличивает throughput, но резко увеличивает queue time, TTFT и p99. Цель Exp1 — найти это колено.

### Closed-loop vs open-loop load generation

| | Closed-loop | Open-loop |
|---|---|---|
| Модель | N клиентов; каждый отправляет следующий запрос после ответа | запросы приходят с rate R (обычно Poisson) независимо от ответов |
| Нагрузка | саморегулируется: сервер медленнее, значит меньше запросов | не ждёт сервер, поэтому очередь может расти без ограничения |
| Хорошо для | «сколько выдержим при N пользователях», Exp1 | поиск saturation, честный queueing, SLO / goodput |
| Риск | **coordinated omission**: скрывает реальную очередь | при R > capacity латентность растёт неограниченно, нужен timeout |

Exp1 начинаем с closed-loop, open-loop добавляем отдельно.

---

## 10. Prefix caching

- Если несколько запросов начинаются с одинакового prefix (system prompt, few-shot, общий документ), KV блоки этого prefix можно **переиспользовать**: блоки адресуются хешем содержимого.
- Эффект: TTFT уменьшается (prefill только для «хвоста»), экономится compute; при общих блоках экономится и KV память.
- Полезно при длинном общем prefix + коротком уникальном хвосте; бесполезно для несвязанных prompt (накладные расходы небольшие).
- В текущем vLLM V1 prefix caching (automatic prefix caching) обычно включается по умолчанию для поддерживаемых моделей; effective setting обязательно фиксируем для выбранной версии и модели. Наблюдаем через `rate(prefix_cache_hits) / rate(prefix_cache_queries)`.
- **Ловушка для бенчмарков:** одинаковые prompt в серии дают нереалистично хороший TTFT. Для Exp0–Exp4 prompts должны быть уникальными (или prefix caching явно выключен через `--no-enable-prefix-caching`), а это должно быть записано в protocol.

---

## 11. GPU utilization и VRAM: почему метрики вводят в заблуждение

- **`nvidia-smi` GPU-Util** = доля времени, когда на GPU выполнялся **хоть один kernel**. Не доля занятых SM и не доля FLOPS. Memory-bound decode с batch=1 может показывать ~90–100% «util» при ~1% compute. Более информативные метрики DCGM: `SM_ACTIVE`, `SM_OCCUPANCY`, `DRAM_ACTIVE` (если доступны, profiling metrics).
- **VRAM «полная» сразу после старта:** vLLM **заранее** резервирует значительную часть VRAM согласно `gpu_memory_utilization` (фактическое значение фиксируем для выбранной версии) под weights + KV cache pool. `nvidia-smi` покажет почти полную память ещё до первого запроса. Реальное заполнение видно только через `kv_cache_usage_perc`.
- **Compute свободен, а запросы ждут:** когда KV блоки закончились, scheduler не может принять новые запросы, хотя SM простаивают. Это и есть ответ на вопрос «почему memory, а не compute, ограничивает inference» (Exp4).
- **Низкий GPU util** при нагрузке: узкое место вне GPU (tokenization, Python overhead, сеть, слишком маленькая concurrency у клиента, сам load generator).

---

## 12. Latency vs throughput trade-off

```text
concurrency ↑
  → batch ↑ → output tokens/s ↑ (пока decode memory-bound)
  → TPOT ↑ (каждый шаг тяжелее)
  → при исчерпании KV / max_num_seqs: queue ↑ → TTFT ↑↑, p99 ↑↑
```

- Точка максимального throughput ≠ правильная production operating point, если есть latency SLO.
- Настройки, которые сдвигают trade-off: `max_num_seqs` (верхняя граница batch), `max_num_batched_tokens` (token budget шага и баланс prefill/decode), `gpu_memory_utilization` (объём KV pool), размер модели и quantization (bytes на параметр и на KV элемент).

---

## 13. Training scheduler vs inference scheduler

Сравнение с GPU scheduling для training / fine-tuning workloads (например, LoRA/QLoRA jobs):

| | Training scheduler (cluster-level) | Inference scheduler (vLLM) |
|---|---|---|
| Уровень | cluster: какие jobs на какие GPU | engine: какие токены каких запросов в следующем шаге |
| Гранулярность | минуты–часы | миллисекунды (каждая итерация) |
| Ресурс | GPU целиком / VRAM на job | KV блоки + token budget шага |
| Метрики | JCT, fairness, utilization, cost | TTFT, TPOT, goodput, queue time |
| Preemption | checkpoint/restart job | recompute KV запроса |
| Heterogeneous GPU | выбор GPU под профиль job | выбор replica / GPU pool под профиль запроса (routing, optional) |

Общее: queueing, saturation, fairness, utilization vs latency, стоимость простаивающей GPU.

---

## 14. Гипотезы для экспериментов

Это гипотезы, не результаты. Проверяются в экспериментах из `ROADMAP.md`.

- **Exp0 (baseline):** при concurrency = 1 TPOT выше theoretical bandwidth lower bound (`weights_bytes / bandwidth`, §3); измерение показывает величину реального overhead относительно этой границы (KV reads, kernels, scheduler, framework). Run-to-run variance небольшая (< ~5%?).
- **Exp1 (concurrency):** output tokens/s растёт почти линейно на малых N, затем выходит на плато. TPOT растёт плавно. TTFT p95 резко растёт **раньше**, чем throughput выходит на плато, когда появляются `num_requests_waiting > 0`.
- **Exp2 (context length):** TTFT растёт ~линейно с длиной prompt. KV footprint одного active request растёт с context length, поэтому при прочих равных уменьшается число requests, одновременно помещающихся в KV cache (на это также влияют `max_num_seqs`, token budget и compute).
- **Exp3 (output length):** TTFT почти не зависит от output length. E2E ~линейно растёт с N_out. Длинные выходы дольше держат KV блоки, поэтому раньше возникает очередь.
- **Exp4 (KV pressure):** при `kv_cache_usage_perc` → 100% появляются preemptions и рост waiting, при этом compute не исчерпан. p99 E2E деградирует сильнее, чем p50.

---

## 15. Контрольные вопросы

- What is TTFT? What is TPOT? Чем TPOT отличается от ITL?
- What is the difference between prefill and decode? Какая фаза compute-bound, какая memory-bound, и почему?
- What is KV cache? Why does KV cache consume so much VRAM? Посчитай KV per token для модели из эксперимента.
- What is continuous batching? Чем отличается от static / dynamic batching?
- Why does batching improve throughput? Why can batching hurt latency?
- What happens as concurrency increases? How do you identify saturation?
- Why can GPU utilization be low? Почему `nvidia-smi` util 100% ещё ничего не значит?
- Why can VRAM be full while GPU compute is not saturated?
- What changes with longer context? Why does output length matter?
- Что такое preemption в vLLM и как её заметить?
- Когда prefix caching помогает, а когда нет? Как он может исказить бенчмарк?
- Closed-loop vs open-loop: что такое coordinated omission?
- How would you monitor a vLLM service? Какие 5 метрик на dashboard первыми?
- How would you autoscale it? Why is queue length useful? Почему CPU-style autoscaling плохо переносится на GPU serving?
- What happens during pod restart? Why is model cold start expensive?
- When do replicas help? When would tensor parallelism be needed?
- Training scheduler vs inference scheduler?
- Which workloads are compute-bound vs memory-bound?
