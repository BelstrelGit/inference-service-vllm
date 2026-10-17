# EXP_000 — First measured request (M1)

**Дата:** 2026-10-17
**Статус:** done
**Series:** `l4-qwen7b-bf16-m1`

---

## Question

Сколько занимают TTFT и TPOT одного запроса без конкуренции на нашей конфигурации (L4 + Qwen2.5-7B bf16 + vLLM 0.30.0) и воспроизводимо ли измерение?

Это первый замер, а не полноценный Exp0: 1 повтор, без серверных метрик. Его задача — проверить измерительную цепочку и получить порядок величин.

## Environment

- Series: `l4-qwen7b-bf16-m1`
- Meta: `results/raw/l4-qwen7b-bf16-m1/20261017-094239_single_n20_r1.meta.json`
- Environment: B (GKE, `us-central1-a`), `g2-standard-8` + 1 × NVIDIA L4 (23034 MiB), driver 580.173.02
- vLLM 0.30.0 (`vllm/vllm-openai:v0.30.0@sha256:8a69ffad…`), `Qwen/Qwen2.5-7B-Instruct` @ `a09a3545…`, bf16
- Server: `k8s/serving/vllm.yaml` (Deployment + Service, commit `8518e2e`): `max-model-len 8192`, `gpu-memory-utilization 0.90`, `max-num-seqs 256`, `max-num-batched-tokens 2048`, prefix caching on, chunked prefill on, `generation-config vllm`
- KV cache: 2.87 GiB → 53,696 tokens (startup log, `docs/ENVIRONMENT.md`)
- Client: `client.run` в Pod на системной ноде (`e2-standard-2`), путь Pod → ClusterIP Service → vLLM Pod; client commit `8518e2e`, clean

## Hypothesis

При одном запросе decode упирается в пропускную способность памяти: на каждый токен GPU читает все веса (~15.2 GB). L4 даёт ~300 GB/s, значит **TPOT ≥ ~50 мс**, на практике **50–60 мс**. Prefill ~45 токенов стоит примерно столько же, сколько один шаг decode, поэтому **TTFT ≈ 1 шаг модели + накладные расходы (HTTP, токенизация, планировщик), то есть < 150 мс**.

## Setup

- Load mode: sequential (concurrency = 1, следующий запрос после завершения предыдущего)
- Endpoint: `/v1/completions`, текстовый prompt: фиксированный текст плюс уникальный hex-префикс, чтобы prefix cache не давал попаданий. Получилось 41–45 prompt tokens.
- Max output tokens: 256, `ignore_eos=true`, `temperature=0`
- Request count: 20 measure
- Warm-up: 3 запроса, отброшены (`phase=warmup`)
- Repetitions: 1
- Команда запуска:
  ```bash
  scripts/run-bench.sh run --series l4-qwen7b-bf16-m1 --scenario single -n 20 --warmup 3 --max-tokens 256
  ```
  (на момент запуска скрипт вызывался без первого аргумента `run`, интерфейс изменился в M2)

## Controlled variables

| Variable | Value |
|---|---|
| GPU / model / dtype | L4 / Qwen2.5-7B-Instruct / bf16 |
| vLLM image | v0.30.0, digest зафиксирован |
| Prompt | ~43 tokens, уникальный на запрос |
| Output | ровно 256 tokens |
| Concurrency | 1 |
| Sampling | greedy (`temperature=0`) |

## Independent variable

Нет: одиночная точка.

## Results

| Level | req/s | output tok/s | TTFT p50 | TTFT p90 | TTFT max | TPOT p50 | TPOT p90 | E2E p50 | error rate |
|---|---|---|---|---|---|---|---|---|---|
| c=1 | 0.069 | 17.6 | 81.2 ms | 104.8 ms | 124.2 ms | 56.6 ms | 56.6 ms | 14.51 s | 0/20 |

Распределение ITL по всем 5100 интервалам между токенами: p50 56.57 мс, p90 56.95 мс, p99 57.35 мс, min 53.75 мс, max 59.56 мс.

Прогрев: первый запрос после старта сервера TTFT = 160.8 мс, следующие два ~80 мс.

## Graphs

Нет: одиночная точка, таблицы достаточно.

## Interpretation

- **TPOT 56.6 мс, гипотеза подтвердилась.** В приближении «за шаг читаются только веса» 15.23 GB / 0.0566 s ≈ **269 GB/s** (weights-only implied bandwidth), ~90% пиковых ~300 GB/s у L4. Это косвенная оценка, а не измерение: пропускную способность памяти мы не мерили, а реальный decode читает ещё KV cache и создаёт прочий memory traffic. Но TPOT очень близок к простой bandwidth-модели и хорошо поддерживает гипотезу memory-bound decode: быстрее одиночный поток (~17.7 ток/с) на этой карте практически не получить. Рост суммарной пропускной способности может дать только batching (Exp1), потому что за один проход по весам считается больше токенов.
- **Шаг decode почти детерминирован**: p99/p50 у ITL = 1.014. Вся вариативность latency при c=1 сидит не в GPU.
- **TTFT ~80 мс, разброс до 124 мс.** Значения группируются около ~80 и ~100 мс, связи с длиной prompt (41–45) нет. Раз шаг модели стабилен, лишние 20–40 мс возникают на пути HTTP → токенизация → `EngineCore` → планировщик → SSE → сеть. Источник по этим данным не определить: нужно сравнить клиентский TTFT с серверным `vllm:time_to_first_token_seconds` (будет в M2).
- **E2E сходится с моделью** TTFT + 255 × TPOT: 80 + 255 × 56.6 ≈ 14 513 мс, измерено 14 510 мс. Формулы клиента согласованы.
- **Первый запрос после старта медленнее** (161 мс): одноразовая инициализация. В smoke-тесте в логе была JIT-компиляция Triton-ядра на первом запросе. Прогрев обязателен.

## Limitations

- 1 повтор, 20 запросов: о run-to-run variance выводов нет (это Exp0).
- Серверные метрики не снимались: разложить TTFT на очередь, prefill и накладные расходы нельзя.
- Prompt текстовый, длина плавает (41–45 tokens), а не задана точно.
- Модель скачивается при каждом старте (emptyDir), а клиент и сервер на разных нодах: в TTFT входит сетевой hop внутри VPC (ожидаемо < 1 мс, отдельно не измерялся).
- Одна GPU, одна модель, одна версия vLLM: результат описывает только эту конфигурацию.

## Next question

Как растут output tokens/s и TTFT/TPOT с concurrency, и где наступает насыщение (Exp1)? Сначала Exp0 (variance при c=1 на prompt 512 / output 128) и проверка генератора нагрузки против `vllm bench serve`.
