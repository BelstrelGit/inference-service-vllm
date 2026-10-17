# Load generator (M2)

Собственный асинхронный генератор нагрузки для controlled experiments: сценарий в YAML, сырые записи по каждому запросу, снимки серверных метрик, агрегация и графики. Правила измерений описаны в [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), а здесь — как этим пользоваться и как это устроено.

## Компоненты

| Файл | Что делает |
|---|---|
| `client/stream.py` | один streaming-запрос, точки замера, ретраи до первого токена (экспоненциальный backoff с jitter) |
| `client/records.py` | схема записи, формулы TTFT / TPOT / ITL / E2E, перцентили, сводка run |
| `client/prompts.py` | синтетические prompt с точным числом токенов (`token_ids` или `text`) |
| `client/load.py` | YAML-сценарий → closed-loop / open-loop прогоны, sweep по уровням и повторам |
| `client/server_metrics.py` | снимки vLLM `/metrics` раз в `metrics_interval_s` во время прогона |
| `client/mock_server.py` | mock vLLM без GPU: очередь при `max_num_seqs`, замедление от размера батча, `/tokenize` |
| `analysis/aggregate.py` | raw → `runs.csv`, `levels.csv`, `levels.md` (медиана по повторам и серверные метрики за окно замера) |
| `analysis/plot.py` | графики sweep и временные ряды очереди/KV |
| `analysis/compare_vllm_bench.py` | сверка прогона с `vllm bench serve` |
| `configs/scenarios/*.yaml` | сценарии экспериментов |
| `scripts/run-bench.sh` | запуск клиента в кластере и копирование результатов |
| `scripts/validate-vs-vllm-bench.sh` | наш клиент и `vllm bench serve` на одной нагрузке |

## Режимы нагрузки

**closed-loop** (`mode: closed`, `concurrency: N`): N асинхронных воркеров; каждый шлёт следующий запрос сразу после завершения предыдущего. В системе одновременно не больше N запросов. Когда сервер замедляется, клиенты просто ждут: нагрузка сама подстраивается под сервер (coordinated omission). Отвечает на вопрос «сколько выдаёт сервер при N одновременных пользователях».

**open-loop** (`mode: open`, `rate: R`): запросы приходят по процессу Пуассона, интервалы экспоненциальные со средним 1/R, независимо от ответов. Если R больше пропускной способности, очередь и TTFT растут весь прогон. Расписание прибытий детерминировано (`seed`) и одинаково для всех повторов. `send_lag` (насколько запрос ушёл позже расписания) показывает, успевает ли сам генератор: если его p99 сравним с TTFT, результат недостоверен.

Подсчёт запросов синхронный, `claim()` выполняется без `await`: даже при плотном потоке прибытий генератор не отправит больше `num_requests`.

**Синхронные волны в closed-loop.** Если все N воркеров стартуют одновременно, а длины запросов одинаковые, воркеры так и идут в ногу: каждая «волна» из N запросов приходит разом, и TTFT отражает худший случай. Это видно в сверке: при c=8 TTFT принимает три значения, ~300 / ~700 / ~1100 мс ([VALIDATION_VLLM_BENCH.md](experiments/VALIDATION_VLLM_BENCH.md)). `vllm bench serve --request-rate inf` ведёт себя так же. Опция `stagger_s` разносит старт воркеров: воркер i стартует через i × `stagger_s` / N. При `stagger_s` около одной E2E прибытия распределяются равномерно, и, поскольку длины одинаковые, сдвиги сохраняются. `stagger_s` не может превышать `warmup_s`: разгон должен уложиться в прогрев.

## Prompt с точным числом токенов

- `token_ids` (по умолчанию): prompt отправляется списком token id (`/v1/completions` это принимает). Сервер получает ровно `input_len` токенов, токенизация из пути выпадает. Id случайные из `[1000, 150000)`, внутри обычного словаря Qwen2.5.
- `text`: случайные id → `/detokenize` → `/tokenize` → подгонка, пока текст не закодируется ровно в `input_len` токенов. Используется токенайзер самого сервера. Ближе к реальному трафику и к `vllm bench serve`. Prompt строятся заранее, поэтому нужен `num_requests`.
- Prompt случайные и общих префиксов не имеют, поэтому prefix cache не даёт попаданий. Общий префикс задаётся явно через `shared_prefix_len` (для экспериментов с prefix caching).

## Прогрев и окно замера

Запрос относится к прогреву (`phase: warmup`), пока не выполнено **оба** условия: отправлено `warmup_requests` запросов и прошло `warmup_s` секунд. Это реализует правило протокола «N = 10 или 30 с, что больше». Дальше идёт окно замера: `num_requests` запросов либо `duration_s` секунд. После окна новые запросы не отправляются, а начатые доводятся до конца.

Окно для пропускной способности: от отправки первого measure-запроса до завершения последнего. Серверные метрики берутся за то же окно: часы клиента переводятся в wall clock через `started_at_unix`.

`output_token_throughput` считает только токены measure-запросов. Если окно сравнимо с E2E, метрика смещена: в начале окна сервер ещё дорабатывает прогревочные запросы, а в конце нагрузка спадает. Поэтому `analysis.aggregate` считает ещё `steady_output_tok_s`: все выходные токены любой фазы между первой и последней отправкой measure-запроса, делённые на этот интервал. В пилоте Exp1 (окно 45 с) две метрики расходились до 19% ([EXP_002](experiments/EXP_002_CONCURRENCY.md)).

## Сценарий

```yaml
name: exp1_concurrency          # входит в run_id
series: l4-qwen7b-bf16-v1       # каталог results/raw/<series>/
mode: closed                    # closed | open
concurrency: [1, 2, 4, 8]       # число или список (sweep); для open — rate
duration_s: 90                  # или num_requests (ровно одно из двух)
num_requests: null
warmup_requests: 10
warmup_s: 30
repeats: 3
input_len: 512
output_len: 128                 # max_tokens; с ignore_eos ровно столько
prompt_format: token_ids        # token_ids | text
endpoint: completions           # completions | chat (chat требует text)
```

Остальные ключи и значения по умолчанию — в `client/load.py` (`Scenario`). Неизвестный ключ — ошибка, так что опечатка не пройдёт молча. Любой ключ можно переопределить из CLI: `--set concurrency=8 --set repeats=1` (значение разбирается как YAML, пробелов внутри быть не должно).

## Выход

```text
results/raw/<series>/<run_id>.jsonl         запись на каждый запрос
results/raw/<series>/<run_id>.meta.json     сценарий, уровень, сервер (/version, /v1/models), git, сводка
results/raw/<series>/<run_id>.server.jsonl  снимки /metrics: gauges, counters, histogram _sum/_count
results/aggregated/<series>/{runs,levels}.csv, levels.md
results/figures/<series>/*.png
```

`run_id = <YYYYMMDD-HHMMSS>_<scenario>_<level>_r<rep>`, например `20261020-143005_exp1_concurrency_c16_r2`.

## Локально (без GPU)

```bash
uv run python -m client.mock_server --port 8011 --max-num-seqs 4 --batch-slowdown 0.05
uv run python -m client.load configs/scenarios/mock_smoke.yaml --base-url http://127.0.0.1:8011
uv run python -m analysis.aggregate results/raw/mock-dev
uv run --extra analysis python -m analysis.plot sweep results/aggregated/mock-dev
```

Mock — игрушечная модель задержек, а не модель vLLM. Он нужен, чтобы проверить весь конвейер; его цифры ничего не говорят о GPU.

## В кластере

```bash
kubectl apply -f k8s/serving/vllm.yaml                          # vLLM Deployment + Service
scripts/run-bench.sh load configs/scenarios/exp0_baseline.yaml  # клиент в Pod на системной ноде
uv run python -m analysis.aggregate results/raw/l4-qwen7b-bf16-v1
```

`run-bench.sh` отправляет текущие `client/` и `configs/scenarios/` в ConfigMap, запускает `k8s/bench/client-pod.yaml`, показывает вывод, копирует `/out` в `results/raw/` и удаляет Pod. Если терминал отвалился во время долгого sweep, прогон продолжается в кластере, а `scripts/run-bench.sh --collect` заберёт результаты.

## Сверка с `vllm bench serve`

```bash
scripts/validate-vs-vllm-bench.sh
```

Оба инструмента по очереди запускаются внутри Pod vLLM на `localhost` с одной нагрузкой (`configs/scenarios/validation_vllm_bench.yaml`: c=8, 200 запросов, 512 → 128, text prompts). Скрипт печатает таблицу расхождений, ожидаем в пределах ~5–10%. Известные различия в определениях (TTFT по первому chunk, конец запроса по последнему токену или по `[DONE]`) описаны в `analysis/compare_vllm_bench.py`.

## Порядок запуска на GPU (gates)

Каждый следующий шаг — только если предыдущий прошёл.

1. **Live M2 smoke** — `configs/scenarios/m2_live_smoke.yaml` (c=1 и c=8, ~3 мин). Проверить, что `.jsonl`, `.server.jsonl`, агрегация и графики корректны на настоящем vLLM.
2. **Сверка с `vllm bench serve`** — `scripts/validate-vs-vllm-bench.sh` (c=8, 200 запросов, 512 → 128).
3. Если TTFT / TPOT / throughput сходятся в пределах ~5–10%, **замораживаем измерительный инструмент**: дальше изменения `client/` начинают новую серию.
4. **Exp0** — `configs/scenarios/exp0_baseline.yaml` (c=1, 5 повторов).
5. **Pilot Exp1** — `configs/scenarios/exp1_pilot.yaml` (c = 1, 8, 32, 64, 128, по одному короткому прогону). Смотрим, где примерно насыщение и не упирается ли клиент: `client CPU` в строке прогона, `kubectl -n vllm top pod bench-client`, клиентский TTFT против серверного.
6. **Полный Exp1** — `configs/scenarios/exp1_concurrency.yaml` (1…128 × 3 повтора).

## Как читать очередь

- `num_requests_waiting > 0` означает, что запросу не хватило ресурса планировщика на этой итерации. Причин несколько: свободные KV-блоки, `max_num_seqs`, token budget `max_num_batched_tokens` (при prompt 512 и budget 2048 за итерацию проходит prefill не больше 4 запросов).
- `num_requests_waiting_by_reason` в vLLM 0.30 различает только `capacity` (все перечисленные причины вместе) и `deferred` (LoRA, передача KV, блокировка). KV и token budget он **не разделяет**. Причину выводим по сочетанию метрик: `kv_cache_usage_perc` около 1 и рост `num_preemptions` указывают на KV; очередь при невысоком KV usage — на token budget или `max_num_seqs`.
- При chunked prefill запрос быстро переходит из `waiting` в `running`, а затем ждёт свои куски prefill в следующих итерациях. Поэтому задержка из-за бюджета токенов видна в `server_prefill_mean_ms` (`vllm:request_prefill_time_seconds`), а не в `waiting` или queue time. В сверке при c=8: `waiting` 0, очередь 2 мс, prefill 712 мс.
- `client_cpu_util` (в meta и `runs.csv`) — доля одного ядра, которую занял процесс клиента. Если она близка к 1, клиент сам может искажать TTFT/ITL.

## Ограничения

- Клиент — один процесс asyncio. На системной ноде (`e2-standard-2`) у него ~0.9 CPU. При больших concurrency разбор SSE может стать узким местом клиента, и это завысит TTFT/ITL. Признаки: `send_lag` в open-loop и сравнение клиентского TTFT с серверным.
- GPU-метрик (utilization, VRAM) в снимках нет, они придут из DCGM в M3/M5.
- Гистограммы vLLM хранятся только как `_sum/_count`, поэтому с сервера берём средние за окно, а перцентили считаем по клиентским записям.
