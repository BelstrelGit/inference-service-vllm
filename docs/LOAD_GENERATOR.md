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

## Prompt с точным числом токенов

- `token_ids` (по умолчанию): prompt отправляется списком token id (`/v1/completions` это принимает). Сервер получает ровно `input_len` токенов, токенизация из пути выпадает. Id случайные из `[1000, 150000)`, внутри обычного словаря Qwen2.5.
- `text`: случайные id → `/detokenize` → `/tokenize` → подгонка, пока текст не закодируется ровно в `input_len` токенов. Используется токенайзер самого сервера. Ближе к реальному трафику и к `vllm bench serve`. Prompt строятся заранее, поэтому нужен `num_requests`.
- Prompt случайные и общих префиксов не имеют, поэтому prefix cache не даёт попаданий. Общий префикс задаётся явно через `shared_prefix_len` (для экспериментов с prefix caching).

## Прогрев и окно замера

Запрос относится к прогреву (`phase: warmup`), пока не выполнено **оба** условия: отправлено `warmup_requests` запросов и прошло `warmup_s` секунд. Это реализует правило протокола «N = 10 или 30 с, что больше». Дальше идёт окно замера: `num_requests` запросов либо `duration_s` секунд. После окна новые запросы не отправляются, а начатые доводятся до конца.

Окно для пропускной способности: от отправки первого measure-запроса до завершения последнего. Серверные метрики берутся за то же окно: часы клиента переводятся в wall clock через `started_at_unix`.

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

## Ограничения

- Клиент — один процесс asyncio. На системной ноде (`e2-standard-2`) у него ~0.9 CPU. При больших concurrency разбор SSE может стать узким местом клиента, и это завысит TTFT/ITL. Признаки: `send_lag` в open-loop и сравнение клиентского TTFT с серверным.
- GPU-метрик (utilization, VRAM) в снимках нет, они придут из DCGM в M3/M5.
- Гистограммы vLLM хранятся только как `_sum/_count`, поэтому с сервера берём средние за окно, а перцентили считаем по клиентским записям.
