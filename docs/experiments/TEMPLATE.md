# EXP_00X — <Название>

**Дата:** YYYY-MM-DD
**Статус:** planned / running / done
**Series:** `<series_id>`

---

## Question

Какой вопрос исследуем — одно предложение.

## Environment

- Series: `<series_id>`
- Meta: `results/raw/<series_id>/<run_id>.meta.json`
- Environment: A (Colab, exploratory) / B (GKE) / C (VM)
- GPU / model / vLLM version — кратко; полная конфигурация в meta.

## Hypothesis

Что ожидаем увидеть и почему (с опорой на механизм: prefill/decode, batching, KV cache…).

## Setup

- Load mode: closed-loop (concurrency) / open-loop (rate)
- Prompt tokens:
- Max output tokens / `ignore_eos`:
- Duration или request count:
- Warm-up:
- Repetitions:
- Команда запуска:

## Controlled variables

| Variable | Value |
|---|---|
| | |

## Independent variable

`<variable>`: уровни `…`

## Results

| Level | req/s | output tok/s | TTFT p50 | TTFT p95 | TPOT p50 | E2E p95 | error rate | KV-cache usage | GPU util |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

Агрегаты: `results/aggregated/<series_id>/…`

## Graphs

![](../../results/figures/<series_id>/<figure>.png)

## Interpretation

Что показали данные, со ссылкой на конкретные цифры. Подтвердилась ли гипотеза.

## Limitations

- …

## Next question

Какой вопрос следует из результатов.
