# Monitoring (M4): Prometheus + Grafana + DCGM

**Статус:** установлено 2026-10-21, после замеров Exp4A (дизайн 2026-10-20). Что пошло не по плану и результаты проверки — в разделе «Установка 2026-10-21».

Цель — самому собрать и увидеть весь путь метрики, без управляемых сервисов GKE:

```text
vLLM /metrics      → Prometheus scrape → TSDB → PromQL → Grafana panel
L4 → DCGM exporter → Prometheus scrape → TSDB → PromQL → Grafana panel
```

Поэтому self-hosted `kube-prometheus-stack` (Prometheus Operator, Prometheus, Grafana, kube-state-metrics, node-exporter), а не Managed Service for Prometheus. GPU-метрики отдаёт `dcgm-exporter`, который GKE сам запускает на GPU-нодах: свой экспортер с ним не уживается (см. «Установка 2026-10-21»), и наш Prometheus снимает метрики с экспортера GKE. Вся конфигурация хранится в Git: кластер можно удалить и поднять заново, мониторинг восстановится из репозитория.

## Когда ставится

Только после замеров Exp4 (bf16 и, если делается, FP8), когда новые performance-эксперименты заморожены. Exp4 сравнивается с Exp2 при неизменном окружении. Prometheus добавил бы второй поток запросов к `/metrics` vLLM и новые поды на нодах, а это новая переменная. В ту же оплачиваемую сессию GKE можно, но после последнего прогона Exp4.

Числа экспериментов по-прежнему считаются из сырых записей клиента и его снимков `/metrics` раз в секунду (`results/raw/`). Prometheus и Grafana нужны, чтобы видеть процесс глазами и для операционного сценария (рестарт пода под нагрузкой), а не как источник чисел для отчётов.

## Topology и размещение

```text
vllm-lab
├── default-pool  e2-standard-2 × 1        системные поды GKE, bench-client (900m CPU)
├── obs-pool      e2-standard-2 × 0/1      Prometheus, Grafana, operator, kube-state-metrics
│                 taint workload=observability:NoSchedule
└── l4-pool       g2-standard-8 + L4 × 0–1 vLLM, dcgm-exporter (ставит GKE, namespace gke-managed-system)
node-exporter: DaemonSet на всех нодах
```

- **Отдельный `obs-pool`.** Генератор нагрузки, inference и observability не должны делить CPU и память одной ноды. На `default-pool` после системных подов GKE остаётся ~1.07 CPU, и почти всё забирает `bench-client` (`k8s/bench/client-pod.yaml`). Taint не пускает на `obs-pool` посторонние поды, а компоненты мониторинга получают toleration и `nodeSelector`.
- **Одна зона для `obs-pool`.** Диск Prometheus (pd-balanced, StorageClass `standard-rwo`, WaitForFirstConsumer) зональный: он создаётся в зоне ноды, на которую Prometheus попал первым, и после этого PV привязывает Prometheus к этой зоне. Поэтому `obs-pool` создаётся в одной зоне (`--node-locations $ZONE`), и нода всегда поднимается там же, где диск.
- **Размер `obs-pool` задаётся вручную** (`resize 0/1`, как `default-pool`), без autoscaler: иначе ожидающие Prometheus и Grafana поднимали бы ноду обратно после каждого `resize 0`.
- **node-exporter — на всех нодах, включая L4.** У GPU-ноды taint `nvidia.com/gpu`, поэтому у DaemonSet node-exporter `tolerations: [{operator: Exists}]`; при установке проверяется, что DESIRED = числу нод (`monitoring/README.md`, Check).
- **DCGM: так было задумано, на практике используется экспортер GKE** (см. «Установка 2026-10-21»). Исходный план: **`dcgm-exporter` — DaemonSet только на GPU-ноде** (affinity по `cloud.google.com/gke-accelerator`, toleration `nvidia.com/gpu`). `nvidia.com/gpu` он **не запрашивает**: единственная L4 занята vLLM. Доступ к GPU — по схеме из документации GKE ([NVIDIA DCGM, Managed Prometheus exporters](https://docs.cloud.google.com/stackdriver/docs/managed-prometheus/exporters/nvidia-dcgm)): privileged-контейнер, hostPath `/home/kubernetes/bin/nvidia` → `/usr/local/nvidia`, `LD_LIBRARY_PATH=/usr/local/nvidia/lib64`, hostPath `/var/lib/kubelet/pod-resources` (сопоставление GPU с подом). В примере GKE два DaemonSet: отдельный `nv-hostengine` и exporter, который к нему подключается. Сначала пробуем один DaemonSet со встроенным hostengine (`monitoring/dcgm/exporter.yaml`); если он не видит GPU, переходим на схему GKE из двух (`monitoring/dcgm/hostengine.yaml`, две правки описаны в его заголовке). Это открытый вопрос реализации, решается на конкретной запиненной версии при установке. Образы в примере GKE старые (3.3.0); у нас самый свежий тег на nvcr.io на 2026-10-20, `dcgm-exporter:4.5.2-4.8.1-ubuntu22.04`.
- **Нода `l4-pool` есть только во время сессии.** Пока её нет, таргеты vLLM и DCGM отсутствуют. Это нормально, не алерт.

Стоимость `obs-pool`: ~$0.07/ч в сессии, 0 между сессиями (пул масштабируется в 0 вместе с остальными).

**Граница объёма M4.** Обязательно: Prometheus, Grafana, ServiceMonitor vLLM, DCGM и один дашборд. Потом операционный сценарий: рестарт пода под нагрузкой (failures, retries, readiness, recovery). Не делаем: Alertmanager и алерты, Ingress, Loki, tracing / OpenTelemetry, Thanos, remote storage, дополнительные дашборды.

## Resource requests / limits

Масштаб маленький: один под vLLM, одна GPU, ~40 метрик vLLM, ~10 полей DCGM, 5 таргетов. Значения стартовые; если не хватит, увеличиваем по факту (`kubectl top`, OOMKilled).

| Компонент | Нода | requests CPU / mem | limits mem |
|---|---|---|---|
| Prometheus | obs-pool | 300m / 1Gi | 2Gi |
| Prometheus Operator | obs-pool | 50m / 128Mi | 256Mi |
| Grafana (+ sidecar дашбордов) | obs-pool | 100m / 256Mi | 512Mi |
| kube-state-metrics | obs-pool | 50m / 64Mi | 128Mi |
| node-exporter | все ноды | 20m / 32Mi | 64Mi |
| dcgm-exporter (GKE) | l4-pool | задаёт GKE | задаёт GKE |
| Alertmanager | — | выключен | — |

- Сумма на `obs-pool` ~0.5 CPU и ~1.5Gi requests, e2-standard-2 (2 vCPU, 8 GB) с запасом.
- На `default-pool` и `l4-pool` добавляется только node-exporter (20m). dcgm-exporter GKE работал на L4-ноде и раньше, во всех прогонах Exp1–Exp4.
- CPU limits не ставим: throttling exporter'ов искажает интервалы scrape. Memory limits ставим.
- Alertmanager не нужен: алертов в объёме работ нет. Включается одной строкой в values, если понадобится.

**Хранилище Prometheus:** PVC 10Gi pd-balanced, retention 7 дней и `retentionSize` 8GB. Диск переживает масштабирование пулов в 0, поэтому сессии можно сравнивать задним числом. Стоит ~$1 в месяц. В чеклисте End `k8s/cluster/README.md` после этого ожидается один диск PVC Prometheus, а не «нет дисков». Grafana без PVC: дашборды и datasource провижнятся из Git, локальных правок в UI не храним.

## Scrape

- **vLLM:** `ServiceMonitor` на Service `vllm` (namespace `vllm`, label `app: vllm`, порт `http`, path `/metrics`), interval 5s.
- **DCGM:** `PodMonitor` на экспортер GKE (`monitoring/dcgm/podmonitor-gke-managed.yaml`), interval 5s. Интервал сбора внутри экспортера задаёт GKE и не проверялся; если он больше 5 с, соседние точки на графике повторяют одно значение. В своём экспортере было 5 с (`DCGM_EXPORTER_INTERVAL=5000`).
- **node-exporter, kube-state-metrics, kubelet/cAdvisor:** по умолчанию чарта, 30s.
- **Выключить в values** таргеты, недоступные на GKE: control plane управляется Google (`kubeControllerManager`, `kubeScheduler`, `kubeEtcd`, `kubeProxy`), иначе они висят как down.
- `serviceMonitorSelectorNilUsesHelmValues: false` и то же для PodMonitor: иначе Prometheus подхватывает только мониторы с label релиза чарта, и наш ServiceMonitor молча не применится. Тот самый случай «labels/selectors», который нужно увидеть руками.
- Нагрузка на vLLM: клиент экспериментов уже опрашивает `/metrics` раз в секунду, Prometheus добавит раз в 5 с. Это и есть причина не ставить мониторинг до Exp4.

## Метрики

**vLLM 0.30** (имена взяты из снимков `results/raw/`, а не из документации):

| Метрика | Тип | Зачем |
|---|---|---|
| `vllm:num_requests_running`, `vllm:num_requests_waiting` | gauge | нагрузка, плато running, очередь |
| `vllm:num_requests_waiting_by_reason{reason}` | gauge | `capacity` против `deferred` |
| `vllm:kv_cache_usage_perc` | gauge | заполнение пула |
| `vllm:cache_config_info{kv_cache_size_tokens, cache_dtype, num_gpu_blocks}` | info | K и dtype на дашборде, вариант сервера |
| `vllm:num_preemptions_total` | counter | preemption в секунду |
| `vllm:prompt_tokens_total`, `vllm:generation_tokens_total` | counter | prefill и decode throughput, tok/s |
| `vllm:request_success_total` | counter | завершённые запросы в секунду |
| `vllm:time_to_first_token_seconds` | histogram | TTFT |
| `vllm:request_queue_time_seconds` | histogram | время в очереди |
| `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` | histogram | фазы запроса |
| `vllm:inter_token_latency_seconds` | histogram | ITL ≈ TPOT |
| `vllm:e2e_request_latency_seconds` | histogram | E2E |

Остальные экспортируемые метрики (`prefix_cache_*`, `iteration_tokens_total`, `request_*_tokens`, `estimated_*_per_gpu_total`) Prometheus сохраняет, но в первый дашборд они не входят.

**DCGM** (поля по именам dcgm-exporter; набор полей экспортера GKE — ConfigMap `gke-managed-system/dcgm-exporter-metrics`, все поля ниже в нём есть, кроме `MEM_CLOCK`):

| Поле | Зачем | Оговорка |
|---|---|---|
| `DCGM_FI_DEV_GPU_UTIL` | доля времени, когда работает хотя бы одно ядро | не загрузка: при c=1 decode уже близок к 100% |
| `DCGM_FI_PROF_SM_ACTIVE` | доля времени, когда SM заняты | profiling-поле; на L4 доступно |
| `DCGM_FI_PROF_DRAM_ACTIVE` | занятость памяти, ближе всего к «упору в bandwidth» при decode | то же |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | занятость tensor cores, ожидаемо выше при prefill | то же, опционально |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | доля времени, когда контроллер памяти читал или писал | грубее `PROF_DRAM_ACTIVE`, но не profiling-поле |
| `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE` | VRAM: веса + KV-пул + активации | при Exp4-override видно, что пул меньше |
| `DCGM_FI_DEV_POWER_USAGE` | мощность, Вт (TDP L4 72 Вт) | |
| `DCGM_FI_DEV_GPU_TEMP` | температура | |
| `DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_MEM_CLOCK` | частоты SM и памяти (`MEM_CLOCK` экспортер GKE не отдаёт) | падение частоты SM при упоре в TDP проявилось бы как дрейф TPOT |


## Первый dashboard: `vllm-gpu-overview`

Один дашборд, строки сверху вниз по цепочке, которую мы до этого измеряли численно: нагрузка → running → GPU → KV → очередь → задержки. Все панели — time series с общей осью времени, `$__rate_interval` для rate и перцентилей.

| Строка | Панель | PromQL |
|---|---|---|
| Server | вариант сервера (stat) | `vllm:cache_config_info` → labels `kv_cache_size_tokens`, `cache_dtype` |
| Load | running / waiting | `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| Load | завершённые запросы, req/s | `rate(vllm:request_success_total[$__rate_interval])` |
| Throughput | prefill и decode tok/s | `rate(vllm:prompt_tokens_total[...])`, `rate(vllm:generation_tokens_total[...])` |
| GPU | GPU util, SM active, DRAM active, tensor active | `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` |
| GPU | VRAM used / free | `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE` |
| GPU | power, temperature, clocks | `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_MEM_CLOCK` |
| KV | KV usage | `vllm:kv_cache_usage_perc` |
| KV | preemption/s | `rate(vllm:num_preemptions_total[...])` |
| Queue | queue time p50 / p95 | `histogram_quantile(0.5, sum by (le) (rate(vllm:request_queue_time_seconds_bucket[...])))` и 0.95 |
| Latency | TTFT p50 / p95 | то же для `vllm:time_to_first_token_seconds` |
| Latency | ITL p50 / p95 | то же для `vllm:inter_token_latency_seconds` |
| Latency | prefill и decode time p50 | то же для `vllm:request_prefill_time_seconds`, `vllm:request_decode_time_seconds` |

Перцентили из гистограмм приближённые (интерполяция внутри bucket'ов vLLM) и в отчёты не идут; точные перцентили — у клиента.

## Структура в репозитории

```text
monitoring/
├── README.md                          создание obs-pool, установка, проверка, удаление
├── prometheus/
│   ├── values.yaml                    kube-prometheus-stack: размещение, ресурсы, retention, PVC,
│   │                                  выключенные таргеты, селекторы, Grafana, Alertmanager off
│   └── service-monitor-vllm.yaml
├── dcgm/                              kubectl apply -k
│   ├── podmonitor-gke-managed.yaml    PodMonitor на dcgm-exporter GKE (применяется)
│   ├── exporter.yaml                  свой DaemonSet + PodMonitor: только если DCGM в GKE выключен
│   ├── counters.csv                   поля DCGM для своего экспортера
│   ├── hostengine.yaml                запасная схема для своего экспортера
│   └── kustomization.yaml
└── grafana/                           kubectl apply -k
    ├── kustomization.yaml             configMapGenerator: JSON → ConfigMap с label grafana_dashboard=1
    └── dashboards/
        └── vllm-gpu-overview.json
```

- Чарт `kube-prometheus-stack` 92.11.0 запинен (`monitoring/README.md`). Версию dcgm-exporter GKE задаёт сам (на 2026-10-21 `gke-dcgm-exporter:4.4.1-4.6.0-gke.27`).
- Datasource Prometheus провижнит сам чарт. Дашборды подхватывает sidecar Grafana по label ConfigMap.
- Пароль admin Grafana не в Git: Secret создаётся командой при установке.
- Доступ только через `kubectl port-forward` (Grafana 3000, Prometheus 9090). Ни Ingress, ни внешнего IP.

## Проверка после установки

1. `Status → Targets` в Prometheus: vLLM, DCGM, node-exporter, kube-state-metrics, kubelet — `up`. node-exporter есть на каждой ноде, в том числе на L4 (toleration на `nvidia.com/gpu`).
2. `vllm:kv_cache_usage_perc` и `DCGM_FI_DEV_FB_USED` есть в Explore; `cache_config_info` показывает тот же K, что `scripts/check-server.sh`.
3. Короткий прогон под нагрузкой (например `exp4_control` при c=24, один повтор, серия `<series>-mon`): на дашборде running ≈ 23–24, KV ≈ 0.9; значения сходятся с `levels.csv` клиента в пределах того, что даёт разница интервалов 1s / 5s.
4. Влияние мониторинга (опционально, контроль, а не новый эксперимент): тот же прогон против `exp4_control` без мониторинга; `steady tok/s` и TPOT в пределах допуска контроля Exp4 (3%).

## Установка 2026-10-21

После последнего прогона Exp4A, в той же сессии, сервер `kv2517` (K = 40,272). Установка по `monitoring/README.md`, ~40 мин вместе с созданием `obs-pool`.

**DCGM: свой экспортер не нужен.** Свой `dcgm-exporter` (`exporter.yaml`, встроенный hostengine) поднялся: DCGM и NVML инициализировались, L4 найдена. Но он падал при подписке на поля: `DCGM collector for entity type 'GPU' cannot be initialized; err: error watching fields: The third-party Profiling module returned an unrecoverable error`. Причина — на ноде уже работал `dcgm-exporter` GKE (`gke-managed-system`, включён компонент мониторинга кластера `DCGM`; под жил с момента создания ноды). Он держит profiling-модуль, а второй клиент к profiling-полям не подключается. Экспортер GKE уже собирает нужные поля, включая `PROF_SM_ACTIVE`, `PROF_DRAM_ACTIVE` и `PROF_PIPE_TENSOR_ACTIVE`, поэтому Prometheus снимает метрики с него. Свой экспортер оставлен в репозитории на случай, если компонент DCGM в GKE выключить. Схема из двух DaemonSet не понадобилась.

Следствие для экспериментов: экспортер GKE работал на L4-ноде и во всех прогонах Exp1–Exp4, это часть их окружения, а не изменение.

**Проверка:**
1. Targets: `vllm`, `gke-managed-dcgm-exporter`, node-exporter на всех трёх нодах (default, l4, obs), kube-state-metrics, kubelet ×3, apiserver, Grafana — `up` ([скриншот](images/monitoring/prometheus-targets.png)).
2. Панель варианта сервера: `K=40272 tokens, dtype=auto, blocks=2517, override=2517`, как `check-server.sh`.
3. Под нагрузкой (`exp4_kv2517` c=24, один прогон, серия `l4-qwen7b-bf16-kv2517-v1-mon`): running до ~19, waiting ~4–6, KV usage до ~1.0, preemption, prefill ~2000 tok/s при req/s ~0.85 × 2048. Совпадает с `levels.csv` клиента.
4. Влияние мониторинга: `steady tok/s` 122.4 против 122.2, TPOT p50 143.1 против 143.4 мс у того же c=24 в основной серии `kv2517` без мониторинга. Разница < 0.3% при допуске 3%, влияние не обнаружено.

Скриншоты (c=24, пул 0.75×):
- [обзор: сервер, нагрузка, throughput](images/monitoring/dashboard-overview.png)
- [GPU и KV](images/monitoring/dashboard-gpu-kv.png)
- [очередь и задержки](images/monitoring/dashboard-queue-latency.png)

![GPU и KV при c=24, пул 0.75×](images/monitoring/dashboard-gpu-kv.png)

**Что видно на дашборде** (наблюдения одного прогона, причины не проверялись):
- GPU util 100%, SM active ~95%, tensor active ~75–80%, DRAM active ~45–55%. По profiling-полям L4 под этой нагрузкой занята вычислениями сильнее, чем памятью.
- Мощность держится у ~72 W, это TDP L4, а частота SM падает с ~2040 до ~0.9–1.2 ГГц. Похоже на упор в лимит мощности. Если так было во всех перегруженных прогонах, TPOT на плато включает этот эффект; в прогонах Exp1–Exp4 GPU-метрики не записывались.
- В конце прогона p50/p95 очереди растут, а ITL проваливается вниз. Это артефакт `histogram_quantile`: после конца нагрузки в окне `rate` остаётся мало событий. Поведение сервера тут ни при чём.

## Открытые вопросы

- Интервал сбора внутри экспортера GKE (по умолчанию dcgm-exporter — 30 с).
- Нужен ли PVC: если сравнивать сессии задним числом не понадобится, `emptyDir` проще и диск не остаётся между сессиями.
