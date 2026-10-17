# Environment

Фиксация окружений. Заполняется **до** первого benchmark run в каждом окружении.
Все значения — **TBD**, пока не сняты реальными командами.

---

## Environment A — Google Colab (exploratory)

Используется для learning / prototyping. Результаты отсюда — **exploratory**, в основные серии не входят.

| Параметр | Value | Как получить |
|---|---|---|
| Дата снятия | TBD | — |
| Runtime type / tier | TBD | Runtime → Change runtime type |
| GPU model | TBD | `nvidia-smi --query-gpu=name --format=csv,noheader` |
| VRAM | TBD | `nvidia-smi --query-gpu=memory.total --format=csv,noheader` |
| Driver version | TBD | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| CUDA version | TBD | `nvidia-smi` (header) / `python -c 'import torch; print(torch.version.cuda)'` |
| Python | TBD | `python --version` |
| PyTorch | TBD | `python -c 'import torch; print(torch.__version__)'` |
| vLLM | TBD | `python -c 'import vllm; print(vllm.__version__)'` |

---

## Environment B — GKE (primary)

Основное окружение для benchmark-серий. Снято 2026-10-17 (`k8s/debug/nvidia-smi.yaml`); как создать — `k8s/cluster/README.md`.

Node OS: Container-Optimized OS, kernel 6.12.105+, containerd 2.2.7.

Наблюдение (одиночный замер, не эксперимент): L4 node scale-up 0 → 1 until Pod Scheduled ≈ 1.5 мин (`TriggeredScaleUp` 06:50:33 → `Scheduled` 06:51:57 UTC). Это не полный Pod startup / cold start: после `Scheduled` идут image pull, container start и т. д. Длительность установки драйвера отдельно не измерялась. Второй запуск (2026-10-17, M1): `TriggeredScaleUp` 08:50:30 → `Scheduled` 08:52:40 ≈ 2.2 мин — разброс заметный, одиночные замеры. В `europe-west4` (все 3 зоны) L4 в тот день не было (`GCE out of resources`).

Первый cold start vLLM Pod (одиночное наблюдение, emptyDir, без кэша образа и модели), UTC 2026-10-17:

| | Время | Δ | Источник |
|---|---|---|---|
| Pod создан → `TriggeredScaleUp` | 08:50:26 → 08:50:30 | | events |
| L4 нода → `Scheduled` | 08:52:40 | 2m10s | events (между ними: taint/`Insufficient nvidia.com/gpu`, пока поднимаются драйвер и device plugin) |
| Image pull | 08:52:41 → 08:56:20 | 3m39s | event `Pulled … in 3m39.07s` |
| Container start → engine init | 08:56:20 → 08:57:21 | ~1 мин | log |
| Model download + load | 08:57:21 → 08:59:07 | 1m46s | log `Model loading took 14.29 GiB memory and 106.1 s` (из них `Loading weights took 21.03 s`) |
| Profile + KV + torch.compile + CUDA graphs | 08:59:07 → 09:00:02 | 55s | log `init engine … took 55.30 s (compilation: 23.35 s)` |
| `Application startup complete` → Ready | 09:00:04 → 09:00:10 | | log / Ready condition |
| **Итого Pod created → Ready** | | **≈ 9m44s** | |

### Cluster

| Параметр | Value | Как получить |
|---|---|---|
| Project | `lora-sched-stand` | `gcloud config get-value project` |
| Cluster name | `vllm-lab` | `gcloud container clusters list` |
| Location (zone/region) | `us-central1-a` (zonal control plane); GPU pool spans `us-central1-a/b/c` | `gcloud container clusters list` |
| Mode (Standard / Autopilot) | Standard | `gcloud container clusters describe <cluster> --location <loc> --format='value(autopilot.enabled)'` |
| GKE version | `1.36.4-gke.1391002` (release channel: regular) | `kubectl version` / `gcloud container clusters describe … --format='value(currentMasterVersion)'` |

### GPU node pool

| Параметр | Value | Как получить |
|---|---|---|
| Node pool name | `l4-pool` (autoscaling 0–1, `--location-policy ANY`) | `gcloud container node-pools list --cluster <cluster> --location <loc>` |
| Machine type | `g2-standard-8` (8 vCPU, 32 GB RAM), boot disk 200 GB pd-balanced | `kubectl get node <node> -L node.kubernetes.io/instance-type` |
| GPU type | NVIDIA L4 (`nvidia-l4`), compute capability 8.9 | `kubectl get nodes -L cloud.google.com/gke-accelerator` |
| GPU count per node | 1 | `kubectl describe node <node> \| grep nvidia.com/gpu` |
| VRAM | 23034 MiB | `kubectl exec <pod> -- nvidia-smi --query-gpu=memory.total --format=csv` |
| Driver install method | GKE-managed, `gpu-driver-version=latest` | `gcloud container node-pools describe <pool> … --format='yaml(config.accelerators)'` |
| Driver version | 580.173.02 | `kubectl exec <pod> -- nvidia-smi` |
| CUDA version | 13.0 (max supported by driver; runtime CUDA — из образа vLLM) | `kubectl exec <pod> -- nvidia-smi` |
| Taints | `nvidia.com/gpu=present:NoSchedule` (toleration добавляется GKE автоматически) | `kubectl describe node <node> \| grep Taints` |
| Spot / on-demand | on-demand | `kubectl get node <node> -L cloud.google.com/gke-spot` |

### Container

| Параметр | Value | Как получить |
|---|---|---|
| Image | `vllm/vllm-openai:v0.30.0` (Docker Hub, pushed 2026-09-22; build commit `ced6857a`) | `k8s/standalone/vllm-pod.yaml` |
| Image digest | index `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`; linux/amd64 `sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40` (~8.7 GB compressed) | `kubectl get pod <pod> -o jsonpath='{.status.containerStatuses[0].imageID}'` |
| Python | 3.12.3 | `kubectl exec <pod> -- python3 --version` |
| PyTorch | 2.13.0+cu130 | `kubectl exec <pod> -- python3 -c 'import torch; print(torch.__version__)'` |
| vLLM | 0.30.0 (V1 engine, V2 model runner; attention backend FLASH_ATTN, FlashAttention 2) | `kubectl exec <pod> -- python3 -c 'import vllm; print(vllm.__version__)'` |
| CUDA runtime (image) | 13.0.2 (`CUDA_VERSION` in image config; `torch.version.cuda` = 13.0) | image config / `torch.version.cuda` |

---

## Environment C — GPU VM (not used yet)

Используется, только если Environment B мешает экспериментам. Структура таблицы — как в Environment B (GPU, driver, CUDA, Docker, image digest).

---

## vLLM server configuration

Для каждой серии — отдельная копия этой таблицы (или ссылка на `meta.json` серии).

| Аргумент | Value | Примечание |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-7B-Instruct` | positional arg of `vllm serve` |
| `--revision` | `a09a35458c702b33eeacc393d103063234e8bc28` | commit hash |
| `--tokenizer-revision` | `a09a35458c702b33eeacc393d103063234e8bc28` | = model revision |
| `--dtype` | `bfloat16` | explicit |
| `--quantization` | none | |
| `--max-model-len` | 8192 | explicit (native 32768) |
| `--gpu-memory-utilization` | 0.90 | explicit |
| `--max-num-seqs` | 256 | default (resolved for this GPU; см. примечание ниже) |
| `--max-num-batched-tokens` | 2048 | default (resolved; совпадает с `compile_ranges_endpoints: [2048]` в логе) |
| `--enable-prefix-caching` | True | default (лог: `enable_prefix_caching=True`; `vllm:cache_config_info`), hash `sha256` |
| `--block-size` | 16 | default (`vllm:cache_config_info`: `block_size="16"`, `num_gpu_blocks="3356"`) |
| chunked prefill | True | default (лог: `enable_chunked_prefill=True`); scheduler policy `fcfs`, async scheduling on |
| `--seed` | 0 | explicit; sampling для benchmark-запросов задаётся отдельно (`temperature=0`) |
| другие аргументы | — | `kv_cache_dtype=auto` (bf16), `enforce_eager=False` (torch.compile + CUDA graphs `FULL_AND_PIECEWISE`, capture sizes 1…512) |
| Полная команда запуска | `k8s/standalone/vllm-pod.yaml` | `non-default args` в логе = только явные аргументы |

Фактические значения (включая defaults) брать из лога старта vLLM, а не только из manifest.

Примечание (2026-10-17, первый запуск):
- `max_num_seqs` / `max_num_batched_tokens` в лог старта не печатаются. Значения получены разрешением того же конфига внутри пода (`EngineArgs(...).create_engine_config(UsageContext.OPENAI_API_SERVER)`); defaults зависят от GPU. Перед Exp0 задать явно.
- **Sampling defaults переопределены `generation_config.json` модели**: `temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.05` (WARNING в логе). Запрос с `temperature=0` перекрывает только temperature — `repetition_penalty=1.05` остаётся. Для бенчмарков: `--generation-config vllm` или явные sampling-параметры в запросе.
- Первый запрос вызвал JIT-компиляцию Triton-ядра (`_gumbel_sample_kernel`) → всплеск латентности. Ещё одна причина для warmup перед замерами.

Память GPU (лог `gpu_worker`): device 22.03 GiB × 0.90 = 19.83 GiB; weights + non-torch 15.53 GiB, peak activation 1.43 GiB, CUDA graphs 0.24 GiB → KV 2.87 GiB. `nvidia-smi` под нагрузкой 0: 19514 MiB used.

---

## Model

| Параметр | Value | Как получить |
|---|---|---|
| Model id (HF) | `Qwen/Qwen2.5-7B-Instruct` (not gated) | — |
| Revision (commit hash) | `a09a35458c702b33eeacc393d103063234e8bc28` (lastModified 2025-01-12) | `huggingface-cli` / страница модели → Files → commit |
| Tokenizer revision | `a09a35458c702b33eeacc393d103063234e8bc28` | обычно = model revision |
| Parameters | ~7.6B; weights 15.23 GB (bf16, 4 safetensors shards) | model card |
| Architecture | `Qwen2ForCausalLM`: 28 layers, 28 attention heads, 4 KV heads (GQA), hidden 3584, head_dim 128 → KV 56 KiB/token (bf16) | `config.json` |
| Native context length | 32768 (`max_position_embeddings`) | `config.json` → `max_position_embeddings` |
| Weights size in VRAM | 14.29 GiB | лог vLLM: `Loading model weights took … GB` |
| Available KV-cache memory | 2.87 GiB → 53,696 tokens (3356 blocks × 16) | лог vLLM: `Available KV cache memory` / `GPU KV cache size: … tokens` |
| Max concurrency at max_model_len | 6.55× (8192 tokens per request) | лог vLLM: `Maximum concurrency for … tokens per request` |
| License | Apache-2.0 (HF card metadata) | model card |
