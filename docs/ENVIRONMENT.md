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

Наблюдение: scale-up L4-ноды 0 → 1 вместе с установкой драйвера до запуска пода ≈ 1.5 мин (TriggeredScaleUp 06:50:33 → Scheduled 06:51:57 UTC). В `europe-west4` (все 3 зоны) L4 в тот день не было (`GCE out of resources`).

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
| Image | TBD (`vllm/vllm-openai:<tag>`) | manifest |
| Image digest | TBD | `kubectl get pod <pod> -o jsonpath='{.status.containerStatuses[0].imageID}'` |
| Python | TBD | `kubectl exec <pod> -- python3 --version` |
| PyTorch | TBD | `kubectl exec <pod> -- python3 -c 'import torch; print(torch.__version__)'` |
| vLLM | TBD | `kubectl exec <pod> -- python3 -c 'import vllm; print(vllm.__version__)'` |

---

## Environment C — GPU VM (not used yet)

Используется, только если Environment B мешает экспериментам. Структура таблицы — как в Environment B (GPU, driver, CUDA, Docker, image digest).

---

## vLLM server configuration

Для каждой серии — отдельная копия этой таблицы (или ссылка на `meta.json` серии).

| Аргумент | Value | Примечание |
|---|---|---|
| `--model` | TBD | |
| `--revision` | TBD | commit hash |
| `--tokenizer-revision` | TBD | |
| `--dtype` | TBD | на L4: `bfloat16` |
| `--quantization` | TBD | none? |
| `--max-model-len` | TBD | |
| `--gpu-memory-utilization` | TBD | default зависит от версии — записать effective value |
| `--max-num-seqs` | TBD | |
| `--max-num-batched-tokens` | TBD | |
| `--enable-prefix-caching` | TBD | default зависит от версии и модели — записать effective value |
| `--block-size` | TBD | effective KV block size (PagedAttention) |
| chunked prefill | TBD | |
| `--seed` | TBD | |
| другие аргументы | TBD | |
| Полная команда запуска | TBD | скопировать из manifest / лога |

Фактические значения (включая defaults) брать из лога старта vLLM, а не только из manifest.

---

## Model

| Параметр | Value | Как получить |
|---|---|---|
| Model id (HF) | TBD | — |
| Revision (commit hash) | TBD | `huggingface-cli` / страница модели → Files → commit |
| Tokenizer revision | TBD | обычно = model revision |
| Parameters | TBD | model card |
| Architecture | TBD | `config.json` |
| Native context length | TBD | `config.json` → `max_position_embeddings` |
| Weights size in VRAM | TBD | лог vLLM: `Loading model weights took … GB` |
| Available KV-cache memory | TBD | лог vLLM: `Available KV cache memory` / `GPU KV cache size: … tokens` |
| Max concurrency at max_model_len | TBD | лог vLLM: `Maximum concurrency for … tokens per request` |
| License | TBD | model card |
