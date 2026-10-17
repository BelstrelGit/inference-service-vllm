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

Основное окружение для benchmark-серий.

### Cluster

| Параметр | Value | Как получить |
|---|---|---|
| Project | TBD | `gcloud config get-value project` |
| Cluster name | TBD | `gcloud container clusters list` |
| Location (zone/region) | TBD | `gcloud container clusters list` |
| Mode (Standard / Autopilot) | TBD | `gcloud container clusters describe <cluster> --location <loc> --format='value(autopilot.enabled)'` |
| GKE version | TBD | `kubectl version` / `gcloud container clusters describe … --format='value(currentMasterVersion)'` |

### GPU node pool

| Параметр | Value | Как получить |
|---|---|---|
| Node pool name | TBD | `gcloud container node-pools list --cluster <cluster> --location <loc>` |
| Machine type | TBD (L4 → `g2-standard-*`) | `kubectl get node <node> -L node.kubernetes.io/instance-type` |
| GPU type | TBD (кандидат: **nvidia-l4**) | `kubectl get nodes -L cloud.google.com/gke-accelerator` |
| GPU count per node | TBD | `kubectl describe node <node> \| grep nvidia.com/gpu` |
| VRAM | TBD | `kubectl exec <pod> -- nvidia-smi --query-gpu=memory.total --format=csv` |
| Driver install method | TBD (GKE-managed `gpu-driver-version=default/latest` или DaemonSet) | `gcloud container node-pools describe <pool> … --format='yaml(config.accelerators)'` |
| Driver version | TBD | `kubectl exec <pod> -- nvidia-smi` |
| CUDA version | TBD | `kubectl exec <pod> -- nvidia-smi` |
| Taints | TBD | `kubectl describe node <node> \| grep Taints` |
| Spot / on-demand | TBD | `kubectl get node <node> -L cloud.google.com/gke-spot` |

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

Используется, только если Environment B мешает экспериментам (см. план, раздел 4). Структура таблицы — как в Environment B (GPU, driver, CUDA, Docker, image digest).

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
