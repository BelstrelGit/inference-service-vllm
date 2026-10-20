# GKE cluster

Minimal Standard GKE cluster: one small system node + one autoscaled L4 node pool (0–1 nodes), so the GPU node runs only during experiments.

```text
vllm-lab (zonal control plane in us-central1-a)
├── default-pool   e2-standard-2 × 1            system pods, later Prometheus/Grafana
└── l4-pool        g2-standard-8 + 1 × L4, 0–1  vLLM; driver installed by GKE; any zone of us-central1
```

## Variables

```bash
export PROJECT=lora-sched-stand
export ZONE=us-central1-a
export CLUSTER=vllm-lab
```

## Create

```bash
gcloud container clusters create $CLUSTER \
  --project $PROJECT --zone $ZONE \
  --release-channel regular \
  --machine-type e2-standard-2 --num-nodes 1 \
  --disk-type pd-balanced --disk-size 50 \
  --workload-pool $PROJECT.svc.id.goog

gcloud container node-pools create l4-pool \
  --project $PROJECT --zone $ZONE --cluster $CLUSTER \
  --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
  --disk-type pd-balanced --disk-size 200 \
  --node-locations us-central1-a,us-central1-b,us-central1-c \
  --num-nodes 0 --enable-autoscaling --total-min-nodes 0 --total-max-nodes 1 --location-policy ANY

gcloud container clusters get-credentials $CLUSTER --project $PROJECT --zone $ZONE
```

Notes:

- GKE taints GPU nodes with `nvidia.com/gpu=present:NoSchedule` and automatically adds the toleration to pods that request `nvidia.com/gpu`.
- `disk-size 200`: vLLM image (~10 GB+) and model weights (~16 GB for a 7–8B bf16 model) land on the node disk until a model cache volume is added.
- L4 capacity varies per zone (`GCE out of resources` in pod events). The pool spans all three us-central1 zones and `--location-policy ANY` lets the autoscaler pick whichever has capacity; `--total-max-nodes 1` keeps it within the 1-GPU quota (per-zone `--max-nodes` would allow one node per zone).
- If the whole region is out of L4, probe other zones (create a minimal `g2-standard-4` VM per zone and delete it immediately; a failed attempt costs nothing), then recreate the cluster in a region that has capacity. A zonal cluster cannot have nodes outside its region. On 2026-10-17 `europe-west4` had no L4 in any zone; `us-central1-a` had.
- The GCP project and its 1-GPU quota (`GPUS_ALL_REGIONS = 1`) are shared with other workloads: only one GPU can be held at a time.

## Start / stop the GPU node

The GPU node is started only by the autoscaler: it adds the L4 node when a pod requests a GPU and removes it ~10 min after the node is idle. Do not use `resize --num-nodes 1` to start it: in a multi-zone pool `--num-nodes` is per zone, so it would request one node in every zone.

Force it down (0 is safe in any zone count):

```bash
gcloud container clusters resize $CLUSTER --node-pool l4-pool --num-nodes 0 --project $PROJECT --zone $ZONE --quiet
```

## Pause the whole cluster between sessions

Both pools can be scaled to 0; the cluster and its config stay, nodes stop billing. The control plane of one zonal cluster is covered by the GKE free tier.

```bash
# end of session
gcloud container clusters resize $CLUSTER --node-pool default-pool --num-nodes 0 --project $PROJECT --zone $ZONE --quiet
# start of session
gcloud container clusters resize $CLUSTER --node-pool default-pool --num-nodes 1 --project $PROJECT --zone $ZONE --quiet
```

Rough cost (verify in the GCP pricing calculator): e2-standard-2 ≈ $0.07/h (≈ $55/month if always on), g2-standard-8 + L4 ≈ $0.9/h.

## Experiment session checklist

The GPU is shared with another project (`GPUS_ALL_REGIONS = 1`): before a session check that no other g2 / GPU node is running (`gcloud compute instances list --project $PROJECT`). Load runs execute inside the cluster (`scripts/run-bench.sh`), but the script waits on `kubectl exec` from the laptop: keep the laptop awake (`caffeinate -i` on macOS) or re-attach with `scripts/run-bench.sh --collect`.

Start:

```bash
gcloud container clusters resize $CLUSTER --node-pool default-pool --num-nodes 1 --project $PROJECT --zone $ZONE --quiet
gcloud container clusters get-credentials $CLUSTER --project $PROJECT --zone $ZONE
kubectl apply -f k8s/serving/vllm.yaml            # the GPU request makes the autoscaler add the L4 node
kubectl -n vllm rollout status deploy/vllm --timeout=30m
scripts/check-server.sh 53696                     # baseline: no KV args, bf16 KV, 53,696 tokens
```

Switching the server (Exp4 overlays): one `apply` replaces the pod (strategy Recreate). Check the pool before every scenario.

| Server | Apply | `check-server.sh` | Scenarios |
|---|---|---|---|
| baseline | `kubectl apply -f k8s/serving/vllm.yaml` | `53696` | Exp1–Exp3, `exp4_control` |
| 0.5× pool | `kubectl apply -k k8s/exp4/kv1678` | `26848` | `exp4_kv1678` |
| 0.75× pool | `kubectl apply -k k8s/exp4/kv2517` | `40272` | `exp4_kv2517` |
| FP8, same pool | `kubectl apply -k k8s/exp4/fp8-kv3356` | `53696` | `exp4_fp8_kv3356` |
| FP8, full pool | `kubectl apply -k k8s/exp4/fp8` | no argument: record the size | `exp4_fp8` |

```bash
kubectl apply -k k8s/exp4/kv1678
kubectl -n vllm rollout status deploy/vllm --timeout=30m
scripts/check-server.sh 26848 --save results/raw/l4-qwen7b-bf16-kv1678-v1   # keeps args, pool and backend next to the runs
scripts/run-bench.sh load configs/scenarios/exp4_kv1678.yaml
```

The order of runs and the pilot commands are in each experiment doc (`docs/experiments/EXP_00X_*.md`, Pilot / Setup).

End:

```bash
kubectl delete -f k8s/serving/vllm.yaml           # removes the Deployment of any variant
gcloud container clusters resize $CLUSTER --node-pool l4-pool --num-nodes 0 --project $PROJECT --zone $ZONE --quiet
gcloud container clusters resize $CLUSTER --node-pool default-pool --num-nodes 0 --project $PROJECT --zone $ZONE --quiet
gcloud compute instances list --project $PROJECT  # expect no vllm-lab nodes
gcloud compute disks list --project $PROJECT      # expect no leftover node disks
```

## Verify GPU

```bash
kubectl get nodes -L cloud.google.com/gke-accelerator,node.kubernetes.io/instance-type
kubectl apply -f k8s/debug/nvidia-smi.yaml
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/nvidia-smi --timeout=15m
kubectl logs nvidia-smi
kubectl delete -f k8s/debug/nvidia-smi.yaml
```

## Delete

```bash
gcloud container clusters delete $CLUSTER --project $PROJECT --zone $ZONE
```
