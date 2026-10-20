# Monitoring

Prometheus + Grafana (kube-prometheus-stack) and NVIDIA dcgm-exporter for `vllm-lab`. Design, metric choice and dashboard: [docs/MONITORING.md](../docs/MONITORING.md).

Install only after the performance experiments of the session are done: monitoring adds a second `/metrics` scraper and new pods, and the experiments compare against runs made without it.

```text
monitoring/
├── prometheus/values.yaml               kube-prometheus-stack: placement, resources, retention, PVC
├── prometheus/service-monitor-vllm.yaml vLLM Service, 5 s
├── dcgm/                                dcgm-exporter DaemonSet + PodMonitor (kustomize), counters.csv
└── grafana/                             dashboards as ConfigMaps (kustomize)
```

Variables as in [k8s/cluster/README.md](../k8s/cluster/README.md) (`PROJECT`, `ZONE`, `CLUSTER`).

## Once: node pool for observability

```bash
gcloud container node-pools create obs-pool \
  --project $PROJECT --zone $ZONE --cluster $CLUSTER \
  --machine-type e2-standard-2 --disk-type pd-balanced --disk-size 30 \
  --node-locations $ZONE --num-nodes 0 \
  --node-labels workload=observability --node-taints workload=observability:NoSchedule
```

- One zone: the Prometheus PVC is a zonal pd-balanced disk. It is created in the zone of the node Prometheus first runs on, and afterwards Prometheus can only run in that zone.
- No autoscaling, sized by hand like `default-pool`: with autoscaling, the pending Prometheus and Grafana pods would bring the node back after every `resize --num-nodes 0`.

## Install

```bash
gcloud container clusters resize $CLUSTER --node-pool obs-pool --num-nodes 1 --project $PROJECT --zone $ZONE --quiet

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace monitoring
kubectl -n monitoring create secret generic grafana-admin \
  --from-literal=admin-user=admin --from-literal=admin-password="$(openssl rand -base64 18)"
helm install kps prometheus-community/kube-prometheus-stack --version 92.11.0 \
  -n monitoring -f monitoring/prometheus/values.yaml

kubectl apply -f monitoring/prometheus/service-monitor-vllm.yaml
kubectl apply -k monitoring/dcgm      # pods appear only while the L4 node exists
kubectl apply -k monitoring/grafana
```

Chart 92.11.0 (Prometheus v3.15.0, operator v0.94.1, Grafana 13.2.4, node-exporter v1.12.1): checked with `helm template` against these values on 2026-10-20, not installed yet. dcgm-exporter `4.5.2-4.8.1-ubuntu22.04`.

## Check

```bash
kubectl -n monitoring get pods -o wide            # Prometheus/Grafana/operator/KSM on obs-pool
kubectl -n monitoring get ds                      # node-exporter DESIRED = number of nodes, incl. the L4 node
kubectl -n monitoring logs ds/dcgm-exporter | head
kubectl -n monitoring port-forward svc/kps-kube-prometheus-stack-prometheus 9090 &
kubectl -n monitoring port-forward svc/kps-grafana 3000:80 &
kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.admin-password}' | base64 -d; echo
```

1. Prometheus, Status → Targets: `vllm`, `dcgm-exporter`, node-exporter on **every** node (the L4 node too: its `nvidia.com/gpu` taint must be tolerated), kube-state-metrics, kubelet are `up`.
2. Explore: `vllm:kv_cache_usage_perc` and `DCGM_FI_DEV_FB_USED` return data; the "vLLM server variant" panel shows the same K as `scripts/check-server.sh`.
3. `DCGM_FI_PROF_*`: present or not on L4 (`curl localhost:9400/metrics` inside the exporter pod). If absent, record it in docs/MONITORING.md.
4. Under load: dashboard `vLLM + GPU overview` against the client's `levels.csv` (docs/MONITORING.md, "Проверка после установки").

Service names (`kps-...`) follow the Helm release name `kps`; check with `kubectl -n monitoring get svc`.

## If dcgm-exporter cannot see the GPU

Switch to the two-DaemonSet layout from Google's GKE example: see the header of `monitoring/dcgm/hostengine.yaml`.

## End of session

```bash
gcloud container clusters resize $CLUSTER --node-pool obs-pool --num-nodes 0 --project $PROJECT --zone $ZONE --quiet
```

The Helm release, its config and the Prometheus PVC stay; Prometheus resumes with its data when the pool is resized to 1. `gcloud compute disks list` then shows one disk for the PVC (~$1/month).

## Remove

```bash
kubectl delete -k monitoring/grafana
kubectl delete -k monitoring/dcgm
kubectl delete -f monitoring/prometheus/service-monitor-vllm.yaml
helm uninstall kps -n monitoring
kubectl -n monitoring delete pvc --all            # deletes the Prometheus disk
kubectl delete namespace monitoring
gcloud container node-pools delete obs-pool --project $PROJECT --zone $ZONE --cluster $CLUSTER
```

The CRDs of the operator stay after `helm uninstall` (`kubectl get crd | grep monitoring.coreos.com`).
