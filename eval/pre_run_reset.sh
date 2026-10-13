#!/usr/bin/env bash
# pre_run_reset.sh -- put the stack back to a clean memory baseline before an evaluation run.
#
# Why: pods' memory, the WSL page cache and (found in heavy-006) the WSL swap all stay occupied between runs.
# heavy-006 started with 1.9 GB already in swap and filled all 4 GB of it by the 20th question minute; major page
# faults then ran at 200-300/s and a 90 s answer took 450-4000 s. A run starts from a clean baseline or its latency
# numbers are not comparable with the others.
#
# Run it interactively (it uses sudo to drop the page cache and clear swap), then start the evaluation:
#     eval/pre_run_reset.sh
#     eval/run_with_sampler.sh llamacpp-heavy <group> -- --note "..."
# The llama.cpp pod reloads its model on the first request (cold start, logged separately by the runner).
set -euo pipefail
NS="${1:-code-doc}"

echo "== memory before"; free -m; swapon --show || true
echo "== restarting every Deployment and StatefulSet in namespace $NS"
kubectl rollout restart deployment -n "$NS"
kubectl rollout restart statefulset -n "$NS"
for r in $(kubectl get deployment,statefulset -n "$NS" -o name); do
    kubectl rollout status "$r" -n "$NS" --timeout=900s
done

echo "== dropping the WSL page cache (sudo)"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

echo "== clearing swap (sudo): pages swapped out by earlier runs would otherwise be faulted back during this one"
for dev in $(swapon --noheadings --show=NAME); do
    sudo swapoff "$dev" && sudo swapon "$dev" || echo "WARNING: could not recycle swap on $dev"
done
sleep 20

echo "== waiting for MLflow to answer quickly (the runner aborts when a request takes more than 5 s)"
APP="$(kubectl get deployment -n "$NS" -o name | grep -- '-app$' | head -1)"
mlflow_ok=0
for i in $(seq 1 60); do
    if kubectl exec -i -n "$NS" "$APP" -- python - <<'PY' 2>/dev/null
import time, mlflow
fast = 0
for _ in range(3):
    t = time.time()
    mlflow.search_experiments()
    fast += (time.time() - t) < 2
raise SystemExit(0 if fast == 3 else 1)
PY
    then echo "MLflow answers quickly"; mlflow_ok=1; break; fi
    echo "  MLflow not ready ($i/60)"; sleep 10
done

echo "== memory after (wanted: ~4-5 GB used, >6 GB available, swap used under ~200 MB)"
free -m; swapon --show || true
echo "== pods"; kubectl get pods -n "$NS"

swap_used=$(free -m | awk '/^Swap:/ {print $3}')
avail=$(free -m | awk '/^Mem:/ {print $7}')
ok=1
[ "$mlflow_ok" = "1" ] || { echo "NOT READY: MLflow did not answer quickly"; ok=0; }
[ "${swap_used:-0}" -le 300 ] || { echo "NOT READY: ${swap_used} MB of swap still in use"; ok=0; }
[ "${avail:-0}" -ge 6000 ] || { echo "NOT READY: only ${avail} MB available"; ok=0; }
[ "$ok" = "1" ] && echo "READY: start the run" || exit 1
