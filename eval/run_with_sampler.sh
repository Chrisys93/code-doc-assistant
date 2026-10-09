#!/usr/bin/env bash
# Run ONE evaluation configuration unattended, with the host sampler around it, and log everything to MLflow.
#
#   eval/run_with_sampler.sh <config> [group] [-- extra run_eval.py args]
#   eval/run_with_sampler.sh llamacpp-heavy heavy-001
#   eval/run_with_sampler.sh llamacpp-heavy heavy-002 -- --stage A --only Q1,Q2
#
# What it does (all from the WSL2 shell, where nvidia-smi and kubectl work):
#   1. starts host_sampler.py (GPU / host RAM / pod CPU+memory, 1 sample per second) and lets it record an idle baseline
#   2. starts run_eval.py INSIDE the app pod under nohup, so a dropped `kubectl exec` / closed terminal cannot kill it
#   3. polls until the run signals completion (a .done marker in the pod), then records a short idle tail
#   4. stops the sampler (always, also on error or Ctrl+C), copies the results out of the pod
#   5. runs join_resources.py --mlflow: res_* metrics on every question run, peaks/means + the raw samples
#      on the parent run
# To walk away: run it under tmux or `nohup eval/run_with_sampler.sh ... > run.out 2>&1 &`.
#
# Environment: NS (code-doc)  SETTLE (30 s idle before)  COOLDOWN (20 s idle after)  POLL (30 s)
#              EXPERIMENT (code-doc-assistant-eval)  PYTHON (python)  MLFLOW_TRACKING_URI (default: a port-forward to the in-cluster MLflow)
#              JOIN_MLFLOW (1)  POD (default: first app pod)
set -uo pipefail

if [[ $# -lt 1 || "$1" == -h || "$1" == --help ]]; then sed -n 2,22p "$0" | sed 's/^# \{0,1\}//'; exit 2; fi
CONFIG="$1"; shift
GROUP="${CONFIG}-$(date +%Y%m%d-%H%M)"
if [[ $# -gt 0 && "$1" != "--" ]]; then GROUP="$1"; shift; fi
[[ "${1:-}" == "--" ]] && shift
EXTRA=""; for a in "$@"; do EXTRA+="$(printf '%q ' "$a")"; done

NS="${NS:-code-doc}"; SETTLE="${SETTLE:-30}"; COOLDOWN="${COOLDOWN:-20}"; POLL="${POLL:-30}"
EXPERIMENT="${EXPERIMENT:-code-doc-assistant-eval}"
PYTHON="${PYTHON:-python}"; JOIN_MLFLOW="${JOIN_MLFLOW:-1}"
REMOTE_ROOT="${REMOTE_ROOT:-/app/eval/results}"; RUN_EVAL="${RUN_EVAL:-/app/eval/run_eval.py}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="eval/results/$GROUP"; SAMPLES="$OUT/samples-$CONFIG.csv"; REMOTE="$REMOTE_ROOT/$GROUP"
mkdir -p "$OUT"

POD="${POD:-$(kubectl get pods -n "$NS" -o name | grep -- '-app-' | head -1 | cut -d/ -f2)}"
[[ -n "$POD" ]] || { echo "no app pod found in namespace $NS"; exit 1; }
echo "[$(date +%T)] config=$CONFIG group=$GROUP pod=$POD namespace=$NS experiment=$EXPERIMENT"

SAMPLER_PID=""; PF_PID=""
stop_sampler() {
  if [[ -n "$SAMPLER_PID" ]] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
    echo "[$(date +%T)] sampler stopped"
  fi
}
cleanup() { stop_sampler; [[ -n "$PF_PID" ]] && kill "$PF_PID" 2>/dev/null; }
trap cleanup EXIT
trap 'echo; echo "interrupted - stopping the sampler (the evaluation keeps running in the pod: $REMOTE)"; exit 130' INT TERM

# 1. sampler + idle baseline
"$PYTHON" eval/host_sampler.py --out "$SAMPLES" --k8s-namespace "$NS" > "$OUT/sampler.log" 2>&1 &
SAMPLER_PID=$!
sleep 2
kill -0 "$SAMPLER_PID" 2>/dev/null || { echo "sampler failed to start:"; cat "$OUT/sampler.log"; exit 1; }
grep -q "nvidia-smi not found" "$OUT/sampler.log" && echo "WARNING: nvidia-smi not found - no GPU columns (is this the WSL2 shell?)"
echo "[$(date +%T)] sampler running (pid $SAMPLER_PID); idle baseline for ${SETTLE}s"
sleep "$SETTLE"

# 2. start the evaluation in the pod, detached
kubectl exec -n "$NS" "$POD" -- sh -c "mkdir -p '$REMOTE' && rm -f '$REMOTE/.done' '$REMOTE/.exit' && \
  ( env MLFLOW_EXPERIMENT='$EXPERIMENT' python '$RUN_EVAL' --config '$CONFIG' --group '$GROUP' $EXTRA > '$REMOTE/run.log' 2>&1; \
    echo \$? > '$REMOTE/.exit'; touch '$REMOTE/.done' ) > /dev/null 2>&1 &" \
  || { echo "could not start the evaluation in the pod"; exit 1; }
echo "[$(date +%T)] evaluation started in the pod (log: $REMOTE/run.log)"

# 3. wait for completion
fails=0; n=0
while true; do
  if kubectl exec -n "$NS" "$POD" -- test -f "$REMOTE/.done" 2>/dev/null; then break; fi
  if ! kubectl get pod -n "$NS" "$POD" > /dev/null 2>&1; then
    fails=$((fails+1))
    if (( fails >= 10 )); then echo "pod $POD is gone - giving up (results inside it are lost if it was replaced)"; exit 1; fi
  else fails=0; fi
  n=$((n+1))
  if (( n % 4 == 0 )); then
    last="$(kubectl exec -n "$NS" "$POD" -- sh -c "grep -E '^  (WARMUP|Q[0-9])|cold start' '$REMOTE/run.log' | tail -1" 2>/dev/null)"
    echo "[$(date +%T)] running... ${last:-}"
  fi
  sleep "$POLL"
done
EXIT_CODE="$(kubectl exec -n "$NS" "$POD" -- cat "$REMOTE/.exit" 2>/dev/null || echo '?')"
echo "[$(date +%T)] evaluation finished (exit code $EXIT_CODE); idle tail ${COOLDOWN}s"
sleep "$COOLDOWN"

# 4. stop the sampler, fetch the results
stop_sampler
kubectl exec -n "$NS" "$POD" -- tar cf - -C "$REMOTE_ROOT" "$GROUP" | tar xf - -C eval/results \
  || { echo "could not copy the results out of the pod"; exit 1; }
echo "[$(date +%T)] results copied to $OUT"
tail -n 12 "$OUT/run.log" 2>/dev/null

# 5. join the resources and log to MLflow
FLAGS=""
if [[ "$JOIN_MLFLOW" == "1" ]]; then
  FLAGS="--mlflow"
  if [[ -z "${MLFLOW_TRACKING_URI:-}" ]]; then
    kubectl port-forward -n "$NS" svc/cda-code-doc-assistant-mlflow 5000:5000 > "$OUT/portforward.log" 2>&1 &
    PF_PID=$!; export MLFLOW_TRACKING_URI="http://localhost:5000"
    for _ in $(seq 1 15); do curl -fs "$MLFLOW_TRACKING_URI/health" > /dev/null 2>&1 && break; sleep 1; done
  fi
fi
shopt -s nullglob
joined=0
for d in "$OUT"/*/; do
  [[ -f "$d/summary.json" ]] || continue
  "$PYTHON" eval/join_resources.py --samples "$SAMPLES" --results "${d%/}" $FLAGS && joined=1
done
(( joined )) || echo "nothing joined: no <config>/summary.json found under $OUT"
echo "[$(date +%T)] done. Results: $OUT   (eval exit code $EXIT_CODE)"
[[ "$EXIT_CODE" == "0" ]] || exit 1
