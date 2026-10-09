# Evaluation

Two levels of evaluation, both recorded in MLflow:

| Level | Question it answers | Tool |
| --- | --- | --- |
| **Answer quality** | Does the pipeline give correct, grounded answers on known repos? | `run_eval.py` (+ optional `judge.py`) |
| **Serving / resources** | How fast and how heavy is each backend and model? | `bench.py`, `host_sampler.py`, `join_resources.py` |

## Files

| File | Role |
| --- | --- |
| `questions.json` | The question set: stages, reference notes, automatic checks, grading rules |
| `configs.json` | Named configurations (backend, tier, env overrides) |
| `run_eval.py` | Headless runner: indexes the stage's repos, asks the questions, records parent + child MLflow runs |
| `checks.py` | Automatic checks (citations, key facts, file recall) |
| `bench.py` | Raw serving benchmark for any OpenAI-compatible endpoint (TTFT, prefill, decode) |
| `host_sampler.py` | GPU / host RAM / pod CPU+memory sampler, run on the host |
| `join_resources.py` | Attaches the samples to each question's time window |
| `judge.py` | Optional LLM-as-judge over saved results (blind to backend/model) |
| `test_eval.py` | Self-test with fakes; needs no GPU, Ollama or Chroma |

## The question set

Stage A indexes `code-doc-assistant` and `IcarusRepoSEND`; stage B adds `chapter3` (an unrelated Stata
codebase) and re-asks Q1, Q4 and Q10 as **dilution probes**: if their scores or sources change after chapter3 is
added, retrieval is being diluted by unrelated chunks (every question searches all active collections).

Grading: each question has automatic checks (`checks.py`) and, except Q6, a manual 0-2 score. Q6 is
recorded and compared, not graded. Put the manual score in the question's result JSON as `manual_score`.

## Configurations (`configs.json`)

| Config | Status | How to bring the stack to it |
| --- | --- | --- |
| `llamacpp-heavy` | ready | `helm upgrade cda ... -f values-wsl2-minikube.yaml --set app.image.tag=<tag> --set modelTier=heavy` (DeepSeek-Coder-V2-Lite, ctx 4096, 24 GPU layers) |
| `llamacpp-full` | ready | same with `--set modelTier=full` (Mistral-Nemo 12B, ctx 8192, all layers on GPU; first start downloads the GGUF) |
| `ollama-heavy`, `ollama-full` | ready on GPU | add `-f values-ollama-gpu.yaml` to the llama.cpp command (turns llama.cpp off, gives Ollama the GPU). Without the overlay the Ollama pod is CPU-only. Check `ollama_vram_fraction` in the cold-start line: ~1.0 = fully on GPU, 0.0 = CPU |
| `strata-heavy` | **not implemented** | placeholder; needs a `strata` backend in the app (OpenAI-compatible) and a Strata server the cluster can reach |
| `llamacpp-heavy-topk5` | ready | same server as `llamacpp-heavy`, `TOP_K=5` |
| `vllm` | placeholder | |

llama-server serves only the GGUF it was launched with, so the runner **refuses** a `llamacpp-*` config when the
server's model belongs to another tier (override: `--allow-tier-mismatch`). Run one GPU backend at a time: the 12 GB
card cannot hold two models, so scale the other backend down first.

**Compare within a tier only** (`llamacpp-heavy` vs `ollama-heavy`, `llamacpp-full` vs `ollama-full`). Never rank
heavy against full: different models, context sizes and GPU offload make that comparison unfair. Both backends
use the app's per-tier `num_ctx`, so context is equal within a tier.

The tiers also differ in context size (heavy 4096, full 8192). That is a real part of each tier's default, but it is a
confound when comparing models: for a like-for-like model comparison set the same `llamacpp.contextSize` on both.

## Cold start

The first question after a deploy pays the model load. The runner asks a fixed neutral question first and reports it as
the cold start: `WARMUP.json`, a row in `summary.csv`, `cold_start_*` metrics on the parent run and its own child run
(`eval.warmup=true`). It is not part of the latency statistics, but it is part of the run's record.

## Running it

Run the runner **inside the app pod** (it reaches Chroma, the model server and MLflow by their service names).
From the WSL shell, one command does the whole measurement unattended:

```
eval/run_with_sampler.sh llamacpp-heavy heavy-001
eval/run_with_sampler.sh llamacpp-heavy heavy-002 -- --stage A --only Q1,Q2     # extra run_eval.py args after --
```

It starts the host sampler (and records an idle baseline), launches `run_eval.py` in the pod under `nohup` (a dropped
`kubectl exec` or closed terminal cannot kill it), polls until it finishes, records an idle tail, **stops the sampler
itself**, copies the results out and runs `join_resources.py --mlflow`. To walk away, run it under `tmux` or
`nohup eval/run_with_sampler.sh ... > run.out 2>&1 &`. Settings (environment): `NS`, `SETTLE`, `COOLDOWN`, `POLL`,
`EXPERIMENT` (default `code-doc-assistant-eval`: eval runs are kept apart from the UI's runs), `PYTHON`, `MLFLOW_TRACKING_URI` (default: a port-forward to the in-cluster MLflow), `JOIN_MLFLOW=0` to skip the upload.

**MLflow artifacts need the proxied artifact root** (`mlflow.serveArtifacts`, the chart default since this change). With
a plain-path root every client writes artifacts to its own disk and the UI's Artifacts tab stays empty. An experiment
that already exists keeps the location it was created with, which is why the eval uses a fresh experiment.

Where things end up in MLflow:

* **question (child) runs**: latency per node (`lat_*_ms`), `gen_ms`, `eval_wall_ms`, the automatic checks, then
  `res_*` (GPU / VRAM / power / host RAM during that question, and `*_peak_over_idle`) after the join. Artifacts:
  `result.json` (the answer, sources, plan) and `eval_checks.json`. The warm-up has its own child run.
* **parent run**: summary latency statistics, cold start, stage index times, and after the join the run-level
  `res_*` peaks / means, `res_idle_*`, plus the raw `samples-<config>.csv` and `resources.csv` under `resources/`.
* The sampler runs on the host on purpose: the pods cannot see the GPU. The `res_*` metrics are written after the
  evaluation ends, so they appear a few minutes after the last question.

Manual steps, if you want them (the script does exactly this):

```
python eval/host_sampler.py --out eval/results/samples.csv --k8s-namespace code-doc      # in WSL, ~30 s before
kubectl exec -n code-doc deploy/cda-code-doc-assistant-app -- \
    python /app/eval/run_eval.py --config llamacpp-heavy --stage all --group heavy-001
# Ctrl+C the sampler, then:
kubectl exec -n code-doc <app-pod> -- tar cf - -C /app/eval/results heavy-001 | tar xf - -C eval/results
MLFLOW_TRACKING_URI=http://localhost:5000 \
python eval/join_resources.py --samples eval/results/samples.csv --results eval/results/heavy-001/llamacpp-heavy --mlflow
```

To read an answer straight from the pod (note the `-i`: without it a heredoc is not passed to `kubectl exec`, and
the command prints nothing):

```
kubectl exec -i -n code-doc deploy/cda-code-doc-assistant-app -- python - <<'PY'
import json
d = json.load(open("/app/eval/results/<group>/<config>/Q1.json"))     # WARMUP.json for the cold-start question
print(d["checks"]); print(d["answer"]); print(d["sources"])
PY
```

Useful options: `--stage A|B|all`, `--only Q1,Q4`, `--repeat 3` (latency statistics), `--no-warmup`, `--dry-run`.
The first question after a model load pays the load time, so one warm-up question is asked first; its MLflow run is
tagged `eval.warmup=true` and is excluded from every statistic.

Serving benchmark (the model server alone, synthetic prompts of several sizes):

```
kubectl exec -n code-doc deploy/cda-code-doc-assistant-app -- \
    sh -c 'python /app/eval/bench.py --base-url "$LLAMACPP_HOST/v1" --label heavy-ctx4096 --prompt-tokens 256,1024,3000'
```

Judge (optional, external API; only for repos you may share):

```
ANTHROPIC_API_KEY=... python eval/judge.py eval/results/heavy-001/llamacpp-heavy --calibrate
```

## What to compare in MLflow

* Parent runs (`tags.run.kind = 'eval_parent'`): one per configuration: `auto_pass_rate`, `wall_ms_p50/p95`,
  `stage_B_index_ms`, and every setting as a parameter (`rt.model`, `rt.n_ctx`, `top_k`, ...).
* Child runs (`tags.run.kind = 'eval_question'`): per-node latency (`lat_<node>_ms`), `gen_ms`, the automatic checks,
  `res_*` resource metrics after joining, `judge_score` after judging. Group by `tags.eval.question` to compare a
  question across configurations.
* Bench runs (`tags.run.kind = 'bench'`): `p<size>_ttft_s_median`, `_prefill_tps_median`, `_decode_tps_median`.

## Record with every configuration

Quantisation, `--ctx-size`, GPU layers and Flash Attention are launch settings of the server. The runner records
what the server reports (`rt.model`, `rt.n_ctx`, `rt.gguf`, slots) and the chart/env overrides in `configs.json`;
write the rest into `--note` ("ctx 4096, 24 GPU layers, FA off") so a run is reproducible from MLflow alone.

## Prompt budget (all configurations)

Retrieved context gets 60% of the model's real window, converted to characters at `CHARS_PER_TOKEN` (default 3.0,
code is roughly 3 chars/token), and generation is capped (`LLAMACPP_MAX_TOKENS`, default 1024). For llamacpp the
window is the smaller of what the server reports (`GET /props`) and the `--ctx-size` the chart launched it with
(`LLAMACPP_CTX`); the `context_assembly` trace line names the source (`[server+config]`, `[config]`, ...). If it
ever shows `tier-table-UNVERIFIED` the budget is a guess and the question gets a warning. Smoke run 2 failed this
way: the server held 4096 but the app budgeted for the tier table's 8192 (prompt 4217 > 4096). Keep these values
the same in every configuration you compare; they are part of the setup, not the backend.

## Known limits

* `host_sampler.py` sees the whole machine's GPU, not one process; run nothing else on the GPU during a measurement.
* Retrieved chunks do not carry their repository, so the repo mix in the context is not measured; the dilution
  probes compare answers and cited files instead.
* One configuration per process: most settings are read at import time.
