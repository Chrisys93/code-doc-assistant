# Future Directions — Identified, Not Yet Actioned

Running notes from the `dev`-branch llama.cpp integration session. Nothing here has been implemented — this is a capture of ideas and problems worth returning to deliberately, each in its own right-sized session.

---

## llama.cpp — repo contribution candidates

1. **Documentation gap**: `--served-model-name` (vLLM's flag) vs `-a`/`--alias` (llama-server's actual equivalent). We personally lost real time to this exact confusion. A small, low-risk doc PR (`docs/docker.md` or server README) explicitly calling this out for anyone migrating from vLLM would be a realistic first contribution — grounded in firsthand pain, not scraped.

2. **Issue #22364** (open, ggml-org/llama.cpp): `--models-preset` creates a phantom `"default"` model entry in the `/models` API that clients can't actually use. Bigger lift than the doc fix — requires reading the actual C++ preset/router implementation. Worth attempting once ready to go deeper into the codebase.

3. **`--n-cpu-moe` vs our current `-ngl`-based `heavy`-tier config**: our current `heavy` tier (DeepSeek-Coder-V2-Lite) partial-offloaded via `--n-gpu-layers 24` when this was written (the 2026-10-04 runs logged ngl=28), cutting whole transformer layers (including attention) to CPU. `--n-cpu-moe N` is purpose-built for MoE models — moves only the sparse, rarely-touched expert feed-forward weights to CPU RAM while keeping attention/KV-cache/embeddings on GPU. Likely a materially better default. Test current config first, then upgrade to `--n-cpu-moe` as an isolated, separately-verified change.

4. **Dynamic expert placement** — a genuine, buildable feature proposal: `--n-cpu-moe` today is static (set once at launch, tuned by manual sweep). A session-adaptive version — adjusting which experts stay GPU-resident based on observed routing frequency at runtime — would be a real contribution, and is structurally the same idea as the LRU multi-resident-model logic already built for the Ollama backend this session, one level deeper into the model itself.

5. **OpenVINO NPU backend** (`ggml/src/ggml-openvino`, upstream, official, work-in-progress): runs the same GGUF workflow across Intel CPU/GPU/NPU via `GGML_OPENVINO_DEVICE`. Its own docs list concrete, currently-open limitations worth treating as a ready-made contribution list:
   - No model caching yet
   - `-np > 1` (multiple parallel sequences) unsupported on NPU
   - NPU runs in stateless-only mode
   - Accuracy validation incomplete for several quant formats
   - Directly testable on our own hardware if the MSI's integrated NPU is Intel-based (Core Ultra / "AI Boost") — worth confirming the exact CPU model before starting.

---

## Research direction — MoE expert placement as a third VAOS substrate

Real, non-coincidental connection identified this session: MoE expert placement (which experts stay GPU-resident, PCIe transaction monitoring/nature, dynamic data-distribution strategy) is structurally the same problem VAOS already formalizes for LLM serving orchestration, and ReStorEdge/SEND formalized before that for edge storage.

- **Expert "hotness"** (frequently-activated, worth keeping resident) = the same locality question as LSH-bucketed reuse in ReStorEdge.
- **Predicting which experts the next token needs**, to prefetch across PCIe before they're required rather than reactively = GAUSS's queueing-prediction problem, applied to expert-routing instead of request-routing.
- **"Which representation lives where, for how long, with what validity"** (SEND's core reframing) is literally the expert-placement question, at a smaller substrate.

This is the same research programme (thesis → SEND/EDR → ReStorEdge → VAOS) finding a third substrate, not a new direction from scratch. Candidate outcomes once we're in a dedicated research session (with Spyros in the loop):
- Its own position paper, OR
- A section/extension feeding into the existing VAOS paper, OR
- A subsection of `RESEARCH_orchestrated.md` in the research repo, as groundwork before either of the above.

Decide the shape once actually back in that context — not scoped further here.

---

## MLflow / "Snowflake-like platform" — repo contribution / dev direction

Origin, for the record (corrected from an earlier misattribution): this traces back to the user's own observation in an earlier session — noting MLflow's evaluation-run tracking resembles Snowflake's evaluation GUI/API, initially raised as a demo/interview-prep framing device, not a platform-building proposal. The platform idea itself is new, first raised this session.

Three framings worth keeping on record, primary noted first:

1. **[Primary]** A multi-tenant, queryable experiment-tracking backend — MLflow's own registry, scaled up. The most directly buildable of the three; closest to MLflow's actual architecture today.
2. A full data-warehouse-style storage/compute separation for the artifacts MLflow generates — bigger infrastructure lift, Snowflake's actual defining architectural trait.
3. Snowflake's specific eval/observability tooling (Cortex) reimplemented as a feature set on top of MLflow.

Target repo: [`mlflow/mlflow`](https://github.com/mlflow/mlflow) — confirmed, not ambiguous.

Not scoped further — worth its own dedicated session once ready, likely after the llama.cpp / heavy-tier / NPU work settles.

---

## Backend × model evaluation in MLflow — measure what deployment does to results

Origin: the heavy-tier bring-up (DeepSeek-Coder-V2-Lite-Instruct Q4_K_M on llama.cpp, 12GB laptop GPU) showed that the *same weights* behave very differently depending on how they are served, in latency, resource use and planning behaviour. Observed on one machine in one evening: generation anywhere from ~190 tok/s down to ~0.25 tok/s on the same model and settings (cause not established — it appeared after use and cleared on a server restart), and a context window of 4096 on the server against 8192 assumed by the app.

Idea: use MLflow as the comparison harness. ARCHITECTURE already records `inference_backend` and `model_name` per run (MLflow is system-level observability, not tied to a backend), so the missing pieces are the metrics and a fixed workload:
- **Matrix:** backend (ollama / llamacpp / vllm) × tier/model × placement (`--n-gpu-layers`, `--n-cpu-moe`, `--ctx-size`, quantisation).
- **Fixed workload:** a small set of queries run through the headless graph (`smoke_retrieval.py` is the seed for this), retrieval held constant.
- **Metrics:** planner latency, generation tok/s (from the server's own timings), peak VRAM/RAM, tool-call validity (JSON parses, number of planner arguments the code had to ignore), retrieval coverage per repo, and whether real sources reached the model.
- **Why it is worth doing:** it turns "the model felt worse on this backend" into a number, and it feeds the MoE expert-placement direction above with real placement-versus-latency data.

Not scoped or started. Natural to pick up after the heavy-tier work settles.

---

## Per-model planner adaptation — a deployment / infra-automation step, not a one-off bug fix

What happened (2026-10-04, heavy tier): the tool-selection planner (same LLM as generation) emitted `vector_search` calls that were individually plausible but wrong, and each wrongness produced the same user-visible symptom, "no context":
- invented a `filter_file` value (`'code-doc-assistant'`, not a path) → zero hits;
- split the two repos across `collection_name` and `collections` — the tool prefers `collections`, so one repo was silently dropped;
- guessed `chroma_host` as `localhost:8000` (already force-overridden in code).

Separately, an empty search result was being wrapped as a fake chunk (`"[]"`, source `codebase`, confidence 1.0), which hid all of the above from the supervisor.

What was changed: the code, not the model, now owns every argument that sets scope or recall. `vector_search` always covers all active repos; planner-supplied `score_threshold` / `filter_file` / `collection_name` are ignored and listed in the trace; empty results stay empty; retrieved-context budget is read from the running llama-server (`/props`) instead of the tier table.

Why it belongs in the deployment notes: a different model, or the same model behind a different backend, will fail this contract differently. Candidate automation, per model/backend deployment:
1. A tool-call conformance check before a tier is declared usable (`smoke_retrieval.py --no-generate` is the start of it: it prints the raw plan and the chunks it produced).
2. Take facts from the server instead of config: context size (`/props`) and the served model name (the app's `LLAMACPP_MODEL` default said `mistral-nemo` while the server was serving DeepSeek).
3. Treat MoE models as multi-part deployments — attention/KV on GPU, experts in RAM (`--n-cpu-moe`) — so memory placement is part of the per-model adaptation, next to the prompt/tool contract.

Not yet done: a general argument-validation layer in the tool registry (the fix above is specific to `vector_search`), (the Helm heavy-tier settings are now in chart 0.2.0).

---

## Finding: an Ollama embedding model on the same GPU collapses llama-server (resolved by placement)

Measured twice on the 12 GB laptop GPU with the heavy tier resident (VRAM ~11.6 of 12.2 GB): decode speed was 132-177 tok/s after a clean llama-server restart; one Ollama `/api/embed` call (`nomic-embed-text`, loaded on the GPU, VRAM +180-200 MiB) dropped it to 0.49 / 6.7 / 6.7 tok/s, and it stayed there until llama-server was restarted. The same call with `options: {"num_gpu": 0}` left `size_vram: 0`, VRAM unchanged and speed at 160-165 tok/s. Mechanism not established (WDDM memory pressure is the working guess); the placement dependence is.

Resolution: `src/embedding.py` is the single place the embedding client is built (used by retrieval and ingestion) and runs it on CPU by default (`EMBED_NUM_GPU`, 0 = CPU, -1 = Ollama decides). Results do not depend on placement, only latency does.

Worth keeping for the backend x model evaluation: embedding placement is a deployment variable like `--n-gpu-layers`, and it interacts with the answering model's VRAM budget.

---

## Indexing / retrieval quality — what is done and what must be measured

Done: identical chunks are dropped at ingestion (`dedupe_nodes`) and collapsed at query time before the top-k cut (collections ingested earlier had up to a third exact repeats: 2418 stored / 1602 unique in the Icarus collection).

Not done, deliberately, because each needs a measurement rather than an assumption:
- **Task prefixes for nomic-embed-text** (`search_query: ` / `search_document: `). The model was trained with them and Ollama does not add them. Adopting them needs a re-index and the same prefix scheme at query time, so compare retrieval on a fixed query set first (the backend x model harness above is the place to do it).
- **Relevance across file types**: on a mixed repo, Helm/YAML chunks outrank the source that actually answers an "embedding model" question. Candidates: per-file-type score weighting, a second retrieval pass restricted to code, or query rewriting. Measure before choosing.
- **Stale collections**: collections for repo refs no longer used (one empty) accumulate; needs a clean-up policy, not a one-off delete.

---

## Finding: silent failures across layers — what is done, what is next

Done (see ARCHITECTURE Phase 22): the Ollama init container no longer reports success after a failed pull; indexing preflights the embedding model and treats 0 stored chunks as an error; the app hard-stops on indexing failure; empty retrieval skips generation with an explanation and is logged as `no_retrieval`; context assembly truncates an oversize first chunk instead of dropping everything; traces persist per answer.

The general lesson: a pipeline whose components each "succeed" can still produce nothing useful. Every stage that can legitimately return empty needs an explicit empty/degraded status that reaches the user and the tracking layer.

Next:
- **Indexing as a background job** with visible progress (chunks embedded / total) instead of a blocking call in the chat; the first index of a repo took several minutes on the Ollama CPU. Check whether embeddings can use the GPU without colliding with llama-server (see the Ollama-embedding finding above).
- **Readiness for the app**: expose a `/health`-style check (Ollama embedding model present, ChromaDB reachable, llama-server ready, collections non-empty) and use it in the chart's probes and in a UI banner.
- **Same treatment for other stages**: low-confidence-only retrieval, a planner that always falls back, a supervisor that never passes. Each should warn, not stay `ok`.
- **Context-assembly policy**: the budget currently drops lower-ranked chunks after the first that does not fit (`break`). Compare `continue` (fill the budget with smaller chunks) and per-repo quotas on the evaluation set.
- **Eval runner**: a parent MLflow run per evaluation with one child run per question, a fixed question file (about 10 questions over the two repositories), and a list of configurations (backend, tier, top-k) swapped sequentially so results can be compared directly. Needs the question set first; quantisation and Flash Attention status should be recorded as parameters.

---

## Evaluation plan — questions noted for the eval runner

Findings from real traces that the evaluation runs should measure, not assume:

- **Context window vs. VRAM.** The heavy tier runs llama-server at `--ctx-size 4096` (24 GPU layers) on a 12 GB card, so the context budget is about 2,450 tokens (60% of the window). A typical retrieval of 10 chunks keeps 6 and drops 4. DeepSeek-Coder-V2-Lite supports a much larger window, but the KV cache costs VRAM (10.4 of 12.2 GB already used). Measure answer quality, latency and VRAM at 4096 / 8192 / 16384, with and without Flash Attention (reported unsupported for this GGUF on the CUDA build) and with `--n-cpu-moe`.
- **`break` vs. `continue` in context assembly.** Assembly stops at the first chunk that does not fit, which left about 1,100 budget tokens unused while 4 chunks were dropped. Compare `continue` (skip the oversize chunk, keep filling), per-repo quotas and smaller chunks. Dropping lower-ranked chunks may be the right call, so compare on the question set. The trace now warns when more than 30% of the chunks are dropped (`CONTEXT_DROP_WARN_FRACTION`), which makes the affected questions easy to filter.
- **Planner arguments.** The planner keeps proposing arguments the pipeline then ignores (`filter_file='IcarusSEND'` is a repository name, not a file; `collections=[...]`). They are dropped and logged; count how often per model, and whether a planner prompt change reduces it.
- **Retrieval relevance by file type and the nomic task prefixes** (see the indexing section above), using the same question set.
- **Embeddings on CPU vs. GPU:** first-index time (about 1,600 chunks per repo, several minutes on CPU) against the VRAM collision with llama-server.
- Record, as parameters of every run: model, quantisation, `--ctx-size`, GPU layers, Flash Attention status, top-k, context budget, embedding model.

---

## Higher-level roadmap after the llama.cpp bring-up — partial reconstruction

The original list from the earlier session could not be recovered (it is not in the Project, the searchable chats, or this session's transcript). Below is what is confirmed by the project owner's description on 2026-10-05 and by the sections above; add the missing items when they are remembered.

1. **Per-model llama.cpp load settings and compatibility**: `--n-gpu-layers` / `--n-cpu-moe` / `--ctx-size` / quantisation / alias per model, with facts read back from the server (`/props`, `/v1/models`) instead of config, and a tool-call conformance check before a tier is declared usable (see the per-model planner adaptation section).
2. **Dynamic model exchange on llama.cpp**: already implemented for the Ollama backend (several models resident, LRU eviction, switching at runtime). llama-server holds one model per process, so this is much harder; options to evaluate, not designed: one llama-server per model with an app-side manager that starts and stops them within the VRAM budget; llama-server's multi-model preset mode (`--models-preset`, see issue #22364 above); a small swap proxy in front of the container.
3. **Backend x model evaluation in MLflow** (section above): Ollama vs llama.cpp on the same model, Nemo vs DeepSeek on the same backend.
4. **Indexing and collection optimisation** (section above).
5. **Pipeline readiness for multi-agent integration** (named by the project owner on 2026-10-05; not yet scoped): the graph is already one node per responsibility with a shared typed state, human-review interrupts and per-question MLflow runs, which are the seams an agent-to-agent hand-off would need. Not designed: which roles become separate agents (planner, retriever, writer, reviewer), how they exchange state, and how a run that spans agents is still ONE tracked run.
6. **Tooling**: extend the tool registry beyond `vector_search` (MCP-capable tools already exist in `build_tool_registry`), with the tool-call conformance check from item 1 deciding which tools a given model may be offered.
7. **RLHF / feedback loop**: the second human-review step already records a 1-5 satisfaction score, a decision (accept / regenerate / add context) and format notes, and the first records approve / modify / re-plan / reject with the reviewer's expectations. These are logged per run in MLflow and are the raw material for preference data. Not designed: how to turn them into a reward or ranking signal, and how to keep that apart from the backend x model evaluation.
8. "...and so on": the project owner said more items were on the original list; they are still not recovered. Add them here when remembered.

### Status as of 2026-10-05 (from the project owner's two lists)

Done: Ollama runtime model hot-swap; multi-repo retrieval fixes (host forcing, tool-argument hints, `#branch` syntax); live pipeline visualisation (now repainting once per node with the running node marked); balanced/heavy tier correction; the mlflow compose-profile bug; llama.cpp wired to the full tier; BASE/QUANT escaping in the compose bootstrap; GUI / human-review / pipeline-repaint checks on the real stack; per-question MLflow runs (`tracking.py`, written and tested headless; not yet seen against the real mlflow container); automatic deployment defaults (`src/deployment.py`: with `INFERENCE_BACKEND=auto` the app follows the running llama-server / vLLM / Ollama instead of needing variables set by hand); MLflow `--allowed-hosts` in docker-compose.dev.yml (the server refused the app's `mlflow:5000` Host header with 403, silently).

Done, Helm (chart 0.2.0, `helm/code-doc-assistant/`; checked with a stand-in Go-template renderer over ~100 assertions AND run for real: `helm lint`/`helm upgrade` on the user's machine and the full stack on a minikube GPU cluster on 2026-10-06, a question answered through llama.cpp on the GPU):
- The chart on `dev` could not render at all: `app-deployment.yaml` called a `chunkingStrategy` helper that `_helpers.tpl` did not define. Added.
- `heavy` tier (Ollama model tag, resources, context/timeout) and `modelTier: heavy` for llama.cpp.
- llama.cpp: Deployment (`/app/llama-server`, Recreate strategy, long startup probe, GPU limit), Service, optional models PVC / hostPath / existing claim, optional GGUF download init container, tier defaults mirroring compose (heavy: ctx 4096 / 24 layers), `extraArgs` for `--n-cpu-moe` / `--flash-attn` experiments. Image path fixed to `ghcr.io/ggml-org/llama.cpp:server-cuda`.
- `inferenceBackend` accepts `auto`; `llamacpp.enabled` / `vllm.enabled` deploy a backend next to Ollama (for the backend comparison). The Ollama pod only pulls the chat model when Ollama may answer (`ollama.pullLlm` overrides).
- MLflow `--allowed-hosts` generated from the in-cluster Service names + localhost + `mlflow.allowedHosts`; probes pin `Host: localhost:<port>`.
- GPU on Kubernetes: `llamacpp.gpu.mode` (`resource` | `runtime`), `gpu.wsl2` (runtime mode + hostPath mount of `libdxcore.so`), because on Windows/WSL2 the GPU is paravirtualised and the NVIDIA device plugin fails NVML init ("Not Supported"); found and verified on minikube (docker driver, `--gpus=all`), not on Docker Desktop's kind cluster. Startup budget `llamacpp.startupFailureThreshold` (default 180 x 10 s): first load of the 10 GB heavy model took ~15 min (cold), 3.5 min warm, and the original 15-min budget killed the pod (exit 137).
- App env: `LLAMACPP_HOST`, `LLAMACPP_MODEL` (fallback), `EMBED_NUM_GPU`, `MAX_REPLANS`, `MLFLOW_UI_URL`, `MLFLOW_EXPERIMENT`, `EVAL_GROUP`; app resources now come from the existing `appResources` helper (none on `deploymentTarget: local`).

Remaining, Helm: GPU scheduling on a real multi-node Linux cluster with the NVIDIA device plugin (default `resource` mode renders and is tested, but has not been run); vLLM end to end on the chart; a `values-wsl2-minikube.yaml` preset (the install flags are currently in the README); Flash Attention is reported unsupported for the DeepSeek GGUF on the CUDA build (attention falls back to the slower path) -- worth a look together with `--n-cpu-moe`; faster first load (the 15 min cold read is storage-bound: PVC/hostPath on the minikube node, 9 GB node memory vs a 9.9 GB file); generic `helm` tooling note: delete `:Zone.Identifier` files before helm commands (now in `.gitignore`); `llamacpp.modelPath` was replaced by `llamacpp.modelFile` (file name inside `/models`).

Remaining, MLflow: per-node / per-LLM-call traces (MLflow 3 GenAI view; `mlflow.langchain.autolog()` is the likely route). Not needed for the backend x model evaluation, which uses the per-question runs (params, metrics, `result.json`) already logged. Test before enabling: traces change what is logged and how runs and traces link when a run is re-activated per node.

Remaining, other: llama.cpp for the other tiers, heavy first (`--n-gpu-layers` controllability); upstream llama.cpp contributions (section at the top); committing and pushing the dev changes (nothing pushed yet); stale test count in the README ("62 tests") -- the suite now also has the app-flow, deployment and chart-render tests.

