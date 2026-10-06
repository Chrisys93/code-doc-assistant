# Architecture & Design Decisions

This document captures the thinking behind the component choices and design patterns in the Code Documentation Assistant. It's written chronologically — decisions are recorded as they were made, not reconstructed after the fact. The aim is to preserve the actual reasoning, including dead ends and trade-offs, rather than presenting a sanitised post-hoc narrative.

A plan was made at the outset, defining the approach in four phases: LLM provider selection, pipeline planning, deployment format, and component selection. Implementation and testing followed as phases 5 and 6. The `dev` branch evolution — agent pipeline, HITL, vLLM integration, MLflow, Argo Workflows — is documented in phases 7 onwards.

---

## Phase 1: LLM Provider — Why Ollama?

**Decision: Ollama with an open-source model.**

The first fork was whether to use a hosted API (OpenAI, Anthropic) or a self-hosted open-source stack. The considerations:

- **Self-contained**: no API keys, no paid accounts required to run or evaluate the system
- **Infrastructure ownership**: standing up the full inference stack demonstrates CI/CD, deployment, provisioning, and ML/AI footprint awareness — not just API consumption
- **Privacy**: for a code documentation tool, keeping code local is often a hard requirement. Many organisations cannot send proprietary source code to external APIs. Self-hosting handles this by default

Trade-off acknowledged: hosted APIs produce noticeably better responses for complex code reasoning, especially architectural questions. For a production system where output quality is paramount and privacy constraints are relaxed, a hybrid approach makes sense — local model for routine queries, API fallback for complex reasoning. The self-hosted path is the right default for this use case.

**Evolution of thinking — from "which API" to "own the stack":**

The initial framing was: which hosted API should be used? The reframing came from recognising that the purpose of this system isn't to show which API produces the best answers, but to engineer the appropriate solution and demonstrate depth across the stack. Wrapping an API is a weekend project; standing up the full inference stack — model serving, embedding pipeline, vector storage, Kubernetes-native deployment — demonstrates infrastructure ownership and full-pipeline awareness. The privacy argument reinforced the decision: for a tool that ingests proprietary codebases, self-hosting isn't a nice-to-have, it's a requirement many organisations would insist on.

---

## Phase 2: README-Driven Development

**Decision: README as a living design document, developed alongside the code.**

Rather than coding first and documenting later, the README was developed in parallel, capturing decision points — especially inflexion points — as they happened.

**Evolution of thinking:**

Writing the README *during* development captures the actual thought process: blind alleys explored, trade-offs weighed, moments where understanding shifted. It also functions as a rubber duck — several technical choices (especially around embedding model selection and vector DB architecture) were refined because writing them down forced sharper thinking.

---

## Phase 3: Deployment — Docker Compose and Helm

**Decision: both, serving different purposes.**

| Aspect | Docker Compose | Helm Chart |
|--------|---------------|------------|
| Purpose | Local dev, single-command startup | Production-grade K8s deployment |
| Audience | Anyone with Docker installed | K8s cluster operators |
| What it shows | Containerisation | Architecture as infrastructure-as-code |

The more interesting reason to have both is what the Helm chart forces you to think about. When you model the system as Kubernetes objects, the architecture becomes explicit:

- **Ollama** → StatefulSet (model weights are persistent state)
- **ChromaDB** → StatefulSet (vector index is persistent state)
- **Application** → Deployment (stateless, horizontally scalable)

The Helm chart also surfaced the composability patterns (the `_helpers.tpl` tier system) that wouldn't have emerged from Docker Compose alone — Compose doesn't have the same templating, system-wide programmability, or automation-ready power.

---

## Phase 4: Component Selection

### 4a. LLM Model Selection — A Tiered Approach

**Decision: tiered model strategy — the system is model-agnostic, model name is a configuration value.**

A single `modelTier` value cascades through the entire system:

```mermaid
flowchart LR
    MT["modelTier=heavy"] --> M["deepseek-coder-v2:16b-lite-instruct"]
    Q["quantisation=q4_K_M"] --> TAG["final tag:\ndeepseek-coder-v2:16b-lite-instruct-q4_K_M"]
    M --> TAG
    TAG --> RES["resource limits\nCPU/GPU allocation"]
    MT --> CTX["ctx 8192\ntimeout 120s"]
    MT --> CHUNK["AST chunking"]
```

**Task framing**: this is a *code comprehension + explanation* task, not code generation. The model must read retrieved code chunks, understand what they do, reason about relationships (dependencies, API endpoints, architecture), and explain in natural language. Reasoning capability — following multi-step logic across files — scales with parameter count and is the key differentiator between tiers.

**Models evaluated**:

| Model | Parameters | Strengths | Reasoning | Weaknesses |
|-------|-----------|-----------|-----------|------------|
| Mistral Nemo | 12B | Excellent code comprehension + explanation | Strong multi-step; traces cross-file dependencies | Needs GPU |
| DeepSeek-Coder V2 Lite | 16B (MoE) | MoE excels at polyglot codebases | Strong within-context reasoning | Variable memory patterns |
| Qwen2.5-Coder 7B | 7B | Best code comprehension at 7B tier | Adequate for single-file reasoning | May struggle with complex multi-module chains |
| Phi-3.5 Mini | 3.8B | Runs almost anywhere | Best for straightforward "what does this function do" queries | Not code-specialised |

*CodeLlama 7B was evaluated and excluded — Qwen2.5-Coder 7B supersedes it on modern benchmarks.*

**Tiered defaults** (matches `config.py`'s `_MODEL_TIER_BASE`):

0. **Heavy** (explicit, opt-in) — DeepSeek-Coder V2 Lite (16B, MoE). MoEs are particularly effective for multi-language tasks, treating programming language diversity analogously to natural language multilingualism ([Wang et al., 2025](https://arxiv.org/abs/2508.19268)). It has its own slot because its VRAM footprint is dominated by total, not active, parameters (see Phase 19).
1. **Full** — Mistral Nemo (12B). Best balance of explanation quality and code understanding.
2. **Balanced** — Qwen2.5-Coder 7B (dense). DeepSeek V2 Lite briefly held this slot and was moved to `heavy` (Phase 19).
3. **Lightweight** — Phi-3.5 Mini (3.8B). Edge, CPU-only, or resource-constrained deployments. Fine-tuning candidate.
4. **Minimal** — Qwen2.5-Coder 3B. Tightest memory footprint; CI pipelines and very constrained machines.

*Note: Qwen2.5-Coder 7B was the original `balanced` candidate, was replaced by DeepSeek V2 Lite for a while, and is `balanced` again (Phase 19). The 3B Qwen2.5-Coder variant serves `minimal`.*

**Hardware reality check**: frontier models reach trillions of parameters — three orders of magnitude above these. But without a multi-GPU system, models above ~16B aren't practical to serve. These tiers reflect models genuinely usable on realistic hardware.

**Evolution of thinking — from fixed to composable:**

The initial approach was to pick the best model and hard-code it. The shift came from asking: what does configurability actually mean in a Kubernetes-native deployment? Not just swapping a string, but a single high-level intent (`modelTier=lightweight`) that cascades through every dependent decision — which model to pull, memory to request, GPU requirements, context window, timeout. The `_helpers.tpl` implements this. A deployer expresses "I want the lightweight tier" and the system resolves the rest. This is the difference between configuration and *composable system design* — the same principle behind Kubernetes operators and Terraform modules.

### 4b. Embedding Model

**Decision: `nomic-embed-text` as default, with codebase-aware configuration.**

**Critical constraint — embedding compatibility**: you cannot mix embeddings from different models into a single index. Changing the embedding model requires full re-ingestion. The `_helpers.tpl` derives vector dimension from the embedding model choice automatically, preventing silent misconfiguration.

**Codebase-aware embedding selection** — two axes characterise the input:

| Axis | States | Implication |
|------|--------|-------------|
| **Language distribution** | Primary-code (>90% one language) vs. Multi-code | Multi-code benefits from polyglot models; connects to DeepSeek LLM choice |
| **Documentation state** | No-docs / Partial-docs / Review-and-revise | Review-and-revise is most demanding — must handle inconsistencies between code and prose |

**Available models**:

| Model | Dimensions | Best for |
|-------|-----------|----------|
| `nomic-embed-text` | 768 | Partial-documentation codebases (default) |
| `all-minilm` | 384 | Lightweight tier; resource-constrained |
| `mxbai-embed-large` | 1024 | Complex codebases with dense documentation |

**Evolution of thinking — from infrastructure choice to input-driven configuration:**

Initially the embedding model was treated as a pure infrastructure decision. The shift came when recognising that the *nature of the codebase* should inform the choice — and that the embedding model determines much of what's possible downstream. A raw-code-only repo needs strong code-native embeddings. A heavily-documented repo needs good code+text understanding. A polyglot repo needs polyglot awareness. This reframing — embedding selection as a property of the *input*, not the *infrastructure* — led to the two-axis characterisation. The implementation stays simple; the systematic understanding is documented for operators making informed deployment choices.

### 4c. Vector Database

**Decision: ChromaDB as default, behind a thin abstraction layer.**

An important distinction emerged during evaluation: **FAISS is a search index library, not a database**. It provides indexing algorithms (LSH, HNSW, IVF) but no persistence, metadata filtering, or API. Vector databases like ChromaDB and Qdrant use HNSW internally and wrap it with database functionality.

| Solution | What it is | Persistence | Metadata filtering | Best fit |
|----------|-----------|-------------|-------------------|----------|
| **FAISS** | Search index library | None — you build it | None — you build it | Raw performance; custom systems needing LSH |
| **ChromaDB** | Vector database (HNSW) | Built-in | Built-in | Developer convenience; small-to-medium codebases |
| **Qdrant** | Vector database (HNSW) | Built-in | Built-in (richer) | Production; large codebases |

For a code documentation assistant, **metadata filtering matters** — filtering by file type, directory, or language when searching. ChromaDB provides this out of the box. FAISS would require building all of that plumbing manually, or pairing it with a separate database for persistence and metadata.

The abstraction layer preserves the ability to swap in FAISS+LSH or Qdrant later without changing application code. Pragmatic in implementation, extensible in design.

On `dev`, this vector store is one of two retrieval mechanisms, used for semantic/approximate queries; Kuzu (Phase 16) handles structural/exact queries:

```mermaid
flowchart LR
    subgraph RETRIEVAL["Retrieval"]
        VS["Vector Search\nChromaDB\nsemantic / approximate"]
        GT["Graph Traverse\nKuzu\nstructural / exact"]
    end

    Q1["'How does X work?'"] --> VS
    Q2["'What calls X?'"] --> GT
    Q3["'What changed with X?'"] --> GT
```

**Evolution of thinking — FAISS, LSH, and knowing when to stop:**

LSH came up during evaluation as an alternative indexing strategy. Building persistence, metadata filtering, and CRUD on top of FAISS is substantial engineering with no practical benefit at this project's scale. ChromaDB behind an abstraction layer is the right trade-off — not over-engineering for hypothetical requirements, but not closing the door on them either.

### 4d. Orchestration Framework

**Decision: LlamaIndex for the RAG pipeline; LangGraph adopted in `dev` for the agent pipeline.**

LlamaIndex is purpose-built for RAG: native `CodeSplitter` with AST-aware chunking, tree-structured indexes, lighter weight than LangChain for pure retrieval-and-respond workflows.

**Production orchestration**: MLflow (prototyping) → W&B (production monitoring) → Ray on K8s (distributed compute). Ray Serve wraps Ollama for load balancing; Ray Data enables parallel ingestion for large codebases; KubeRay deploys natively on K8s.

**Evolution of thinking — from "which framework" to "what question am I actually answering":**

The instinct was to reach for LangChain — it's the default answer for AI orchestration. But LangChain is a general-purpose framework; this project is focused RAG. The real orchestration question isn't "which framework chains my prompts" but "how does this system scale operationally?" That's answered by MLflow/W&B for tracking and Ray/K8s for compute, not by a prompt-chaining library. LangChain — specifically LangGraph — enters the picture when the *application scope* grows (agents, tool use, HITL, conditional graphs), which is exactly what the `dev` branch does.

### 4e. Code Chunking Strategy

**Decision: AST-based chunking via tree-sitter (LlamaIndex's `CodeSplitter`), with fixed-window fallback.**

| Strategy | Strengths | Weaknesses |
|----------|-----------|------------|
| AST-based (tree-sitter) | Preserves logical units; language-aware | Requires valid, parseable code |
| Heuristic / pattern-based | Works on broken code | Fragile; misses nested structures |
| Fixed-window | Language-agnostic; never fails | Splits functions mid-body |
| Hybrid (AST + fallback) | Best of both | Slightly more complex |

The hybrid is the right choice: tree-sitter for clean code, graceful degradation to fixed-window for generated code, config files, and partial snippets.

The chunking strategy is also **tier-configurable**: full/balanced tiers default to AST chunking; the lightweight tier defaults to text-based chunking (lower CPU/memory during ingestion). The `_helpers.tpl` resolves this from `modelTier` and cascades through to the app via the `CHUNKING_STRATEGY` environment variable — the same composability pattern applied to model selection.

**Pipeline validation confirmed**: tree-sitter produced 41 semantic chunks from 7 Python files; the fallback handled text and config files correctly. Chunk distribution (e.g., `ingest.py` → 9 chunks, `config.py` → 2 chunks) shows the AST splitter respects logical boundaries rather than imposing uniform size.

**Evolution of thinking — from "just split the text" to language-aware semantic boundaries:**

Fixed-window chunking is the naive approach. A function split mid-body produces chunks that are individually meaningless. The AST-aware approach ensures a function, class, or method is always a complete unit. The tier-configurable fallback emerged later: for the lightweight tier already resource-constrained running Phi-3.5 on CPU, spending resources on AST parsing during ingestion may not be the right trade-off. This led to making chunking strategy part of the tier cascade — the same composability principle applied one more time.

### 4f. Interface

**Decision: Streamlit.**

Streamlit provides a ChatGPT-style interface with `st.chat_input()` and `st.chat_message()` in pure Python — functional and clean.

`master`: simple chat UI with sidebar ingestion controls.
`dev`: tabbed layout — **Chat** (conversation + HITL widgets), **Pipeline** (full-width graph diagram with live execution trace highlighting), **Session** (accumulated preference profile, supervisor audit trail, MLflow run link).

**Access patterns**:

| Method | Context | Helm config |
|--------|---------|-------------|
| `kubectl port-forward` | Single developer, same machine | Default (ClusterIP) |
| NodePort | Team on a private network | `--set app.service.type=NodePort` |
| Ingress with TLS | Production / public access | `--set ingress.enabled=true` |

**Evolution of thinking — access patterns and network realities:**

The initial Helm setup offered ClusterIP + optional Ingress. Experience with multi-node private network deployments surfaced the NodePort gap: a team externally accessing the tool from other machines without an ingress controller. NodePort has a "dirty quick fix" reputation mainly because it's not secured for public-facing interfaces. For an internal code documentation tool on a private network — exactly where a tool handling proprietary code would run — NodePort is perfectly pragmatic. The "textbook" answer (always use Ingress) isn't always right; deployment context and network topology are the deciding factors.

### 4g. RAG Quality and Limitations

RAG is not unconditionally beneficial. Research shows retrieval noise can actively degrade output quality — irrelevant context can sometimes be worse than no context at all ([Gupta et al., 2024](https://arxiv.org/abs/2410.12837)).

**Code documentation-specific RAG risks:**
- **Stale context**: chunks from a previous version may contradict current code
- **Partial context**: a function without its imports leads to incorrect explanations
- **Cross-file confusion**: similar naming across modules causes conflation

**Mitigations implemented**: similarity score cutoff (0.3), metadata preservation (file paths and languages in prompt), source attribution in every response.

The `dev` branch extends this with two supervisor-level guardrails: a pre-generation confidence gate (retrieval quality check before the generation node runs) and a post-generation quality gate (rubric-scored LLM evaluation when `OUTPUT_REVIEW_MODE=supervisor`).

**Evolution of thinking — from "RAG always helps" to understanding when it hurts:**

The initial assumption was straightforward: retrieve relevant context, feed it to the LLM, get better answers. The key insight for code: the *quality* of retrieval matters more than the *quantity*. A function chunk without its import context may lead the LLM to hallucinate dependencies. A chunk from a similarly-named function in a different module may cause conflation. The similarity cutoff and metadata preservation are direct responses to these failure modes — not just "nice to have" filtering, but guardrails against specific RAG failure modes applied to code.

### 4h. Guardrails

For a code documentation tool, guardrails are domain-specific.

**Hallucination prevention**: prompt template instructs the model to say "I don't have enough context" rather than guess; source attribution enables verification; similarity cutoff prevents irrelevant context from triggering confabulation.

**Sensitive data protection**: code often contains credentials, API keys, and tokens. The system should detect and redact common patterns before including chunks in responses. Not implemented in the current version but a critical production requirement.

**Bias and consistency**: tokenisation may weight variable naming conventions differently across languages. Mitigation: normalise code formatting before embedding; monitor response consistency.

**Evolution of thinking — from "add a filter" to understanding code-specific risks:**

Guardrails in general LLM applications focus on content moderation. For code documentation the risks are different — hallucinated file paths that don't exist, confidently wrong architectural explanations, leaked credentials embedded in code chunks. Source attribution is itself a guardrail: when the developer can see *which files* informed the answer, they can verify claims against actual code. This transforms the system from a black-box oracle into a transparent assistant.

---

## Phase 5: Implementation

Key outcomes:
- **5 Python modules** (`config.py`, `vector_store.py`, `ingest.py`, `query_engine.py`, `app.py`) — each mapping to a distinct concern
- **ChromaDB abstraction layer** — `VectorStoreBase` ABC with `ChromaVectorStoreImpl`; swappable to FAISS or Qdrant
- **Tier-aware configuration** — `config.py` mirrors the Helm `_helpers.tpl` logic for Docker Compose parity, including model selection, resource allocation, embedding dimension, and chunking strategy
- **AST chunking with tier-configurable fallback** — tree-sitter via LlamaIndex's `CodeSplitter` for full/balanced tiers; `SentenceSplitter` for lightweight tier; AST→text fallback always available as a safety net
- **Pipeline validation test** (`tests/test_pipeline.py`) — end-to-end test using ChromaDB in embedded mode with lightweight local embeddings, validating the full ingest→chunk→store→retrieve pipeline without requiring external services

One implementation detail worth noting: `tree-sitter-language-pack` was discovered as a missing dependency during pipeline validation — LlamaIndex's `CodeSplitter` requires it for AST parsing, but it's not listed as a dependency of `llama-index-core`. Added to `requirements.txt` after confirming the fallback path worked correctly and the AST path needed the additional package.

---

## Phase 6: Testing & Refinement

**Resource constraints**: the full tier (Mistral Nemo 12B) requires a GPU and was not fully integration-tested. The lightweight tier (Phi-3.5, CPU-only) is the recommended tier for testing without GPU access. The code is tier-independent — switching tiers changes only model and resources, not application logic.

**Pipeline validation results**:
- All module imports validated ✅
- Config tier resolution tested across all tiers ✅
- File discovery: 18 files found (8 code, 10 text/config), correctly classified ✅
- AST chunking: 41 code chunks from 7 Python files via tree-sitter ✅
- ChromaDB storage: 58 total chunks stored in embedded mode ✅
- Retrieval: 4/6 test queries hit expected files with test embeddings ✅

Full local integration was also tested against a real repository. One issue encountered: the model returned `Error generating response: model requires more system memory (50.0 GiB) than is available (7.7 GiB)` when asked to document a specific file. This surfaced the gap between VRAM (what the GPU inference needs) and system RAM (what the error reports) — model quantisation and architecture can affect both independently. The cloud resource estimates in the README account for this with headroom above theoretical minimums.

**Evolution of thinking — honesty over impression management:**

The temptation was to gloss over resource constraints and imply thorough testing. The pipeline validation test was designed specifically to exercise as much of the codebase as possible *without* requiring external services. The 4/6 retrieval hit rate with character-trigram embeddings validates the pipeline mechanics; real Ollama embeddings would resolve the remaining misses. The principle throughout: be clear about what was tested, clear about what wasn't, and show you know the difference.

---

## Phase 7: `dev` Branch — Why an Agent Pipeline?

The `master` branch answers "what does this code do?" reliably via a linear pipeline: ingest → embed → retrieve → respond. The `dev` branch asks a harder question: what if the system needed to *decide how* to find the answer — choosing between grep, AST parsing, vector search, or GitHub fetch depending on the query — and what if a human needed to review that plan before any tools ran?

This changes the execution model from a linear pipeline (fixed stages, predictable path) to a conditional graph (node routing, feedback loops, human interrupt points). The right tool for this is LangGraph.

**Evolution of thinking — from "add tools" to "redesign the execution model":**

The instinct was to layer tool use on top of the existing RAG pipeline. The reframe came from recognising that tool selection is itself a decision that benefits from human oversight — not because the LLM can't choose tools, but because in a code documentation context the human often has knowledge the tool-selection agent doesn't: which files are the canonical source of truth, which directories are generated and should be skipped, what the current task's scope actually is. HITL-1 (tool plan approval) exists to transfer that contextual knowledge into the pipeline before any tools execute.

---

## Phase 8: Agent Graph Design

**Decision: LangGraph `StateGraph` with nine nodes and two configurable interrupt points.**

```mermaid
flowchart TD
    START(["START"])
    TS["tool_selection"]
    HITL1["hitl_checkpoint\n[HITL-1]"]
    TE["tool_execution"]
    SUP["supervisor"]
    CA["context_assembly"]
    GEN["generation"]
    OR["output_review\n[HITL-2]"]
    END_(["END"])

    START --> TS
    TS --> HITL1
    HITL1 --> TE
    TE --> SUP
    SUP -->|retry| TE
    SUP --> CA
    CA --> GEN
    GEN --> OR
    OR -->|regenerate| TS
    OR --> END_
```

`[HITL-1]` = `interrupt_before` on `hitl_checkpoint` when `HITL_ENABLED=true`
`[HITL-2]` = behaviour controlled by `OUTPUT_REVIEW_MODE`

**Graph topology is fixed regardless of `OUTPUT_REVIEW_MODE`**: the `output_review` node always exists and is always connected. The mode controls whether `interrupt_before` is set on it and what logic runs inside it at runtime. This is intentional — the graph structure is a stable contract; runtime behaviour is a configuration concern. It also means `get_graph_mermaid()` always renders the same topology in the Pipeline tab, regardless of mode.

**The supervisor has two distinct responsibilities:**

Pre-generation: evaluates retrieval confidence scores, injects `SessionPreferences` biases (preferred files, response format, verbosity), retries with adjusted parameters if confidence is below threshold.

Post-generation (when `OUTPUT_REVIEW_MODE=supervisor`): evaluates the generated output against a rubric scoring accuracy (0–4), completeness (0–3), and attribution (0–3). Silently retries if total score is below `QUALITY_GATE_THRESHOLD`; accepts if it passes. Human feedback from previous queries (when `OUTPUT_REVIEW_MODE=human`) updates the `SessionPreferences` profile in `AgentState`, which persists across queries within the thread and biases subsequent supervisor behaviour.

**`OUTPUT_REVIEW_MODE` values:**

| Mode | Behaviour | Use case |
|------|-----------|----------|
| `human` | HITL-2 interrupt — human rates and decides (accept / regenerate / add context) | Default; highest oversight |
| `supervisor` | Rubric LLM evaluates output; silent retry if below threshold | Automated quality gate |
| `self` | Generation LLM self-critiques before emitting; no extra node | Lightweight sanity check |
| `off` | Passthrough; immediate accept | CI/automated pipelines |

---

## Phase 9: Tool Registry

**Decision: nine tools split between shell-level and semantic.**

Shell tools (`grep`, `cat`, `find`, `git_log`, `git_blame`, `stat`) often outperform semantic search for code documentation tasks — finding a function definition is a `grep` problem, not a vector similarity problem. Semantic tools (`vector_search`, `ast_parse`) handle conceptual questions and cross-file relationship queries. `github_fetch` handles cases where the repo isn't locally mounted.

All shell tools validate paths against `REPO_PATH` and use an allowlisted flag set — no path escape, no arbitrary flag injection. The tool registry pattern (`TOOL_REGISTRY` dict mapping name → `{fn, description, required_args, optional_args}`) keeps tool logic in `tools.py` and graph logic in `agent_graph.py` with no coupling between them.

---

## Phase 10: Inference Backend Factory

**Decision: `INFERENCE_BACKEND` env var switches between Ollama, vLLM, and llama-server at the `_get_llm()` factory — no other code changes required.**

```mermaid
flowchart LR
    BACKEND["INFERENCE_BACKEND\nenv var"]

    BACKEND -->|ollama| OL["ChatOllama\n→ Ollama REST API\nmodel management\nCPU fallback"]
    BACKEND -->|vllm| VL["ChatOpenAI\n→ VLLM_HOST/v1\nOpenAI-compatible\nGPU / PagedAttention"]
    BACKEND -->|llamacpp| LC["ChatOpenAI\n→ LLAMACPP_HOST/v1\nOpenAI-compatible\nCPU-native GGUF"]

    OL & VL & LC --> NODES["All nine graph nodes\n(backend-agnostic)"]
```

All nine graph nodes call `_get_llm()` and are backend-agnostic. The `api_key` field is set to `"not-required"` for vLLM and llama-server — both ignore it entirely.

`MODEL_TIER` continues to work as the single high-level control across all backends. `LLAMACPP_MODEL` defaults to the minimal tier model; `VLLM_MODEL` falls back to `OLLAMA_MODEL`.

**Backend affinity by tier:**
- `full` / `balanced` → Ollama or vLLM (GPU recommended)
- `lightweight` / `minimal` → Ollama or llama-server (CPU-native GGUF; llama-server skips Ollama's model management overhead, useful in CI and constrained environments)

**`max_tokens` is conservative for llama-server** (2048 vs 4096 for vLLM): llama-server's context size is set at startup via `--ctx-size` and cannot be exceeded at the API level — a conservative default prevents silent truncation.

**Evolution of thinking — from "vLLM as a future note" to three wired-in backends:**

The initial ARCHITECTURE.md mentioned vLLM as a "what I'd do with more time" item. In `dev`, the right abstraction became clear: the `_get_llm()` factory is the single point where a LangChain model is constructed, so backend-switching belongs there. llama.cpp/llama-server was added after recognising that Ollama wraps llama.cpp internally for most models — the efficiency difference is small for `full`/`balanced` tiers, but meaningful for `minimal` on CPU-only machines where skipping the Ollama daemon layer reduces startup overhead and memory footprint.

---

## Phase 11: MLflow — System-Level Observability

**Decision: MLflow is a system-level observability layer, not associated with any inference backend.**

MLflow logs the behaviour of the system as a whole — regardless of which inference backend is active (Ollama, vLLM, or llama-server), which model tier is deployed, or which tools were called. The inference backend is one *attribute* logged per run, not a dependency of the logging itself.

```mermaid
flowchart TD
    subgraph SYSTEM["System (any tier, any backend)"]
        AGENT["Agent run\n(LangGraph)"]
        INGEST["Ingestion run\n(LlamaIndex + Kuzu)"]
    end

    MLFLOW[("MLflow\nTracking Server\n:5000")]

    AGENT -->|"inference_backend · model_name\nHITL decisions · tool_calls\nretrieval_confidence · quality_gate_scores\nuser_satisfaction · latency · run_id"| MLFLOW

    INGEST -->|"commit_sha · embedding_model\nchunk_count · collection_name\ngraph_build_status · run_id"| MLFLOW
```

`dev`: MLflow is always-on in `docker-compose_dev.yml`. Every query creates a run. The Trace tab in the Streamlit UI links directly to the last run's MLflow page.

`master`: MLflow is opt-in via `--profile observability`. A plain `docker compose up` is unchanged for existing users. The app handles a missing `MLFLOW_TRACKING_URI` gracefully throughout (all MLflow calls are in `try/except`).

**What each run logs:**

Agent runs log: inference backend, resolved model name, HITL settings, output review mode, retrieval confidence metrics, quality gate scores, user satisfaction rating (1–5), generation latency, tool calls executed, supervisor adjustment audit trail.

Ingestion runs log: commit SHA, embedding model, chunk count, collection name, graph build status (dependency graph + co-change graph). This makes MLflow run IDs traceable links between code versions, embedding indexes, and the preference data accumulated from HITL — directly relevant for the DPO training pipeline in the `orchestrated` research branch.

**MLflow vs W&B**: W&B requires a licence for team use at scale. MLflow paired with Argo Workflows (Argo for orchestration, MLflow for tracking) covers the same functional ground with no licence cost and cleaner architectural separation of concerns.

---

## Phase 12: Argo Workflows

**Decision: Argo `WorkflowTemplate` replaces the `ollama-bootstrap` one-shot container with a proper DAG.**

```mermaid
flowchart LR
    CLONE["clone-or-mount"]
    DISCOVER["discover-files"]
    CHUNK_CODE["chunk-code"]
    CHUNK_TEXT["chunk-text"]
    EMBED["embed-and-store"]
    LOG["log-to-mlflow"]

    CLONE --> DISCOVER
    DISCOVER --> CHUNK_CODE & CHUNK_TEXT
    CHUNK_CODE --> EMBED
    CHUNK_TEXT --> EMBED
    EMBED --> LOG
```

`chunk-code` and `chunk-text` run in parallel (different file type sets). A `CronWorkflow` (suspended by default, enabled in production) handles nightly re-ingestion. This moves ingestion from "a thing that happens on container startup" to "a scheduled, observable, retryable pipeline with a logged artefact in MLflow."

The `log-to-mlflow` step records the commit SHA, embedding model, chunk count, and graph build status — making each ingestion run a traceable link between a specific code version and the embedding index built from it.

---

## Next Steps

These are concrete, scoped extensions grounded in what's already built.

### Advanced RAG Mitigations

The similarity cutoff and metadata guardrails are a first line of defence. The next layer:

- **CRAG (Corrective RAG)** — filtering low-confidence retrievals at inference time, reducing retrieval errors by 12–18%
- **Self-RAG** — the model learns to critique its own retrieval usage, deciding whether retrieved context is relevant before incorporating it
- **Re-ranking** — a secondary model re-scores retrieved chunks before they enter the prompt; computationally cheap relative to inference, but meaningfully improves retrieval precision
- **Context windowing** — prioritising recently-modified files for timeliness; stale chunks are a known failure mode for active codebases

### Chunking Granularity and Adaptive Retrieval

The current implementation chunks at function/class level — well-suited for "what does this function do?" but less effective for "how do these modules interact?" An adaptive retrieval approach would maintain multiple index granularities simultaneously: function-level for local questions, file-level for module-level questions, cross-file summaries for architectural questions. The retrieval layer would select the appropriate granularity based on the query.

### vLLM, Quantisation, and Production-Grade Inference

Ollama is the right choice for developer convenience and self-contained deployment. In a production environment with known infrastructure, a different set of optimisations becomes relevant.

**vLLM**: continuous batching, PagedAttention for efficient KV-cache management, native tensor parallelism across multiple GPUs. Where Ollama optimises for developer convenience, vLLM optimises for throughput and latency under concurrent load — critical when serving a team rather than a single developer. It supports OpenAI-compatible API endpoints, making it a drop-in replacement with minimal changes (already wired in `dev` via `_get_llm()`).

**Quantisation** (GPTQ, AWQ, GGUF): reduces model memory footprint by 50–75% with minimal quality loss for code comprehension tasks. This would allow running the balanced tier on hardware currently limited to the lightweight tier, or the full tier on a single T4 via 4-bit quantisation. Beyond memory savings, quantisation can also increase context capacity and concurrent user support for the same resource envelope — a different set of trade-offs worth evaluating per deployment.

**Context and sequence length**: larger context windows (32K+ tokens) enable ingesting entire files or multi-file contexts in a single query, directly addressing the chunking granularity problem. vLLM's PagedAttention makes long-context inference practical without proportional memory scaling.

**Multi-GPU deployment**: tensor parallelism (splitting model layers across GPUs) and pipeline parallelism (splitting pipeline stages) become options with multiple GPUs. The embedding model and LLM could run on separate GPUs, eliminating the shared-VRAM constraint described in the README. Architecture-specific optimisations — FlashAttention-2 for Ampere+, INT8 on Turing, INT4 on Ampere, native FP4 on Blackwell (B100/B200/GB200) — continue to push the boundary of viable model sizes per GPU generation.

**Network and switching**: in distributed deployments (separate nodes for inference, vector DB, and app), network topology matters — NVLink for multi-GPU communication, RDMA/InfiniBand for inter-node model parallelism, co-location of the app and vector DB to minimise retrieval latency.

### Model Fine-Tuning

The RAG approach means the model receives relevant context at query time, which is sufficient for most questions without fine-tuning. Fine-tuning becomes relevant if base models consistently fail on specific languages or domains, or if a particular documentation style is required. This requires training data (code Q&A pairs), compute, and iteration — guided by observed performance gaps, not assumed in advance.

The `fine-tuning` branch (planned) would use LoRA/QLoRA adapters trained on HITL feedback accumulated in MLflow — cross-session, offline, and gated. This is not within-session fine-tuning: the feedback volume per session is too low and catastrophic forgetting risk is real. The right trigger is when MLflow contains enough preference signal to justify a training run.

### Credential and Sensitive Data Redaction

Code often contains credentials, API keys, and tokens embedded as literals or in config files. The current system includes chunks in responses without scanning for sensitive patterns. Detecting and redacting common credential formats before including chunks in the prompt is a critical production requirement, not a nice-to-have.

---

## Wider Vision

This section is explicitly a tangent — a larger architectural idea that emerged from building this system, kept separate to avoid scope creep but documented because it shows where this thinking leads.

The code documentation assistant is a self-contained, useful system. It is also, viewed differently, a **reference implementation of a modular AI subsystem**: it has a well-defined interface (ingest a codebase, answer questions about it), a composable configuration model, clear abstraction boundaries, and a tiered deployment story. These properties make it a natural candidate for composition into a larger platform.

### A Platform for Bespoke AI System Development

The broader vision is a framework for building, fine-tuning, deploying, and continuously improving AI systems — not just for code documentation, but for any domain where a team needs a context-aware, locally-deployed AI assistant. The code documentation assistant would be one instantiation of this framework; others might include a test generation assistant, a security audit assistant, a migration planning assistant, and so on.

The relationship between this project and that platform runs in both directions:

**This system as a subsystem of the platform**: the code documentation assistant plugs into the platform as a module — its ingestion pipeline, embedding model, and query engine are reusable components. The platform orchestrates multiple such modules, routes queries to the appropriate assistant, and manages the shared infrastructure (Ollama cluster, vector DB fleet, model registry).

**The platform as an optimisation service for this system**: conversely, the platform could provide services back to this assistant — a fine-tuning pipeline that trains on accumulated developer interactions, a model registry that manages tier upgrades, a distributed index that aggregates knowledge across teams and codebases. The assistant consumes these services without needing to implement them itself.

This bidirectional composability — each system can be a client or a provider depending on the context — is the core architectural principle.

### Branching Structure for a Multi-Environment System

A natural evolution of the current project into a multi-branch, multi-environment system:

- **`master`** — stable RAG pipeline; reviewer-friendly; single-command startup
- **`dev`** — agent pipeline; LangGraph; HITL; vLLM; MLflow; Argo Workflows
- **`fine-tuning`** — model adaptation; training data pipelines; LoRA/QLoRA experiments on Q&A pairs accumulated from HITL interactions
- **`research`** (separate repo) — online preference learning; federated preference aggregation; autonomous multi-context agent orchestration; explicitly not production-bound

The research repo is where the platform vision above would be prototyped — agentic pipelines that can autonomously ingest, embed, and index new codebases; agent-to-agent (A2A) coordination between specialised assistants; federated embeddings across teams where each node contributes to a shared index without centralising proprietary code.

### LSH and Distributed AI

During vector DB evaluation, LSH (Locality-Sensitive Hashing) came up as an alternative to HNSW. LSH offers compact binary representations, O(1) lookup, and sub-linear search time — relevant at scales well beyond this project. The broader question: can LSH enable **collectively and distributedly intelligent systems** where computational nodes and network topology contribute to a shared, scalable understanding of data, rather than centralising in a single vector database?

Research directions being followed:
- Reformer architecture's use of LSH attention for efficient long-sequence processing
- GPU-optimised LSH with Winner-Take-All hashing and Cuckoo hash tables ([Shi et al., 2018](https://arxiv.org/abs/1806.00588))
- PipeANN on aligning graph-based vector search with SSD characteristics for billion-scale datasets ([Guo & Lu, OSDI '25](https://www.usenix.org/system/files/osdi25-guo.pdf))

In the platform context, LSH-based distributed indexing becomes a more compelling option — federated embeddings across teams, where each node contributes to a shared index without centralising proprietary code. The distributed AI question connects directly to the platform vision: scaling *understanding* across an organisation, not just scaling infrastructure.

### Research Questions

These are genuinely research-level questions — not implementation items — but they define the territory that the platform vision would need to address.

**Study 1: Documentation state vs. compute requirements** — does a well-documented codebase require less inference compute to generate useful answers? Quantifying this relationship across model tiers × embedding models × codebase types would produce actionable guidance for platform operators: *for this type of codebase, this configuration is sufficient*.

**Study 2: Autonomous continuous improvement across environments** — a code documentation assistant deployed at branch level learns from developer interactions (which answers were useful, which were wrong), and that learning propagates from branch → team → enterprise. Scaling *understanding*, not just infrastructure. This is the research-level version of the fine-tuning branch described above, and connects directly to the federated embeddings and distributed AI questions.

---

## Engineering Standards

- Python 3.11+, type hints throughout
- Modular design: each source file maps to a single concern
- Abstraction layers: `VectorStoreBase` ABC for DB-agnostic design; `_get_llm()` factory for backend-agnostic inference
- Configuration: environment-variable driven, with parity between Docker Compose and Helm at every layer
- Testing: pipeline validation with embedded ChromaDB; AST chunking verification; graph topology validation (`dev`)
- Logging: structured logging via Python `logging` module, configurable level; MLflow for experiment-level tracking

---

## Phase 13: RLHF Pipeline — Preference Learning from HITL Feedback

### Why this belongs in the `orchestrated` branch, not a separate track

The RLHF pipeline is not an independent research concern — it is the natural downstream of what the HITL design was always pointing toward. `PostGenerationFeedback` already captures the preference signal (accepted vs rejected/regenerated responses, satisfaction scores 1–5, format and context notes). MLflow already logs it across runs. The infrastructure for collecting preference pairs exists; RLHF is what you do with them.

The `orchestrated` research branch is the right home for this — as an extension of the PEFT work already scoped there, not as a standalone track. The fine-tuning branch uses static Q&A pairs; the orchestrated branch closes the loop by using live preference signal to drive continuous adaptation. The connection is direct.

### Preference data already being collected

Every HITL-2 interaction produces a structured preference record:

- **Chosen**: the response the developer accepted (satisfaction ≥ threshold, decision = `accept`)
- **Rejected**: the response they asked to regenerate (decision = `regenerate`), with notes on why
- **Context**: which tools were called, which chunks were retrieved, what the supervisor adjusted

This is a DPO-compatible preference dataset being generated passively during normal use. MLflow run IDs tie each preference pair to its full execution trace, making the dataset auditable and filterable.

### PEFT approaches under consideration

Three PEFT families are relevant, in increasing order of complexity:

**LoRA / QLoRA** — the baseline approach already scoped in the `fine-tuning` branch. Low-rank adapter layers trained on Q&A pairs. QLoRA extends this to quantised base models (4-bit), making fine-tuning feasible on a single consumer GPU. Well-understood, widely supported, the right starting point.

**Soft prompts (prompt tuning)** — a lighter-weight alternative to weight updates. Instead of adapting model weights, a set of learnable continuous token embeddings is prepended to every prompt. The model weights are frozen; only the soft prompt vectors are trained. Key properties:
- Significantly lower memory and compute requirements than LoRA
- No weight merging or adapter management — the soft prompt is a small tensor, easily versioned and swapped
- Effective at steering model behaviour toward a particular style or domain without catastrophic forgetting risk
- Less expressive than LoRA for large distribution shifts, but well-suited to the narrower task here (code documentation style alignment)
- Natural fit for the `format_notes` field in `PostGenerationFeedback` — style preferences accumulate into a learnable prompt bias

**DPO (Direct Preference Optimisation)** — trains directly on preference pairs (chosen vs rejected) without a separate reward model. Simpler and more stable than PPO-based RLHF, and directly compatible with the MLflow preference dataset. This is the primary RLHF technique to implement in the `orchestrated` branch. Soft prompts and DPO are complementary: soft prompts handle style alignment, DPO handles quality alignment.

**PPO-based RLHF** — the full pipeline: reward model trained on preference pairs, PPO updates the policy. Most powerful but most complex. Deferred to a later phase once DPO results are characterised.

### Relationship to the branching structure

```
fine-tuning branch   → LoRA/QLoRA on static Q&A pairs
orchestrated branch  → soft prompts + DPO on live HITL preference pairs from MLflow
```

The jump from "collect feedback" to "train on feedback" is smaller than it appears — the infrastructure is already there. This is not a new research direction; it is the completion of what the HITL design implied from the beginning.

---

## Phase 14: Helm Deployment Knobs — Composable Configuration Model

### Three orthogonal axes

The dev branch introduces three independently composable configuration values that together define the full deployment profile:

```yaml
modelTier:       "heavy" | "full" | "balanced" | "lightweight" | "minimal"
quantisation:    "q4_K_M" | "q8_0" | "fp16"
deploymentTarget: "local" | "cluster"
```

These are orthogonal by design: `modelTier` is the capability selector, `quantisation` is the resource selector, and `deploymentTarget` is the infrastructure selector. You can tune memory vs quality vs deployment complexity independently at install time.

### `modelTier` — capability selector

| Tier | Model | Notes |
|---|---|---|
| `full` | mistral-nemo:12b-instruct | Best quality, GPU + 12Gi+ RAM |
| `heavy` | deepseek-coder-v2:16b-lite-instruct | Explicit/opt-in, ~13GB VRAM for a full GPU fit (partial offload on 12GB), MoE, strong polyglot reasoning; see Phase 19 |
| `balanced` | qwen2.5-coder:7b | Best code understanding at mid-range |
| `lightweight` | phi3.5 | Edge/low-resource, ~4Gi RAM |
| `minimal` | qwen2.5-coder:3b-instruct | CI pipelines / very constrained dev |

The `minimal` tier is specifically intended for CI pipeline runs and constrained developer machines where the goal is pipeline validation, not output quality.

### `quantisation` — resource selector

Composed with `modelTier` by `_helpers.tpl` to produce the final Ollama model tag:

| Value | Suffix | Memory impact |
|---|---|---|
| `q4_K_M` | `-q4_K_M` | ~50% reduction, minimal quality loss |
| `q8_0` | `-q8_0` | ~25% reduction, near lossless |
| `fp16` | (none) | Full precision, full footprint |

`phi3.5` and `qwen2.5-coder:3b` use Ollama's default quantisation (already Q4), so no suffix is appended for `lightweight` and `minimal` tiers.

Example compositions:
- `heavy` + `q4_K_M` → `deepseek-coder-v2:16b-lite-instruct-q4_K_M` (~13Gi)
- `balanced` (any quantisation) → `qwen2.5-coder:7b` (~4.5Gi; bare tag, no quant suffix)
- `full` + `q8_0` → `mistral-nemo:12b-instruct-q8_0` (~12Gi)
- `full` + `fp16` → `mistral-nemo:12b-instruct` (~14Gi)

### `deploymentTarget` — infrastructure selector

Controls resource profiles, storage backends, healthcheck intervals, and ChromaDB HNSW tuning.

**`local`** (developer laptop / single-node):
- No resource requests/limits on app or Ollama containers (meaningless on dev machines, can cause unnecessary scheduling friction)
- Lighter healthcheck intervals to reduce startup noise
- ChromaDB HNSW `searchEf` reduced to 20 (faster queries, less accurate — acceptable for single-user dev)
- ChromaDB persistence uses host-path volumes
- MLflow uses local SQLite backend

**`cluster`** (production Kubernetes):
- Resource requests/limits enforced, scaled by `modelTier` + `quantisation`
- Full HNSW `searchEf` from `values.yaml` (better recall for concurrent load)
- PVCs for all stateful services (ChromaDB, MLflow, Ollama)
- External MLflow URI supported
- GPU node selector applied for `full` tier

### ChromaDB HNSW tuning

HNSW parameters are now explicitly configured in `values.yaml` under `vectordb.hnsw` and propagated via `_helpers.tpl`. Previously only `hnsw:space: cosine` was set; the remaining parameters defaulted to ChromaDB's global defaults (which are not tuned for this workload).

| Parameter | Default | Effect |
|---|---|---|
| `M` | 16 | Bidirectional links per node — higher = better recall, more memory |
| `constructionEf` | 100 | Candidate list size at build time — higher = better recall, slower build |
| `searchEf` | 50 (cluster) / 20 (local) | Candidate list size at query time — higher = better recall, slower query |

These values are reasonable defaults for collections in the tens-of-thousands of chunks range. For significantly larger collections, `M: 32` and `constructionEf: 200` are worth evaluating.


---

## Phase 15: MCP Integration — Split Tool Registry

### Design: split registry over unified wrapper

Tools are divided into two explicit categories with a unified dispatch surface:

```
LOCAL_TOOL_REGISTRY  — plain Python callables, always available, no LLM capability requirement
MCP_TOOL_REGISTRY    — MCP-backed tools, offered only when is_mcp_capable() is True
TOOL_REGISTRY        — unified view built at graph time; what node_tool_selection sees
```

The split is architecturally intentional rather than incidental. A unified wrapper that presents both tool types identically to the LLM would silently degrade for lightweight and minimal tier models — those models often cannot reliably produce the structured JSON that MCP tool calls require. The split makes this explicit: `is_mcp_capable()` returns True only for `full` and `balanced` tiers, and `build_tool_registry()` filters accordingly. Local tools are always available; MCP tools are only offered when the LLM can use them.

This also preserves the duality of approach: different LLMs and different deployment configurations use the same codebase, routing to the tool subset appropriate for their capabilities. A `minimal`-tier CI run gets the full local toolset; a `full`-tier interactive session gets local + MCP.

### MCP servers included on `dev`

**Filesystem MCP** — replaces the bespoke `_safe_path` + subprocess allowlist for file access with a declarative permission map:

| Access tier | Paths |
|---|---|
| read-only | source files (`*.py`, `*.ts`, `*.yaml`, `*.md`, configs) |
| read-write | generated artefacts (`docs/`, `reports/`, `*.generated.*`) |
| execute | blocked entirely |

The local `tool_cat`, `tool_find`, and `tool_stat` remain in `LOCAL_TOOL_REGISTRY` as fallbacks and are used unconditionally by lower-tier models. Filesystem MCP tools (`fs_read`, `fs_write`, `fs_list`) are preferred for MCP-capable LLMs — access control is declarative and auditable at the server level rather than enforced through Python path validation.

**GitHub MCP** — replaces `tool_github_fetch` for external repo operations and extends it:
- `github_get_file` — OAuth-handled file fetch, rate-limit aware
- `github_list_prs` — list open PRs; useful for correlating code with PR descriptions
- `github_get_pr_diff` — full diff for a PR; best for documenting what a feature branch changed
- `github_get_issue` — issue body and comments; understanding intent behind a change

Commit and push operations are explicitly excluded on `dev`. The documentation agent reads and reasons about the codebase; it does not write to it. CI/CD owns that.

**Slack MCP** — serves two distinct purposes on `dev`:

1. **Developer workflow**: query the assistant inline from Slack (`#code-review`, `#docs`) without opening the Streamlit UI. This makes the assistant ambient in the team's existing tool rather than a separate destination.
2. **Self-documentation**: as a side-effect of a documentation run, the agent can post a summary to a relevant channel. The assistant becomes part of the team's knowledge flow, not just a query tool.

This is a `dev`-branch concern, not a production commitment — in production, appropriate integrations depend on the deployment context and the assistant's role. The PoC/prototype framing here means Slack MCP is a demonstration of the pattern.

### MCP servers deferred to research repo

**Memory MCP** — cross-session preference persistence. `SessionPreferences` in `dev` is deliberately session-scoped (cleared on "Clear conversation", never persisted to MLflow as a model artefact). Memory MCP extends this: preferences accumulated across sessions become a persistent prior retrieved at the start of each session. This raises questions about preference drift, session boundary detection, and privacy that are out of scope for `dev`. Relevant to all research branches.

**Qdrant MCP** — if `VECTOR_STORE_BACKEND=qdrant` is added (see Phase 14 extensions), the Qdrant MCP server gives the agent direct query access rather than going through the Python client. Relevant to both `orchestrated` (supervised retrieval strategy adaptation) and `emergent` (agents sharing vector state without a central coordinator). The `VectorStoreBase` abstraction already accommodates a `QdrantVectorStoreImpl` without changes to the agent graph.

### Vector DB knob relevance at scale

The `vectordb.hnsw.*` parameters and the `VECTOR_STORE_BACKEND` knob are modest optimisations for the current single-codebase, single-user context. Where they become significantly more relevant:

- **Multi-agent systems**: agents sharing a vector store surface contention on HNSW index locks, making tuning load-dependent. Qdrant's payload filtering becomes more valuable when multiple agents query the same index with different scoping requirements.
- **Research repo `emergent` branch**: agents coordinating through shared vector state (no central supervisor) need a vector DB that supports concurrent writers without coordination overhead. ChromaDB's single-node HNSW is a bottleneck; Qdrant or a distributed index is the right substrate.
- **Research repo `orchestrated` branch**: retrieval strategy adaptation (adjusting index parameters based on preference signal) requires the vector DB to support parameter updates without full re-indexing. A research-level concern, not a dev concern.


---

## Phase 16: Graph Database — Structural Retrieval Alongside Vector Search

### The core distinction

Vector DB and graph DB are orthogonal retrieval mechanisms, not alternatives:

| | Vector DB (ChromaDB) | Graph DB (Kuzu) |
|---|---|---|
| Query type | "What is semantically similar?" | "What is structurally related?" |
| Answer | Approximate, ranked | Precise, enumerable |
| Best for | Fuzzy conceptual questions | Traversal, multi-hop structural questions |
| Needs embeddings? | Yes | No |
| Example | "How does caching work?" | "What calls `node_supervisor`?" |

Graph DB does not need vector embeddings at all. For structural/relational queries — call graphs, import chains, co-change relationships — it gives exact answers where the vector DB gives approximate ones. The hybrid pattern uses both: graph traversal to find the structural neighbourhood, then vector search within that neighbourhood for semantic depth.

### Do knowledge graphs require vector DBs?

No. This is an important clarification. A graph DB with only explicit, hand-crafted or AST-extracted relationships works entirely without embeddings. Embeddings only enter when you need semantic similarity as an edge — "conceptually related to", "similar purpose as". The three graph structures built here use no embeddings: all edges are structurally derived from AST parses and git history.

### Three graph structures built during ingestion

**Code dependency graph** — built from tree-sitter AST parse (the same parse already running for chunking). Nodes: `File`, `Function`, `Class`. Edges: `IMPORTS`, `DEFINES`, `CALLS`, `INHERITS`, `DEFINES_CLASS`. Enables precise call chain traversal and import relationship queries.

**Co-change graph** — built from `git log --name-only`. Nodes: `File`. Edges: `CHANGED_TOGETHER` with a weight equal to co-occurrence count across commits. Captures *logical coupling* — files that change together even when not structurally linked. Enables: "if I change X, what else will likely need updating?"

**Knowledge graph (documentation layer)** — deferred to research/orchestrated. Nodes: `Concept`, `DesignDecision`, `Component`. Edges: `IMPLEMENTS`, `DEPENDS_ON`, `DOCUMENTED_IN`, `REPLACED_BY`. Source: structured extraction from ARCHITECTURE.md and docstrings. This is where the full power of a knowledge graph — generic and specialised associations, multi-dimensional concept relationships, cross-codebase analogies — becomes relevant. The `orchestrated` branch's retrieval strategy adaptation particularly benefits here: rather than adjusting retrieval parameters blindly, the supervisor can reason about *which structural neighbourhood* is relevant to a query before deciding retrieval strategy.

### Why Kuzu, not Neo4j

Kuzu is not derived from Neo4j — they share only the openCypher query language (an open standard). The right mental model is "Neo4j : PostgreSQL :: Kuzu : SQLite". Kuzu is embedded (no server process), written in C++, Python-native, built at University of Waterloo (2022). Zero new services, zero new infrastructure — the graph DB is a directory on disk alongside the ChromaDB persistence volume, and a single `pip install kuzu`.

### Integration in the pipeline

Graph construction runs as the final step of `ingest_codebase()`, after the vector index is built, using the same `files` list. It is additive and non-blocking — if it fails, the vector index is returned successfully. Controlled by `GRAPH_ENABLED` env var (default: true when kuzu is installed).

```
ingest_codebase()
  └── discover_files()          → files list
  └── load_and_chunk_files()    → nodes
  └── build_index()             → VectorStoreIndex  ← unchanged
  └── build_graphs()            → KuzuGraphStore     ← new, parallel output
        └── build_dependency_graph()   (AST-derived)
        └── build_co_change_graph()    (git history)
```

### New tool: `graph_traverse`

Added to `LOCAL_TOOL_REGISTRY` (always available, no MCP capability requirement). Seven query types:

| query_type | Question answered |
|---|---|
| `callees` | What functions does this function call? (depth-limited) |
| `callers` | What functions call this function? (depth-limited) |
| `dependencies` | What files does this file import? |
| `dependents` | What files import this file? |
| `co_changed` | What files frequently change together with this one? |
| `symbols` | What functions and classes are defined in this file? |
| `cypher` | Raw Cypher for advanced/custom traversals |

### Locality-sensitive domains and the path toward knowledge graphs

Your intuition about locality-sensitive domains is the right framing. For queries with a natural locality in the code graph — "everything in this module", "everything that calls into this subsystem", "everything that changed with this PR" — the graph DB gives exact, bounded answers without relying on the vagaries of semantic similarity. This is particularly valuable in domains where terminology is dense and overloaded (a common function name appears in many contexts; the vector DB can't distinguish them structurally).

The path from code dependency graph → knowledge graph is an enrichment of the same structure: starting with structurally-derived edges (CALLS, IMPORTS), adding semantically-derived edges (SIMILAR_PURPOSE, CONCEPTUALLY_RELATED), then adding provenance edges (MOTIVATED_BY, REPLACED_BY). As the graph becomes richer, it enables increasingly generic and increasingly specialised queries simultaneously — the knowledge graph is the same structure, just with more edge types and more heterogeneous nodes.

This is explicitly `orchestrated` branch territory in the research repo: graph-aware retrieval strategy adaptation, where the supervisor doesn't just adjust retrieval parameters but reasons about which part of the graph is relevant to a query before retrieval begins.


---

## Phase 17: llama.cpp — Third Inference Backend

### Why a third backend

Ollama wraps llama.cpp for most models — so in practice, the CPU efficiency difference between Ollama and llama.cpp directly is smaller than it appears for most tiers. The distinction matters specifically for `lightweight` and `minimal` tiers on constrained machines where:

- You want to run a specific GGUF file without Ollama's model management layer
- You need maximum control over context size, thread count, and GPU layer offload
- You are running in a CI pipeline or very constrained environment where even the Ollama daemon adds overhead

`llama-server` (llama.cpp's HTTP server mode) exposes an OpenAI-compatible `/v1/chat/completions` API identical to vLLM's. This means the `llamacpp` backend reuses `ChatOpenAI` pointed at `LLAMACPP_HOST` — no new LangChain client, no new interface, just a different endpoint.

### Backend decision tree

```
GPU available, serving multiple users  → vllm
CPU-only or constrained, need control  → llamacpp
Want model management / easy pulls     → ollama (default)
```

### Tier affinity

| Backend | Recommended tiers | Notes |
|---|---|---|
| ollama | full, balanced, lightweight, minimal | Default; model management included |
| vllm | full, balanced | GPU required; high-throughput |
| llamacpp | lightweight, minimal (CPU); heavy (GPU, partial offload) | GGUF; lowest overhead; heavy tier verified on a 12 GB GPU with 24 offloaded layers |

`lightweight` and `minimal` suppress the quantisation suffix (already Q4 by default), so llama-server receives e.g. `phi3.5` or `qwen2.5-coder:3b-instruct` as the served model name.

### Models must be downloaded manually

Unlike Ollama, llama-server does not pull models. GGUF files must be placed in `./models/` before starting the `llamacpp` profile. The `docker-compose_dev.yml` service definition includes a download example in comments.

### max_tokens is conservative for llamacpp

`_get_llm()` sets `max_tokens=2048` for the llamacpp backend (vs 4096 for vllm). llama-server's context size is set at startup via `--ctx-size` and cannot be exceeded at the API level — a conservative default prevents silent truncation.

---

## Phase 18: Testing Infrastructure

### Scope

`test_pipeline.py` is a single-file, no-external-services test suite covering the full dev branch stack. It runs entirely in-process — ChromaDB embedded, Kuzu embedded, LLM calls mocked via importlib reload, local trigram embeddings substituted for Ollama embeddings.

### Test classes and what they cover

| Class | Coverage |
|---|---|
| `TestConfig` | All tier/quant/backend combinations, HNSW params, deployment target, MCP flags |
| `TestDiscovery` | File discovery, chunking, metadata, skip-dir behaviour |
| `TestVectorStore` | HNSW metadata applied correctly for local vs cluster; in-process roundtrip |
| `TestGraphStore` | Kuzu init, upsert, import/call/co-change edges, raw Cypher, reset |
| `TestToolRegistry` | Local tools present, MCP registry present, MCP capability gating per tier, run_tool dispatch |
| `TestAgentState` | Dataclass integrity, SessionPreferences.update(), running average, format propagation |
| `TestInferenceBackend` | _get_llm() returns correct type per backend, host/port separation, max_tokens |
| `TestRetrievalQuality` | End-to-end embed→store→retrieve, per-file hit assertions, regression guard |
| `TestMinimalDeployment` | All knobs consistent for minimal + llamacpp + local: model tag, chunking, HNSW, registry, backend |

### Bugs found during test authoring

Three real bugs were caught by the tests that would have caused silent failures in production:

1. **Kuzu `mkdir` before init** — `Path.mkdir()` pre-creating the graph path caused Kuzu to fail with "cannot be a directory". Fixed: only `mkdir` the parent; Kuzu creates the database path itself.
2. **Kuzu `reset()` using `shutil.rmtree`** — Kuzu stores the database as a file, not a directory. `rmtree` raised `NotADirectoryError`. Fixed: detect file vs directory and use `path.unlink()` accordingly.
3. **`end` is a reserved word in Kuzu Cypher** — `RETURN fn.end_line AS end` caused a parser exception. Fixed: renamed to `end_line` throughout `graph_store.py` and `tools.py`.
4. **`ingest.py` top-level Ollama import** — `from llama_index.embeddings.ollama import OllamaEmbedding` at module level prevented import in test environments without the llama-index extras package. Fixed: moved to lazy import inside `build_index()` and `load_existing_index()`.

### Running the tests

```bash
# Full suite (no external services required)
python test_pipeline.py

# Specific class
python -m pytest test_pipeline.py::TestMinimalDeployment -v

# Smoke test (fast summary, no unittest verbosity)
python test_pipeline.py smoke

# Test a specific deployment profile
MODEL_TIER=minimal INFERENCE_BACKEND=llamacpp DEPLOYMENT_TARGET=local python test_pipeline.py
```

---

## Phase 19: `balanced` Tier Reverted — MoE VRAM Footprint ≠ Compute Cost

### What happened

`balanced` was changed from `qwen2.5-coder:7b` to `deepseek-coder-v2:16b-lite-instruct` on the reasoning that DeepSeek-Coder-V2-Lite's Mixture-of-Experts architecture is compute-efficient (~2.4B active parameters per token out of 16B total) and effective for polyglot codebases.

That reasoning is true for **compute** but doesn't apply to **VRAM**. MoE VRAM footprint is dominated by total parameters, not active parameters — every expert has to sit resident in memory even though only a subset fires per token. A 16B MoE model costs roughly 16B-worth of VRAM regardless of how "lite" its per-token compute is. Verified in practice on a 12GB card: `ollama ps` showed `deepseek-coder-v2:16b-lite-instruct-q4_K_M` at 13GB, split `20%/80% CPU/GPU` — it didn't fit, and Ollama silently fell back to partial CPU offload rather than erroring. This produced a multi-minute hang with no visible error (compounded by the `num_ctx` fix in Phase 17/18-era work actually requesting the full 8192-token context for the first time, where previously the silent-2048-token default had been masking the problem by keeping the KV cache small enough to still barely fit).

### Fix

`balanced` reverted to `qwen2.5-coder:7b` (dense, ~4.5Gi at Q4), restoring the intended `heavy` (opt-in) > `full` (12B) > `balanced` (7B) > `lightweight` (3.8B) > `minimal` (3B) size ordering. `qwen2.5-coder:7b` also does **not** compose a quantisation suffix (joins `lightweight`/`minimal` in this respect) — no verified `qwen2.5-coder:7b-<quant>` Ollama tag was confirmed to exist, and guessing one risks reintroducing the exact "model not found" 404 class of bug this session spent considerable effort diagnosing and fixing (see Phase 17/18 test coverage). The bare tag is the one confirmed working, going back to the original `master`-branch design.

DeepSeek-Coder-V2-Lite was **not discarded** — it now has its own explicit, separately-labelled `heavy` tier, above `full` in the dropdown, rather than silently occupying `balanced`'s slot. `heavy` *does* compose a quantisation suffix — `deepseek-coder-v2:16b-lite-instruct-q4_K_M` was empirically confirmed to exist and pull successfully during this session's testing (visible directly in `ollama ps` output), so unlike `balanced`'s case there's no guessing involved. `heavy` is explicit/opt-in specifically because its ~13GB footprint needs real VRAM headroom (16GB+) that a 12GB card doesn't have; the UI surfaces a warning when it's selected, both in the dropdown help text and in the active-override info banner, rather than letting the person rediscover the partial-CPU-offload hang the hard way.

### Evolution of thinking — active parameters vs resident parameters

The distinction that was missed: "lite" in an MoE model's name describes its *compute* cost, not its *memory* cost. This is a genuinely easy mistake to make when reasoning about model selection from benchmarks and architecture descriptions alone, without checking actual `ollama ps`/`nvidia-smi` output against real hardware. The concrete lesson for this project: tier definitions should be validated against actual VRAM measurements on representative hardware, not just parameter-count-weighted compute estimates — a gap that motivates the VRAM-fit capability check proposed as follow-up work (querying `ollama ps`'s size/processor split before committing to a hot-swap, rather than after).

<!-- docs-update:phases-mlflow-helm -->
---

## Phase 20: Per-Question MLflow Runs and Automatic Deployment Defaults

### Per-question runs

**Problem.** The UI advances the LangGraph pipeline one node per Streamlit rerun (so the Pipeline tab can repaint), and a human-review pause can last minutes. MLflow's fluent API keeps the active run per thread, so a run opened with `with mlflow.start_run():` cannot span those reruns. Node-level `mlflow.log_*` calls then each opened a stray run, and run-level parameters were never set.

**Decision.** `src/tracking.py` keeps one run open per question on the server and re-activates it for exactly the duration of each node:

```
run_id = tracking.start_query_run(...)        # once per question -> state["mlflow_run_id"]
with tracking.activate(run_id): node(state)   # around every node (build_graph does this)
tracking.finish_run(run_id, final_state)      # once: answered / rejected / failed
```

Every public function is failure tolerant: an unreachable or missing MLflow turns tracking into a no-op with one warning, and client calls fail fast so a dead server cannot stall a question. A run is the unit later used for backend x model evaluation (params, metrics and `result.json` per question, `eval.group` to group a batch).

**Gotcha found on the real stack.** MLflow's allowed-hosts check returned 403 for the app's `mlflow:5000` Host header, silently. `--allowed-hosts` is now set in compose and generated in Helm.

### Saving conversations

A conversation can be downloaded as a Markdown transcript (thread, repositories, resolved model, messages). Since Phase 22 the transcript also carries the indexing outcome and each answer's pipeline trace, so a saved conversation shows which steps warned and what was retrieved, not only the answers.

### Human review: three-way tool-plan rejection

Rejecting a tool plan used to mean "stop". It is now **end**, **re-plan**, or **re-plan with feedback**, where the reviewer's note is handed to the planner. A cap (`MAX_REPLANS`) keeps a plan/reject loop finite, and each decision is logged on the question's run.

### Automatic deployment defaults

**Problem.** The "container default" backend and tier were whatever environment variables happened to be set, which drifted from what was actually running (start the llama.cpp profile with a heavy GGUF and the app still said Ollama / full).

**Decision.** `src/deployment.py` is the single source of the container default. With `INFERENCE_BACKEND=auto` it probes the stack in priority order (llama-server, then vLLM, then Ollama), reads the served model from the server and infers the tier from the model name (first match wins, specific names first). Positive answers are cached for 20 s and misses for 4 s, so a server that starts later is noticed quickly. Explicit `INFERENCE_BACKEND` values pin the backend and Helm always sets one.

**Placement finding.** An Ollama embedding model on the same 12 GB GPU as the resident heavy model collapsed llama-server throughput (132-177 tok/s down to under 7 until restart). `src/embedding.py` is the one place the embedding client is built and runs it on CPU by default (`EMBED_NUM_GPU`). Placement is a deployment variable on a par with `--n-gpu-layers`, and belongs in any backend x model evaluation.

---

## Phase 21: Helm Chart 0.2.0 — llama.cpp and GPUs on Kubernetes

**Goal.** Reach compose parity on Kubernetes: the same backends, tiers and observability, composed from values rather than edited per environment.

### What changed in the chart

- The `dev` chart could not render at all: `app-deployment.yaml` used a `chunkingStrategy` helper `_helpers.tpl` did not define. Added.
- `heavy` tier everywhere tiers are resolved (model tag, resources, context and timeout; llama.cpp + heavy gets a 180 s timeout).
- llama.cpp as a first-class component: Deployment, Service, models volume (PVC, hostPath or existing claim), optional GGUF download init container, tier defaults mirroring compose, `extraArgs`, configurable startup probe.
- `inferenceBackend: auto`; `llamacpp.enabled` / `vllm.enabled` deploy a backend beside Ollama for comparisons; the Ollama pod skips the chat-model pull unless Ollama may answer.
- MLflow: generated `--allowed-hosts`, probes with a pinned Host header, configurable resources.
- App env mirrors compose (`LLAMACPP_*`, `EMBED_NUM_GPU`, `MAX_REPLANS`, `MLFLOW_*`, `EVAL_GROUP`).

### GPU access is a platform property, not a chart property

The first question was whether the chart can be "platform independent". It can for everything except how a GPU reaches a container, so that is exposed as one small, explicit setting (`llamacpp.gpu.mode` / `wsl2`) rather than hidden. On a standard Linux GPU node the default (`nvidia.com/gpu` via the device plugin) just works. On Windows + WSL2 it does not, for reasons found one at a time:

1. WSL2 exposes the GPU through `/dev/dxg` plus Windows driver libraries under `/usr/lib/wsl`, not through NVIDIA device nodes.
2. Docker Desktop's built-in (kind) cluster cannot expose it. **minikube with the docker driver and `--gpus=all`** can: the node container sees the GPU and `nvidia-smi` works in it.
3. The NVIDIA device plugin (v0.20.0) then fails NVML initialisation ("Not Supported"), so `nvidia.com/gpu` is never advertised. `gpu.mode=runtime` (container runtime injects the GPU, no resource request) avoids the plugin but still failed with `Failed to initialize NVML`.
4. Root cause: nested containers do not receive `libdxcore.so`. Mounting the node's copy (`/usr/lib/x86_64-linux-gnu/libdxcore.so`, hostPath type File) into the container fixes it; this was verified by hand with a plain `docker run` in the node before being encoded as `gpu.wsl2=true`.

Result: llama-server ran the heavy tier on the RTX 5070 Ti (about 10.4 GB VRAM in use) and answered through the app. Neither symlinks into `/usr/lib/wsl/lib` nor `--device /dev/dxg` helped.

### Startup is storage-bound

The first model load took about 15 minutes (cold read of a 10 GB file through the minikube volume) and exceeded the original 15-minute startup budget, so the pod was killed (exit 137) and restarted repeatedly. A warm load took 3.5 minutes. The budget is now `llamacpp.startupFailureThreshold` (default 180 x 10 s) rather than a constant.

### Smaller findings

- `--port={{ x | quote }}` renders literal quotes (`--port="5000"`) and crash-loops MLflow; the template is unquoted and a regression test covers it.
- Windows downloads leave `:Zone.Identifier` files that Helm rejects; clean before every helm command.
- Kubernetes has no `src/` bind mount: rebuild the image and `minikube image load` after code changes.
- With Flash Attention reported as unsupported for this model on the CUDA build, attention falls back to the slower path; noted as a performance item.

### How it was verified

- `helm lint` and `helm upgrade` on a real install, and the full stack on a minikube GPU cluster: app, ChromaDB, MLflow, Ollama (embeddings) and llama.cpp all `Running`, a question answered through llama.cpp on the GPU, restart count 0 after the startup-budget change.
- Because the `helm` binary was not available in the authoring sandbox, the chart is also covered by a stand-in Go-template renderer and ~100 render assertions (backend wiring, tier defaults, probe budgets, GPU modes, MLflow flags).

### Not yet covered

GPU scheduling on a real multi-node cluster with the device plugin (the default `resource` mode is rendered and tested, but not run), vLLM on the chart end to end, and per-model llama.cpp load settings read back from the server.

---

## Phase 22: Silent Failures — Awareness Across Layers

**Goal.** A pipeline that *looks* like it works but has silently lost its retrieval must say so. This phase came from a real incident on the Kubernetes deployment.

### The incident

Questions were answered fluently, the Pipeline tab was green and MLflow logged a normal run, yet no Sources were cited and the answers were generic. Cause, found layer by layer:

1. **Chart (init container).** The Ollama pod pulls the embedding model in an init container that started `ollama serve &`, slept a fixed 5 s, then ran `ollama pull`. The pull lost the race ("could not connect to ollama server"), but the script had no error handling, printed "Model pull complete." and exited 0. The pod was `Running` with an empty model store.
2. **Indexing.** Every embedding call returned 404, so ingestion created the ChromaDB collections and stored **0 chunks**. `ensure_indexed` caught the exception and returned `status: "error"`, but the app ignored the status.
3. **Retrieval.** `vector_search` over empty collections returned 0 chunks, which the trace still reported as `ok`. Context assembly produced `[No relevant context found]`, the LLM answered from nothing ("no specific information…") and the run was logged as answered.

No layer raised, so the failure was invisible, and it could stop the whole pipeline's usefulness while every health signal stayed green. (The same applies to *any* empty retrieval: wrong collection, unreadable index, a question that matches nothing.)

### What changed

| Layer | Before | Now |
|---|---|---|
| Ollama init container | fixed sleep, no `set -e`, exit 0 on failed pull | waits for the server (up to 180 s), retries each pull 5x, **exits non-zero** on failure, and verifies the embedding model is listed before reporting completion |
| `repo_index.ensure_indexed` | tried to ingest and swallowed the error | **preflight**: asks Ollama whether the embedding model exists (`embedding_ready`); an ingestion that stores 0 chunks is an error; errors carry a `hint` (the exact `ollama pull` command) |
| App | continued after an indexing error | **hard stop**: shows the error and hint, does not run the pipeline, closes the MLflow run with `outcome=error`; also stops when every selected repo is empty |
| `tool_execution` | `ok` regardless of hits | `warn` when `vector_search` ran but returned 0 chunks |
| `context_assembly` | no trace entry; `break` dropped *every* chunk if the best one exceeded the budget | trace entry (chunks kept / dropped / truncated, tokens used of budget); an oversize first chunk is **truncated, not dropped**; sets `retrieval_empty` when nothing is left; `warn` when more than 30% of the retrieved chunks were dropped (`CONTEXT_DROP_WARN_FRACTION`) |
| `generation` | called the LLM with an empty context | **skipped** when `retrieval_empty`: the answer states that nothing was retrieved, why (per-collection chunk counts) and what to check; metric `retrieval_empty=1` |
| `output_review` | rated the non-answer (and could push a 5/5 into the preference profile) | skipped with a `warn` entry |
| MLflow | outcome `answered` | outcome `no_retrieval` (or `error` for an indexing hard stop) |
| UI | the 📂 indexing lines and the live progress vanished at the end of the run | indexing results persist in the chat; every answer carries a collapsible **Pipeline trace** with ⚠️ warnings; `warn` nodes are orange in the diagram |

### Design notes

- **Fail where the cause is, say it where the user looks.** The init container fails the pod (cause); the app turns the same condition into a message with the fix (symptom). Neither depends on the other.
- **Warn is a status, not a log line.** `warn` flows through the trace, the diagram, the persisted answer and MLflow, so evaluation runs can filter on it.
- **StatefulSet stays.** A Deployment would not remove the race (it is an init-script problem) and would give up the stable volume for the model store.
- **Indexing is still blocking.** The first index of a repo embeds every chunk on the Ollama CPU (about 1,600 chunks per repo here) and can take minutes; the status box tells the user so and the per-chunk progress is in the app log. Running ingestion as a background job with visible progress is in `future_directions.md`.

### Verified

App-flow tests (real Streamlit `AppTest`, real graph) cover the hard stop (no LLM call, error shown, run closed as `error`), the empty-retrieval answer (warn entries, no sources, outcome `no_retrieval`) and the persistent trace. Chart-render tests check the init script (waits, retries, fails hard, verifies). On the real cluster the fixed init container pulled and verified `nomic-embed-text`, and indexed repos then produced cited answers from both repositories.

---

## Appendix: Reference Test Infrastructure

Everything in Phases 21 and 22 was built and verified on one concrete rig. It is documented so the results are reproducible and so the parts that are specific to it are easy to adapt.

### The rig

| Layer | What | Notes |
|---|---|---|
| Hardware | Laptop, RTX 5070 Ti (12.2 GB VRAM) | about 10.4 GB used by the heavy tier |
| OS | Windows 11 + WSL2 (Ubuntu) | WSL had about 11 GB RAM available |
| Container runtime | Docker Desktop (WSL integration) | its built-in kind cluster cannot expose the GPU |
| Cluster | minikube, docker driver, `--gpus=all` | the Kubernetes node is itself a container |
| Chart | `helm/code-doc-assistant` 0.2.0 + `values-wsl2-minikube.yaml` | the preset holds everything specific to this rig |
| Inference | llama.cpp `llama-server`, DeepSeek-Coder-V2-Lite Q4_K_M, `--ctx-size 4096`, 24 GPU layers | |
| Embeddings | Ollama `nomic-embed-text` (CPU) | on the GPU it competes with llama-server for VRAM |

### Layering and why it matters

```mermaid
flowchart TB
    BROWSER["Windows browser\nlocalhost:8501 / :5000"] --> PROXY["socat proxy containers\n(Docker, --restart unless-stopped)"]
    PROXY -->|"Docker network 'minikube'"| NODE["minikube node = a container\n(runs the Kubernetes node)"]
    NODE --> SVC["NodePort services\n30501 app · 30500 MLflow"]
    SVC --> PODS["pods: app · ChromaDB · MLflow\nOllama (embeddings) · llama-server (GPU)"]
    GPU["RTX 5070 Ti\n/dev/dxg + libdxcore.so"] -.-> NODE
```

Nesting (Windows, WSL2, Docker, minikube node, pod) explains the platform-specific parts:
- **GPU access:** WSL2 exposes the GPU as `/dev/dxg` plus Windows driver libraries. Nested containers do not get `libdxcore.so`, and the NVIDIA device plugin fails NVML, so the chart's `llamacpp.gpu.wsl2=true` uses runtime injection and mounts the node's copy (Phase 21).
- **Reachability:** the node's IP lives on a Docker-internal network, so NodePorts are not reachable from Windows. Two `alpine/socat` containers on the `minikube` network publish fixed local ports (8501, 5000) and forward to the NodePorts by container name. They survive pod replacement, `helm upgrade` and reboots, unlike `kubectl port-forward`, which is bound to one pod. The alternative, `minikube start --ports=...`, needs the cluster recreated, which wipes the model volume.
- **No bind mount:** app code changes need an image rebuild, a `minikube image load` under a new tag, and an upgrade with that tag (an image in use cannot be removed).
- **Cold start:** the first load of the 10 GB model took about 15 minutes (storage-bound), a warm load about 3.5 minutes; the startup probe budget is configurable.
- **Windows file markers:** `:Zone.Identifier` files and stray non-template files in `templates/` break `helm upgrade`.

### Adapting it

- **Linux GPU node:** drop the preset's `wsl2` line; the default `nvidia.com/gpu` resource mode (device plugin) applies. No proxies are needed: use NodePort or an Ingress.
- **Docker Desktop kind, Docker Compose:** no GPU through kind; Compose is the simpler path on one machine (`docker compose`, see README).
- **Smaller GPU:** lower `llamacpp.gpuLayers` or use a smaller tier; larger `--ctx-size` costs KV-cache VRAM (see `future_directions.md`).
- **Other platforms:** keep the same checks: the model-pull init log shows the embedding model listed, `nvidia-smi` shows `llama-server` holding VRAM, the first question's trace shows `context_assembly` with sources.
