# Architecture & Design Decisions

This document captures the thinking behind the component choices and design patterns in the Code Documentation Assistant. It's written chronologically — decisions are recorded as they were made, not reconstructed after the fact. The aim is to preserve the actual reasoning, including dead ends and trade-offs, rather than presenting a sanitised post-hoc narrative.

A plan was made at the outset, defining the approach in four phases: LLM provider selection, pipeline planning, deployment format, and component selection. Implementation and testing followed as phases 5 and 6.

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

**Task framing**: this is a *code comprehension + explanation* task, not code generation. The model must read retrieved code chunks, understand what they do, reason about relationships (dependencies, API endpoints, architecture), and explain in natural language. Reasoning capability — following multi-step logic across files — scales with parameter count and is the key differentiator between tiers.

**Models evaluated**:

| Model | Parameters | Strengths | Reasoning | Weaknesses |
|-------|-----------|-----------|-----------|------------|
| Mistral Nemo | 12B | Excellent code comprehension + explanation | Strong multi-step; traces cross-file dependencies | Needs GPU |
| DeepSeek-Coder V2 Lite | 16B (MoE) | MoE excels at polyglot codebases | Strong within-context reasoning | Variable memory patterns |
| Qwen2.5-Coder 7B | 7B | Best code comprehension at 7B tier | Adequate for single-file reasoning | May struggle with complex multi-module chains |
| Phi-3.5 Mini | 3.8B | Runs almost anywhere | Best for straightforward "what does this function do" queries | Not code-specialised |

*CodeLlama 7B was evaluated and excluded — Qwen2.5-Coder 7B supersedes it on modern benchmarks.*

**Tiered defaults**:

1. **Full** — Mistral Nemo (12B). Best balance of explanation quality and code understanding. For polyglot codebases, DeepSeek-Coder V2 Lite is the recommended swap — recent research on MoE architectures confirms they're particularly effective for multi-language tasks, treating programming language diversity analogously to natural language multilingualism ([Wang et al., 2025](https://arxiv.org/abs/2508.19268)).
2. **Balanced** — Qwen2.5-Coder 7B. Best-in-class at 7B. Right choice for ~8GB VRAM.
3. **Lightweight** — Phi-3.5 Mini (3.8B). Edge, CPU-only, or resource-constrained deployments. Fine-tuning candidate.

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

**Evolution of thinking — FAISS, LSH, and knowing when to stop:**

LSH came up during evaluation as an alternative indexing strategy. Building persistence, metadata filtering, and CRUD on top of FAISS is substantial engineering with no practical benefit at this project's scale. ChromaDB behind an abstraction layer is the right trade-off — not over-engineering for hypothetical requirements, but not closing the door on them either.

### 4d. Orchestration Framework

**Decision: LlamaIndex for the RAG pipeline; LangChain documented as a future growth path.**

LlamaIndex is purpose-built for RAG: native `CodeSplitter` with AST-aware chunking, tree-structured indexes, lighter weight than LangChain for pure retrieval-and-respond workflows.

**Production orchestration**: MLflow (prototyping) → W&B (production monitoring) → Ray on K8s (distributed compute). Ray Serve wraps Ollama for load balancing; Ray Data enables parallel ingestion for large codebases; KubeRay deploys natively on K8s.

**Evolution of thinking — from "which framework" to "what question am I actually answering":**

The instinct was to reach for LangChain — it's the default answer for AI orchestration. But LangChain is a general-purpose framework; this project is focused RAG. The real orchestration question isn't "which framework chains my prompts" but "how does this system scale operationally?" That's answered by MLflow/W&B for tracking and Ray/K8s for compute, not by a prompt-chaining library. LangChain enters the picture when the *application scope* grows (agents, tool use, CI/CD integration), not when the infrastructure scales.

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

## Next Steps

These are concrete, scoped extensions to the current system — things that would improve this specific assistant with more time and resources. They are grounded in what's already built and address known gaps.

### Advanced RAG Mitigations

The similarity cutoff and metadata guardrails in the current implementation are a first line of defence. The next layer of RAG quality improvements:

- **CRAG (Corrective RAG)** — filtering low-confidence retrievals at inference time, reducing retrieval errors by 12–18%; rather than passing all retrieved chunks to the LLM, a corrective pass identifies and discards chunks below a confidence threshold
- **Self-RAG** — the model learns to critique its own retrieval usage, deciding whether retrieved context is relevant before incorporating it
- **Re-ranking** — a secondary model re-scores retrieved chunks before they enter the prompt; computationally cheap relative to inference, but meaningfully improves retrieval precision
- **Context windowing** — prioritising recently-modified files for timeliness; stale chunks are a known failure mode for active codebases

### Chunking Granularity and Adaptive Retrieval

The current implementation chunks at function/class level — well-suited for "what does this function do?" but less effective for "how do these modules interact?" An adaptive retrieval approach would maintain multiple index granularities simultaneously: function-level for local questions, file-level for module-level questions, cross-file summaries for architectural questions. The retrieval layer would select the appropriate granularity based on the query.

### vLLM, Quantisation, and Production-Grade Inference

Ollama is the right choice for developer convenience and self-contained deployment. In a production environment with known infrastructure, a different set of optimisations becomes relevant.

**vLLM**: continuous batching, PagedAttention for efficient KV-cache management, native tensor parallelism across multiple GPUs. Where Ollama optimises for developer convenience, vLLM optimises for throughput and latency under concurrent load — critical when serving a team rather than a single developer. It supports OpenAI-compatible API endpoints, making it a drop-in replacement with minimal changes to `query_engine.py`.

**Quantisation** (GPTQ, AWQ, GGUF): reduces model memory footprint by 50–75% with minimal quality loss for code comprehension tasks. This would allow running the balanced tier on hardware currently limited to the lightweight tier, or the full tier on a single T4 via 4-bit quantisation. Beyond memory savings, quantisation can also increase context capacity and concurrent user support for the same resource envelope — a different set of trade-offs worth evaluating per deployment.

**Context and sequence length**: larger context windows (32K+ tokens) enable ingesting entire files or multi-file contexts in a single query, directly addressing the chunking granularity problem. vLLM's PagedAttention makes long-context inference practical without proportional memory scaling.

**Multi-GPU deployment**: tensor parallelism (splitting model layers across GPUs) and pipeline parallelism (splitting pipeline stages) become options with multiple GPUs. The embedding model and LLM could run on separate GPUs, eliminating the shared-VRAM constraint described in the README. Architecture-specific optimisations — FlashAttention-2 for Ampere+, INT8 on Turing, INT4 on Ampere, native FP4 on Blackwell (B100/B200/GB200) — continue to push the boundary of viable model sizes per GPU generation.

**Network and switching**: in distributed deployments (separate nodes for inference, vector DB, and app), network topology matters — NVLink for multi-GPU communication, RDMA/InfiniBand for inter-node model parallelism, co-location of the app and vector DB to minimise retrieval latency.

### Model Fine-Tuning

The RAG approach means the model receives relevant context at query time, which is sufficient for most questions without fine-tuning. Fine-tuning becomes relevant if base models consistently fail on specific languages or domains, or if a particular documentation style is required. This requires training data (code Q&A pairs), compute, and iteration — guided by observed performance gaps, not assumed in advance.

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

A natural evolution of the current single-branch project into a multi-branch, multi-environment system:

- **`dev`** — active development, experimental features, frequent iteration; uses lightweight tier by default
- **`fine-tuning`** — model adaptation branch; training data pipelines, evaluation harnesses, LoRA/QLoRA experiments against the codebase-specific Q&A pairs accumulated from production interactions
- **`production`** — stable, versioned deployments; full Helm chart, GPU-provisioned, monitoring enabled
- **`research`** — exploratory work; agentic and A2A (agent-to-agent) system prototypes, optimisation experiments, customer-specific system definitions; explicitly not production-bound

The research branch is where the platform vision above would be prototyped — agentic pipelines that can autonomously ingest, embed, and index new codebases; A2A coordination between specialised assistants; and the infrastructure for bespoke, customer-specific system definitions.

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
- Abstraction layers: vector store interface for DB-agnostic design
- Configuration: environment-variable driven, with parity between Docker Compose and Helm
- Testing: pipeline validation with embedded ChromaDB and AST chunking verification
- Logging: structured logging via Python `logging` module, configurable level
