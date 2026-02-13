# Code Documentation Assistant

> A conversational AI assistant that ingests a codebase (GitHub repo or local files) and answers questions about the code — how it works, where functionality is implemented, API endpoints, dependencies, etc.

---

## Architecture Overview

```
                                    ┌─────────────────────────────┐
                                    │         Ollama              │
                                    │  ┌───────────────────────┐  │
                                    │  │ Tier 1: Mistral Nemo  │  │
                                    │  │ Tier 2: Qwen2.5-Coder │  │
                                    │  │ Tier 3: Phi-3.5 Mini  │  │
                                    │  └───────────────────────┘  │
                                    │  ┌───────────────────────┐  │
                                    │  │ Embeddings:           │  │
                                    │  │  nomic-embed-text     │  │
                                    │  │  all-minilm           │  │
                                    │  │  mxbai-embed-large    │  │
                                    │  └───────────────────────┘  │
                                    └──────────┬──────────────────┘
                                               │ ▲
                                    Embeddings │ │ Generated
                                    + Queries  │ │ Responses
                                               ▼ │
┌─────────────┐    Questions    ┌──────────────────────────────┐
│   Web UI    │ ──────────────▶ │       App Server             │
│ (Streamlit) │ ◀────────────── │  ┌────────────────────────┐  │
│             │    Answers +    │  │ RAG Pipeline            │  │
│  Chat UI    │    Sources      │  │  - Ingestion (clone/    │  │
│  Sidebar    │                 │  │    discover/chunk)      │  │
│  ingestion  │                 │  │  - AST Chunking         │  │
│  controls   │                 │  │    (tree-sitter)        │  │
└─────────────┘                 │  │  - Query + Retrieval    │  │
                                │  │  - Prompt Assembly      │  │
                                │  │  - Guardrails           │  │
                                │  └────────────────────────┘  │
                                └──────────────┬───────────────┘
                                               │ ▲
                                  Store chunks │ │ Retrieve
                                  (embed time) │ │ top-k
                                               ▼ │
                                    ┌─────────────────────────┐
                                    │  Vector DB (ChromaDB)   │
                                    │  - HNSW index           │
                                    │  - Metadata filtering   │
                                    │  - Persistent storage   │
                                    └─────────────────────────┘
```

**Deployment options:**
- **Docker Compose**: All components in local containers (`docker compose up`)
- **Helm/K8s**: Ollama as StatefulSet, ChromaDB as StatefulSet, App as Deployment
- **Access**: localhost (port-forward) | NodePort (private network) | Ingress (production)

---

## Quick Setup

### Prerequisites
- Docker & Docker Compose
- (Optional) A Kubernetes cluster + Helm for the Helm-based deployment
- (Optional) NVIDIA GPU + drivers for full-tier model performance (Ollama falls back to CPU)

### Local Development (Docker Compose)
```bash
git clone <repo-url>
cd code-doc-assistant

# Default: full tier (Mistral Nemo 12B)
docker compose up

# Or specify a lighter tier:
MODEL_TIER=balanced docker compose up     # Qwen2.5-Coder 7B
MODEL_TIER=lightweight docker compose up  # Phi-3.5 Mini 3.8B

# With a custom embedding model:
EMBEDDING_MODEL=all-minilm MODEL_TIER=lightweight docker compose up
```
Then open `http://localhost:8501` in your browser.

On first startup, the `ollama-bootstrap` service pulls the LLM and embedding models — this may take a few minutes depending on your connection. Models are persisted in a Docker volume, so subsequent starts are fast.

### Production Deployment (Helm)
```bash
# Default: full tier
helm install code-doc-assistant ./helm/code-doc-assistant

# Lightweight tier
helm install code-doc-assistant ./helm/code-doc-assistant --set modelTier=lightweight

# Custom combination
helm install code-doc-assistant ./helm/code-doc-assistant \
  --set modelTier=balanced \
  --set embeddingModel=lightweight

# Access via port-forward (single developer)
kubectl port-forward svc/code-doc-assistant-app 8501:8501

# Access via NodePort (team on private network)
helm install code-doc-assistant ./helm/code-doc-assistant \
  --set app.service.type=NodePort \
  --set app.service.nodePort=30501
```

---

## Productionisation Considerations

### Cloud Resources by Model Tier

The resource estimates below account for the **full pipeline** — not just the LLM, but the embedding model, ChromaDB, and the Streamlit application running concurrently. A single GPU must serve both the LLM and the embedding model via Ollama, while ChromaDB and the app consume additional CPU and RAM. CPU-only deployment is technically possible for the lightweight tier but would not deliver a responsive user experience — even for demonstration purposes, a Lead AI Engineer should provision for GPU inference.

| Tier | AWS Instance | GCP Instance | GPU | System RAM | Estimated Cost/hr |
|------|-------------|-------------|-----|------------|-------------------|
| Full (Mistral Nemo 12B) | `g5.2xlarge` | `a2-highgpu-1g` | 1x A10G (24GB) / A100 (40GB) | 32Gi+ | ~$1.50–$5.00 |
| Balanced (Qwen2.5-Coder 7B) | `g5.xlarge` / `g4dn.xlarge` | `n1-standard-8` + T4 | 1x T4 (16GB) / A10G | 16Gi+ | ~$0.75–$2.00 |
| Lightweight (Phi-3.5 3.8B) | `g4dn.xlarge` | `n1-standard-8` + T4 | 1x T4 (recommended) | 8Gi+ | ~$0.50–$1.00 |

**Why these are larger than the "minimum LLM" estimates**: The LLM alone may fit in the quoted VRAM, but Ollama also loads the embedding model (nomic-embed-text: ~274MB in VRAM), and the system needs headroom for ChromaDB indexing, the Streamlit process, and OS overhead. For the full tier, Mistral Nemo 12B uses ~8-10GB VRAM for inference — add the embedding model and you're well past a T4's 16GB, hence the A10G (24GB) recommendation. For production, provisioning one size up from the theoretical minimum is standard practice.

### Scaling

- **HPA (Horizontal Pod Autoscaler)** on the app Deployment — the stateless Streamlit app scales horizontally
- **Ollama scaling** — model replication (multiple StatefulSet replicas) or request queuing; for high-throughput, Ray Serve wrapping Ollama provides load balancing
- **Vector DB** — for large codebases, migrate from self-hosted ChromaDB to managed options (Pinecone, Weaviate Cloud) or self-hosted Qdrant with persistent volumes and replication

### Infrastructure & Operations

- **Observability**: Structured JSON logging, Prometheus metrics, Grafana dashboards, tracing (OpenTelemetry)
- **CI/CD**: GitHub Actions → build container images → push to ECR/GCR → Helm upgrade
- **Security**: Network policies between pods, secrets management (Vault/AWS Secrets Manager), RBAC
- **Agent sandboxing**: Docker Sandboxes (GA January 2026) provide microVM-based isolation for AI agents, with per-sandbox Docker daemons, network allow/deny lists, and workspace syncing ([Docker Blog, Jan 2026](https://www.docker.com/blog/docker-sandboxes-run-claude-code-and-other-coding-agents-unsupervised-but-safely/)). This is directly relevant for a code documentation tool where the ingestion pipeline executes in proximity to proprietary codebases — sandboxed execution would prevent a compromised or misbehaving pipeline from accessing the host filesystem beyond the mounted workspace. The broader ecosystem is converging on microVM isolation as the standard for untrusted code execution ([Northflank, 2026](https://northflank.com/blog/how-to-sandbox-ai-agents)), with Kubernetes-native options like the CNCF Agent Sandbox controller also emerging for cluster-level isolation.

### Self-Hosting vs. API Quality Trade-off

| Approach | Pros | Cons |
|----------|------|------|
| **Self-hosted (Ollama)** | Full control; no API costs; code stays local (privacy); no external dependency | Lower output quality at small parameter counts; requires GPU infrastructure; operational burden |
| **Hosted API (Claude, GPT-4)** | Highest quality reasoning; no infrastructure to manage; easy to scale | API costs; code sent to external service (privacy concern); vendor dependency |
| **Hybrid** | Best of both — local for simple/frequent queries, API for complex reasoning | More complex routing logic; two systems to maintain |

For a code documentation tool specifically, keeping code local is a real-world concern — many organisations cannot send proprietary code to external APIs. The self-hosted approach addresses this by default.

---

## How AI Tools Were Used in Development

This project was developed with the assistance of Claude (Anthropic) as a conversational development partner:
- **Architecture decisions** were suggested by the developer (me), then discussed and debated with Claude — the LLM provider choice, deployment strategy, and component selection
- **Code generation** was mainly generated by Claude, with the developer reviewing, modifying, and testing all the outputs
- **README content** was defined mainly by the developer, with table and figure generation, major text shifts, user guidance, system overview, and large formatting tasks handled by Claude — the journey log mainly captures the conversation flow, which the developer wrote in their own voice

The key principle: AI tools accelerated development, but every decision and the main documentation details (especially the "Evolution of thinking" sections) were made by the developer based on their own experience, thoughts and judgment. The runtime itself uses Ollama (open-source, self-hosted) specifically to demonstrate full-stack engineering ownership rather than API dependency.

---

## Journey Log (Development Process)

This section documents the decision-making process chronologically, as it happened during development. Each phase includes an **"Evolution of thinking"** subsection — these are the most important parts, capturing how the system was developed and improved through conversation, debate, and real-world experience.

To start with, a plan was made, on how to approach the development of the agentic system's pipeline, and an approach in four phases was defined:
**1: LLM Provider Selection**
**2: Planning the system's pipeline**
**3: Deployment Format (& Automation)**
**4: Component Selection**

### Phase 1: LLM Provider Selection — Why Ollama?

**Decision: Ollama with an open-source model.**

As a first step in the process of defining the system, a decision needed to be made on whether to use an API-called model/chat (OpenAI, Anthropic/Claude) or provide a self-hosted open-source approach. Thus, I considered the following considerations for the requirements of the assignment (with some reasoning as to why and how each could be proven):
- **Self-contained repo**: A reviewer can clone and run the project without needing API keys or paid accounts. This approach also makes the solution easier to evaluate.
- **Engineering depth**: Standing up the full inference stack locally can showcase my expertise in CI/CD, deployment and provisioning, infrastructure-aware development and ML/AI design and footprint considerations.
- **Alignment with Lead role**: A Lead AI Engineer should be comfortable with the full model-serving stack, not just API consumption and/or model architecture design and evaluation.
- **Cost & privacy**: For a code documentation tool, keeping code local (not sent to external APIs) is a real-world concern. Self-hosting addresses this by default.

It must be mentioned that, through the process, I (the developer) acknowledged an important trade-off: Hosted API agents (especially Claude, GPT-4) produce higher-quality responses for complex code reasoning. For a production system, a hybrid approach (local for simple queries, API fallback for complex ones) might be ideal. I chose to optimise for demonstrating engineering capability over raw output quality.

**Evolution of thinking — from "which API" to "own the stack":**

The initial framing was: which hosted API should be used? Claude (Code) and GPT-4 (possibly via Cursor) are obvious choices. However, re-reading the assignment brief and determining that a Lead AI Engineer would have to have a broader scope than just agents' definition and use, and the skills to be demonstrated have to demonstrate both depth and breadth in the implemented pipelines. The purpose of this piece of work and documentation isn't to show which API produces the best answers, but to demonstrate the skills needed through engineering the appropriate solution. Wrapping an API is a weekend (or even Saturday morning) project; standing up the full inference stack — model serving, embedding pipeline, vector storage, Kubernetes-native deployment — demonstrates infrastructure ownership, as well as AI pipeline and workflow needs and development stack awareness. The privacy argument then reinforced the decision: for a tool that ingests proprietary codebases, self-hosting isn't just a nice-to-have, it's a requirement many organisations would insist on. This reframing — from output quality to engineering depth plus real-world constraints — was the key engineering decision and turning point. The provider and model selection was effectively chosen based on general knowledge of the landscape. Later, I started thinking about vLLM as a provider, however Ollama is still better as a starting point here.

### Phase 2: Planning the Approach — README-Driven Development

Rather than coding first and documenting later, I got Claude to develop the README and codebase in parallel, leaving "breadcrumbs" behind, and documenting the main decision points, especially inflexion points which showed my direct involvement, experience and engineering acumen.

**Decision: README as a living design document, developed alongside the code.**

**Evolution of thinking — why document-first matters for this assignment:**

The decision was made so that the README is written as a record of how decisions were made. Writing the README *during* development captures the actual thought process: (certain) blind alleys explored (I am only human and sometimes chats can miss certain details, too), trade-offs found or defined, weighed and/or exposed as configuration parameters (if dependent on deployment-time characteristics and modifiers), and the moments where understanding shifted were noted and shown here. This is README-driven development — document as a design artefact, not just documentation. It also functions as a kind of "rubber duckie method". Several technical choices (especially around embedding model selection and vector DB architecture and research) were refined because writing them down forced sharper thinking.

### Phase 3: Deployment Format & Automation — Docker Compose AND Helm

**Decision: Both Docker Compose and Helm chart, serving different purposes.**

| Aspect | Docker Compose | Helm Chart |
|--------|---------------|------------|
| **Purpose** | Local dev & reviewer convenience | Production-grade deployment model |
| **Audience** | `docker compose up` and see it work | K8s cluster / cloud deployment |
| **What it demonstrates** | "I can containerize an app" | "I think in deployable, scalable units" |

The Helm chart models the system as separate concerns:
- **Ollama** → StatefulSet (model weights are state that persists across restarts)
- **Vector DB (ChromaDB)** → StatefulSet (index data is persistent state)
- **Application** → Deployment (stateless, horizontally scalable)

This separation *is* the architecture, expressed as infrastructure-as-code.

**Evolution of thinking — from "one or the other" to "both, for different audiences":**

The initial approach was based on Docker Compose. Having extensive experience with deploying containers and full environments within more complex environments, like networked, on-prem servers and whole managed environments, I thought that a Helm chart will offer a much more configurable and automation-friendly (and scalable) format for the deployment. However, the docker combined with docker-compose solution was much more convenient for testing this as a "proof-of-concept". The answer, thus, became obvious when considering who uses each, as the reviewer who clones the repo can just type `docker compose up` and "Hey presto!" — the solution is up and working (hopefully) in one command. The Helm chart is for demonstrating production thinking: how would this deploy to a real K8s cluster, with persistent volumes, resource limits, health checks, and configurable access patterns? The Helm chart shows my Kubernetes fluency, which also should be expected at the Lead level. Writing both forced me to think about the system from two perspectives simultaneously: developer convenience *and* operational reality. The Helm chart also naturally surfaced the composability patterns (the `_helpers.tpl` tier system) that wouldn't have emerged from Docker Compose alone, because Compose doesn't have the same templating, system-wide programmability/configurability and automation-ready power.

### Phase 4: Component Selection

#### 4a. LLM Model Selection — A Tiered, Configurable Approach

**Decision: Tiered model strategy — the system is model-agnostic, with the model as a configuration value, not a hard dependency.**

**Task analysis**: Code comprehension + explanation/documentation **(not code generation)**. The model must read code chunks, understand what they do, reason about relationships (dependencies, API endpoints, architecture), and explain in natural language.

**Reasoning capability matters**: The model must follow multi-step logic — "function A calls B, which depends on C, and C is where the configuration is loaded." This reasoning depth scales with parameter count and is a key differentiator between tiers.

**Models evaluated**:

| Model | Parameters | Strengths | Weaknesses | Reasoning | Ollama availability |
|-------|-----------|-----------|------------|-----------|-------------------|
| Mistral Nemo | 12B | Excellent code comprehension AND natural language explanation; broad community adoption | Higher resource needs than 7B models | Strong multi-step reasoning; can trace cross-file dependencies and explain architectural relationships | `ollama pull mistral-nemo` ✅ |
| DeepSeek-Coder V2 Lite | 16B (MoE) | MoE architecture excels at multi-language codebases; strong polyglot handling | Variable memory patterns; heavier footprint | Strong reasoning within code context; MoE may route reasoning tasks to specialised experts | `ollama pull deepseek-coder-v2:16b` ✅ |
| Qwen2.5-Coder 7B | 7B | Best code comprehension at the 7B tier; modern benchmarks; lightweight | Less widely adopted; narrower community | Adequate for single-file reasoning; may struggle with complex multi-module dependency chains | `ollama pull qwen2.5-coder:7b` ✅ |
| Phi-3.5 Mini | 3.8B | Extremely lightweight; runs on almost anything | Not code-specialised; may need fine-tuning | Limited multi-step reasoning; best for straightforward "what does this function do" queries | `ollama pull phi3.5` ✅ |

*Note: CodeLlama 7B was evaluated but excluded — Qwen2.5-Coder 7B supersedes it on modern benchmarks.*

**Final tiered ranking**:

1. **Tier 1 — Full (default)**: **Mistral Nemo (12B)** — Best balance of code comprehension and natural language explanation. For polyglot codebases, **DeepSeek-Coder V2 Lite** is the recommended swap — recent research (MultiPL-MoE, [Wang et al., 2025](https://arxiv.org/abs/2508.19268)) confirms MoE architectures are particularly effective for multi-programming-language tasks, treating programming language diversity analogously to natural language multilingualism.
2. **Tier 2 — Balanced**: **Qwen2.5-Coder 7B** — Best-in-class at 7B. Go-to choice for ~8GB VRAM environments.
3. **Tier 3 — Lightweight / edge**: **Phi-3.5 Mini (3.8B)** — Smallest footprint, still-decent performance. Fine-tuning candidate.

**Hardware reality check**: Frontier models reach trillions of parameters — three orders of magnitude above these. But without a multi-GPU system, models larger than DeepSeek-Coder V2 Lite (16B) aren't practical to serve. These tiers reflect models genuinely usable on realistic hardware: a single consumer GPU or even CPU-only for the lightweight tier.

**Configurability**: A single `modelTier` value cascades through the entire system via `_helpers.tpl` (Helm) or environment variables (Docker Compose): model selection, resource allocation, context window, timeouts.

```bash
helm install code-doc-assistant ./helm/code-doc-assistant --set modelTier=lightweight
```

**Evolution of thinking — from fixed to composable:**

This decision (right from the start) was where the approach for helm automation was inspired from. The initial approach was to pick the best model, hard-code it. Then I realised that, since we may be dealind with a more configurable and scalable environment, within a production environment, there could be merit in making the model choice more flexible, based on needs and resource constraints. That's thinking in terms of *swapping a string* and obtaining a lot more configurability in a Kubernetes-native deployment. That way, changing a line, or even half of a YAML file becomes a single high-level intent (`modelTier=lightweight`) that cascades through every dependent decision: which model to pull, how much memory to request, GPU requirements, context window, timeout. The `_helpers.tpl` implements this — a deployer expresses "I want the lightweight tier" and the system resolves the rest. This is the beauty of defining a few go variables and leveraging them in a helm chart's configuration and shows the difference between configuration and *composable system design*, the same principle behind Kubernetes operators and Terraform modules.

#### 4b. Embedding Model

**Decision: `nomic-embed-text` via Ollama as default, with codebase-aware configuration.**

Key architectural principle — **embedding compatibility is a hard constraint**: you CANNOT mix embeddings from different models into a single index. Changing the embedding model requires full re-ingestion. The `_helpers.tpl` derives vector dimension from the embedding model choice, preventing silent failures. More on this in the **Evolution of thinking** subsection.

**Codebase-aware embedding selection** — two axes characterise the input:

| Axis | States | Embedding implication |
|------|--------|----------------------|
| **Language distribution** | Primary-code (>90% one lang) vs. Multi-code | Multi-code benefits from polyglot models; connects to DeepSeek LLM choice |
| **Documentation state** | No-docs / Partial-docs / Review-and-revise | Review-and-revise is most demanding — must detect inconsistencies between code and prose |

**Available embedding models:**

| Model | Dimensions | Best for |
|-------|-----------|----------|
| `nomic-embed-text` | 768 | Partial-documentation codebases (default) |
| `all-minilm` | 384 | Lightweight tier; resource-constrained |
| `mxbai-embed-large` | 1024 | Complex codebases with dense documentation |

**Evolution of thinking — from infrastructure choice to input-driven configuration:**

Initially, the embedding model was treated as a pure infrastructure decision — pick one that works, move on. There then was a shift in thinking when recognising that the *nature of the codebase* should inform the choice, and that the embedding model itself determines (and may restrict) much of the further deployments and/or leverage possible within the pipeline. Further, because of the above-mentioned *nature of the codebase*, I determined that some classification or due diligence on codebase assessment can make a world of a difference to determining the type of deployment needed. For example, a raw-code-only repo needs strong code-native embeddings. A heavily-documented repo needs good code+text understanding. A multi-language (with or without documentation) repo needs polyglot awareness. This reframing — embedding selection as a property of the *input*, not the *infrastructure* — led to a two-axis characterisation (language distribution × documentation state) of the embedding model needs. The implementation stays simple, but the systematic understanding is documented for operators making informed design and deployment choices.

#### 4c. Vector Database

**Decision: ChromaDB as default, behind a thin abstraction layer.**

An important distinction emerged during evaluation: **FAISS is a search index library, not a database**. It provides indexing algorithms (LSH, HNSW, IVF) but no persistence, metadata filtering, or API. Vector databases like ChromaDB and Qdrant use indexing algorithms *internally* (ChromaDB uses HNSW) and wrap them with database functionality. FAISS doesn't replace a vector DB — it replaces the *indexing engine inside one*.

| Solution | What it is | Index algorithm | Persistence | Metadata filtering | Best fit |
|----------|-----------|-----------------|-------------|-------------------|----------|
| **FAISS** | Search index library | LSH/HNSW/IVF (configurable) | None — you build it | None — you build it | Raw performance; custom systems where you need LSH |
| **ChromaDB** | Vector database | HNSW (built-in) | Built-in | Built-in | Developer convenience; small-to-medium codebases |
| **Qdrant** | Vector database | HNSW (built-in) | Built-in | Built-in (richer) | Production deployments; large codebases |

For a code documentation assistant, **metadata filtering matters** — filtering by file type, directory, language when searching. ChromaDB provides this out of the box. FAISS would require building all of that manually, or pairing FAISS with a separate database (e.g., PostgreSQL, SQLite) for persistence and metadata — essentially building a custom vector DB.

**Evolution of thinking — FAISS, LSH, and knowing when to stop:**

Since I did some research on LSH-based data and computation capability management, we explored whether FAISS should be the primary indexing library, given LSH support. However, building persistence, metadata filtering, and CRUD on top of FAISS would be substantial engineering for no practical benefit at this project's scale. ChromaDB behind an abstraction layer is the right trade-off, offering pragmatism in implementation and extensibility by design. The abstraction costs almost nothing but preserves the ability to swap in FAISS+LSH or Qdrant later. This is the kind of systems' efficiency and scalability decisions I like making, at the same time trying not to over-engineer solutions, but not closing the door on them either.

#### 4d. Orchestration Framework

**Decision: LlamaIndex for RAG pipeline; LangChain documented as future growth path.**

LlamaIndex is purpose-built for RAG: native `CodeSplitter` with AST-aware chunking, tree-structured indexes, lighter weight than LangChain for pure retrieval-and-respond workflows.

**Production orchestration**: MLflow (prototyping) → W&B (production monitoring) → Ray on K8s (distributed compute). Ray Serve wraps Ollama for load balancing; Ray Data enables parallel ingestion; KubeRay deploys natively on K8s.

**Evolution of thinking — from "which framework" to "what question am I actually answering":**

The instinct was to reach for LangChain — it's the default answer for AI orchestration. But LangChain is a general-purpose framework; this project is focused RAG. The real orchestration question isn't "which framework chains my prompts" but "how does this system scale operationally?" That's answered by MLflow/W&B for tracking and Ray/K8s for compute, not by a prompt-chaining library. LangChain enters the picture when the *application scope* grows (agents, tools, CI/CD integration), not when the infrastructure scales.

#### 4e. Code Chunking Strategy

**Decision: AST-based chunking via tree-sitter (LlamaIndex's `CodeSplitter`), with fixed-window fallback.**

| Strategy | Strengths | Weaknesses |
|----------|-----------|------------|
| **AST-based (tree-sitter)** | Preserves logical units; language-aware | Requires valid parseable code |
| **Heuristic / pattern-based** | Works on broken code | Fragile; misses nested structures |
| **Fixed-window** | Language-agnostic; never fails | Splits functions mid-body |
| **Hybrid (AST + fallback)** | Best of both | Slightly more complex |

**Evolution of thinking — from "just split the text" to language-aware semantic boundaries:**

The naive approach is fixed-window chunking — split every N tokens. However, a function split mid-body can produce chunks that are individually meaningless. As AST-aware chunking via tree-sitter ensures a function, class, or method is always a complete unit and it includes a fallback which covers all fronts, as not all files parse cleanly (generated code, config files, partial snippets), the system degrades gracefully to fixed-window, when it could otherwise fail. The pipeline validation confirmed this: tree-sitter produced 41 semantic chunks from 7 Python files, while the fallback handled text/config files correctly. The chunk distribution (e.g., `ingest.py` → 9 chunks, `config.py` → 2 chunks) shows the AST splitter respects logical boundaries rather than imposing uniform size. 

For further refinement and configurability, I thought that the hybrid fallback for the CodeSplitter implementation is definitely something that could be accounted for in the helm chart, to account for system complexity and resource-usage within the resource constraints of the deployment. For the lightweight tier (already resource-constrained running Phi-3.5 on CPU), spending resources on AST parsing during ingestion may not be the best trade-off. This led to making the chunking strategy **tier-configurable**: full/balanced tiers default to AST chunking, while the lightweight tier defaults to text-based chunking. The `_helpers.tpl` resolves this automatically from `modelTier`, and it cascades through to the app via the `CHUNKING_STRATEGY` environment variable — the same composability pattern applied to model selection. The AST fallback still exists as a safety net regardless of configuration.

#### 4f. Interface

**Decision: Streamlit for the web UI.**

The assignment explicitly states UI/UX is not a judging criterion. Streamlit provides a ChatGPT-style interface with `st.chat_input()` and `st.chat_message()` in pure Python — functional and clean without consuming development time.

**Access patterns:**

| Access Method | Context | Helm config |
|--------------|---------|-------------|
| `kubectl port-forward` | Developer on the same machine | Default (ClusterIP) |
| NodePort (30000-32767) | Team on a private network | `--set app.service.type=NodePort` |
| Ingress with TLS | Production / public access | `--set ingress.enabled=true` |

**Evolution of thinking — access patterns and network realities:**

The initial Helm setup offered ClusterIP + optional Ingress. At this point, my experience with multi-node cluster environment orchestration reminded me about multi-node private network deployments, and accounting for a team externally accessing the tool from other machines without an ingress controller, in which case I used NodePorts beforehand. Yes, NodePort has a "dirty quick fix" reputation, however that is mainly because it is not secured, potentially exposing the system through public-facing interfaces (to threats). For an internal code documentation tool, on a private network (exactly where a tool handling proprietary code would run), NodePort is perfectly pragmatic. This reflects my real-world DevOps and network management and configuration experience, showing that the "textbook" answer (always use Ingress) isn't always right, and deployment context — network topology, security posture, access control, data management — can sometimes be the main access pattern enablers/blockers.

#### 4g. RAG Quality and Limitations

RAG is not unconditionally beneficial. Research shows retrieval noise can actively degrade output quality — "misinformation can be worse than no information at all" ([Gupta et al., 2024](https://arxiv.org/abs/2410.12837)). Paradoxically, including irrelevant documents can sometimes increase accuracy by over 30% ([Gupta et al., 2024](https://arxiv.org/html/2410.12341v2)).

**Code documentation-specific RAG risks:**
- **Stale context**: Chunks from a previous version may contradict current code
- **Partial context**: A function without its imports leads to incorrect explanations
- **Cross-file confusion**: Similar naming across modules causes conflation

**Mitigations implemented**: Similarity score cutoff (0.3), metadata preservation (file paths/languages in prompt), source attribution in every response.

**Evolution of thinking — from "RAG always helps" to understanding when it hurts:**

The initial assumption was straightforward: retrieve relevant context, feed it to the LLM, get better answers. Research forced a more nuanced view. The key insight for code documentation: the *quality* of retrieval matters more than the *quantity*. A function chunk without its import context may lead the LLM to hallucinate dependencies. A chunk from a similarly-named function in a different module may cause the LLM to conflate them. The similarity cutoff and metadata preservation are direct responses to these risks — they're not just "nice to have" filtering, they're guardrails against the specific failure modes of RAG applied to code. Further mitigations (CRAG, Self-RAG, re-ranking) are documented in "What I'd Do Differently."

#### 4h. Guardrails

For a code documentation tool, guardrails are domain-specific:

**Hallucination prevention**: Prompt template instructs "I don't have enough context" over guessing; source attribution enables verification; similarity cutoff prevents irrelevant context from triggering confabulation.

**Sensitive data protection**: Code often contains credentials, API keys, tokens. The system should detect and redact common patterns before including chunks in responses. Not implemented in the deliverable but a critical production requirement.

**Bias and consistency**: Tokenisation may weight variable naming conventions differently across languages. Mitigation: normalise code formatting before embedding, monitor response consistency.

**Evolution of thinking — from "add a filter" to understanding code-specific risks:**

Guardrails in general LLM applications focus on content moderation. For code documentation, the risks are different — hallucinated file paths that don't exist, confidently wrong architectural explanations, and leaked credentials embedded in code chunks. The prompt-level guardrail ("say you don't have enough context") is the first line of defence, but the deeper insight is that source attribution is itself a guardrail — when the developer can see *which files* informed the answer, they can verify claims against actual code. This transforms the system from a black-box oracle into a transparent assistant. Credential redaction is flagged as a production requirement because it is a real-world risk that's easy to miss in a prototype.

### Phase 5: Implementation

The implementation phase followed the architecture decisions above. Key outcomes:
- **5 Python modules** (`config.py`, `vector_store.py`, `ingest.py`, `query_engine.py`, `app.py`) — each mapping to a distinct concern
- **ChromaDB abstraction layer** — `VectorStoreBase` ABC with `ChromaVectorStoreImpl`; swappable to FAISS (with custom persistence) or Qdrant
- **Tier-aware configuration** — `config.py` mirrors the Helm `_helpers.tpl` logic for Docker Compose parity, including model selection, resource allocation, embedding dimension, and chunking strategy
- **AST chunking with tier-configurable fallback** — tree-sitter via LlamaIndex's `CodeSplitter` for full/balanced tiers, `SentenceSplitter` as default for lightweight tier (lower CPU/memory during ingestion), with AST→text fallback always available as a safety net regardless of configuration
- **Pipeline validation test** (`tests/test_pipeline.py`) — end-to-end test using ChromaDB in embedded mode with lightweight local embeddings, validating the full ingest→chunk→store→retrieve pipeline without requiring external services

There isn't much of an **Evolution of thinking** here, since the tests and implementation instructions are already there and there isn't much to think of, but I did have to do a bit of "maintenance" and make sure that every part of the code is where it should and make sure that the system itself was actually functional, with the interfacing and eventually some code examples, as well. One notable implementation detail: the `tree-sitter-language-pack` dependency was discovered during pipeline validation — LlamaIndex's `CodeSplitter` requires it for AST parsing, but it's not listed as a dependency of `llama-index-core`. This was added to `requirements.txt` after testing confirmed the fallback path worked correctly but the AST path needed the additional package.

### Phase 6: Testing & Refinement

**Resource constraints**: I don't have significant local compute resources (GPU, large RAM).

- The **lightweight tier** (Phi-3.5, CPU-only) is the recommended tier for reviewers without GPU access
- The **full tier** (Mistral Nemo 12B) requires a GPU and was not fully integration-tested
- The code is tier-independent — switching tiers changes only model and resources, not application logic

**Pipeline validation** (run in a constrained CI-like environment):
- All module imports validated ✅
- Config tier resolution tested across all tiers ✅
- File discovery: found 18 files (8 code, 10 text/config), correctly classified ✅
- AST chunking: 41 code chunks from 7 Python files via tree-sitter ✅
- ChromaDB storage: 58 total chunks stored in embedded mode ✅
- Retrieval: 4/6 test queries hit expected files with test embeddings ✅

**What requires local/cloud resources**: Full Ollama inference, real embedding quality, Streamlit UI interaction, Docker Compose / Helm deployment.

**Evolution of thinking — honesty over impression management:**

The temptation was to gloss over the resource constraints and imply thorough testing. But the assignment asks for engineering judgment, and honest that the assessment of what was and wasn't tested demonstrates that judgment more than a false claim of full coverage. 

The pipeline validation test (`tests/test_pipeline.py`) was designed specifically to exercise as much of the codebase as possible *without* requiring external services — ChromaDB in embedded mode, lightweight test embeddings, the full ingest→chunk→store→retrieve pipeline. The 4/6 retrieval hit rate with character-trigram embeddings (not real semantic embeddings) validates the pipeline mechanics; real Ollama embeddings would resolve the remaining misses. This is the same principle applied throughout: be honest about what you tested, clear about what you didn't, and show you know the difference.

Later on, I managed to fully test the pipeline on my own laptop, with full integration and functionality. However, more problems occured after successful code ingestion of [one of my repositories](https://github.com/Chrisys93/IcarusRepoSEND) (via the embedding model) - the model could not respond to the request "Could you please document and produce comments for the main functions in the models/strategy/repo_storage_mgmt_app.pyfile?", giving an error: `Error generating response: model requires more system memory (50.0 GiB) than is available (7.7 GiB) (status code: 500)`. With this, I am not sure about whether the previously-quoted machines could support the (associated) model(s), however VRAM is also different from normal compute RAM, as well, and the efficiency of model execution and optimisation could depend on architecture, model quantisation - as below - and on data/knowledge distillation, encoding, mapping and compression.

---

## What I'd Do Differently With More Time

### Model Fine-Tuning

The RAG approach means the model receives context at query time, sufficient for most questions without fine-tuning. Fine-tuning would be relevant if base models consistently failed on specific languages/domains, or if a particular documentation style was required. This requires training data (code Q&A pairs), compute, and iteration — guided by observed performance gaps, not assumed in advance.

### Chunking Granularity and Adaptive Retrieval

Function-level chunks answer "what does this function do?" well but struggle with "how do modules interact?" File-level chunks capture more context but may exceed embedding windows. An adaptive approach — chunking at multiple granularities and letting retrieval pick the right level — is the research-level version.

### Advanced RAG Mitigations

- **CRAG (Corrective RAG)** — filtering low-confidence retrievals at inference time, reducing retrieval errors by 12–18%
- **Self-RAG** — the model learns to critique its own retrieval usage
- **Context windowing** — prioritising recently-modified files for timeliness
- **Re-ranking** — secondary model re-scores retrieved chunks before they enter the prompt

### LSH and Distributed AI — A Research Tangent

This is explicitly a tangent, not a justified requirement for the current project, but it signals the direction of thinking.

During vector DB evaluation, LSH came up as an alternative to HNSW. LSH offers compact binary representations, O(1) lookup, and sub-linear search — relevant at scales beyond this project. The broader question: can LSH enable **collectively and distributedly intelligent systems** where computational nodes contribute to shared, scalable understanding rather than centralising in a single vector DB?

Research followed:
- Reformer architecture's LSH attention ([W&B: Methods LSH](https://wandb.ai/fastai_community/reformer-fastai/reports/Methods-LSH--Vmlldzo0Mjc2ODQ))
- GPU-optimised LSH with Winner-Take-All hashing ([Shi et al., 2018](https://arxiv.org/abs/1806.00588))
- PipeANN for billion-scale vector search on SSDs ([Guo & Lu, OSDI '25](https://www.usenix.org/system/files/osdi25-guo.pdf))

### Preliminary Research Directions

**Study 1: Documentation state vs. resource efficiency** — Does a well-documented codebase require less compute for useful answers? Quantifying this across model tiers × embedding models × codebase types would produce actionable guidance.

**Study 2: Autonomous continuous improvement** — A code documentation assistant deployed at branch level learns from developer interactions, propagating from branch → team → enterprise environments. Scaling *understanding*, not just infrastructure.

### With Known Infrastructure: vLLM, Quantisation, and Production-Grade Inference

The current implementation uses Ollama as the inference server — specifically chosen for delivering a self-contained, reviewer-friendly deliverable. However, in a production environment where the infrastructure is known (GPU type, count, budget, expected load), a different set of optimisations becomes relevant. This section documents some of the avenues I would explore with that knowledge.

**vLLM as inference server**: vLLM provides continuous batching, PagedAttention for efficient KV-cache management, and native support for tensor parallelism across multiple GPUs. Where Ollama optimises for developer convenience (single command, automatic model management), vLLM optimises for throughput and latency under concurrent load — critical when the code documentation tool serves a team rather than a single developer. vLLM also supports OpenAI-compatible API endpoints, making it a drop-in replacement in the architecture (the `query_engine.py` would need minimal changes). The trade-off is operational complexity: vLLM requires explicit model loading, (multi-)GPU memory management, and doesn't auto-pull models like Ollama.

**Quantisation techniques**: For deployment-constrained environments, quantisation (GPTQ, AWQ, GGUF) can reduce model memory footprint by 50-75% with minimal quality loss for code comprehension tasks. This would allow running the balanced tier (Qwen2.5-Coder 7B) on hardware currently limited to the lightweight tier, or running the full tier (Mistral Nemo 12B) on a single T4 (16GB) via 4-bit quantisation. The quality impact on code explanation tasks specifically (vs. general benchmarks) would need evaluation. There could, on the other hand, be positive effects on the context ingestion and/or number of users per session, in the case of model quantisation, for the same resource usage/availability (some other trade-offs are "born"/emerge).

**Context and sequence length optimisation**: Larger context windows (32K+ tokens) enable ingesting entire files or multi-file contexts in a single query, reducing the chunking granularity problem. vLLM's PagedAttention makes long-context inference practical without proportional memory scaling. This connects directly to the adaptive retrieval question — with sufficient context window, the system could retrieve at file-level granularity rather than function-level, answering architectural questions more effectively.

**Multi-GPU and hardware-aware deployment**: With multiple GPUs available, tensor parallelism (splitting model layers across GPUs) and pipeline parallelism (splitting the pipeline stages) become options. The embedding model and LLM could run on separate GPUs, eliminating the shared-VRAM constraint noted in the cloud resources table, and enabling more complex, helm-based (or, why not, Argo Workflow) deployments — yes, I am aware that these are the main use cases where the scalable design assumed from the start of this design exercise would truly shine. Architecture-specific optimisations (FlashAttention-2 for Ampere+ GPUs, INT8 inference on Turing GPUs) and hardware placement (co-locating the vector DB on NVMe SSDs, placing Ollama/vLLM on GPU nodes) would further improve throughput. And speaking of hardware placement and throughputs, some other very interesting problems (that on-prem would be even more interesting to look at and explore) would be the use of InfiniBand technologies (that NVidia now own), for NVLink and ConnectX switching, but also the sizing and system optimisation of inter-server/inter-node network resources and management.

**Network and switching optimisation**: In distributed deployments (separate nodes for inference, vector DB, and app), network topology matters. NVLink for multi-GPU communication, RDMA/InfiniBand for inter-node model parallelism, and even simple considerations like co-locating the app and vector DB to minimise retrieval latency. These are the kinds of infrastructure decisions that separate a working prototype from a production system.

This section overlaps with fine-tuning (documented above) in that both aim to improve model performance for deployment — fine-tuning changes the model weights, while quantisation and inference optimisation change how those weights are served. Both are guided by observed performance requirements, not assumed in advance.

---

## Engineering Standards

- **Python 3.11+**, type hints used throughout
- **Modular design**: each source file maps to a single concern
- **Abstraction layers**: vector store interface for DB-agnostic design
- **Configuration**: environment-variable driven, parity between Docker Compose and Helm
- **Testing**: pipeline validation with embedded ChromaDB, AST chunking verification
- **Logging**: structured logging via Python `logging` module, configurable level

---

## Reviewer Note

I am willing to provide the full conversation transcripts from the development sessions with Claude, which helped develop this project. These transcripts would show the unedited back-and-forth — including corrections and the moments where ideas were realigned and redirected — and may provide additional context for evaluating the development process and decision-making documented in this "Journey Log".
