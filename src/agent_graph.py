"""
agent_graph.py — LangGraph StateGraph for the code documentation agent pipeline.

Graph topology
──────────────
                    ┌─────────────────────┐
                    │       START         │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  tool_selection      │  LLM proposes tool plan
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  hitl_checkpoint     │  [HITL-1] Human reviews tool plan
                    └─────────┬───────────┘  interrupt_before (when HITL_ENABLED=true)
              approved │           rejected → END
                    ┌─────────▼───────────┐
                    │  tool_execution      │  Runs tools in sandbox
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  supervisor          │  Confidence check + preference injection
                    └─────────┬───────────┘
             proceed │               retry → tool_execution
                    ┌─────────▼───────────┐
                    │  context_assembly    │  Dedup, rank, trim
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  generation          │  Documentation LLM
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  output_review       │  Behaviour determined by OUTPUT_REVIEW_MODE:
                    └─────────┬───────────┘
                              │
        OUTPUT_REVIEW_MODE ───┤
          "human"             │  interrupt_before: human rates + decides (accept/regen/add_context)
          "supervisor"        │  supervisor LLM quality-gates against rubric; silent retry if fail
          "self"              │  generation node self-critiques before emitting (handled in generation)
          "off"               │  passthrough → END immediately
                              │
              accept/pass ────┤──── END
              regenerate ─────┤──── context_assembly
              add_context ────┤──── tool_execution

OUTPUT_REVIEW_MODE is resolved at build_graph() time from the OUTPUT_REVIEW_MODE env var,
which cascades from Helm values.yaml → _helpers.tpl → app container env → here.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from typing import Any, Literal, Optional

import mlflow
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

logger = logging.getLogger(__name__)

from agent_state import (
    AgentState, Chunk, HITLCheckpoint, PostGenerationFeedback,
    SessionPreferences, SupervisorAdjustment, ToolCall
)
from tools import build_tool_registry, run_tool
import tracking
import deployment

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral-nemo")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
HITL_ENABLED = os.environ.get("HITL_ENABLED", "true").lower() == "true"

# Inference backend: "ollama" (default) | "vllm" | "llamacpp"
#
# ollama   → ChatOllama → Ollama REST API. Default; model management included.
#            Best for: full/balanced tiers, ease of use.
# vllm     → ChatOpenAI → vLLM OpenAI-compatible API. GPU required.
#            Best for: high-throughput serving at scale.
# llamacpp → ChatOpenAI → llama-server OpenAI-compatible API. CPU-native.
#            Best for: lightweight/minimal tiers on constrained machines.
#            llama-server exposes the same /v1/chat/completions API as vLLM;
#            we reuse ChatOpenAI pointed at LLAMACPP_HOST.
#
# "auto" (the dev-compose default) means: follow what the stack is actually running -- see deployment.py.
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "auto").lower()


def _default_backend() -> str:
    """The deployment's backend: INFERENCE_BACKEND if it names one, otherwise whatever is running."""
    return deployment.backend()
VLLM_HOST = os.environ.get("VLLM_HOST", "http://localhost:8080")
VLLM_MODEL = os.environ.get("VLLM_MODEL", os.environ.get("OLLAMA_MODEL", "mistral-nemo"))
LLAMACPP_HOST = os.environ.get("LLAMACPP_HOST", "http://localhost:8081")
# LLAMACPP_MODEL is only the FALLBACK name, used when llama-server cannot be asked.
# The name actually used (and logged) comes from the server's own /v1/models -- see
# _llamacpp_served_model() -- because llama-server serves whichever GGUF it was launched
# with (its -a alias, or the GGUF filename stem) and ignores the request's `model` field.
LLAMACPP_MODEL = os.environ.get("LLAMACPP_MODEL", "mistral-nemo-instruct-2407-q4_k_m")

# Output review mode — controls post-generation quality gate behaviour.
# "human"      → HITL-2: interrupt and wait for human rating + decision
# "supervisor" → Supervisor LLM evaluates output against a rubric; silent retry if fail
# "self"       → Generation LLM self-critiques before emitting (no extra node)
# "off"        → No post-generation check; accept immediately
OUTPUT_REVIEW_MODE: Literal["human", "supervisor", "self", "off"] = (
    os.environ.get("OUTPUT_REVIEW_MODE", "human")  # type: ignore[assignment]
)

# Supervisor thresholds
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
MAX_RETRIEVAL_ATTEMPTS = int(os.environ.get("MAX_RETRIEVAL_ATTEMPTS", "3"))
MAX_GENERATION_ATTEMPTS = int(os.environ.get("MAX_GENERATION_ATTEMPTS", "3"))
# How many times a reviewer may send the tool plan back for re-planning before the run ends.
MAX_REPLANS = int(os.environ.get("MAX_REPLANS", "3"))

# Supervisor quality-gate rubric score below which output is rejected (0–10)
QUALITY_GATE_THRESHOLD = float(os.environ.get("QUALITY_GATE_THRESHOLD", "6.0"))

# ---------------------------------------------------------------------------
# LLM client — backend-agnostic factory
# ---------------------------------------------------------------------------

# Model tier -> base model name, mirrors config.py / _helpers.tpl resolution.
# Lets a runtime tier override (from the UI dropdown) resolve to a concrete
# model tag without a container restart.
#
# "heavy" is explicit and opt-in: deepseek-coder-v2:16b-lite-instruct is a
# genuinely strong choice for polyglot codebases (MoE, good multi-language
# reasoning), but its MoE VRAM footprint is dominated by TOTAL parameters
# (16B), not active parameters (~2.4B/token) — it needs ~13GB regardless of
# "lite" compute cost, which does NOT fit a 12GB card alongside context and
# silently falls back to partial CPU inference (see ARCHITECTURE.md Phase 19
# for the full incident writeup). It briefly occupied "balanced" by mistake;
# it now has its own clearly-labelled slot instead, above "full", for anyone
# with the VRAM headroom (16GB+) to actually run it entirely on GPU.
_MODEL_TIER_BASE = {
    "heavy":       "deepseek-coder-v2:16b-lite-instruct",
    "full":        "mistral-nemo:12b-instruct",
    "balanced":    "qwen2.5-coder:7b",
    "lightweight": "phi3.5",
    "minimal":     "qwen2.5-coder:3b-instruct",
}


# Context window per tier, mirrors _helpers.tpl's inferenceConfig helper.
# NOTE: this was previously only emitted as an env var for the Helm chart
# and never actually consumed anywhere in this file — ChatOllama() ran on
# Ollama's small built-in default (2048) regardless of tier or model. That
# was invisible with mistral-nemo + short single-repo prompts; it surfaces
# immediately with a bigger model + a fair-merged multi-repo context that
# no longer fits in 2048 tokens. Wired in properly below.
_MODEL_TIER_CTX = {
    "heavy":       8192,
    "full":        8192,
    "balanced":    8192,
    "lightweight": 4096,
    "minimal":     2048,
}

# Fallback for the container's own default tier (no hot-swap override),
# read once at import — OLLAMA_NUM_CTX env var wins if explicitly set,
# otherwise derived from MODEL_TIER the same way OLLAMA_MODEL already is.
OLLAMA_NUM_CTX = int(os.environ.get(
    "OLLAMA_NUM_CTX",
    str(_MODEL_TIER_CTX.get(os.environ.get("MODEL_TIER", "full"), 8192)),
))


def _llamacpp_props() -> dict | None:
    """The running llama-server's own description of itself (GET /props), or None."""
    try:
        import requests
        return requests.get(f"{LLAMACPP_HOST}/props", timeout=3).json()
    except Exception:
        return None


def _llamacpp_server_ctx() -> int | None:
    """Real per-slot context of the running llama-server (GET /props), or None.

    llama-server is launched with its own --ctx-size (4096 for the heavy tier),
    independent of _MODEL_TIER_CTX below. Budgeting retrieved context against the
    tier table (8192) while the server holds 4096 overflows the prompt as soon as
    retrieval returns real chunks. Asking the server removes that config coupling.
    """
    props = _llamacpp_props() or {}
    n = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
    try:
        return int(n) if n else None
    except (TypeError, ValueError):
        return None


_SERVED_MODEL_CACHE: dict = {"at": 0.0, "name": None}


def _llamacpp_served_model(max_age_s: float = 30.0) -> str | None:
    """Model id the running llama-server actually serves (GET /v1/models), or None.

    llama-server holds ONE model, chosen by its launch command, and ignores the `model`
    field of requests. Reading the id from the server -- instead of from the app's own
    LLAMACPP_MODEL default -- keeps the trace, the MLflow params and any comparison between
    models honest: previously the app labelled DeepSeek answers "mistral-nemo". Cached for a
    short time so it costs nothing per call but still follows a server that was relaunched
    with another model.
    """
    now = time.time()
    if now - _SERVED_MODEL_CACHE["at"] < max_age_s and _SERVED_MODEL_CACHE["name"]:
        return _SERVED_MODEL_CACHE["name"]
    name = None
    try:
        import requests
        data = (requests.get(f"{LLAMACPP_HOST}/v1/models", timeout=3).json() or {}).get("data") or []
        name = data[0].get("id") if data else None
    except Exception:
        name = None
    if name:
        _SERVED_MODEL_CACHE.update(at=now, name=name)
    return name or _SERVED_MODEL_CACHE["name"]


def _effective_ctx_window(state: "AgentState") -> int:
    """Context window the answering model really has, for context budgeting."""
    backend = (state.get("active_backend") or _default_backend() or "").lower()
    if backend == "llamacpp":
        n = _llamacpp_server_ctx()
        if n:
            return n
    return _resolve_ctx_for_tier(state.get("active_model_tier"))


def _resolve_ctx_for_tier(tier: str | None) -> int:
    """Context window for a given tier; falls back to the container default."""
    if tier:
        return _MODEL_TIER_CTX.get(tier, OLLAMA_NUM_CTX)
    return OLLAMA_NUM_CTX


def _resolve_model_for_tier(tier: str, quantisation: str = "q4_K_M") -> str:
    """Compose an Ollama-style model tag from a tier (+ quantisation), same
    logic as config.py._resolve_ollama_model().

    lightweight/minimal/balanced suppress the quant suffix — lightweight and
    minimal already default to Q4 in Ollama; balanced's qwen2.5-coder:7b has
    no verified compound tag (see Phase 19), so it uses its known-working
    bare tag rather than a guessed one.

    heavy KEEPS quant composition — deepseek-coder-v2:16b-lite-instruct-q4_K_M
    was empirically confirmed to exist and pull successfully during this
    session's testing (visible in `ollama ps` output), so composing the
    suffix here is safe, unlike the balanced case above."""
    base = _MODEL_TIER_BASE.get(tier, _MODEL_TIER_BASE["full"])
    if tier in ("lightweight", "minimal", "balanced") or quantisation == "fp16":
        return base
    return f"{base}-{quantisation}"


# Resident-model budget: how many distinct Ollama models we're willing to
# keep warm at once. Today, single GPU, this is 1 — meaning hot-swap always
# fully replaces whatever was resident (current behaviour, unchanged).
#
# The hook for later: bump OLLAMA_MAX_RESIDENT_MODELS once there's more than
# one GPU (or enough VRAM to genuinely hold multiple models), and the SAME
# code below automatically becomes LRU eviction instead of full replacement
# — keep the N most-recently-used models resident, only evict the rest.
# No code change needed at that point, just the env var (and presumably
# Ollama's own OLLAMA_MAX_LOADED_MODELS server-side setting, which this is
# deliberately meant to track rather than duplicate/fight).
OLLAMA_MAX_RESIDENT_MODELS = int(os.environ.get("OLLAMA_MAX_RESIDENT_MODELS", "1"))


def _unload_other_ollama_models(keep_model: str, host: str) -> None:
    """
    Free up Ollama-resident model slots for keep_model, respecting
    OLLAMA_MAX_RESIDENT_MODELS.

    budget=1 (default, single-GPU today): every OTHER resident model is
      unloaded — hot-swap fully replaces what's warm. This is today's
      behaviour and needs no multi-GPU awareness to be correct.

    budget>1 (future, multi-GPU/more VRAM): keeps the (budget - 1)
      most-recently-used OTHER models resident alongside keep_model and
      only evicts the least-recently-used beyond that — true concurrent
      multi-model access rather than replacement. Recency is taken from
      Ollama's own `expires_at` on each /api/ps entry (it resets on every
      use of that model), so no separate LRU tracking is needed here.

    Ollama keeps a loaded model resident for its keep_alive window (default
    5 minutes idle) after each request. Without this, every tier you've
    tried this session stays resident, competing for the same GPU/RAM,
    until it happens to idle out on its own.

    Set OLLAMA_AUTO_UNLOAD=false to disable this entirely.
    Best-effort: any failure here is logged and swallowed — this is a
    memory optimisation, not something that should block a generation
    request that would otherwise succeed.
    """
    if os.environ.get("OLLAMA_AUTO_UNLOAD", "true").lower() != "true":
        return
    try:
        import ollama
        client = ollama.Client(host=host)
        running = client.ps().get("models", [])
        others = [m for m in running if (m.get("model") or m.get("name")) != keep_model]
        if not others:
            return

        budget_for_others = max(OLLAMA_MAX_RESIDENT_MODELS - 1, 0)
        if budget_for_others > 0:
            # Multi-model path: keep the most-recently-used `budget_for_others`
            # others resident, evict the rest. expires_at is reset by Ollama on
            # every use, so sorting by it descending approximates true LRU order.
            others.sort(key=lambda m: m.get("expires_at") or "", reverse=True)
            to_unload = others[budget_for_others:]
        else:
            # budget=1 path (today): unload everything that isn't keep_model.
            to_unload = others

        for m in to_unload:
            tag = m.get("model") or m.get("name")
            if not tag:
                continue
            logger.info(f"Unloading idle Ollama model {tag!r} to free memory for {keep_model!r}")
            try:
                # keep_alive=0 tells Ollama to unload immediately after this
                # (empty, no-op) request rather than waiting out its idle timeout.
                client.generate(model=tag, prompt="", keep_alive=0)
            except Exception as ue:
                logger.warning(f"Could not unload {tag!r}: {ue}")
    except Exception as e:
        logger.warning(f"Could not query/unload running Ollama models (non-fatal): {e}")


def _ensure_ollama_model_available(model: str, host: str) -> None:
    """
    Check whether `model` is already pulled on the Ollama server at `host`;
    if not, pull it now (blocking).

    This is what makes runtime model hot-swapping actually work end-to-end.
    Picking a new tier/backend in the dropdown resolves a new model TAG
    immediately (that part always worked), but Ollama still needs that exact
    tag physically pulled before it can serve a request for it. Without this
    check, the first request after a hot-swap fails deep inside
    langchain_ollama with a raw 404 "model not found" — which is exactly
    what surfaced once the error-persistence fix made it visible.

    Only called from the Ollama branch of _get_llm(), and only when a
    runtime override was actually supplied (see call site) — the normal,
    no-override path (container's own default model, pulled once at
    startup by ollama-bootstrap) incurs no extra API call or latency.
    """
    try:
        import ollama
        client = ollama.Client(host=host)
        existing = set()
        for m in client.list().get("models", []):
            tag = m.get("model") or m.get("name")
            if tag:
                existing.add(tag)
        if model in existing:
            return
        logger.info(
            f"Model {model!r} not found on Ollama host {host} — pulling now "
            f"(first use of this tag can take a few minutes)..."
        )
        client.pull(model)
        logger.info(f"Pull complete: {model!r}")
    except Exception as e:
        # Re-raise as a clear, actionable message rather than letting the
        # underlying ollama/list/pull exception (or the original 404) reach
        # the caller — this is what the app's persisted error panel shows.
        raise RuntimeError(
            f"Model {model!r} is not available on Ollama ({host}) and could "
            f"not be auto-pulled: {e}. Try `ollama pull {model}` manually on "
            f"the Ollama host, or pick a different tier/backend."
        ) from e


def _get_llm(
    temperature: float = 0.1,
    backend: str | None = None,
    model: str | None = None,
    model_tier: str | None = None,
):
    """
    Return a LangChain chat model pointed at the configured inference backend.

    Runtime hot-swap
    ─────────────────
    backend / model / model_tier, when provided, override the INFERENCE_BACKEND /
    OLLAMA_MODEL / VLLM_MODEL / LLAMACPP_MODEL env vars for this call only —
    same live-state pattern as hitl_enabled/output_review_mode elsewhere in this
    file: the env var is only the fallback default, resolved once at import;
    the actual per-request value is read from AgentState (active_backend /
    active_model_tier) via _llm_kwargs_from_state() and passed in here on every
    node call, so a mid-conversation model change (e.g. the Streamlit dropdown)
    takes effect without restarting the container. If model is not given but
    model_tier is, the tag is resolved from the tier. If neither is given,
    falls back to the env-var defaults exactly as before — fully backward
    compatible with existing callers and tests.

    Host resolution is NOT overridden — OLLAMA_HOST/VLLM_HOST/LLAMACPP_HOST stay
    fixed per deployment; only which backend/model is targeted changes at runtime.

    Hot-swap only fully works for Ollama. Ollama's API can pull an arbitrary
    model tag on demand (see _ensure_ollama_model_available below) — the
    first request after switching to a not-yet-pulled tag will block for a
    few minutes while it downloads, then succeed. vLLM and llama-server are
    each started with ONE fixed model baked into their launch command
    (docker-compose_dev.yml `command:` args) — they cannot serve a different
    model without the container being restarted with new args. Switching
    the backend dropdown to vllm/llamacpp changes which server this
    conversation talks to, but not what that server is currently running;
    if it's serving a different model than what got resolved here, the
    request will fail at the backend itself rather than in this file.

    ollama   → ChatOllama. Talks to the Ollama REST API directly.
    vllm     → ChatOpenAI pointed at vLLM's OpenAI-compatible /v1 endpoint.
    llamacpp → ChatOpenAI pointed at llama-server's OpenAI-compatible /v1 endpoint.
               llama-server is started separately (see docker-compose_dev.yml --profile
               llamacpp). No API key required for either vLLM or llama-server.

    All three return the same LangChain BaseLanguageModel interface —
    all nodes are backend-agnostic.

    Tier affinity (not enforced, but documented):
      llamacpp → lightweight / minimal  (CPU-native GGUF, lowest memory)
      ollama   → any tier               (model management included)
      vllm     → full / balanced        (GPU, high throughput)
    """
    resolved_backend = (backend or _default_backend()).lower()

    if resolved_backend == "vllm":
        resolved_model = model or (
            _resolve_model_for_tier(model_tier) if model_tier else VLLM_MODEL
        )
        return ChatOpenAI(
            base_url=f"{VLLM_HOST}/v1",
            api_key="not-required",
            model=resolved_model,
            temperature=temperature,
            max_tokens=4096,
        )
    if resolved_backend == "llamacpp":
        # model_tier is deliberately ignored here (unlike the ollama branch below):
        # llama-server is started with ONE fixed model baked into its launch
        # command and cannot switch at runtime. _resolve_model_for_tier() would
        # produce an Ollama-style tag (e.g. "mistral-nemo:12b-instruct-q4_K_M")
        # that doesn't match whatever --served-model-name llama-server was
        # actually started with -- sending that would risk a model-name
        # mismatch against the real server. `model` (an explicit override) still
        # works if the caller genuinely knows the served name.
        # The served model is read from the server (/v1/models); LLAMACPP_MODEL is only the
        # fallback when the server cannot be reached.
        resolved_model = model or _llamacpp_served_model() or LLAMACPP_MODEL
        return ChatOpenAI(
            base_url=f"{LLAMACPP_HOST}/v1",
            api_key="not-required",
            model=resolved_model,
            temperature=temperature,
            # full tier (mistral-nemo, 8192 ctx) benefits from more generous
            # output headroom than the old 2048 default, which was sized for
            # the tiny minimal-tier model this used to point at.
            max_tokens=4096,
        )
    # Default: Ollama
    resolved_model = model or (
        _resolve_model_for_tier(model_tier) if model_tier else OLLAMA_MODEL
    )
    # Only check/pull when this call came from a runtime override (backend,
    # model, or model_tier explicitly passed) — the ordinary env-default path
    # already has its one model pulled once at container startup by
    # ollama-bootstrap, so skipping the check there avoids an extra API call
    # on every single node invocation.
    if backend or model or model_tier:
        _unload_other_ollama_models(resolved_model, OLLAMA_HOST)
        _ensure_ollama_model_available(resolved_model, OLLAMA_HOST)
    return ChatOllama(
        base_url=OLLAMA_HOST,
        model=resolved_model,
        temperature=temperature,
        num_ctx=_resolve_ctx_for_tier(model_tier),
    )


def _llm_kwargs_from_state(state: "AgentState") -> dict[str, Any]:
    """
    Extract the runtime model override (if any) from AgentState, ready to
    unpack into _get_llm(): `_get_llm(temperature=X, **_llm_kwargs_from_state(state))`.
    Returns {} when no override is set, giving the env-var default behaviour —
    same pattern as `state.get("output_review_mode") or OUTPUT_REVIEW_MODE`
    used elsewhere in this file, just as kwargs instead of an `or` fallback
    since _get_llm needs to distinguish "unset" from "explicitly default".
    """
    kwargs: dict[str, Any] = {}
    backend = state.get("active_backend")
    tier = state.get("active_model_tier")
    if backend:
        kwargs["backend"] = backend
    if tier:
        kwargs["model_tier"] = tier
    return kwargs


# ---------------------------------------------------------------------------
# Node: tool_selection
# ---------------------------------------------------------------------------

TOOL_SELECTION_SYSTEM = """You are a tool-selection agent for a code documentation assistant.
Your job is to decide which tools to use to answer the user's query about a codebase.

Available tools:
{tool_descriptions}

Rules:
1. Choose the minimum set of tools that will answer the query well.
2. For specific file/function questions: prefer grep + cat + ast_parse.
3. For conceptual/architectural questions (e.g. "how does X work", "could A integrate
   with B", "what does this repo do"): prefer vector_search over any other tool.
4. For questions about change history: use git_log or git_blame.
5. Combine tools when needed (e.g. find → grep → cat is a common chain).
6. If the query mentions "Indexed repos" below, those repos are ALREADY embedded and
   searchable — use vector_search against them. Do NOT use github_fetch to read a
   single file (like README.md) from an indexed repo as a substitute for search;
   github_fetch is only for repos/files that are NOT in the indexed list, or for a
   specific named file the user explicitly asked to see verbatim.

Respond ONLY with a JSON array of tool calls. Each element must have:
  {{"tool_name": str, "args": {{...}}, "reasoning": str}}

Example:
[
  {{"tool_name": "grep", "args": {{"repo_path": "/data/repos/myrepo", "pattern": "def process_request", "include": "*.py"}}, "reasoning": "Locate the function definition first"}},
  {{"tool_name": "cat", "args": {{"repo_path": "/data/repos/myrepo", "file_path": "src/handler.py", "start_line": 45, "end_line": 90}}, "reasoning": "Read the function body once grep finds it"}}
]
"""


def _scope_vector_search_args(args: dict, active_cols: list, human_edited: bool = False) -> list[str]:
    """
    Normalise a vector_search call in place and return what was ignored, as readable strings.

    The code, not the model, owns every argument that sets search scope or recall. A weak
    planner mis-uses the schema in ways that each end as "no context": it puts one repo in
    `collection_name` and the other in `collections` (silently dropping a repo), invents a
    `filter_file` that is not a path, or guesses a `score_threshold` that excludes everything,
    and it cannot know the internal Chroma host (it guesses "localhost:8000"). So, for a
    planner-made plan:
      * `collections` becomes ALL active per-repo collections;
      * `score_threshold` and `filter_file` are dropped (config / tool defaults apply);
      * `chroma_host` is always the real internal address.
    This runs when the plan is PROPOSED, so the human reviewing it sees exactly what will run,
    and the ignored planner values are returned for the trace (they are the measure of how well
    a given model follows the tool contract).

    human_edited=True means a person changed the plan in the review step: their `collections`,
    `filter_file` and `score_threshold` are respected and only the host is forced (and
    `collections` is filled in if left empty).
    """
    dropped: list[str] = []
    if active_cols:
        planned = args.get("collections")
        if isinstance(planned, str):
            planned = [x.strip() for x in planned.split(",") if x.strip()]
        planned = list(planned or [])
        if args.get("collection_name"):
            planned.append(args["collection_name"])
        if not human_edited:
            if planned and set(planned) != set(active_cols):
                dropped.append(f"collections={planned!r} (searching all {len(active_cols)} active)")
            args["collections"] = list(active_cols)
            args.pop("collection_name", None)
        elif not planned:
            args["collections"] = list(active_cols)
            args.pop("collection_name", None)
    args["chroma_host"] = CHROMA_HOST
    if not human_edited:
        for k in ("score_threshold", "filter_file"):
            if k in args:
                dropped.append(f"{k}={args.pop(k)!r}")
    return dropped


def node_tool_selection(state: AgentState) -> dict[str, Any]:
    """LLM proposes a tool plan for the given query."""
    # build_tool_registry() filters MCP tools by is_mcp_capable() + server enabled state,
    # so the LLM is only offered tools it can actually use.
    registry = build_tool_registry()
    tool_descs = "\n".join(
        f"  - {name}: {meta['description']}\n"
        f"      required args: {', '.join(meta.get('required_args', [])) or '(none)'}\n"
        f"      optional args: {', '.join(meta.get('optional_args', [])) or '(none)'}"
        for name, meta in registry.items()
    )
    system_msg = TOOL_SELECTION_SYSTEM.format(tool_descriptions=tool_descs)
    active_cols = state.get("active_collections") or []
    if active_cols:
        indexed_note = f"\nIndexed repos (already embedded, use vector_search): {', '.join(active_cols)}"
    else:
        indexed_note = ""
    user_msg = f"Query: {state['query']}\nRepo path: {state['repo_path']}{indexed_note}"
    # Re-plan: the reviewer sent the previous plan back. Show the model what was rejected and what
    # the reviewer expected, so the new plan is different on purpose, not a re-roll.
    replan_feedback = state.get("planner_feedback")
    if replan_feedback is not None:
        prev = [{"tool_name": tc.tool_name, "args": tc.args} for tc in state.get("proposed_tool_calls") or []]
        user_msg += ("\n\nA human reviewer REJECTED this previous plan: "
                     + json.dumps(prev, default=str)[:1500]
                     + (f"\nWhat the reviewer expected: {replan_feedback}" if replan_feedback.strip()
                        else "\n(No reason given - propose a meaningfully different plan.)")
                     + "\nPropose a new plan that addresses this.")

    llm = _get_llm(temperature=0.0, **_llm_kwargs_from_state(state))
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    planner_fell_back = False
    try:
        plan = json.loads(raw)
        tool_calls = [
            ToolCall(tool_name=tc["tool_name"], args=tc["args"])
            for tc in plan
        ]
        if not tool_calls:
            raise ValueError("model proposed zero tool calls")
    except Exception:
        planner_fell_back = True
        tool_calls = [
            ToolCall(
                tool_name="vector_search",
                args={"query": state["query"], "chroma_host": CHROMA_HOST},
            )
        ]

    # Normalise before anyone sees the plan (see _scope_vector_search_args): the human review
    # step then shows what will really run, and what the planner got wrong stays in the trace.
    ignored: list[str] = []
    for tc in tool_calls:
        if tc.tool_name == "vector_search":
            ignored.extend(_scope_vector_search_args(tc.args, active_cols))

    # Planner-quality signals for the model comparison: did it return a usable plan at all, and how
    # many of its arguments did the code have to override?
    try:
        mlflow.log_metric("planner_json_ok", 0 if planner_fell_back else 1)
        mlflow.log_metric("planner_ignored_args", len(ignored))
    except Exception:
        pass

    trace = list(state.get("execution_trace", []))
    trace.append({"node": "tool_selection", "status": "ok",
                  "detail": f"proposed {len(tool_calls)} tool(s): " + "; ".join(
                      f"{tc.tool_name}({json.dumps(tc.args, default=str)[:160]})"
                      for tc in tool_calls)
                  + (f" | ignored planner args: {', '.join(ignored)}" if ignored else "")})
    out: dict[str, Any] = {"proposed_tool_calls": tool_calls, "execution_trace": trace}
    if replan_feedback is not None:
        out["planner_feedback"] = None            # consumed
    return out


# ---------------------------------------------------------------------------
# Node: hitl_checkpoint
# ---------------------------------------------------------------------------

def node_hitl_checkpoint(state: AgentState) -> dict[str, Any]:
    """
    Human-in-the-loop review of the proposed tool plan.

    hitl_enabled comes from state (set fresh by app.py from the UI toggle on
    every request) so it's a genuine per-request setting, not a value frozen
    at process start. Falls back to the HITL_ENABLED env default only when a
    caller invokes the graph without setting it in state.

    When enabled: pauses execution with interrupt(), waiting for human
    approval via graph.invoke(Command(resume={...})).
    When disabled: auto-approves.
    """
    hitl = state.get("hitl_enabled")
    if hitl is None:
        hitl = HITL_ENABLED
    if not hitl:
        return {
            "approved_tool_calls": state["proposed_tool_calls"],
            "hitl_checkpoint": HITLCheckpoint(
                proposed_tool_calls=state["proposed_tool_calls"],
                decision="approved",
            ),
        }

    # Pause — the Streamlit UI will resume with human feedback
    human_response = interrupt({
        "proposed_tool_calls": [
            {"tool_name": tc.tool_name, "args": tc.args}
            for tc in state["proposed_tool_calls"]
        ],
        "message": "Review the proposed tool plan. Approve, modify, re-plan (with feedback), or end.",
    })

    decision: str = human_response.get("decision", "approved")
    modified_calls_raw: list = human_response.get("tool_calls", [])

    if decision == "approved":
        approved = state["proposed_tool_calls"]
    elif decision == "modified":
        approved = [ToolCall(tool_name=tc["tool_name"], args=tc["args"]) for tc in modified_calls_raw]
    else:  # "rejected" (end here) or "replan" (send back)
        approved = []

    replans = state.get("replan_count") or 0
    limit_hit = decision == "replan" and replans >= MAX_REPLANS
    if limit_hit:
        decision = "rejected"          # no more re-plans allowed: end instead of looping forever

    checkpoint = HITLCheckpoint(
        proposed_tool_calls=state["proposed_tool_calls"],
        decision=decision,
        modified_tool_calls=approved if decision == "modified" else None,
        feedback=human_response.get("feedback"),
    )
    trace = list(state.get("execution_trace", []))
    status = {"rejected": "rejected", "replan": "retry"}.get(decision, "ok")
    detail = f"human decision: {decision}"
    if decision == "replan":
        fb = (human_response.get("feedback") or "").strip()
        detail += f" (re-plan {replans + 1}/{MAX_REPLANS}" + (f", feedback: {fb[:120]}" if fb else ", no feedback") + ")"
    elif limit_hit:
        detail += f" (re-plan limit {MAX_REPLANS} reached - ending)"
    trace.append({"node": "hitl_checkpoint", "status": status, "detail": detail})
    out: dict[str, Any] = {"approved_tool_calls": approved, "hitl_checkpoint": checkpoint,
                           "execution_trace": trace}
    if decision == "replan":
        out["planner_feedback"] = (human_response.get("feedback") or "").strip()
        out["replan_count"] = replans + 1
    return out


def _route_after_hitl(state: AgentState) -> Literal["tool_execution", "tool_selection", "__end__"]:
    """tool_execution if approved; back to tool_selection on a re-plan; otherwise END."""
    if getattr(state.get("hitl_checkpoint"), "decision", None) == "replan":
        return "tool_selection"
    if not state.get("approved_tool_calls"):
        return "__end__"
    return "tool_execution"


# ---------------------------------------------------------------------------
# Node: tool_execution
# ---------------------------------------------------------------------------

def node_tool_execution(state: AgentState) -> dict[str, Any]:
    """Execute all approved tool calls and collect results as Chunks."""
    approved = state.get("approved_tool_calls", [])
    executed: list[ToolCall] = []
    new_chunks: list[Chunk] = []
    vec_calls = vec_hits = 0       # vector_search calls / chunks they returned (0 hits is a warning, not a success)
    active_cols = state.get("active_collections", [])
    dropped_args: list[str] = []
    human_edited = getattr(state.get("hitl_checkpoint"), "decision", None) == "modified"

    for tc in approved:
        # Normally a no-op: node_tool_selection already normalised the plan, so the human
        # reviewed exactly what runs. It still guards plans that did not come from the
        # planner, and re-applies the host. When the human EDITED the plan, their
        # collections / filter_file / score_threshold are honoured, not overridden.
        if tc.tool_name == "vector_search":
            dropped_args.extend(_scope_vector_search_args(
                tc.args, active_cols, human_edited=human_edited))
        result = run_tool(tc.tool_name, tc.args)
        tc.result = result.get("result", "") or json.dumps(result.get("chunks", []))
        tc.success = result.get("success", False)
        tc.error = result.get("error")
        tc.latency_ms = result.get("latency_ms")
        executed.append(tc)

        if tc.tool_name == "vector_search":
            vec_calls += 1
            vec_hits += len(result.get("chunks") or [])

        # Convert results to Chunks
        if tc.tool_name == "vector_search" and result.get("chunks"):
            for c in result["chunks"]:
                new_chunks.append(Chunk(
                    content=c["content"],
                    source_file=c["source_file"],
                    start_line=c.get("start_line"),
                    end_line=c.get("end_line"),
                    chunk_type=c.get("chunk_type", "text"),
                    confidence=c.get("confidence", 0.0),
                ))
        elif tc.tool_name != "vector_search" and tc.success and tc.result:
            # Shell/AST tools: wrap output as a single chunk, high confidence (exact match).
            # vector_search is excluded on purpose: a search with zero hits returns no
            # "result" text, so tc.result falls back to json.dumps([]) == "[]" -- which is
            # truthy and used to become a fake chunk (source "codebase", confidence 1.0).
            # That made the supervisor see perfect retrieval and the model be handed "[]"
            # as its only context.
            source = tc.args.get("file_path", tc.args.get("path", "codebase"))
            new_chunks.append(Chunk(
                content=tc.result,
                source_file=source,
                chunk_type="grep_match" if tc.tool_name == "grep" else "text",
                confidence=1.0,
            ))

    scores = [c.confidence for c in new_chunks]
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "tool_execution", "status": "warn" if (vec_calls and not vec_hits) else "ok",
                  "detail": f"ran {len(executed)} tool(s), got {len(new_chunks)} chunk(s) -- " + "; ".join(
                      f"{t.tool_name}:{'ok' if t.success else 'FAIL'}"
                      + (f" collections={t.args.get('collections')}" if t.tool_name == "vector_search" else "")
                      + (f" error={str(t.error)[:100]}" if t.error else "")
                      for t in executed)
                  + (f" | ignored planner args: {', '.join(dropped_args)}" if dropped_args else "")
                  + (" | WARNING: vector_search returned 0 chunks (empty or unindexed collections?)"
                     if vec_calls and not vec_hits else "")})
    return {
        "executed_tool_calls": executed,
        "retrieved_chunks": new_chunks,
        "confidence_scores": scores,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "execution_trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: supervisor (short-loop optimiser)
# ---------------------------------------------------------------------------

def node_supervisor(state: AgentState) -> dict[str, Any]:
    """
    Short-loop optimisation supervisor (extended).

    Pre-generation responsibilities:
      1. Evaluate retrieval confidence_scores.
      2. Inject session_preferences into tool args (prioritised files, top_k bias).
      3. If confidence < threshold and attempts remaining: adjust and retry.
      4. If proceeding: inject format preference hint into state for generation node.

    Post-generation feedback is handled separately in node_output_review,
    which loops back here via the add_context route.
    """
    scores = state.get("confidence_scores", [])
    attempts = state.get("retrieval_attempts", 1)
    max_attempts = state.get("max_retrieval_attempts", MAX_RETRIEVAL_ATTEMPTS)
    adjustments = list(state.get("supervisor_adjustments", []))
    prefs: Optional[SessionPreferences] = state.get("session_preferences")

    mean_score = sum(scores) / len(scores) if scores else 0.0

    # --- Inject session preferences into the proceed path ---
    if mean_score >= CONFIDENCE_THRESHOLD or attempts >= max_attempts or not scores:
        reason = (
            f"mean_score={mean_score:.2f} >= threshold={CONFIDENCE_THRESHOLD}"
            if mean_score >= CONFIDENCE_THRESHOLD
            else f"max_attempts={max_attempts} reached"
            if attempts >= max_attempts
            else "no retrieval scores (non-semantic tools used)"
        )
        adjustments.append(SupervisorAdjustment(
            reason=reason, action="proceed",
            before={"mean_score": mean_score, "attempts": attempts},
            after={},
        ))

        # Surface preference hints as state — generation node reads these
        format_hint = prefs.preferred_format if prefs else None
        verbosity_hint = prefs.preferred_verbosity if prefs else None

        try:
            mlflow.log_metric("supervisor_mean_score_final", mean_score)
            if prefs:
                mlflow.log_metric("session_avg_satisfaction", prefs.avg_satisfaction)
                mlflow.log_metric("session_feedback_count", prefs.feedback_count)
        except Exception:
            pass

        trace = list(state.get("execution_trace", []))
        trace.append({"node": "supervisor", "status": "ok", "detail": f"proceed — {reason}"})
        return {
            "proceed_to_generation": True,
            "supervisor_adjustments": adjustments,
            "_format_hint": format_hint,
            "_verbosity_hint": verbosity_hint,
            "execution_trace": trace,
        }

    # --- Retry path: adjust tool args ---
    current_calls = state.get("approved_tool_calls", [])
    new_calls: list[ToolCall] = []
    before_params: dict = {}
    after_params: dict = {}

    for tc in current_calls:
        new_args = dict(tc.args)

        if tc.tool_name == "vector_search":
            old_k = new_args.get("top_k", 5)
            new_args["top_k"] = min(old_k + 3, 15)
            new_args["score_threshold"] = max(new_args.get("score_threshold", 0.3) - 0.05, 0.1)
            # Bias toward prioritised files if preferences exist
            if prefs and prefs.prioritised_files and not new_args.get("filter_file"):
                new_args["_prioritised_files"] = prefs.prioritised_files[:3]
            before_params = {"top_k": old_k}
            after_params = {"top_k": new_args["top_k"], "score_threshold": new_args["score_threshold"]}

        elif tc.tool_name == "grep":
            if "include" in new_args:
                before_params = {"include": new_args["include"]}
                del new_args["include"]
                after_params = {"include": "removed (widened search)"}
            # Add prioritised files to search path if preferences exist
            if prefs and prefs.prioritised_files:
                new_args["path"] = prefs.prioritised_files[0]
                after_params["path"] = new_args["path"]

        new_calls.append(ToolCall(tool_name=tc.tool_name, args=new_args))

    adjustments.append(SupervisorAdjustment(
        reason=f"mean_score={mean_score:.2f} < threshold={CONFIDENCE_THRESHOLD}, "
               f"attempt {attempts}/{max_attempts}",
        action="retry with adjusted params",
        before=before_params,
        after=after_params,
    ))

    try:
        mlflow.log_metric("supervisor_mean_score", mean_score, step=attempts)
        mlflow.log_metric("supervisor_retry", 1, step=attempts)
    except Exception:
        pass

    trace = list(state.get("execution_trace", []))
    trace.append({"node": "supervisor", "status": "retry",
                  "detail": f"score={mean_score:.2f} < {CONFIDENCE_THRESHOLD}, retrying"})
    return {
        "approved_tool_calls": new_calls,
        "retrieved_chunks": [],
        "confidence_scores": [],
        "proceed_to_generation": False,
        "supervisor_adjustments": adjustments,
        "execution_trace": trace,
    }


def _route_after_supervisor(state: AgentState) -> Literal["context_assembly", "tool_execution"]:
    if state.get("proceed_to_generation", False):
        return "context_assembly"
    return "tool_execution"


# ---------------------------------------------------------------------------
# Node: context_assembly
# ---------------------------------------------------------------------------

MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "8000"))

CONTEXT_DROP_WARN_FRACTION = float(os.environ.get("CONTEXT_DROP_WARN_FRACTION", "0.3"))


def node_context_assembly(state: AgentState) -> dict[str, Any]:
    """
    Deduplicate, rank by confidence, trim to context window, and
    build the final context string passed to the generation LLM.
    """
    chunks = state.get("retrieved_chunks", [])

    # Trim budget must be tier-aware and leave real headroom for the system
    # prompt (tool descriptions + instructions), the query, and the model's
    # own response — NOT just cap at a flat MAX_CONTEXT_TOKENS regardless of
    # the model's actual context window. Previously this used a flat 8000-token
    # budget with no relation to num_ctx at all; that's exactly what let a
    # fair-merged two-repo context overflow a model's real window even after
    # num_ctx was correctly wired into _get_llm() — the retrieved-context
    # budget alone could still exceed what's left after the system prompt.
    # Retrieved context gets ~60% of the window; the remainder is reserved.
    ctx_window = _effective_ctx_window(state)
    budget_tokens = min(MAX_CONTEXT_TOKENS, int(ctx_window * 0.6))

    # Deduplicate by content hash, preserving arrival order.
    # IMPORTANT: do NOT re-sort by raw confidence here. retrieved_chunks already
    # arrives fair-merged across collections (tool_vector_search interleaves by
    # within-collection rank so a smaller/lower-scoring repo isn't crowded out —
    # see tools.py). A global confidence sort at this stage silently undoes that:
    # if one repo's chunks score systematically higher (different domain, more
    # directly relevant vocabulary), they'd all float to the front and the trim
    # below would truncate the other repo out first — exactly the failure mode
    # multi-repo fair-merge exists to prevent.
    seen: set[int] = set()
    unique: list[Chunk] = []
    for c in chunks:
        h = hash(c.content.strip())
        if h not in seen:
            seen.add(h)
            unique.append(c)

    # Trim to context window (rough token estimate: 1 token ≈ 4 chars)
    max_chars = budget_tokens * 4
    context_parts: list[str] = []
    source_files: list[str] = []
    total_chars = 0
    truncated = 0

    for c in unique:
        part = f"### {c.source_file}" + (
            f" (lines {c.start_line}–{c.end_line})" if c.start_line else ""
        ) + f"\n```\n{c.content}\n```\n"
        if total_chars + len(part) > max_chars:
            if context_parts:
                break
            # The FIRST (best-ranked) chunk alone exceeds the budget. Dropping it -- and with it every
            # later chunk -- used to leave the model with "[No relevant context found]" and no sources
            # even though retrieval had worked. Keep its head instead.
            part = part[:max_chars] + "\n[... truncated to fit the context budget]\n```\n"
            truncated += 1
        context_parts.append(part)
        total_chars += len(part)
        if c.source_file not in source_files:
            source_files.append(c.source_file)

    included = len(context_parts)
    dropped = len(unique) - included
    final_context = "\n".join(context_parts) if context_parts else "[No relevant context found]"
    empty = included == 0
    detail = (f"NOTHING retrieved: 0 chunks reached context assembly (generation will be skipped)" if empty else
              f"{included}/{len(unique)} chunk(s) in context (~{total_chars // 4} of {budget_tokens} budget tokens, "
              f"model window {ctx_window})"
              + (f", {dropped} dropped for budget" if dropped else "")
              + (f", {truncated} truncated" if truncated else "")
              + f"; sources: {len(source_files)}")
    trace = list(state.get("execution_trace", []))
    # Transparency: a retrieval that mostly did not fit the model's window is not a clean success.
    heavy_drop = bool(unique) and dropped / len(unique) > CONTEXT_DROP_WARN_FRACTION
    if heavy_drop:
        detail += (f" | WARNING: more than {int(CONTEXT_DROP_WARN_FRACTION * 100)}% of the retrieved chunks "
                   "did not fit (small context window, or oversize chunks)")
    trace.append({"node": "context_assembly", "status": "warn" if (empty or heavy_drop) else "ok", "detail": detail})
    return {"final_context": final_context, "source_attribution": source_files,
            "retrieval_empty": empty, "execution_trace": trace}


# ---------------------------------------------------------------------------
# Node: generation
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = """You are a code documentation assistant.
You will be given context retrieved from a codebase (files, functions, grep results)
and a user question. Your job is to produce clear, accurate documentation or an answer.

Rules:
1. Base your response ONLY on the provided context. Do not hallucinate file paths or function names.
2. If the context is insufficient, say so explicitly rather than guessing.
3. Use markdown for code blocks and structure.
4. Cite the source file for every claim (e.g. "In `src/ingest.py`, the function...").
5. If asked to produce documentation, format it as docstrings or markdown, as appropriate.
{format_instruction}
{verbosity_instruction}
"""


SELF_CRITIQUE_PROMPT = """Review the documentation you just produced against the original query and context.

Query: {query}

Is your response:
1. Accurate — does it match what's actually in the context?
2. Complete — does it cover the scope of the query?
3. Cited — does it reference the source files?

If you find specific errors or gaps, rewrite the response to fix them.
If the response is already good, return it unchanged.

Respond with only the (possibly revised) documentation, no preamble."""


def _collection_counts(cols: list[str]) -> dict[str, Optional[int]]:
    """Chunks stored per collection (None = could not be read). Used only to explain an empty retrieval."""
    try:
        try:
            from repo_index import _chroma_client
        except ImportError:
            from src.repo_index import _chroma_client
        client = _chroma_client(CHROMA_HOST)
    except Exception:
        return {c: None for c in cols}
    out: dict[str, Optional[int]] = {}
    for c in cols:
        try:
            out[c] = client.get_collection(c).count()
        except Exception:
            out[c] = None
    return out


def _no_retrieval_response(state: AgentState) -> dict[str, Any]:
    """Nothing was retrieved: say so, with the likely reason, instead of asking the model to answer
    from an empty context (it replies "no specific information", which reads like a content problem
    when the real problem is upstream: unindexed repo, missing embedding model, empty collection)."""
    cols = state.get("active_collections") or []
    counts = _collection_counts(cols) if cols else {}
    known = [v for v in counts.values() if v is not None]
    total = sum(known)
    lines = []
    if not cols:
        why = "No repositories are selected for this question, so there was nothing to search."
    elif known and total == 0:
        why = ("The selected repositories contain **0 indexed chunks**, so there is nothing to search. "
               "Indexing most likely failed or produced nothing: check the 📂 line under your question and the "
               "app log, and make sure the embedding model is available in Ollama.")
    elif known:
        why = (f"The search ran over {total} indexed chunks but none came back. Try a more specific question "
               "(file, function or class names), or check the retrieval settings.")
    else:
        why = "The index could not be read, so it is unknown whether the repositories are indexed."
    for name, n in counts.items():
        lines.append(f"- `{name}`: " + ("unreadable" if n is None else f"{n} chunks"))
    response = ("**No answer was generated: nothing was retrieved for this question.**\n\n" + why
                + ("\n\n" + "\n".join(lines) if lines else ""))
    try:
        mlflow.log_metric("retrieval_empty", 1)
    except Exception:
        pass
    attempts = state.get("generation_attempts", 0) + 1
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "generation", "status": "warn",
                  "detail": "LLM call skipped: nothing was retrieved" + (f" ({total} chunks indexed)" if known else "")})
    return {"response": response, "generation_attempts": attempts, "execution_trace": trace,
            "total_latency_ms": 0.0}


def node_generation(state: AgentState) -> dict[str, Any]:
    """
    Generate the final documentation/answer from the assembled context.

    When OUTPUT_REVIEW_MODE="self": appends a self-critique pass — the LLM
    reviews its own output against the query and revises if needed. This adds
    one extra LLM call but no human latency, and catches obvious mis-scoping
    before the response reaches the user or supervisor quality gate.
    """
    if state.get("retrieval_empty"):
        return _no_retrieval_response(state)
    context = state.get("final_context", "[No context]")
    query = state["query"]

    format_hint = state.get("_format_hint") or ""
    verbosity_hint = state.get("_verbosity_hint") or ""
    format_instruction = f"6. Use {format_hint} format for any docstrings." if format_hint else ""
    verbosity_instruction = f"7. Keep responses {verbosity_hint}." if verbosity_hint else ""

    system = GENERATION_SYSTEM.format(
        format_instruction=format_instruction,
        verbosity_instruction=verbosity_instruction,
    ).strip()

    start = time.time()
    llm_overrides = _llm_kwargs_from_state(state)
    llm = _get_llm(temperature=0.2, **llm_overrides)
    resolved_backend = llm_overrides.get("backend", _default_backend())
    resolved_model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])
    draft = response.content
    gen_attempts = state.get("generation_attempts", 0) + 1
    gen_mode = state.get("output_review_mode") or OUTPUT_REVIEW_MODE

    # Self-critique pass (only when output review mode == "self", read live from state)
    final_response = draft
    if gen_mode == "self":
        critique_response = llm.invoke([
            SystemMessage(content=SELF_CRITIQUE_PROMPT.format(query=query)),
            HumanMessage(content=f"Context:\n{context}\n\nYour draft:\n{draft}"),
        ])
        final_response = critique_response.content

    latency = round((time.time() - start) * 1000, 1)

    try:
        mlflow.log_metric("generation_latency_ms", latency, step=gen_attempts)
        mlflow.log_metric("response_length_chars", len(final_response), step=gen_attempts)
        mlflow.log_metric("generation_attempts", gen_attempts)
        mlflow.log_text(final_response, f"response_attempt_{gen_attempts}.txt")
        # Per-turn backend/model — a thread may hot-swap models across turns,
        # unlike the run-level params logged once in run_agent() which only
        # capture the initial container config.
        mlflow.log_param(f"turn_{gen_attempts}_backend", resolved_backend)
        if resolved_model:
            mlflow.log_param(f"turn_{gen_attempts}_model", resolved_model)
    except Exception:
        pass

    trace = list(state.get("execution_trace", []))
    mode_note = " (+ self-critique)" if gen_mode == "self" else ""
    model_note = f" via {resolved_backend}/{resolved_model}" if resolved_model else ""
    trace.append({"node": "generation", "status": "ok",
                  "detail": f"generated {len(final_response)} chars{mode_note}{model_note}"})
    return {
        "response": final_response,
        "total_latency_ms": latency,
        "generation_attempts": gen_attempts,
        "execution_trace": trace,
        "active_backend": resolved_backend,
        "active_model": resolved_model,
    }


# ---------------------------------------------------------------------------
# Node: output_review — behaviour depends on OUTPUT_REVIEW_MODE
# ---------------------------------------------------------------------------

QUALITY_GATE_RUBRIC = """You are evaluating a code documentation response. Score it 0–10 on each criterion:

1. Accuracy (0–4): Does the response accurately reflect what's in the provided context?
   Penalise hallucinated file paths, function names, or behaviours not present in the context.

2. Completeness (0–3): Does it address the full scope of the query?
   A partial answer covering only one aspect of a multi-part question scores low.

3. Attribution (0–3): Are source files cited for specific claims?

Query: {query}
Context provided: {context_summary}
Response to evaluate: {response}

Reply ONLY with a JSON object:
{{"accuracy": <0-4>, "completeness": <0-3>, "attribution": <0-3>, "total": <0-10>,
  "pass": <true|false>, "reason": "<one sentence>"}}
"""


def node_output_review(state: AgentState) -> dict[str, Any]:
    """
    Post-generation output review. Behaviour depends on OUTPUT_REVIEW_MODE:

    "human"      → interrupt_before: human rates + decides (accept/regenerate/add_context)
    "supervisor" → supervisor LLM evaluates against rubric; silent retry or accept
    "self"       → handled in node_generation; this node is a passthrough
    "off"        → immediate passthrough → accept
    """
    prefs: SessionPreferences = state.get("session_preferences") or SessionPreferences()
    gen_attempts = state.get("generation_attempts", 1)
    trace = list(state.get("execution_trace", []))
    mode = state.get("output_review_mode") or OUTPUT_REVIEW_MODE

    # Nothing was retrieved, so generation was skipped: there is no answer to rate, and an automatic
    # "accept, 5/5" (off/self) or a quality-gate retry (supervisor) would only pollute the preference
    # profile. End here; the response already explains what happened.
    if state.get("retrieval_empty"):
        trace.append({"node": "output_review", "status": "warn",
                      "detail": "nothing to review: no answer was generated"})
        return {"post_generation_feedback": None, "execution_trace": trace}

    # --- "off" and "self" modes: passthrough ---
    if mode in ("off", "self"):
        feedback = PostGenerationFeedback(
            response_shown=state.get("response", ""),
            decision="accept",
            satisfaction_score=5,
        )
        prefs.update(feedback)
        trace.append({"node": "output_review", "status": "ok",
                      "detail": f"mode={mode}, auto-accept"})
        return {"post_generation_feedback": feedback, "session_preferences": prefs,
                "execution_trace": trace}

    # --- "supervisor" mode: LLM quality gate ---
    if mode == "supervisor":
        if gen_attempts >= MAX_GENERATION_ATTEMPTS:
            # Max attempts reached — accept whatever we have
            feedback = PostGenerationFeedback(
                response_shown=state.get("response", ""),
                decision="accept",
                satisfaction_score=3,
            )
            prefs.update(feedback)
            trace.append({"node": "output_review", "status": "ok",
                          "detail": f"supervisor: max attempts reached, accepting"})
            return {"post_generation_feedback": feedback, "session_preferences": prefs,
                    "execution_trace": trace}

        # Build a short context summary for the rubric (avoid sending full context)
        context = state.get("final_context", "")
        context_summary = context[:800] + "..." if len(context) > 800 else context
        rubric_prompt = QUALITY_GATE_RUBRIC.format(
            query=state["query"],
            context_summary=context_summary,
            response=state.get("response", ""),
        )

        llm = _get_llm(temperature=0.0, **_llm_kwargs_from_state(state))
        try:
            eval_response = llm.invoke([HumanMessage(content=rubric_prompt)])
            raw = eval_response.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            scores = json.loads(raw)
            total = float(scores.get("total", 0))
            passed = scores.get("pass", total >= QUALITY_GATE_THRESHOLD)
            reason = scores.get("reason", "")
        except Exception as e:
            # If the rubric LLM call fails, accept to avoid infinite loops
            total, passed, reason = 5.0, True, f"rubric eval failed: {e}"

        try:
            mlflow.log_metric("quality_gate_score", total, step=gen_attempts)
            mlflow.log_metric("quality_gate_pass", int(passed), step=gen_attempts)
        except Exception:
            pass

        if passed:
            feedback = PostGenerationFeedback(
                response_shown=state.get("response", ""),
                decision="accept",
                satisfaction_score=min(5, int(total / 2)),
            )
            prefs.update(feedback)
            trace.append({"node": "output_review", "status": "ok",
                          "detail": f"supervisor: score={total:.1f}/10 PASS — {reason}"})
            return {"post_generation_feedback": feedback, "session_preferences": prefs,
                    "execution_trace": trace}
        else:
            feedback = PostGenerationFeedback(
                response_shown=state.get("response", ""),
                decision="regenerate",
                satisfaction_score=max(1, int(total / 2)),
                context_notes=f"Quality gate failed (score={total:.1f}/10): {reason}",
            )
            trace.append({"node": "output_review", "status": "retry",
                          "detail": f"supervisor: score={total:.1f}/10 FAIL — {reason}"})
            return {"post_generation_feedback": feedback, "session_preferences": prefs,
                    "proceed_to_generation": False, "execution_trace": trace}

    # --- "human" mode: interrupt and wait ---
    if gen_attempts >= MAX_GENERATION_ATTEMPTS:
        feedback = PostGenerationFeedback(
            response_shown=state.get("response", ""),
            decision="accept",
            satisfaction_score=3,
        )
        prefs.update(feedback)
        trace.append({"node": "output_review", "status": "ok",
                      "detail": "human: max attempts, auto-accept"})
        return {"post_generation_feedback": feedback, "session_preferences": prefs,
                "execution_trace": trace}

    human_response = interrupt({
        "response": state.get("response", ""),
        "source_attribution": state.get("source_attribution", []),
        "generation_attempt": gen_attempts,
        "message": "Review the generated documentation.",
        "current_preferences": {
            "format": prefs.preferred_format,
            "verbosity": prefs.preferred_verbosity,
            "avg_satisfaction": round(prefs.avg_satisfaction, 2),
        },
    })

    decision = human_response.get("decision", "accept")
    feedback = PostGenerationFeedback(
        response_shown=state.get("response", ""),
        decision=decision,
        satisfaction_score=human_response.get("satisfaction_score", 5),
        context_notes=human_response.get("context_notes"),
        format_notes=human_response.get("format_notes"),
        additional_files=human_response.get("additional_files", []),
    )
    prefs.update(feedback)

    try:
        mlflow.log_metric("user_satisfaction", feedback.satisfaction_score, step=gen_attempts)
        mlflow.log_param("output_decision", decision)
        if feedback.format_notes:
            mlflow.log_param("format_preference", feedback.format_notes)
    except Exception:
        pass

    extra_tool_calls: list[ToolCall] = []
    if decision == "add_context" and feedback.additional_files:
        for fpath in feedback.additional_files:
            extra_tool_calls.append(ToolCall(
                tool_name="cat",
                args={"repo_path": state["repo_path"], "file_path": fpath},
            ))

    trace.append({"node": "output_review", "status": "ok",
                  "detail": f"human: score={feedback.satisfaction_score}/5, decision={decision}"})
    return {
        "post_generation_feedback": feedback,
        "session_preferences": prefs,
        "approved_tool_calls": extra_tool_calls if extra_tool_calls else state.get("approved_tool_calls", []),
        # retrieved_chunks is an append-reducer field (operator.add): returning the existing list here
        # used to DOUBLE every chunk each time this node ran (accept, regenerate...), and returning []
        # never cleared anything. Leaving the key out keeps the chunks exactly as they are.
        "confidence_scores": [] if decision == "add_context" else state.get("confidence_scores", []),
        "proceed_to_generation": False if decision in ("regenerate", "add_context") else True,
        "execution_trace": trace,
    }


def _route_after_output_review(
    state: AgentState,
) -> Literal["__end__", "context_assembly", "tool_execution"]:
    feedback = state.get("post_generation_feedback")
    if not feedback or feedback.decision == "accept":
        return "__end__"
    if feedback.decision == "regenerate":
        return "context_assembly"
    return "tool_execution"


# ---------------------------------------------------------------------------
# Build the graph — topology varies by OUTPUT_REVIEW_MODE
# ---------------------------------------------------------------------------

def _tracked(node_fn):
    """
    Run a node with this question's MLflow run active (state["mlflow_run_id"]), so the node's own
    mlflow.log_* calls land in that run no matter which thread / Streamlit rerun executes it.
    A no-op when the state carries no run id.
    """
    @functools.wraps(node_fn)
    def inner(state, *args, **kwargs):
        with tracking.activate(state.get("mlflow_run_id")):
            return node_fn(state, *args, **kwargs)
    return inner


def make_checkpointer():
    """
    In-memory LangGraph checkpointer that (de)serialises our own state classes explicitly.

    Without this, LangGraph logs "Deserializing unregistered type agent_state.ToolCall ..." on every
    resume and says it will BLOCK such types in a future release -- which would break resuming at a
    human-review pause. The allow-list is built from agent_state itself, so a new state dataclass is
    covered automatically. Falls back gracefully on LangGraph versions with an older serde API.
    """
    import dataclasses
    import agent_state
    from langgraph.checkpoint.memory import MemorySaver
    allowed = [(agent_state.__name__, name) for name, obj in vars(agent_state).items()
               if dataclasses.is_dataclass(obj) and getattr(obj, "__module__", None) == agent_state.__name__]
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError:
        return MemorySaver()
    for kwargs in ({"pickle_fallback": True, "allowed_msgpack_modules": allowed}, {"pickle_fallback": True}):
        try:
            return MemorySaver(serde=JsonPlusSerializer(**kwargs))
        except TypeError:
            continue
    return MemorySaver()


def runtime_info(backend: str | None = None, tier: str | None = None) -> dict[str, Any]:
    """
    What is ACTUALLY answering, for logging (MLflow) and display: the backend and tier after UI overrides,
    and for llama.cpp what the server itself reports (served model, context size, GGUF, slots).
    """
    dep = deployment.resolve()
    b = (backend or dep["backend"]).lower()
    t = tier or dep["tier"]
    info: dict[str, Any] = {"backend": b, "tier": t, "default_source": dep["source"]}
    if b == "llamacpp":
        props = _llamacpp_props() or {}
        info.update(model=_llamacpp_served_model() or LLAMACPP_MODEL, n_ctx=_llamacpp_server_ctx(),
                    gguf=props.get("model_path"), slots=props.get("total_slots"))
    elif b == "vllm":
        info["model"] = _resolve_model_for_tier(tier) if tier else VLLM_MODEL
    else:
        info["model"] = _resolve_model_for_tier(tier) if tier else OLLAMA_MODEL
    try:
        from embedding import EMBEDDING_MODEL, embed_options
        info["embedding_model"] = EMBEDDING_MODEL
        info["embedding_options"] = embed_options()
    except Exception:  # noqa: BLE001
        pass
    return info


def build_graph(checkpointer=None, output_review_mode: str | None = None,
                hitl_enabled: bool | None = None) -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    The graph topology is fixed across all OUTPUT_REVIEW_MODE values — the
    output_review node always exists and is always connected. The mode controls:
      - Whether interrupt_before is set on output_review ("human" only)
      - What logic runs inside node_output_review at runtime

    This means get_graph_mermaid() always shows the same topology regardless
    of mode, which is intentional: the graph structure is stable, only the
    behaviour of one node changes.

    Args:
        checkpointer:       Optional LangGraph checkpointer for persistence.
        output_review_mode: Override OUTPUT_REVIEW_MODE (for testing / notebook use).
        hitl_enabled:       Override HITL_ENABLED (for testing / notebook use).
                            Both env-based defaults are resolved at module import,
                            so a UI toggle changing os.environ AFTER that point has
                            no effect unless passed explicitly here — this is the
                            hook for that. Callers (e.g. app.py) should pass the
                            live toggle value on every build_graph() call.
    """
    mode = output_review_mode or OUTPUT_REVIEW_MODE
    hitl = HITL_ENABLED if hitl_enabled is None else hitl_enabled

    builder = StateGraph(AgentState)
    builder.add_node("tool_selection", _tracked(node_tool_selection))
    builder.add_node("hitl_checkpoint", _tracked(node_hitl_checkpoint))
    builder.add_node("tool_execution", _tracked(node_tool_execution))
    builder.add_node("supervisor", _tracked(node_supervisor))
    builder.add_node("context_assembly", _tracked(node_context_assembly))
    builder.add_node("generation", _tracked(node_generation))
    builder.add_node("output_review", _tracked(node_output_review))

    builder.add_edge(START, "tool_selection")
    builder.add_edge("tool_selection", "hitl_checkpoint")
    builder.add_conditional_edges(
        "hitl_checkpoint",
        _route_after_hitl,
        {"tool_execution": "tool_execution", "tool_selection": "tool_selection", "__end__": END},
    )
    builder.add_edge("tool_execution", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"context_assembly": "context_assembly", "tool_execution": "tool_execution"},
    )
    builder.add_edge("context_assembly", "generation")
    builder.add_edge("generation", "output_review")
    builder.add_conditional_edges(
        "output_review",
        _route_after_output_review,
        {"__end__": END, "context_assembly": "context_assembly", "tool_execution": "tool_execution"},
    )

    compile_kwargs: dict[str, Any] = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    # NOTE: interrupt_before is intentionally NOT used here. It's a compile-time
    # (per-graph) setting and can't vary per-request, which is exactly the bug
    # that made the HITL toggle a no-op: this graph is built once and cached.
    # The dynamic interrupt() calls inside node_hitl_checkpoint / node_output_review
    # are the real, correct pause mechanism — they now read the live toggle value
    # from state (state["hitl_enabled"], state["output_review_mode"]) on every
    # request, set fresh by app.py each time. `hitl` / `mode` above remain as the
    # module-level fallback when a caller invokes the graph without setting them
    # in state (e.g. tests, notebook use).

    return builder.compile(**compile_kwargs)
def get_graph_mermaid() -> str:
    """Return the Mermaid diagram source for the graph (for Streamlit rendering)."""
    g = build_graph()
    return g.get_graph().draw_mermaid()


# ---------------------------------------------------------------------------
# MLflow run wrapper
# ---------------------------------------------------------------------------

def run_agent(query: str, repo_path: str, thread_id: str = "default",
              extra_state: dict | None = None) -> AgentState:
    """
    Run the full agent pipeline with MLflow tracking.

    Args:
        query:      User's question
        repo_path:  Path to the mounted repo
        thread_id:  LangGraph thread ID for checkpointing
        extra_state: Optional state overrides (e.g. max_retrieval_attempts)
    """
    graph = build_graph()
    initial_state: AgentState = {
        "query": query,
        "repo_path": repo_path,
        "proposed_tool_calls": [],
        "hitl_checkpoint": None,
        "approved_tool_calls": [],
        "executed_tool_calls": [],
        "retrieved_chunks": [],
        "confidence_scores": [],
        "retrieval_attempts": 0,
        "max_retrieval_attempts": MAX_RETRIEVAL_ATTEMPTS,
        "supervisor_adjustments": [],
        "proceed_to_generation": False,
        "final_context": "",
        "response": "",
        "source_attribution": [],
        "post_generation_feedback": None,
        "session_preferences": None,
        "generation_attempts": 0,
        "active_backend": None,
        "active_model_tier": None,
        "active_model": None,
        "execution_trace": [],
        "mlflow_run_id": None,
        "total_latency_ms": None,
        **(extra_state or {}),
    }

    # One run per question -- same lifecycle the Streamlit UI uses (see tracking.py).
    runtime = runtime_info(initial_state.get("active_backend"), initial_state.get("active_model_tier"))
    run_id = tracking.start_query_run(
        query, runtime,
        settings={"repo_path": repo_path, "hitl_enabled": initial_state.get("hitl_enabled", HITL_ENABLED),
                  "output_review_mode": initial_state.get("output_review_mode", OUTPUT_REVIEW_MODE)},
        tags={"source": "run_agent", "thread_id": thread_id},
    )
    initial_state["mlflow_run_id"] = run_id
    config = {"configurable": {"thread_id": thread_id}}
    try:
        final_state = graph.invoke(initial_state, config=config)
    except Exception as e:
        tracking.finish_run(run_id, None, error=str(e))
        raise
    tracking.finish_run(run_id, final_state)
    return final_state

