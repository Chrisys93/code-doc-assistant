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

import json
import os
import time
from typing import Any, Literal, Optional

import mlflow
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

from agent_state import (
    AgentState, Chunk, HITLCheckpoint, PostGenerationFeedback,
    SessionPreferences, SupervisorAdjustment, ToolCall
)
from tools import build_tool_registry, run_tool

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
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "ollama").lower()
VLLM_HOST = os.environ.get("VLLM_HOST", "http://localhost:8080")
VLLM_MODEL = os.environ.get("VLLM_MODEL", os.environ.get("OLLAMA_MODEL", "mistral-nemo"))
LLAMACPP_HOST = os.environ.get("LLAMACPP_HOST", "http://localhost:8081")
# LLAMACPP_MODEL is the --served-model-name passed to llama-server,
# or GGUF filename stem if no alias is set. Defaults to minimal tier model.
LLAMACPP_MODEL = os.environ.get("LLAMACPP_MODEL", "qwen2.5-coder-3b-instruct-q4_k_m")

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

# Supervisor quality-gate rubric score below which output is rejected (0–10)
QUALITY_GATE_THRESHOLD = float(os.environ.get("QUALITY_GATE_THRESHOLD", "6.0"))

# ---------------------------------------------------------------------------
# LLM client — backend-agnostic factory
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.1):
    """
    Return a LangChain chat model pointed at the configured inference backend.

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
    if INFERENCE_BACKEND == "vllm":
        return ChatOpenAI(
            base_url=f"{VLLM_HOST}/v1",
            api_key="not-required",
            model=VLLM_MODEL,
            temperature=temperature,
            max_tokens=4096,
        )
    if INFERENCE_BACKEND == "llamacpp":
        return ChatOpenAI(
            base_url=f"{LLAMACPP_HOST}/v1",
            api_key="not-required",
            model=LLAMACPP_MODEL,
            temperature=temperature,
            max_tokens=2048,  # conservative default; llama-server context is GGUF-defined
        )
    # Default: Ollama
    return ChatOllama(
        base_url=OLLAMA_HOST,
        model=OLLAMA_MODEL,
        temperature=temperature,
    )


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
3. For conceptual/architectural questions: prefer vector_search.
4. For questions about change history: use git_log or git_blame.
5. Combine tools when needed (e.g. find → grep → cat is a common chain).

Respond ONLY with a JSON array of tool calls. Each element must have:
  {{"tool_name": str, "args": {{...}}, "reasoning": str}}

Example:
[
  {{"tool_name": "grep", "args": {{"repo_path": "/data/repos/myrepo", "pattern": "def process_request", "include": "*.py"}}, "reasoning": "Locate the function definition first"}},
  {{"tool_name": "cat", "args": {{"repo_path": "/data/repos/myrepo", "file_path": "src/handler.py", "start_line": 45, "end_line": 90}}, "reasoning": "Read the function body once grep finds it"}}
]
"""


def node_tool_selection(state: AgentState) -> dict[str, Any]:
    """LLM proposes a tool plan for the given query."""
    # build_tool_registry() filters MCP tools by is_mcp_capable() + server enabled state,
    # so the LLM is only offered tools it can actually use.
    registry = build_tool_registry()
    tool_descs = "\n".join(
        f"  - {name}: {meta['description']}"
        for name, meta in registry.items()
    )
    system_msg = TOOL_SELECTION_SYSTEM.format(tool_descriptions=tool_descs)
    user_msg = f"Query: {state['query']}\nRepo path: {state['repo_path']}"

    llm = _get_llm(temperature=0.0)
    response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    try:
        plan = json.loads(raw)
        tool_calls = [
            ToolCall(tool_name=tc["tool_name"], args=tc["args"])
            for tc in plan
        ]
    except Exception:
        tool_calls = [
            ToolCall(
                tool_name="vector_search",
                args={"query": state["query"], "chroma_host": CHROMA_HOST},
            )
        ]

    trace = list(state.get("execution_trace", []))
    trace.append({"node": "tool_selection", "status": "ok",
                  "detail": f"proposed {len(tool_calls)} tool(s)"})
    return {"proposed_tool_calls": tool_calls, "execution_trace": trace}


# ---------------------------------------------------------------------------
# Node: hitl_checkpoint
# ---------------------------------------------------------------------------

def node_hitl_checkpoint(state: AgentState) -> dict[str, Any]:
    """
    Human-in-the-loop review of the proposed tool plan.

    When HITL_ENABLED=true: pauses execution with interrupt(), waiting for
    human approval via graph.invoke(Command(resume={...})).

    When HITL_ENABLED=false (CI/automated): auto-approves.
    """
    if not HITL_ENABLED:
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
        "message": "Review the proposed tool plan. Approve, modify, or reject.",
    })

    decision: str = human_response.get("decision", "approved")
    modified_calls_raw: list = human_response.get("tool_calls", [])

    if decision == "approved":
        approved = state["proposed_tool_calls"]
    elif decision == "modified":
        approved = [ToolCall(tool_name=tc["tool_name"], args=tc["args"]) for tc in modified_calls_raw]
    else:  # rejected
        approved = []

    checkpoint = HITLCheckpoint(
        proposed_tool_calls=state["proposed_tool_calls"],
        decision=decision,
        modified_tool_calls=approved if decision == "modified" else None,
        feedback=human_response.get("feedback"),
    )
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "hitl_checkpoint", "status": "ok" if decision != "rejected" else "rejected",
                  "detail": f"human decision: {decision}"})
    return {"approved_tool_calls": approved, "hitl_checkpoint": checkpoint, "execution_trace": trace}


def _route_after_hitl(state: AgentState) -> Literal["tool_execution", "__end__"]:
    """Route to tool_execution if approved, else END."""
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

    for tc in approved:
        result = run_tool(tc.tool_name, tc.args)
        tc.result = result.get("result", "") or json.dumps(result.get("chunks", []))
        tc.success = result.get("success", False)
        tc.error = result.get("error")
        tc.latency_ms = result.get("latency_ms")
        executed.append(tc)

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
        elif tc.success and tc.result:
            # Shell/AST tools: wrap output as a single chunk, high confidence (exact match)
            source = tc.args.get("file_path", tc.args.get("path", "codebase"))
            new_chunks.append(Chunk(
                content=tc.result,
                source_file=source,
                chunk_type="grep_match" if tc.tool_name == "grep" else "text",
                confidence=1.0,
            ))

    scores = [c.confidence for c in new_chunks]
    trace = list(state.get("execution_trace", []))
    trace.append({"node": "tool_execution", "status": "ok",
                  "detail": f"ran {len(executed)} tool(s), got {len(new_chunks)} chunk(s)"})
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

def node_context_assembly(state: AgentState) -> dict[str, Any]:
    """
    Deduplicate, rank by confidence, trim to context window, and
    build the final context string passed to the generation LLM.
    """
    chunks = state.get("retrieved_chunks", [])

    # Deduplicate by content hash
    seen: set[int] = set()
    unique: list[Chunk] = []
    for c in chunks:
        h = hash(c.content.strip())
        if h not in seen:
            seen.add(h)
            unique.append(c)

    # Sort by confidence descending
    unique.sort(key=lambda c: c.confidence, reverse=True)

    # Trim to context window (rough token estimate: 1 token ≈ 4 chars)
    max_chars = MAX_CONTEXT_TOKENS * 4
    context_parts: list[str] = []
    source_files: list[str] = []
    total_chars = 0

    for c in unique:
        part = f"### {c.source_file}" + (
            f" (lines {c.start_line}–{c.end_line})" if c.start_line else ""
        ) + f"\n```\n{c.content}\n```\n"
        if total_chars + len(part) > max_chars:
            break
        context_parts.append(part)
        total_chars += len(part)
        if c.source_file not in source_files:
            source_files.append(c.source_file)

    final_context = "\n".join(context_parts) if context_parts else "[No relevant context found]"
    return {"final_context": final_context, "source_attribution": source_files}


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


def node_generation(state: AgentState) -> dict[str, Any]:
    """
    Generate the final documentation/answer from the assembled context.

    When OUTPUT_REVIEW_MODE="self": appends a self-critique pass — the LLM
    reviews its own output against the query and revises if needed. This adds
    one extra LLM call but no human latency, and catches obvious mis-scoping
    before the response reaches the user or supervisor quality gate.
    """
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
    llm = _get_llm(temperature=0.2)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])
    draft = response.content
    gen_attempts = state.get("generation_attempts", 0) + 1

    # Self-critique pass (only when OUTPUT_REVIEW_MODE="self")
    final_response = draft
    if OUTPUT_REVIEW_MODE == "self":
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
    except Exception:
        pass

    trace = list(state.get("execution_trace", []))
    mode_note = " (+ self-critique)" if OUTPUT_REVIEW_MODE == "self" else ""
    trace.append({"node": "generation", "status": "ok",
                  "detail": f"generated {len(final_response)} chars{mode_note}"})
    return {
        "response": final_response,
        "total_latency_ms": latency,
        "generation_attempts": gen_attempts,
        "execution_trace": trace,
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

    # --- "off" and "self" modes: passthrough ---
    if OUTPUT_REVIEW_MODE in ("off", "self"):
        feedback = PostGenerationFeedback(
            response_shown=state.get("response", ""),
            decision="accept",
            satisfaction_score=5,
        )
        prefs.update(feedback)
        trace.append({"node": "output_review", "status": "ok",
                      "detail": f"mode={OUTPUT_REVIEW_MODE}, auto-accept"})
        return {"post_generation_feedback": feedback, "session_preferences": prefs,
                "execution_trace": trace}

    # --- "supervisor" mode: LLM quality gate ---
    if OUTPUT_REVIEW_MODE == "supervisor":
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

        llm = _get_llm(temperature=0.0)
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
        "retrieved_chunks": [] if decision == "add_context" else state.get("retrieved_chunks", []),
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

def build_graph(checkpointer=None, output_review_mode: str | None = None) -> Any:
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
    """
    mode = output_review_mode or OUTPUT_REVIEW_MODE

    builder = StateGraph(AgentState)
    builder.add_node("tool_selection", node_tool_selection)
    builder.add_node("hitl_checkpoint", node_hitl_checkpoint)
    builder.add_node("tool_execution", node_tool_execution)
    builder.add_node("supervisor", node_supervisor)
    builder.add_node("context_assembly", node_context_assembly)
    builder.add_node("generation", node_generation)
    builder.add_node("output_review", node_output_review)

    builder.add_edge(START, "tool_selection")
    builder.add_edge("tool_selection", "hitl_checkpoint")
    builder.add_conditional_edges(
        "hitl_checkpoint",
        _route_after_hitl,
        {"tool_execution": "tool_execution", "__end__": END},
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

    interrupt_nodes = []
    if HITL_ENABLED:
        interrupt_nodes.append("hitl_checkpoint")
    if mode == "human":
        interrupt_nodes.append("output_review")
    if interrupt_nodes:
        compile_kwargs["interrupt_before"] = interrupt_nodes

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
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("code-doc-assistant-dev")

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
        "execution_trace": [],
        "mlflow_run_id": None,
        "total_latency_ms": None,
        **(extra_state or {}),
    }

    with mlflow.start_run() as run:
        mlflow.log_param("query", query)
        mlflow.log_param("repo_path", repo_path)
        mlflow.log_param("inference_backend", INFERENCE_BACKEND)
        mlflow.log_param("model", (
            VLLM_MODEL if INFERENCE_BACKEND == "vllm"
            else LLAMACPP_MODEL if INFERENCE_BACKEND == "llamacpp"
            else OLLAMA_MODEL
        ))
        mlflow.log_param("hitl_enabled", HITL_ENABLED)
        mlflow.log_param("output_review_mode", OUTPUT_REVIEW_MODE)

        initial_state["mlflow_run_id"] = run.info.run_id
        config = {"configurable": {"thread_id": thread_id}}
        final_state = graph.invoke(initial_state, config=config)

        # Log summary metrics
        mlflow.log_metric("total_tool_calls", len(final_state.get("executed_tool_calls", [])))
        mlflow.log_metric("retrieval_attempts", final_state.get("retrieval_attempts", 0))
        mlflow.log_metric("chunks_retrieved", len(final_state.get("retrieved_chunks", [])))
        mlflow.log_metric("supervisor_adjustments", len(final_state.get("supervisor_adjustments", [])))

    return final_state

