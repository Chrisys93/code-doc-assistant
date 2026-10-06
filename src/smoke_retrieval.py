"""
smoke_retrieval.py -- one-command check that the agent graph retrieves real context.

Runs the SAME LangGraph the UI uses, headless, with human review off, against the
repos you pass in. It prints every node's trace line (including the tool plan and
which collections were searched), then PASS/FAIL on whether real chunks reached
the model. It does not depend on Streamlit, the model tier, or the GPU: retrieval
happens before generation (Ollama embeddings + ChromaDB).

Usage (inside the app container, repos as typed in the UI):
  docker exec -w /app/src code-doc-app python smoke_retrieval.py \
      "https://github.com/Chrisys93/code-doc-assistant.git#dev" \
      "https://github.com/Chrisys93/IcarusRepoSEND" \
      --query "Could IcarusSEND integrate the code-doc-assistant embedding model?"

Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import sys
import time

import tracking
from agent_graph import build_graph, make_checkpointer, runtime_info
from repo_index import active_collections


def report_server(backend=None) -> None:
    """Print what is really answering: for llama.cpp, ask the server (not the app's config)."""
    import agent_graph as ag
    b = (backend or ag._default_backend() or "").lower()
    if b != "llamacpp":
        print(f"backend = {b}")
        return
    props = ag._llamacpp_props() or {}
    print(f"llama-server: model={ag._llamacpp_served_model()} | n_ctx={ag._llamacpp_server_ctx()} "
          f"| gguf={props.get('model_path')} | slots={props.get('total_slots')}")


def plan_and_retrieve_only(query: str, cols: list, backend=None, tier=None) -> int:
    """Planner call + tool execution only. Shows exactly what the planner asked for."""
    from agent_graph import node_tool_selection, node_tool_execution
    run_id = tracking.start_query_run(query, runtime_info(backend, tier), tags={"source": "smoke-plan"})
    state = {"query": query, "repo_path": "", "active_collections": cols,
             "hitl_enabled": False, "execution_trace": [], "mlflow_run_id": run_id,
             "active_backend": backend, "active_model_tier": tier}
    t0 = time.time()
    with tracking.activate(run_id):
        sel = node_tool_selection(state)
    print(f"planner took {time.time() - t0:.1f}s")
    print("selection:", sel["execution_trace"][-1]["detail"])
    for tc in sel["proposed_tool_calls"]:
        print("PLAN (normalised):", tc.tool_name, tc.args)
    state.update(sel)
    state["approved_tool_calls"] = sel["proposed_tool_calls"]
    with tracking.activate(run_id):
        ex = node_tool_execution(state)
    state.update(ex)
    tracking.finish_run(run_id, state, outcome="plan_only")
    for tc in ex["executed_tool_calls"]:
        print(f"RAN: {tc.tool_name} success={tc.success} error={tc.error} args={tc.args}")
    chunks = ex["retrieved_chunks"]
    print("trace:", ex["execution_trace"][-1]["detail"])
    print("chunks:", len(chunks))
    for c in chunks:
        print(f"  {c.confidence:.3f}  {c.source_file}")
    ok = bool(chunks) and not all(c.source_file == "codebase" for c in chunks)
    print("\nPASS" if ok else "\nFAIL: no real chunks")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+", help="repo refs exactly as typed in the UI")
    ap.add_argument("--query", default="What does this repository do, and how is it structured?")
    ap.add_argument("--backend", default=None, choices=["ollama", "llamacpp", "vllm"],
                    help="override the inference backend for this run (A/B the same model)")
    ap.add_argument("--tier", default=None, help="override model tier, e.g. heavy")
    ap.add_argument("--no-generate", action="store_true",
                    help="run only tool planning + retrieval (fast; skips generation)")
    args = ap.parse_args()

    cols = active_collections(args.repos)
    print("active_collections =", cols)
    report_server(args.backend)

    if args.no_generate:
        return plan_and_retrieve_only(args.query, cols, args.backend, args.tier)

    graph = build_graph(checkpointer=make_checkpointer())
    cfg = {"configurable": {"thread_id": f"smoke-{int(time.time())}"}}
    init = {
        "query": args.query, "repo_path": "", "active_collections": cols,
        "hitl_enabled": False, "output_review_mode": "off",
        "active_backend": args.backend, "active_model_tier": args.tier,
        "proposed_tool_calls": [], "hitl_checkpoint": None, "approved_tool_calls": [],
        "executed_tool_calls": [], "retrieved_chunks": [], "confidence_scores": [],
        "retrieval_attempts": 0, "max_retrieval_attempts": 3,
        "supervisor_adjustments": [], "proceed_to_generation": False,
        "final_context": "", "response": "", "source_attribution": [],
        "post_generation_feedback": None, "generation_attempts": 0,
        "execution_trace": [], "mlflow_run_id": None, "total_latency_ms": None,
    }
    init["mlflow_run_id"] = tracking.start_query_run(
        args.query, runtime_info(args.backend, args.tier),
        settings={"repos": args.repos, "collections": cols, "hitl_enabled": False, "output_review_mode": "off"},
        tags={"source": "smoke", "thread_id": cfg["configurable"]["thread_id"]})

    t0 = time.time()
    try:
        r = graph.invoke(init, config=cfg)
    except Exception as e:
        tracking.finish_run(init["mlflow_run_id"], None, error=str(e))
        raise
    tracking.finish_run(init["mlflow_run_id"], r)
    print(f"\n--- trace ({time.time() - t0:.1f}s total) ---")
    for step in r.get("execution_trace", []):
        print(f"[{step.get('status', '?')}] {step['node']}: {step.get('detail', '')}")

    chunks = r.get("retrieved_chunks", [])
    sources = r.get("source_attribution", [])
    ctx = r.get("final_context", "")
    print("\nretrieved_chunks:", len(chunks))
    print("sources:", sources)
    print("context chars:", len(ctx))
    print("response:", (r.get("response") or "")[:400].replace("\n", " "))

    problems = []
    if not chunks:
        problems.append("no chunks retrieved")
    if ctx.strip() in ("", "[No relevant context found]"):
        problems.append("model was given no context")
    if sources and all(s == "codebase" for s in sources):
        problems.append("only the placeholder source 'codebase' (not a real file)")
    if problems:
        print("\nFAIL:", "; ".join(problems))
        return 1
    print("\nPASS: real chunks reached the model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
