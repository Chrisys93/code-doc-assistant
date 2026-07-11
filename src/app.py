"""
app.py — Streamlit UI for the dev branch agent pipeline.

Layout:
  Tab 1 "Chat"     — conversation interface with HITL-1/2 widgets
  Tab 2 "Pipeline" — full-width graph diagram + live execution trace from last query
  Tab 3 "Session"  — accumulated preference profile, supervisor audit trail, MLflow link
"""

from __future__ import annotations
import json, os, time
from typing import Any
import streamlit as st
from langgraph.types import Command

st.set_page_config(
    page_title="Code Doc Assistant — dev",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_graph():
    from agent_graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver
    # Debug-log #15: MemorySaver's default jsonplus serde doesn't know our
    # custom dataclasses (ToolCall, HITLCheckpoint, SupervisorAdjustment),
    # which logs "Deserializing unregistered type ... will be blocked in a
    # future version" on every resume. pickle_fallback lets it (de)serialize
    # them safely now, before that becomes a hard error.
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        checkpointer = MemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
    except TypeError:
        # Pinned LangGraph version doesn't accept pickle_fallback — fall back
        # to the plain saver rather than breaking the build.
        checkpointer = MemorySaver()
    return build_graph(checkpointer=checkpointer)

def _get_mermaid() -> str:
    from agent_graph import get_graph_mermaid
    return get_graph_mermaid()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init():
    defaults = {
        "messages": [],
        "thread_id": f"thread-{int(time.time())}",
        "pending_hitl": None,
        "awaiting_hitl": False,
        "pending_output_review": None,
        "awaiting_output_review": False,
        "last_run_id": None,
        "last_trace": [],          # execution_trace from last completed query
        "last_adjustments": [],
        "session_preferences": None,
        "agent_state": None,
        "output_review_mode": "human",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _finalise(state: dict) -> None:
    rt = state.get("response", "[No response]")
    srcs = state.get("source_attribution", [])
    if srcs:
        rt += "\n\n---\n**Sources:** " + ", ".join(f"`{s}`" for s in srcs)
    st.session_state.messages.append({"role": "assistant", "content": rt})
    st.session_state.last_run_id = state.get("mlflow_run_id")
    st.session_state.last_adjustments = state.get("supervisor_adjustments", [])
    st.session_state.last_trace = state.get("execution_trace", [])
    if state.get("session_preferences"):
        st.session_state.session_preferences = state["session_preferences"]


def _mermaid_html(mermaid_src: str, trace: list[dict] | None = None) -> str:
    """
    Build an HTML snippet rendering the Mermaid graph.
    If trace is provided, inject JS to colour executed nodes:
      green = ok, orange = retry, red = rejected.
    """
    status_colours = {"ok": "#34c759", "retry": "#ff9500", "rejected": "#ff3b30"}
    highlight_js = ""
    if trace:
        visited: dict[str, str] = {}
        for step in trace:
            node = step.get("node", "")
            status = step.get("status", "ok")
            # last status for this node wins
            visited[node] = status_colours.get(status, "#34c759")
        # Build JS that finds SVG node labels and overrides fill
        assignments = "\n".join(
            f'  recolour("{node}", "{colour}");'
            for node, colour in visited.items()
        )
        highlight_js = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
  function recolour(label, colour) {{
    document.querySelectorAll('.node').forEach(function(el) {{
      if (el.textContent.trim().startsWith(label)) {{
        el.querySelector('rect,circle,polygon')?.setAttribute('fill', colour);
      }}
    }});
  }}
  setTimeout(function() {{
{assignments}
  }}, 600);
}});
</script>"""

    return f"""
<div class="mermaid" id="pipeline-graph">
{mermaid_src}
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:'neutral', securityLevel:'loose'}});</script>
{highlight_js}
"""


def _render_hitl1(proposed: list[dict]) -> dict | None:
    st.subheader("🔍 Review proposed tool plan")
    st.caption("Approve, modify args, or reject — no tools have run yet.")
    modified = []
    for i, tc in enumerate(proposed):
        with st.expander(f"Tool {i+1}: `{tc['tool_name']}`", expanded=True):
            st.json(tc["args"])
            raw = st.text_area("Modify args (JSON)", value=json.dumps(tc["args"], indent=2),
                               key=f"h1_{i}", height=100)
            try:
                args = json.loads(raw)
            except Exception:
                args = tc["args"]
            modified.append({"tool_name": tc["tool_name"], "args": args})
    feedback = st.text_input("Optional feedback", key="h1_fb")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ Approve", type="primary", key="h1_approve"):
            return {"decision": "approved", "tool_calls": proposed, "feedback": feedback}
    with c2:
        if st.button("✏️ Use modified", key="h1_modify"):
            return {"decision": "modified", "tool_calls": modified, "feedback": feedback}
    with c3:
        if st.button("❌ Reject", key="h1_reject"):
            return {"decision": "rejected", "tool_calls": [], "feedback": feedback}
    return None


def _render_hitl2(response: str, sources: list, attempt: int) -> dict | None:
    st.subheader(f"📋 Review output (attempt {attempt})")
    with st.expander("Generated response", expanded=True):
        st.markdown(response)
        if sources:
            st.caption("Sources: " + ", ".join(f"`{s}`" for s in sources))
    score = st.slider("Satisfaction (1–5)", 1, 5, 4, key=f"h2_score_{attempt}")
    decision = st.radio("Action", ["✅ Accept", "🔄 Regenerate", "➕ Add context"],
                        key=f"h2_dec_{attempt}")
    fmt, ctx, files = "", "", []
    if "Regenerate" in decision:
        fmt = st.text_input("Format/style notes", key=f"h2_fmt_{attempt}")
        ctx = st.text_input("Context notes", key=f"h2_ctx_{attempt}")
    elif "Add context" in decision:
        raw_files = st.text_area("Files to fetch (one per line, repo-relative)",
                                 key=f"h2_files_{attempt}")
        files = [f.strip() for f in raw_files.splitlines() if f.strip()]
    if st.button("Submit feedback", type="primary", key=f"h2_submit_{attempt}"):
        d = ("accept" if "Accept" in decision
             else "regenerate" if "Regenerate" in decision
             else "add_context")
        return {"decision": d, "satisfaction_score": score,
                "format_notes": fmt or None, "context_notes": ctx or None,
                "additional_files": files}
    return None

# ---------------------------------------------------------------------------
# Config (top bar instead of sidebar)
# ---------------------------------------------------------------------------
with st.expander("⚙️ Configuration", expanded=False):
    cc = st.columns(5)
    with cc[0]:
        repos_raw = st.text_area("Repos (one per line)",
                                 value=os.environ.get("REPO_PATH", "/data/repos/myrepo"),
                                 help="Git URLs or local paths. Each is indexed into its own collection before you query.")
        repos = [r.strip() for r in repos_raw.splitlines() if r.strip()]
        repo_path = repos[0] if repos else ""   # back-compat: existing code still reads repo_path
        st.session_state.repos = repos
    with cc[1]:
        hitl1_on = st.toggle("Tool plan HITL", value=True)
    with cc[2]:
        review_mode = st.selectbox("Output review mode",
                                   ["human", "supervisor", "self", "off"],
                                   key="output_review_mode")
    with cc[3]:
        max_ret = st.slider("Max retrieval retries", 1, 5, 3)
        conf_thr = st.slider("Confidence threshold", 0.1, 0.9, 0.45, 0.05)
    with cc[4]:
        max_gen = st.slider("Max generation attempts", 1, 3, 3)
        gate_thr = st.slider("Quality gate (supervisor)", 0.0, 10.0, 6.0, 0.5)

    # NOTE: os.environ["HITL_ENABLED"] used to be set here, but HITL_ENABLED is
    # read once at agent_graph's module import — setting it per-rerun had no
    # effect after the first request. hitl1_on is now passed straight into the
    # graph's init state below, which node_hitl_checkpoint reads live.
    os.environ["OUTPUT_REVIEW_MODE"] = review_mode
    os.environ["MAX_RETRIEVAL_ATTEMPTS"] = str(max_ret)
    os.environ["MAX_GENERATION_ATTEMPTS"] = str(max_gen)
    os.environ["CONFIDENCE_THRESHOLD"] = str(conf_thr)
    os.environ["QUALITY_GATE_THRESHOLD"] = str(gate_thr)

    if st.button("🗑️ Clear conversation"):
        for k, v in {"messages": [], "pending_hitl": None, "awaiting_hitl": False,
                     "pending_output_review": None, "awaiting_output_review": False,
                     "last_run_id": None, "last_trace": [], "last_adjustments": [],
                     "session_preferences": None, "agent_state": None}.items():
            st.session_state[k] = v
        st.session_state.thread_id = f"thread-{int(time.time())}"
        st.rerun()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_pipeline, tab_session = st.tabs(["💬 Chat", "🔗 Pipeline", "🧠 Session"])

# ===========================================================================
# Tab 2: Pipeline — graph + execution trace
# ===========================================================================
with tab_pipeline:
    try:
        mermaid_src = _get_mermaid()
        trace = st.session_state.last_trace
        html = _mermaid_html(mermaid_src, trace if trace else None)
        # st.components.v1.html is deprecated (removed after 2026-06-01); st.iframe
        # takes a src URL rather than raw HTML, so render via a data: URI.
        import base64
        b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
        st.iframe(src=f"data:text/html;base64,{b64}", height=520, scrolling=False)

        if trace:
            st.subheader("Last execution trace")
            st.caption(f"Thread: `{st.session_state.thread_id}`")
            for step in trace:
                status = step.get("status", "ok")
                icon = "✅" if status == "ok" else "🔄" if status == "retry" else "❌"
                st.markdown(f"{icon} **{step['node']}** — {step.get('detail', '')}")
        else:
            st.info("Run a query to see the execution trace here.")

        st.caption(
            "Node colours after a query: 🟢 completed · 🟠 retried · 🔴 rejected\n\n"
            f"Current mode: HITL-1={'on' if hitl1_on else 'off'} · "
            f"Output review=`{review_mode}`"
        )
    except Exception as e:
        st.warning(f"Graph render unavailable: {e}")

# ===========================================================================
# Tab 3: Session
# ===========================================================================
with tab_session:
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if st.session_state.last_run_id:
        st.markdown(f"📊 [Last MLflow run]({mlflow_uri}/#/experiments/1/runs/{st.session_state.last_run_id})")
    else:
        st.caption(f"MLflow: {mlflow_uri}")

    prefs = st.session_state.session_preferences
    if prefs:
        st.subheader("Session preferences (accumulated)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg satisfaction", f"{prefs.avg_satisfaction:.1f}/5")
        c2.metric("Feedback count", prefs.feedback_count)
        c3.metric("Prioritised files", len(prefs.prioritised_files))
        if prefs.preferred_format:
            st.markdown(f"**Format:** `{prefs.preferred_format}`")
        if prefs.preferred_verbosity:
            st.markdown(f"**Verbosity:** `{prefs.preferred_verbosity}`")
        if prefs.prioritised_files:
            st.markdown("**Prioritised files:** " + ", ".join(f"`{f}`" for f in prefs.prioritised_files))
    else:
        st.info("Session preferences will appear here after your first reviewed response.")

    if st.session_state.last_adjustments:
        st.subheader("Supervisor adjustments (last query)")
        for adj in st.session_state.last_adjustments:
            st.markdown(f"- **{adj.action}**: {adj.reason}")

# ===========================================================================
# Tab 1: Chat
# ===========================================================================
with tab_chat:
    st.caption(f"Thread: `{st.session_state.thread_id}` · Mode: `{review_mode}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- HITL-1 pending ---
    if st.session_state.awaiting_hitl and st.session_state.pending_hitl:
        with st.chat_message("assistant"):
            resp = _render_hitl1(st.session_state.pending_hitl)
        if resp is not None:
            graph = _get_graph()
            cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Running approved tools..."):
                try:
                    graph.invoke(Command(resume=resp), config=cfg)
                    st.session_state.awaiting_hitl = False
                    st.session_state.pending_hitl = None
                    snap = graph.get_state(cfg)
                    if snap.next and "output_review" in snap.next:
                        v = snap.values
                        st.session_state.awaiting_output_review = True
                        st.session_state.pending_output_review = {
                            "response": v.get("response", ""),
                            "source_attribution": v.get("source_attribution", []),
                            "generation_attempts": v.get("generation_attempts", 1),
                        }
                    elif not snap.next:
                        _finalise(snap.values)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.awaiting_hitl = False
            st.rerun()

    # --- HITL-2 pending ---
    elif st.session_state.awaiting_output_review and st.session_state.pending_output_review:
        rev = st.session_state.pending_output_review
        with st.chat_message("assistant"):
            resp = _render_hitl2(
                response=rev["response"],
                sources=rev.get("source_attribution", []),
                attempt=rev.get("generation_attempts", 1),
            )
        if resp is not None:
            graph = _get_graph()
            cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Processing feedback..."):
                try:
                    graph.invoke(Command(resume=resp), config=cfg)
                    st.session_state.awaiting_output_review = False
                    st.session_state.pending_output_review = None
                    snap = graph.get_state(cfg)
                    if snap.next and "output_review" in snap.next:
                        v = snap.values
                        st.session_state.awaiting_output_review = True
                        st.session_state.pending_output_review = {
                            "response": v.get("response", ""),
                            "source_attribution": v.get("source_attribution", []),
                            "generation_attempts": v.get("generation_attempts", 1),
                        }
                    elif snap.next and "hitl_checkpoint" in snap.next:
                        proposed = snap.values.get("proposed_tool_calls", [])
                        st.session_state.awaiting_hitl = True
                        st.session_state.pending_hitl = [
                            {"tool_name": tc.tool_name, "args": tc.args} for tc in proposed
                        ]
                    else:
                        _finalise(snap.values if not isinstance(snap.values, dict)
                                  else snap.values)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.awaiting_output_review = False
            st.rerun()

    # --- New query ---
    elif not st.session_state.awaiting_hitl and not st.session_state.awaiting_output_review:
        if prompt := st.chat_input("Ask about the codebase..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            from repo_index import ensure_indexed, active_collections
            _chroma = os.environ.get("CHROMA_HOST", "http://chromadb:8000")
            with st.spinner("Ensuring repos are indexed…"):
                status = ensure_indexed(st.session_state.get("repos", []), _chroma)
            st.session_state.active_collections = active_collections(st.session_state.get("repos", []))
            for ref, s in status.items():
                st.caption(f"📂 {ref.split('/')[-1]} → {s['status']} ({s['docs']} docs)")

            graph = _get_graph()
            cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
            extra: dict[str, Any] = {}
            if st.session_state.session_preferences:
                extra["session_preferences"] = st.session_state.session_preferences

            init: dict[str, Any] = {
                "query": prompt, "repo_path": repo_path,
                "hitl_enabled": hitl1_on, "output_review_mode": review_mode,
                "active_collections": st.session_state.get("active_collections", []),
                "proposed_tool_calls": [], "hitl_checkpoint": None,
                "approved_tool_calls": [], "executed_tool_calls": [],
                "retrieved_chunks": [], "confidence_scores": [],
                "retrieval_attempts": 0, "max_retrieval_attempts": max_ret,
                "supervisor_adjustments": [], "proceed_to_generation": False,
                "final_context": "", "response": "", "source_attribution": [],
                "post_generation_feedback": None, "generation_attempts": 0,
                "execution_trace": [], "mlflow_run_id": None, "total_latency_ms": None,
                **extra,
            }

            import logging
            logging.getLogger(__name__).warning(
                "DIAG init: repos=%r active_collections=%r hitl_enabled=%r output_review_mode=%r",
                st.session_state.get("repos"), init["active_collections"],
                init["hitl_enabled"], init["output_review_mode"],
            )

            with st.spinner("Agent selecting tools..."):
                try:
                    graph.invoke(init, config=cfg)
                    snap = graph.get_state(cfg)
                    if snap.next and "hitl_checkpoint" in snap.next:
                        proposed = snap.values.get("proposed_tool_calls", [])
                        st.session_state.awaiting_hitl = True
                        st.session_state.pending_hitl = [
                            {"tool_name": tc.tool_name, "args": tc.args} for tc in proposed
                        ]
                    elif snap.next and "output_review" in snap.next:
                        v = snap.values
                        st.session_state.awaiting_output_review = True
                        st.session_state.pending_output_review = {
                            "response": v.get("response", ""),
                            "source_attribution": v.get("source_attribution", []),
                            "generation_attempts": v.get("generation_attempts", 1),
                        }
                    elif not snap.next:
                        _finalise(snap.values)
                except Exception as e:
                    st.error(f"Agent error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
            st.rerun()
