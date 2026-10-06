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
    from agent_graph import build_graph, make_checkpointer
    # make_checkpointer() registers our own state classes with LangGraph's serializer
    # (no "Deserializing unregistered type" warnings, and no break when LangGraph starts blocking them).
    return build_graph(checkpointer=make_checkpointer())

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
        "active_model_tier": None,   # None = use container env default (MODEL_TIER)
        "active_backend": None,      # None = use container env default (INFERENCE_BACKEND)
        "last_active_model": None,   # resolved model tag actually used last turn (display only)
        "last_error": None,          # persisted across the st.rerun() after a failed graph.invoke()
        "streaming_active": False,   # a run is advancing one node per rerun (see _process_one_stream_step)
        "pending_init": None,        # full init dict for the FIRST step of a new query; None afterward
        "pending_resume": None,      # HITL decision payload for the FIRST step after a decision; None afterward
        "running_node": None,        # node executing during this rerun (drawn highlighted on the Pipeline tab)
        "current_run_id": None,      # MLflow run of the question in flight (one run per question)
        "replan_round": 0,           # how many times the tool plan was sent back for this question
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

def _end_run(state: dict | None = None, error: str | None = None, outcome: str | None = None) -> None:
    """Close the MLflow run of the question in flight (answered / rejected / error / abandoned)."""
    rid = st.session_state.get("current_run_id")
    if rid:
        import tracking
        tracking.finish_run(rid, state, outcome=outcome, error=error)
        st.session_state.current_run_id = None


def _finalise(state: dict) -> None:
    _end_run(state)
    rt = state.get("response") or ""
    if not rt.strip():
        # The graph ended without generating: a rejected tool plan is the normal way to get here.
        cp = state.get("hitl_checkpoint")
        if getattr(cp, "decision", None) == "rejected":
            rt = "_Tool plan rejected — nothing was run and no answer was generated._"
        else:
            rt = "_[No response was generated.]_"
    srcs = state.get("source_attribution", [])
    if srcs:
        rt += "\n\n---\n**Sources:** " + ", ".join(f"`{s}`" for s in srcs)
    st.session_state.messages.append({"role": "assistant", "content": rt,
                                      "trace": list(state.get("execution_trace", []))})
    st.session_state.last_run_id = state.get("mlflow_run_id")
    st.session_state.last_adjustments = state.get("supervisor_adjustments", [])
    st.session_state.last_trace = state.get("execution_trace", [])
    if state.get("session_preferences"):
        st.session_state.session_preferences = state["session_preferences"]
    if state.get("active_model"):
        import deployment
        backend = state.get("active_backend") or deployment.backend()
        st.session_state.last_active_model = f"{state['active_model']} ({backend})"


def _process_one_stream_step() -> None:
    """
    Advance the running graph by exactly ONE node, then the caller st.reruns()
    so every tab (Pipeline included) re-renders with the freshly-updated
    st.session_state.last_trace on the next pass -- this is what gives live,
    node-by-node highlighting instead of only a summary shown after the whole
    run finishes.

    Mechanism: LangGraph checkpoints state after every node. Calling
    graph.stream(None, config=cfg) creates a brand-new generator each time,
    but because of the checkpoint it resumes from exactly where the previous
    call left off rather than replaying completed nodes -- verified directly
    against a toy graph (including one with an interrupt()) before wiring
    this in: repeatedly creating a fresh .stream() call and consuming only
    the first yielded item, across many separate calls, produces exactly one
    step of forward progress each time, with no replay and no corruption.

    pending_init / pending_resume supply the input ONLY for the first step of
    a run; every step after that passes None to correctly resume instead of
    restarting. This mirrors exactly what the old single graph.invoke() call
    did, just spread across multiple reruns so each step becomes visible.
    """
    graph = _get_graph()
    cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        if st.session_state.pending_init is not None:
            stream_input = st.session_state.pending_init
            st.session_state.pending_init = None
        elif st.session_state.pending_resume is not None:
            stream_input = Command(resume=st.session_state.pending_resume)
            st.session_state.pending_resume = None
        else:
            stream_input = None

        step = next(iter(graph.stream(stream_input, config=cfg, stream_mode="updates")), None)
        if step:
            node_update = next(iter(step.values()), {})
            if "execution_trace" in node_update:
                st.session_state.last_trace = node_update["execution_trace"]

        snap = graph.get_state(cfg)
        # snap.next only names the node that runs NEXT -- it is true after
        # tool_selection / generation even when the human-review toggle is off.
        # A review is really pending only if that node raised interrupt().
        paused = any(getattr(t, "interrupts", None) for t in (snap.tasks or ()))
        if paused and snap.next and "hitl_checkpoint" in snap.next:
            proposed = snap.values.get("proposed_tool_calls", [])
            st.session_state.awaiting_hitl = True
            st.session_state.pending_hitl = [
                {"tool_name": tc.tool_name, "args": tc.args} for tc in proposed
            ]
            st.session_state.streaming_active = False
            st.session_state.running_node = None      # waiting for the human, nothing is executing
        elif paused and snap.next and "output_review" in snap.next:
            v = snap.values
            st.session_state.awaiting_output_review = True
            st.session_state.pending_output_review = {
                "response": v.get("response", ""),
                "source_attribution": v.get("source_attribution", []),
                "generation_attempts": v.get("generation_attempts", 1),
            }
            st.session_state.streaming_active = False
            st.session_state.running_node = None
        elif not snap.next:
            _finalise(snap.values)
            st.session_state.streaming_active = False
            st.session_state.running_node = None
        else:
            # More steps remain: the node that runs NEXT is the one the following rerun
            # executes, so it is the one to draw as "running" at the top of that rerun.
            st.session_state.running_node = snap.next[0]
    except Exception as e:
        import traceback
        st.session_state.last_error = {"message": str(e), "traceback": traceback.format_exc()}
        st.session_state.streaming_active = False
        st.session_state.running_node = None
        _end_run(error=str(e))


# NOTE: the call that actually runs _process_one_stream_step() lives at the very END of
# this script. It used to sit here, before the tabs were drawn, and finished with
# st.rerun() -- so every intermediate rerun stopped before drawing anything and the
# Pipeline tab only repainted once the whole run was over. At the end of the script the
# tabs are drawn first (with the trace and running node from the previous step) and only
# then does the next node execute.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _export_conversation_markdown() -> str:
    """
    Render the current conversation as a self-contained Markdown transcript —
    thread id, repos in scope, model used, and every message in order.
    This is the only way to actually keep a conversation right now: nothing
    is persisted anywhere else in a directly usable form (MLflow logs each
    turn's response as a separate artifact keyed by run id, which is
    recoverable but not a "save this conversation" feature).
    """
    lines = [
        f"# Conversation — {st.session_state.thread_id}",
        "",
        f"- **Repos:** {', '.join(st.session_state.get('repos', [])) or '(none)'}",
        f"- **Last resolved model:** {st.session_state.get('last_active_model') or '(none yet)'}",
        f"- **Exported:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]
    for msg in st.session_state.messages:
        speaker = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        lines.append(f"**{speaker}:**\n\n{msg['content']}\n")
    return "\n".join(lines)


def _mermaid_svg_url(mermaid_src: str, trace: list[dict] | None = None,
                     running: str | None = None) -> str:
    """
    Build a mermaid.ink SVG URL for the graph -- a static image, not a live
    JS-rendered diagram. Two consecutive approaches using in-browser mermaid.js
    (st.iframe with a data: URI, then st.components.v1.html, which turned out
    to be past its removal deadline and non-functional) both failed to
    reliably render; this sidesteps browser script execution entirely by
    rendering server-side (via mermaid.ink) and embedding as a plain image.

    Trace-based node colouring is done via Mermaid's native `style <id>
    fill:<colour>` syntax appended directly to the diagram source -- a
    standard Mermaid feature, not a post-render JS injection -- so it
    survives static rendering. green = ok, orange = retry, red = rejected.

    mermaid.ink's URL protocol: base64url(zlib_deflate(json({"code": ...,
    "mermaid": {...}}))), prefixed "pako:". Verified against mermaid.ink's
    own docs and the mermaid-live-editor project's reference implementation.
    """
    import base64
    import json
    import zlib

    status_colours = {"ok": "#34c759", "retry": "#ff9500", "warn": "#ff9500", "rejected": "#ff3b30"}
    styled_src = mermaid_src
    if trace:
        visited: dict[str, str] = {}
        for step in trace:
            node = step.get("node", "")
            status = step.get("status", "ok")
            visited[node] = status_colours.get(status, "#34c759")  # last status wins
        style_lines = "\n".join(f"style {node} fill:{colour}" for node, colour in visited.items())
        styled_src = f"{mermaid_src}\n{style_lines}"
    if running:
        # The node executing right now: yellow with a dashed border. Applied last so it also
        # overrides the colour of a node that is being re-run (retry / regenerate).
        styled_src = (f"{styled_src}\nstyle {running} fill:#ffd60a,stroke:#333,"
                      f"stroke-width:3px,stroke-dasharray: 5 5")

    # htmlLabels=False draws node labels as SVG <text> instead of fixed-width HTML boxes: mermaid.ink measures
    # the label in ITS font, the browser then draws it in another, and the HTML box clipped the last letters
    # ("tool_selectior"). Extra padding gives the remaining font difference some slack.
    payload = json.dumps({"code": styled_src,
                          "mermaid": {"theme": "neutral",
                                      "flowchart": {"htmlLabels": False, "padding": 18}}})
    compressor = zlib.compressobj(9, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY)
    deflated = compressor.compress(payload.encode("utf-8")) + compressor.flush()
    encoded = base64.b64encode(deflated).decode("ascii").replace("+", "-").replace("/", "_").rstrip("=")
    return f"https://mermaid.ink/svg/pako:{encoded}"



def _render_hitl1(proposed: list[dict], round_no: int = 0) -> dict | None:
    # Widget keys carry the re-plan round, so a re-proposed plan starts with fresh editors instead of
    # inheriting the text of the plan that was just rejected. Round 0 keeps the plain keys.
    sfx = f"_r{round_no}" if round_no else ""
    st.subheader("🔍 Review proposed tool plan" + (f" (re-plan {round_no})" if round_no else ""))
    st.caption("Approve, modify args, or reject — no tools have run yet.")
    modified = []
    for i, tc in enumerate(proposed):
        with st.expander(f"Tool {i+1}: `{tc['tool_name']}`", expanded=True):
            st.json(tc["args"])
            raw = st.text_area("Modify args (JSON)", value=json.dumps(tc["args"], indent=2),
                               key=f"h1_{i}{sfx}", height=100)
            try:
                args = json.loads(raw)
            except Exception:
                args = tc["args"]
            modified.append({"tool_name": tc["tool_name"], "args": args})
    feedback = st.text_input("Optional feedback", key=f"h1_fb{sfx}")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ Approve", type="primary", key=f"h1_approve{sfx}"):
            return {"decision": "approved", "tool_calls": proposed, "feedback": feedback}
    with c2:
        if st.button("✏️ Use modified", key=f"h1_modify{sfx}"):
            return {"decision": "modified", "tool_calls": modified, "feedback": feedback}
    with c3:
        if st.button("❌ Reject", key=f"h1_reject{sfx}"):
            st.session_state[f"h1_rejecting{sfx}"] = True

    # Rejecting is not the end of the road by default: ask what was expected and let the human choose.
    if st.session_state.get(f"h1_rejecting{sfx}"):
        st.warning("Plan rejected. What should happen next?")
        expect = st.text_area("What did you expect instead? (e.g. which files or kind of search it should use)",
                              key=f"h1_expect{sfx}", height=80)
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🔁 Re-plan with my feedback", type="primary", key=f"h1_replan_fb{sfx}",
                         disabled=not expect.strip()):
                return {"decision": "replan", "tool_calls": [], "feedback": expect.strip()}
        with b2:
            if st.button("🔁 Re-plan (no feedback)", key=f"h1_replan{sfx}"):
                return {"decision": "replan", "tool_calls": [], "feedback": ""}
        with b3:
            if st.button("⛔ End here", key=f"h1_end{sfx}"):
                return {"decision": "rejected", "tool_calls": [], "feedback": expect.strip()}
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
                                 help="Git URLs or local paths. Each is indexed into its own collection "
                                      "before you query. To pin a branch (default is the repo's default "
                                      "branch, usually master/main), append #branch — e.g. "
                                      "'https://github.com/owner/repo.git#dev'. A GitHub web URL "
                                      "like '.../tree/dev' will NOT work as a clone target.")
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

    st.divider()
    st.caption("🔀 Model — hot-swappable mid-conversation. Does not affect repo indexing, "
               "the per-repo ChromaDB collections, or Kuzu — only the tool-selection/generation LLM.")
    import deployment
    _dep = deployment.resolve()
    mc = st.columns(3)
    _tier_options = ["(container default)", "heavy", "full", "balanced", "lightweight", "minimal"]
    _backend_options = ["(container default)", "ollama", "vllm", "llamacpp"]
    with mc[0]:
        picked_tier = st.selectbox(
            "Model tier", _tier_options,
            format_func=lambda o: f"(container default: {_dep['tier']})" if o == "(container default)" else o,
            index=_tier_options.index(st.session_state.active_model_tier or "(container default)")
                  if st.session_state.active_model_tier in _tier_options else 0,
            help="Capability selector — heavy/full/balanced/lightweight/minimal. "
                 f"Container default = what this deployment runs ({_dep['source']}).",
        )
        st.session_state.active_model_tier = None if picked_tier == "(container default)" else picked_tier
    with mc[1]:
        picked_backend = st.selectbox(
            "Inference backend", _backend_options,
            format_func=lambda o: f"(container default: {_dep['backend']})" if o == "(container default)" else o,
            index=_backend_options.index(st.session_state.active_backend or "(container default)")
                  if st.session_state.active_backend in _backend_options else 0,
            help="ollama (default, model mgmt included) · vllm (GPU) · llamacpp (CPU-native GGUF). "
                 "The target backend's server must already be running — switching here only "
                 "changes which one this conversation talks to.",
        )
        st.session_state.active_backend = None if picked_backend == "(container default)" else picked_backend
    with mc[2]:
        st.metric("Last resolved model", st.session_state.last_active_model or "—")

    if st.session_state.active_model_tier or st.session_state.active_backend:
        backend_note = ""
        if st.session_state.active_backend in ("vllm", "llamacpp"):
            backend_note = (
                " ⚠️ vLLM/llama-server run ONE fixed model per container — switching "
                "here only works if that server is already running the resolved model; "
                "it will NOT auto-restart with a different one."
            )
        st.info(
            f"🔀 Overriding container defaults for this thread: "
            f"tier=`{st.session_state.active_model_tier or 'default'}`, "
            f"backend=`{st.session_state.active_backend or 'default'}`. "
            f"Repo indexing and retrieval are unaffected by this change. "
            f"First use of a new Ollama tier/model may take a few minutes "
            f"while it's pulled.{backend_note}",
            icon="ℹ️",
        )

    btn_cols = st.columns([1, 1, 5])
    with btn_cols[0]:
        if st.button("🗑️ Clear conversation"):
            _end_run(outcome="abandoned")
            for k, v in {"messages": [], "pending_hitl": None, "awaiting_hitl": False,
                         "pending_output_review": None, "awaiting_output_review": False,
                         "last_run_id": None, "last_trace": [], "last_adjustments": [],
                         "session_preferences": None, "agent_state": None,
                         "last_error": None, "streaming_active": False, "running_node": None,
                         "pending_init": None, "pending_resume": None, "replan_round": 0}.items():
                st.session_state[k] = v
            st.session_state.thread_id = f"thread-{int(time.time())}"
            st.rerun()
    with btn_cols[1]:
        st.download_button(
            "💾 Save conversation",
            data=_export_conversation_markdown(),
            file_name=f"{st.session_state.thread_id}.md",
            mime="text/markdown",
            disabled=not st.session_state.messages,
            help="Downloads the full transcript as Markdown. This is currently "
                 "the only way to keep a conversation — nothing is auto-persisted "
                 "beyond what's needed to resume the live session.",
        )

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
        # Third attempt at rendering this diagram: st.iframe(data:...) and
        # st.components.v1.html both failed to actually execute the mermaid.js
        # CDN script in the browser (the latter turned out to be past its
        # removal deadline and silently non-functional despite still being
        # importable). Rendering server-side as a static SVG via mermaid.ink
        # sidesteps browser script execution entirely.
        running_node = st.session_state.running_node if st.session_state.streaming_active else None
        svg_url = _mermaid_svg_url(mermaid_src, trace if trace else None, running=running_node)
        st.image(svg_url, width=650)
        if st.session_state.streaming_active:
            st.caption(f"⏳ Running — now executing `{running_node or '…'}`. "
                       "This diagram updates one node at a time as the pipeline executes "
                       "(green = done, yellow dashed = running).")

        if trace:
            st.subheader("Last execution trace")
            st.caption(f"Thread: `{st.session_state.thread_id}`")
            for step in trace:
                status = step.get("status", "ok")
                icon = "✅" if status == "ok" else "🔄" if status == "retry" else "⚠️" if status == "warn" else "❌"
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
    mlflow_uri = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000")
    import tracking as _tracking
    _ok, _why = _tracking.health()
    (st.success if _ok else st.error)(("✅ " if _ok else "⚠️ ") + _why)
    st.markdown(f"📊 [Open the MLflow UI]({mlflow_uri}) — your runs are under **Model training → Runs** "
                "(the GenAI tab only lists traces)")
    if st.session_state.last_run_id:
        import tracking
        st.markdown(f"↳ [This question's run]({tracking.run_url(st.session_state.last_run_id)})")

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

    if st.session_state.last_error:
        with st.expander("⚠️ Last query failed — click for details", expanded=True):
            st.error(st.session_state.last_error["message"])
            st.code(st.session_state.last_error["traceback"])
            if st.button("Dismiss"):
                st.session_state.last_error = None
                st.rerun()

    _ICON = {"ok": "✅", "retry": "🔄", "warn": "⚠️"}
    for msg in st.session_state.messages:
        if msg.get("kind") == "index":          # repo indexing outcome: stays visible after the answer
            st.caption(msg["content"])
            continue
        with st.chat_message(msg["role"]):
            if msg.get("kind") == "error":
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("trace"):
                warns = sum(1 for t in msg["trace"] if t.get("status") == "warn")
                with st.expander(f"Pipeline trace ({len(msg['trace'])} steps"
                                 + (f", ⚠️ {warns} warning(s)" if warns else "") + ")"):
                    for t in msg["trace"]:
                        st.markdown(f"{_ICON.get(t.get('status'), '❌')} **{t.get('node')}** — {t.get('detail', '')}")

    if st.session_state.streaming_active:
        st.info(f"⏳ Agent running — now executing `{st.session_state.running_node or '…'}`. "
                "Live progress is on the 🔗 Pipeline tab.")

    # --- HITL-1 pending ---
    if st.session_state.awaiting_hitl and st.session_state.pending_hitl:
        with st.chat_message("assistant"):
            resp = _render_hitl1(st.session_state.pending_hitl, st.session_state.replan_round)
        if resp is not None:
            graph = _get_graph()
            cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Running approved tools..."):
                try:
                    if resp.get("decision") == "replan":
                        st.session_state.replan_round += 1
                    st.session_state.pending_resume = resp
                    st.session_state.pending_init = None
                    st.session_state.awaiting_hitl = False
                    st.session_state.pending_hitl = None
                    st.session_state.streaming_active = True
                    st.session_state.running_node = "hitl_checkpoint"
                except Exception as e:
                    import traceback
                    st.session_state.last_error = {"message": str(e), "traceback": traceback.format_exc()}
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
                    st.session_state.pending_resume = resp
                    st.session_state.pending_init = None
                    st.session_state.awaiting_output_review = False
                    st.session_state.pending_output_review = None
                    st.session_state.streaming_active = True
                    st.session_state.running_node = "output_review"
                except Exception as e:
                    import traceback
                    st.session_state.last_error = {"message": str(e), "traceback": traceback.format_exc()}
                    st.session_state.awaiting_output_review = False
            st.rerun()

    # --- New query ---
    elif not st.session_state.awaiting_hitl and not st.session_state.awaiting_output_review:
        # Disabled while a run is advancing: the script now draws the whole page before each
        # step, so an enabled box could accept a new prompt in the middle of a run.
        if prompt := st.chat_input("Ask about the codebase...",
                                   disabled=st.session_state.streaming_active):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            from repo_index import ensure_indexed, active_collections
            _chroma = os.environ.get("CHROMA_HOST", "http://chromadb:8000")
            with st.status("Ensuring repos are indexed… (first index of a repo embeds every chunk and can "
                           "take minutes; progress is in the app log)", expanded=True) as _ix:
                status = ensure_indexed(st.session_state.get("repos", []), _chroma)
                _failed = {r: x for r, x in status.items() if x["status"] == "error"}
                for ref, x in status.items():
                    st.write(f"📂 {ref.split('/')[-1]} → {x['status']} ({x['docs']} docs)")
                _ix.update(label="Indexing failed" if _failed else "Repos indexed",
                           state="error" if _failed else "complete", expanded=bool(_failed))
            st.session_state.active_collections = active_collections(st.session_state.get("repos", []))
            for ref, x in status.items():           # persisted as messages: a bare st.caption vanishes on rerun
                icon = "❌" if x["status"] == "error" else "📂"
                st.session_state.messages.append({"role": "assistant", "kind": "index",
                    "content": f"{icon} {ref.split('/')[-1]} → {x['status']} ({x['docs']} docs)"})
            _no_docs = [r for r, x in status.items() if x["docs"] == 0]
            _hard = bool(_failed) or (bool(status) and len(_no_docs) == len(status))

            graph = _get_graph()
            cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
            extra: dict[str, Any] = {}
            if st.session_state.session_preferences:
                extra["session_preferences"] = st.session_state.session_preferences
            if st.session_state.active_model_tier:
                extra["active_model_tier"] = st.session_state.active_model_tier
            if st.session_state.active_backend:
                extra["active_backend"] = st.session_state.active_backend

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
                "active_backend": None, "active_model_tier": None, "active_model": None,
                "execution_trace": [], "mlflow_run_id": None, "total_latency_ms": None,
                **extra,
            }

            import logging
            logging.getLogger(__name__).warning(
                "DIAG init: repos=%r active_collections=%r hitl_enabled=%r output_review_mode=%r",
                st.session_state.get("repos"), init["active_collections"],
                init["hitl_enabled"], init["output_review_mode"],
            )

            # One MLflow run per question: opened here, re-activated around every node by the graph,
            # closed by _finalise()/_end_run() when the question is answered, rejected, or fails.
            import tracking
            from agent_graph import runtime_info
            run_id = tracking.start_query_run(
                prompt,
                runtime_info(extra.get("active_backend"), extra.get("active_model_tier")),
                settings={
                    "repos": st.session_state.get("repos", []),
                    "collections": st.session_state.get("active_collections", []),
                    "hitl_enabled": hitl1_on, "output_review_mode": review_mode,
                    "max_retrieval_attempts": max_ret, "max_generation_attempts": max_gen,
                    "confidence_threshold": conf_thr, "quality_gate": gate_thr,
                    "top_k": os.environ.get("TOP_K", "10"),
                },
                tags={"source": "ui", "thread_id": st.session_state.thread_id},
            )
            init["mlflow_run_id"] = run_id
            st.session_state.current_run_id = run_id
            if not run_id:
                st.warning(f"MLflow is not recording this question — {tracking.last_error()}")
            st.session_state.replan_round = 0

            if _hard:
                # HARD STOP: a repo failed to index (or every selected repo is empty). Answering would run
                # the whole pipeline on empty context and read like a content problem -- say what failed.
                _bad = _failed or {r: status[r] for r in _no_docs}
                lines = ["**Indexing failed, so the question was not run.**"]
                for r, x in _bad.items():
                    lines.append(f"- `{r.split('/')[-1]}`: {x.get('error') or 'collection is empty (0 chunks)'}")
                    if x.get("hint"):
                        lines.append(f"  - {x['hint']}")
                st.session_state.messages.append({"role": "assistant", "kind": "error", "content": "\n".join(lines)})
                _end_run(None, error="indexing failed: " + "; ".join(
                    f"{r}: {x.get('error', 'empty')}" for r, x in _bad.items()), outcome="error")
                st.rerun()

            with st.spinner("Agent selecting tools..."):
                st.session_state.last_error = None
                st.session_state.last_trace = []
                st.session_state.pending_init = init
                st.session_state.pending_resume = None
                st.session_state.streaming_active = True
                st.session_state.running_node = "tool_selection"
            st.rerun()

# ---------------------------------------------------------------------------
# Advance a running query by ONE node. Deliberately the LAST thing in the script: by now
# every tab above has been drawn from the state left by the previous step, so each rerun
# shows the pipeline one node further along (and which node is executing right now).
# ---------------------------------------------------------------------------
if st.session_state.streaming_active:
    _process_one_stream_step()
    st.rerun()
