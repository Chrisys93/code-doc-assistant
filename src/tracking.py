"""
tracking.py -- MLflow run lifecycle for the agent: ONE run per question, whichever way it is driven.

Why this module exists
----------------------
The Streamlit UI advances the graph one node per script rerun (so the Pipeline tab can repaint),
and a human-review pause can last minutes. MLflow's fluent API keeps the "active run" per thread, so
a run opened with `with mlflow.start_run():` cannot span those reruns -- the node-level
`mlflow.log_*` calls would each open a stray run, and the run-level parameters would never be set.

The fix is to keep the run open on the server and re-activate it for exactly the duration of each node:

    run_id = tracking.start_query_run(...)        # once per question  -> stored in state["mlflow_run_id"]
    with tracking.activate(run_id): node(state)   # around every node (agent_graph.build_graph does this)
    tracking.finish_run(run_id, final_state)      # once, when the question is answered / rejected / fails

Every public function is failure-tolerant: if MLflow is unreachable, or not installed, tracking quietly
turns itself into a no-op (one warning in the log) and the agent keeps answering.

Environment
-----------
MLFLOW_TRACKING_URI   where to log (compose sets http://mlflow:5000)
MLFLOW_EXPERIMENT     experiment name              (default "code-doc-assistant-dev")
MLFLOW_UI_URL         browser-reachable MLflow URL (default http://localhost:5000) -- only for UI links
EVAL_GROUP            optional label stored as tag `eval.group`, to group runs of one evaluation batch
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

# A dead MLflow server must not stall a question: fail fast instead of retrying for minutes.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

logger = logging.getLogger(__name__)

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "code-doc-assistant-dev")

_experiment_id: Optional[str] = None
_warned: set[str] = set()
_last_error: Optional[str] = None


def _explain(exc: Exception) -> str:
    """The error text plus, for the failures seen in practice, what to do about them."""
    text = f"{type(exc).__name__}: {exc}"
    if "Invalid Host header" in text:
        text += (" -- the MLflow server only accepts Host headers listed in its --allowed-hosts; "
                 "add this container's address for it (e.g. mlflow:5000) to the mlflow service command")
    elif "Connection" in text or "Max retries" in text or "Name or service not known" in text:
        text += " -- the MLflow server is not reachable at MLFLOW_TRACKING_URI"
    return text


def _warn_once(key: str, msg: str, exc: Exception) -> None:
    global _last_error
    _last_error = f"{msg}: {_explain(exc)}"
    if key not in _warned:
        _warned.add(key)
        logger.warning("MLflow tracking disabled for this step (%s): %s", msg, _explain(exc))


def last_error() -> Optional[str]:
    """Why the most recent tracking call failed (None if the last one worked)."""
    return _last_error


def health() -> tuple[bool, str]:
    """Is the MLflow server reachable AND accepting this client? For the UI status line."""
    global _last_error, _experiment_id
    uri = os.environ.get("MLFLOW_TRACKING_URI", "(default)")
    try:
        _experiment_id = None            # re-check from scratch, do not trust a cached success
        _setup()
        _client().search_experiments(max_results=1)
        _last_error = None
        return True, f"MLflow reachable (tracking server {uri.split('://', 1)[-1]})"   # no scheme: the internal name is not a browser link
    except Exception as e:  # noqa: BLE001
        _last_error = f"health check: {_explain(e)}"
        return False, f"MLflow NOT usable at {uri} -- {_explain(e)}"


def _mlflow():
    import mlflow
    return mlflow


def _client():
    from mlflow import MlflowClient
    return MlflowClient()


def _setup() -> Optional[str]:
    """Point MLflow at the configured server and make sure the experiment exists. Returns its id."""
    global _experiment_id
    if _experiment_id:
        return _experiment_id
    mlflow = _mlflow()
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    _experiment_id = mlflow.set_experiment(EXPERIMENT).experiment_id
    return _experiment_id


def _str(v: Any, limit: int = 500) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str) if isinstance(v, (list, dict, tuple)) else str(v)
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def start_query_run(query: str, runtime: dict, settings: Optional[dict] = None,
                    tags: Optional[dict] = None) -> Optional[str]:
    """
    Open the run for one question and log everything that describes HOW it will be answered, so runs
    can be grouped and compared later (backend x model x tier x settings). Returns the run id, or
    None if MLflow is unavailable.

    runtime  -- what is answering: agent_graph.runtime_info() (backend, tier, model, n_ctx, ...)
    settings -- everything else that shapes the answer (repos, review modes, thresholds, embedding)
    tags     -- free labels, e.g. {"source": "ui", "thread_id": ...}
    """
    try:
        exp = _setup()
        name = " | ".join(str(x) for x in (runtime.get("backend"), runtime.get("model") or runtime.get("tier"),
                                           _str(query, 48)) if x)
        all_tags = {"source": "unknown", "start_ms": str(int(time.time() * 1000))}
        if os.environ.get("EVAL_GROUP"):
            all_tags["eval.group"] = os.environ["EVAL_GROUP"]
        all_tags.update({k: _str(v) for k, v in (tags or {}).items() if v is not None})
        client = _client()
        run = client.create_run(exp, tags=all_tags, run_name=name)
        from mlflow.entities import Param
        params = {"query": query, **{f"rt.{k}": v for k, v in runtime.items()},
                  **{k: v for k, v in (settings or {}).items()}}
        client.log_batch(run.info.run_id, params=[Param(k, _str(v)) for k, v in params.items() if v is not None])
        global _last_error
        _last_error = None
        return run.info.run_id
    except Exception as e:  # noqa: BLE001 - tracking must never break a question
        _warn_once("start", "could not start a run", e)
        return None


@contextmanager
def activate(run_id: Optional[str]) -> Iterator[None]:
    """
    Make `run_id` the active run while the body executes, so plain `mlflow.log_metric(...)` calls inside a
    node land in the right run -- from any thread, in any Streamlit rerun. The run's status is NOT changed
    (it stays RUNNING between nodes and while a human review is pending); finish_run() ends it.
    """
    if not run_id:
        yield
        return
    mlflow = _mlflow() if _importable() else None
    started = False
    if mlflow is not None:
        try:
            _setup()
            cur = mlflow.active_run()
            if cur is None or cur.info.run_id != run_id:
                mlflow.start_run(run_id=run_id)
                started = True
        except Exception as e:  # noqa: BLE001
            _warn_once("activate", "could not activate the run", e)
    try:
        yield
    finally:
        if started:
            try:
                mlflow.end_run(status="RUNNING")   # deactivate without marking it finished/failed
            except Exception:  # noqa: BLE001
                pass


def _importable() -> bool:
    try:
        _mlflow()
        return True
    except Exception:  # noqa: BLE001
        return False


def _outcome(state: Optional[dict], error: Optional[str]) -> str:
    if error:
        return "error"
    if not state:
        return "abandoned"
    if state.get("retrieval_empty"):
        return "no_retrieval"      # the pipeline ran but nothing was retrieved: not an answer
    if (state.get("response") or "").strip():
        return "answered"
    if getattr(state.get("hitl_checkpoint"), "decision", None) == "rejected":
        return "rejected"
    return "no_response"


def finish_run(run_id: Optional[str], state: Optional[dict] = None, outcome: Optional[str] = None,
               error: Optional[str] = None) -> None:
    """
    Close the run: summary metrics, a `result.json` artifact (answer, sources, the tool plan that actually
    ran, retrieved chunks, execution trace) and the terminal status. Safe to call with run_id=None.
    """
    if not run_id:
        return
    try:
        from mlflow.entities import Metric
        client = _client()
        outcome = outcome or _outcome(state, error)
        now = int(time.time() * 1000)
        state = state or {}
        chunks = state.get("retrieved_chunks") or []
        calls = state.get("executed_tool_calls") or []
        conf = [float(getattr(c, "confidence", 0.0) or 0.0) for c in chunks]
        start_ms = int(client.get_run(run_id).data.tags.get("start_ms", now))
        values = {
            "wall_ms": now - start_ms,   # includes any time a human spent in a review pause
            "chunks_retrieved": len(chunks),
            "unique_source_files": len({getattr(c, "source_file", "") for c in chunks}),
            "mean_chunk_confidence": (sum(conf) / len(conf)) if conf else 0.0,
            "top_chunk_confidence": max(conf) if conf else 0.0,
            "tool_calls": len(calls),
            "tool_failures": sum(1 for t in calls if not getattr(t, "success", False)),
            "retrieval_attempts": state.get("retrieval_attempts") or 0,
            "generation_attempts": state.get("generation_attempts") or 0,
            "supervisor_adjustments": len(state.get("supervisor_adjustments") or []),
            "tool_plan_replans": state.get("replan_count") or 0,
            "response_chars": len(state.get("response") or ""),
            "answered": int(outcome == "answered"),
        }
        client.log_batch(run_id, metrics=[Metric(k, float(v), now, 0) for k, v in values.items()])
        client.set_tag(run_id, "outcome", outcome)
        cp = state.get("hitl_checkpoint")
        if getattr(cp, "decision", None):
            client.set_tag(run_id, "hitl.tool_plan_decision", cp.decision)
        if error:
            client.set_tag(run_id, "error", _str(error, 1000))

        result = {
            "outcome": outcome,
            "query": state.get("query"),
            "response": state.get("response"),
            "sources": state.get("source_attribution"),
            "tool_plan_run": [{"tool": t.tool_name, "args": t.args, "success": t.success, "error": t.error,
                               "latency_ms": t.latency_ms} for t in calls],
            "chunks": [{"source_file": c.source_file, "lines": [c.start_line, c.end_line],
                        "type": c.chunk_type, "confidence": round(float(c.confidence or 0), 4)} for c in chunks],
            "trace": state.get("execution_trace"),
        }
        client.log_dict(run_id, json.loads(json.dumps(result, default=str)), "result.json")
        client.set_terminated(run_id, "FAILED" if error else "FINISHED")
    except Exception as e:  # noqa: BLE001
        _warn_once("finish", "could not finish the run", e)


def run_url(run_id: Optional[str]) -> Optional[str]:
    """Browser link to a run (uses MLFLOW_UI_URL, not the container-internal tracking URI)."""
    if not run_id:
        return None
    try:
        exp = _client().get_run(run_id).info.experiment_id
    except Exception:  # noqa: BLE001
        exp = _experiment_id or "0"
    base = os.environ.get("MLFLOW_UI_URL", "http://localhost:5000").rstrip("/")
    return f"{base}/#/experiments/{exp}/runs/{run_id}"
