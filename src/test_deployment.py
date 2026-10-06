"""
test_deployment.py -- the "container default" backend/tier must follow what the stack is running.

    docker exec -w /app/src code-doc-app python test_deployment.py        # exit code 0 = all passed

Starts tiny fake llama-server / vLLM HTTP servers on local ports; needs no GPU, Ollama or Chroma.
"""
import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import deployment  # noqa: E402


def serve(routes):
    """Fake server answering GET <path> with the given JSON; returns (url, stop)."""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = routes.get(self.path)
            self.send_response(200 if body is not None else 404)
            self.end_headers()
            self.wfile.write(json.dumps(body if body is not None else {}).encode())
        def log_message(self, *a): pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}", srv.shutdown


def llama(model, gguf=None):
    return {"/props": {"model_path": gguf or f"/models/{model}.gguf", "total_slots": 4},
            "/v1/models": {"data": [{"id": model}]}}


DEAD = "http://127.0.0.1:9"      # nothing listens here


def env(**kw):
    for k in ("INFERENCE_BACKEND", "MODEL_TIER", "LLAMACPP_HOST", "VLLM_HOST"):
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in kw.items() if v is not None})
    deployment._cache.update(at=0.0, ttl=0.0, value=None, key=None)


print("1. A backend named explicitly is respected (Helm, or anyone pinning it) - no probing")
env(INFERENCE_BACKEND="ollama", MODEL_TIER="balanced", LLAMACPP_HOST=DEAD, VLLM_HOST=DEAD)
r = deployment.resolve()
assert (r["backend"], r["tier"]) == ("ollama", "balanced") and "set by this deployment" in r["source"], r

print("2. auto with nothing running -> Ollama")
env(INFERENCE_BACKEND="auto", MODEL_TIER="full", LLAMACPP_HOST=DEAD, VLLM_HOST=DEAD)
r = deployment.resolve()
assert (r["backend"], r["tier"]) == ("ollama", "full"), r

print("3. auto + a llama-server serving DeepSeek -> llama.cpp + heavy, with no variable set by hand")
url, stop = serve(llama("deepseek-coder-v2-lite-instruct-q4_k_m"))
env(INFERENCE_BACKEND="auto", MODEL_TIER="full", LLAMACPP_HOST=url, VLLM_HOST=DEAD)   # MODEL_TIER is the compose fallback
r = deployment.resolve()
assert (r["backend"], r["tier"], r["model"]) == ("llamacpp", "heavy", "deepseek-coder-v2-lite-instruct-q4_k_m"), r
assert "llama-server" in r["source"]
stop()

print("4. ...and the same app follows the stack when a different model is served (Nemo -> full)")
url, stop = serve(llama("mistral-nemo-instruct-2407-q4_k_m"))
env(INFERENCE_BACKEND="auto", MODEL_TIER="heavy", LLAMACPP_HOST=url, VLLM_HOST=DEAD)
r = deployment.resolve()
assert (r["backend"], r["tier"]) == ("llamacpp", "full"), r
stop()

print("5. A server that starts AFTER the app is picked up without restarting it")
env(INFERENCE_BACKEND="auto", LLAMACPP_HOST=DEAD, VLLM_HOST=DEAD)
assert deployment.resolve()["backend"] == "ollama"
url, stop = serve(llama("deepseek-coder-v2-lite-instruct-q4_k_m"))
os.environ["LLAMACPP_HOST"] = url
deployment._cache["at"] -= deployment._TTL_MISS + 1          # let the 'not found' answer expire
assert deployment.resolve()["backend"] == "llamacpp"
stop()

print("6. llama.cpp wins over vLLM; vLLM wins over Ollama")
url_l, stop_l = serve(llama("deepseek-coder-v2-lite-instruct-q4_k_m"))
url_v, stop_v = serve({"/v1/models": {"data": [{"id": "mistralai/Mistral-7B-Instruct"}]}})
env(INFERENCE_BACKEND="auto", LLAMACPP_HOST=url_l, VLLM_HOST=url_v)
assert deployment.resolve()["backend"] == "llamacpp"
env(INFERENCE_BACKEND="auto", LLAMACPP_HOST=DEAD, VLLM_HOST=url_v)
assert deployment.resolve()["backend"] == "vllm"
stop_l(); stop_v()

print("7. Tier from the model name; unknown names fall back to MODEL_TIER")
for name, tier in [("deepseek-coder-v2-lite-instruct-q4_k_m", "heavy"), ("mistral-nemo-instruct-2407-q4_k_m", "full"),
                   ("qwen2.5-coder-3b-instruct-q4_k_m", "minimal"), ("Phi-3.5-mini-instruct", "lightweight"),
                   ("qwen2.5-coder-7b-instruct", "balanced"), ("something-else", None)]:
    assert deployment.tier_for_model(name) == tier, (name, deployment.tier_for_model(name))
url, stop = serve(llama("my-own-finetune"))
env(INFERENCE_BACKEND="auto", MODEL_TIER="lightweight", LLAMACPP_HOST=url, VLLM_HOST=DEAD)
assert deployment.resolve()["tier"] == "lightweight"
stop()

print("8. The graph and the logged run use the same answer (agent_graph.runtime_info)")
for _mod, _cls in (("mlflow", None), ("langchain_ollama", "ChatOllama"), ("langchain_openai", "ChatOpenAI")):
    try:
        __import__(_mod)
    except ImportError:
        m = types.ModuleType(_mod)
        if _cls: setattr(m, _cls, object)
        sys.modules[_mod] = m
ri = types.ModuleType("repo_index"); ri.ensure_indexed = lambda *a, **k: {}; ri.active_collections = lambda r: []
sys.modules["repo_index"] = ri
import agent_graph as ag  # noqa: E402
url, stop = serve(llama("deepseek-coder-v2-lite-instruct-q4_k_m"))
env(INFERENCE_BACKEND="auto", LLAMACPP_HOST=url, VLLM_HOST=DEAD)
ag.LLAMACPP_HOST = url
info = ag.runtime_info()
assert (info["backend"], info["tier"]) == ("llamacpp", "heavy") and "llama-server" in info["default_source"], info
assert ag._default_backend() == "llamacpp"
assert ag.runtime_info("ollama", "full")["backend"] == "ollama", "an explicit UI override still wins"
stop()

print("ALL DEPLOYMENT CHECKS PASSED")
