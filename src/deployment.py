"""
deployment.py -- what is THIS deployment actually running? The single source of the "container default".

The default inference backend and model tier must follow how the stack was started, with nobody
setting variables by hand. So unless INFERENCE_BACKEND names a backend explicitly (Helm does, and so
can anyone who wants to pin it), the app asks the stack what is up:

    llama-server reachable at LLAMACPP_HOST   ->  backend = llamacpp, model = what it serves,
                                                  tier = inferred from that model
    else vLLM reachable at VLLM_HOST          ->  backend = vllm
    else                                      ->  backend = ollama

Starting the `llamacpp` profile with a heavy-tier GGUF therefore makes llama.cpp + heavy the default
by itself, however the app container was (re)started. The check is re-done every few seconds, so a
llama-server that comes up after the app is picked up without a restart.

Environment
-----------
INFERENCE_BACKEND   "auto" (default in docker-compose.dev.yml) | ollama | vllm | llamacpp
MODEL_TIER          only used when the tier cannot be inferred from the running model
LLAMACPP_HOST, VLLM_HOST
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Optional

_TTL_FOUND = 20.0      # seconds a positive answer is reused
_TTL_MISS = 4.0        # a miss is re-checked sooner, so a starting server is noticed quickly
_cache: dict[str, Any] = {"at": 0.0, "ttl": 0.0, "value": None, "key": None}

# Substring of the served model name / GGUF file -> tier. First match wins, so keep specific names first.
_TIER_BY_MODEL = [
    ("deepseek-coder-v2", "heavy"),
    ("mistral-nemo", "full"),
    ("qwen2.5-coder-7b", "balanced"), ("qwen2.5-coder:7b", "balanced"),
    ("qwen2.5-coder-3b", "minimal"), ("qwen2.5-coder:3b", "minimal"),
    ("phi", "lightweight"),
]


def tier_for_model(name: Optional[str]) -> Optional[str]:
    """Tier a served model belongs to, from its name (None if unrecognised)."""
    low = (name or "").lower()
    for needle, tier in _TIER_BY_MODEL:
        if needle in low:
            return tier
    return None


def _get_json(url: str, timeout: float = 1.5) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001 - "not reachable" is an answer, not an error
        return None


def _model_name(host: str) -> Optional[str]:
    data = _get_json(f"{host}/v1/models") or {}
    models = data.get("data") or []
    return (models[0].get("id") if models else None)


def resolve(force: bool = False) -> dict[str, Any]:
    """
    {"backend", "tier", "model", "source"} for this deployment.
    `source` says in words where the answer came from, for the UI.
    """
    configured = (os.environ.get("INFERENCE_BACKEND") or "auto").strip().lower()
    env_tier = os.environ.get("MODEL_TIER") or "full"
    if configured != "auto":
        return {"backend": configured, "tier": env_tier, "model": None,
                "source": f"set by this deployment: INFERENCE_BACKEND={configured}, MODEL_TIER={env_tier}"}

    llama = os.environ.get("LLAMACPP_HOST", "http://localhost:8081").rstrip("/")
    vllm = os.environ.get("VLLM_HOST", "http://localhost:8080").rstrip("/")
    key = (llama, vllm, env_tier)
    now = time.time()
    if (not force and _cache["value"] is not None and _cache["key"] == key
            and now - _cache["at"] < _cache["ttl"]):
        return _cache["value"]

    props = _get_json(f"{llama}/props")
    if props is not None:
        name = _model_name(llama)
        tier = tier_for_model(name) or tier_for_model(props.get("model_path")) or env_tier
        value = {"backend": "llamacpp", "tier": tier, "model": name,
                 "source": f"detected: llama-server at {llama} is serving {name or 'a model'}"}
        ttl = _TTL_FOUND
    elif (v := _get_json(f"{vllm}/v1/models")) is not None:
        name = ((v.get("data") or [{}])[0]).get("id")
        value = {"backend": "vllm", "tier": tier_for_model(name) or env_tier, "model": name,
                 "source": f"detected: vLLM at {vllm} is serving {name or 'a model'}"}
        ttl = _TTL_FOUND
    else:
        value = {"backend": "ollama", "tier": env_tier, "model": None,
                 "source": "no llama-server or vLLM reachable, so Ollama"}
        ttl = _TTL_MISS
    _cache.update(at=now, ttl=ttl, value=value, key=key)
    return value


def backend() -> str:
    return resolve()["backend"]


def tier() -> str:
    return resolve()["tier"]
