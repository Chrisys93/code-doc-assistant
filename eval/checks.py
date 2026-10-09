"""
checks.py -- the automatic part of grading (see questions.json -> "grading").

These are cheap, deterministic and blind to which model answered. They catch the failures that need no
judgement (the answer cites nothing, never states the key fact, names 4 of 17 files). They do NOT decide
whether an answer is good: the manual score (0-2) and the optional LLM judge do that.
"""
from __future__ import annotations

import re
from typing import Iterable


def _norm(p: str) -> str:
    return (p or "").replace("\\", "/").strip().lstrip("./").lower()


def cites(sources: Iterable[str], wanted: str) -> bool:
    """True when a reported source IS the wanted file (path suffix match, either direction)."""
    w = _norm(wanted)
    for s in sources:
        n = _norm(s)
        if n == w or n.endswith("/" + w) or w.endswith("/" + n):
            return True
    return False


def _mentions(text: str, group: list[str]) -> bool:
    t = text.lower()
    return any(g.lower() in t for g in group)


def named_files(text: str, universe: Iterable[str]) -> list[str]:
    """Which files of `universe` the answer names (by full path, or by basename when that is unambiguous)."""
    t = _norm(text)
    names = list(universe)
    base_count: dict[str, int] = {}
    for f in names:
        b = _norm(f).rsplit("/", 1)[-1]
        base_count[b] = base_count.get(b, 0) + 1
    out = []
    for f in names:
        full = _norm(f)
        base = full.rsplit("/", 1)[-1]
        if full in t or (base_count[base] == 1 and base in t):
            out.append(f)
    return out


def evaluate(q: dict, answer: str, sources: list[str], retrieved_files: list[str] | None = None) -> dict:
    """Run the question's automatic checks. Returns flat, loggable values plus a per-check breakdown."""
    answer = answer or ""
    sources = list(sources or [])
    pool = sources + list(retrieved_files or [])
    detail: dict[str, object] = {}

    cite_all = q.get("cite_all") or []
    cite_any = q.get("cite_any") or []
    detail["cite_all"] = {f: cites(pool, f) for f in cite_all}
    detail["cite_any"] = {f: cites(pool, f) for f in cite_any}
    cite_all_ok = all(detail["cite_all"].values()) if cite_all else True
    cite_any_ok = any(detail["cite_any"].values()) if cite_any else True

    groups = q.get("must_mention") or []
    detail["must_mention"] = [{"any_of": g, "hit": _mentions(answer, g)} for g in groups]
    mention_ok = all(x["hit"] for x in detail["must_mention"]) if groups else True

    bad = q.get("must_not") or []
    detail["must_not"] = [{"pattern": b, "hit": bool(re.search(b, answer, re.I))} for b in bad]
    must_not_ok = not any(x["hit"] for x in detail["must_not"])

    out = {
        "cite_all_ok": cite_all_ok, "cite_any_ok": cite_any_ok,
        "mention_ok": mention_ok, "must_not_ok": must_not_ok,
        "answer_chars": len(answer), "sources_n": len(sources),
        "detail": detail,
    }

    truth = q.get("ground_truth_files")
    if truth:
        named = named_files(answer, truth)
        # precision: of the repo files the answer names, how many are in the truth set
        out["file_recall"] = len(named) / len(truth)
        out["files_named_n"] = len(named)
        out["files_truth_n"] = len(truth)
        detail["files_named"] = named
        detail["files_missing"] = [f for f in truth if f not in named]

    out["auto_pass"] = bool(cite_all_ok and cite_any_ok and mention_ok and must_not_ok and bool(answer.strip()))
    if q.get("grading") == "ungraded":
        out["auto_pass"] = None        # recorded, never pass/fail
    return out
