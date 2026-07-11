"""
agent_state.py — Shared state schema for the LangGraph agent pipeline.

All nodes read from and write to AgentState. The TypedDict fields are the
single source of truth for what flows through the graph. Annotated fields
use operator.add so parallel nodes can append to lists without clobbering.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Tool call record — one entry per tool invocation
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool_name: str                    # "grep", "vector_search", "ast_parse", etc.
    args: dict[str, Any]              # arguments passed to the tool
    result: Optional[str] = None      # raw output (None until executed)
    success: bool = False
    error: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class Chunk:
    content: str
    source_file: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    chunk_type: str = "text"          # "function", "class", "text", "grep_match"
    confidence: float = 0.0           # similarity score (0–1); 1.0 for exact matches


# ---------------------------------------------------------------------------
# Supervisor adjustment record — audit trail of short-loop optimisations
# ---------------------------------------------------------------------------

@dataclass
class SupervisorAdjustment:
    reason: str                       # why the adjustment was made
    action: str                       # what was changed
    before: dict[str, Any]           # parameter values before
    after: dict[str, Any]            # parameter values after


# ---------------------------------------------------------------------------
# Human-in-the-loop checkpoint result — tool plan review (pre-generation)
# ---------------------------------------------------------------------------

HumanDecision = Literal["approved", "modified", "rejected"]

@dataclass
class HITLCheckpoint:
    proposed_tool_calls: list[ToolCall]
    decision: Optional[HumanDecision] = None
    modified_tool_calls: Optional[list[ToolCall]] = None  # set if decision == "modified"
    feedback: Optional[str] = None


# ---------------------------------------------------------------------------
# Post-generation feedback — human review of the documentation output
#
# This is the second HITL point, after the Documentation LLM has responded.
# The human signals whether the output is satisfactory, and if not, provides
# structured feedback that the supervisor uses to adjust context strategy for
# the next attempt (more context, different files, different format, etc.).
#
# satisfaction: 1–5 rating (1 = wrong/useless, 5 = perfect)
# regenerate:   True → supervisor will adjust and re-run from tool_execution
# context_notes: "focus on the __init__ method", "include the parent class", etc.
# format_notes:  "use NumPy docstring style", "more concise", "add examples"
# ---------------------------------------------------------------------------

OutputSatisfaction = Literal["accept", "regenerate", "add_context"]

@dataclass
class PostGenerationFeedback:
    response_shown: str                            # the response the human reviewed
    decision: OutputSatisfaction = "accept"
    satisfaction_score: int = 5                    # 1–5
    context_notes: Optional[str] = None           # what context was missing or wrong
    format_notes: Optional[str] = None            # documentation style / format preferences
    additional_files: list[str] = field(default_factory=list)  # specific files to add to context


# ---------------------------------------------------------------------------
# Session preference profile — accumulated across post-generation feedback
#
# The supervisor reads this at the start of each retrieval attempt and uses
# it to bias tool selection and context assembly toward what has worked in
# this session. Persists across queries within a thread.
# ---------------------------------------------------------------------------

@dataclass
class SessionPreferences:
    preferred_format: Optional[str] = None         # e.g. "NumPy docstrings", "Google style"
    preferred_verbosity: Optional[str] = None      # "concise" | "detailed"
    prioritised_files: list[str] = field(default_factory=list)   # files that produced good results
    deprioritised_files: list[str] = field(default_factory=list) # files that produced noise
    avg_satisfaction: float = 0.0
    feedback_count: int = 0

    def update(self, feedback: "PostGenerationFeedback") -> None:
        """Incorporate a new feedback event into the running preference profile."""
        # Running average satisfaction
        self.feedback_count += 1
        self.avg_satisfaction = (
            (self.avg_satisfaction * (self.feedback_count - 1) + feedback.satisfaction_score)
            / self.feedback_count
        )
        # Format / verbosity — last explicit preference wins
        if feedback.format_notes:
            self.preferred_format = feedback.format_notes
        # Boost files the human explicitly added
        for f in feedback.additional_files:
            if f not in self.prioritised_files:
                self.prioritised_files.append(f)


# ---------------------------------------------------------------------------
# Main agent state — flows through every LangGraph node
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # --- Input ---
    query: str                                      # user's original question
    repo_path: str                                  # mounted repo path (e.g. /data/repos/myrepo)
    active_collections: list[str]                   # per-repo ChromaDB collections to search (multi-repo)
    hitl_enabled: bool                              # live UI toggle — read by node_hitl_checkpoint
    output_review_mode: str                         # live UI selector — read by node_generation/node_output_review

    # --- Tool selection ---
    proposed_tool_calls: list[ToolCall]             # what the tool-selection agent wants to run
    hitl_checkpoint: Optional[HITLCheckpoint]       # human review of tool plan (pre-generation)
    approved_tool_calls: list[ToolCall]             # after human approval / modification

    # --- Execution (appendable — parallel tool calls safe) ---
    executed_tool_calls: Annotated[list[ToolCall], operator.add]

    # --- Retrieval results ---
    retrieved_chunks: Annotated[list[Chunk], operator.add]
    confidence_scores: list[float]                  # per-chunk; updated after each retrieval attempt

    # --- Short-loop supervisor ---
    retrieval_attempts: int                         # how many retrieve→supervisor loops so far
    max_retrieval_attempts: int                     # configurable ceiling (default: 3)
    supervisor_adjustments: list[SupervisorAdjustment]
    proceed_to_generation: bool                     # supervisor signals go/no-go

    # --- Generation ---
    final_context: str                              # assembled, deduplicated, trimmed context
    response: str                                   # final LLM output
    source_attribution: list[str]                  # file paths that contributed to the response

    # --- Post-generation HITL + preference learning ---
    post_generation_feedback: Optional[PostGenerationFeedback]   # set after human reviews output
    session_preferences: Optional[SessionPreferences]            # accumulated across the session
    generation_attempts: int                                     # how many regeneration loops

    # --- Execution trace — for graph visualisation highlighting in the UI ---
    # Each entry: {"node": str, "status": "ok"|"retry"|"rejected", "detail": str}
    execution_trace: list[dict[str, Any]]

    # --- MLflow trace metadata ---
    mlflow_run_id: Optional[str]
    total_latency_ms: Optional[float]
