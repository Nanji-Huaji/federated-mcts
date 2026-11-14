"""TypedDict / dataclass contracts for Game24 candidate-flow metrics.

All type definitions live here so the construction/reconciliation module
stays under 250 pure LOC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

GAME24_METRICS_VERSION: int = 1
"""Schema version; reject unknown versions at the boundary."""

# ── Stage / rejection / disposition literals ──────────────────────

StageLiteral = Literal[
    "proposed",
    "parsed",
    "locally_valid",
    "heuristic_rejected",
    "evaluator_scored",
    "selected",
    "expanded",
    "terminal_solved",
    "terminal_unsolved",
    "terminal_invalid",
]

RejectionReasonLiteral = Literal[
    "empty_content",
    "malformed_format",
    "parse_error",
    "non_integral_division",
    "divide_by_zero",
    "number_reuse",
    "wrong_result",
    "heuristic_rejected",
    "exact_prune_unreachable",
    "vote_parse_failed",
    "vote_out_of_range",
]

DispositionLiteral = Literal[
    "none",
    "selected",
    "expanded",
    "terminal_solved",
    "terminal_unsolved",
    "terminal_invalid",
]


# ── Frozen value objects ──────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CandidateLineage:
    """Immutable provenance key for one candidate.

    The ``id`` property produces a deterministic string suitable as
    a stable cross-run identifier.
    """

    task_id: str
    depth: int
    parent_lineage_id: str
    generation_ordinal: int

    @property
    def id(self) -> str:
        if not self.parent_lineage_id:
            return (
                f"{self.task_id}:d{self.depth}:root:"
                f"ord{self.generation_ordinal}"
            )
        return (
            f"{self.task_id}:d{self.depth}:"
            f"{self.parent_lineage_id}:ord{self.generation_ordinal}"
        )


# ── JSON-safe TypedDict contracts ─────────────────────────────────

class CandidateEvent(TypedDict, total=False):
    """Single candidate event, stable across runs.

    Audit fields are namespaced with ``oracle_`` so they cannot be
    confused with decision-affecting fields.
    """

    lineage_id: str
    task_id: str
    depth: int
    parent_lineage_id: str
    generation_ordinal: int
    stage: str
    disposition: str
    raw_content: str | None
    canonical_text: str | None
    rejection_reason: str | None
    evaluator_score: float | None
    evaluator_mode: str | None
    vote_count: int | None
    is_selected: bool
    is_terminal: bool
    oracle_is_reachable: bool | None
    oracle_exact_pruned: bool | None


class RejectionCounts(TypedDict, total=False):
    """Map from stable rejection reason strings to integer counts."""

    empty_content: int
    malformed_format: int
    parse_error: int
    non_integral_division: int
    divide_by_zero: int
    number_reuse: int
    wrong_result: int
    heuristic_rejected: int
    exact_prune_unreachable: int
    vote_parse_failed: int
    vote_out_of_range: int


class StepCandidateMetrics(TypedDict):
    """Per-depth aggregate with all candidate events."""

    version: int
    depth: int
    parent_count: int
    proposed_count: int
    parsed_count: int
    locally_valid_count: int
    heuristic_rejected_count: int
    evaluated_count: int
    selected_count: int
    rejection_counts: RejectionCounts
    candidates: list[CandidateEvent]


class TaskCandidateMetrics(TypedDict):
    """Per-task summary with step breakdown and exact denominators."""

    version: int
    task_id: str
    task_input: str
    solved: bool
    total_steps: int
    total_proposed: int
    total_parsed: int
    total_selected: int
    total_evaluated: int
    total_candidates: int
    rejection_counts: RejectionCounts
    step_details: list[StepCandidateMetrics]


class ExperimentCandidateMetrics(TypedDict):
    """Experiment-level aggregate with exact denominators."""

    version: int
    task_count: int
    solved_task_count: int
    success_rate: float
    total_proposed: int
    total_parsed: int
    total_selected: int
    total_evaluated: int
    total_candidates: int
    rejection_counts: RejectionCounts
