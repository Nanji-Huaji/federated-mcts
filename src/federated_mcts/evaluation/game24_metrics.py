"""Versioned Game24 candidate-flow and audit metric contracts.

Construction, aggregation, reconciliation, and boundary validation
functions.  All type contracts live in ``game24_metrics_types.py``.

Follows the pattern from ``blocksworld_metrics.py`` but operates on
typed candidate-event records rather than symbolic oracle evaluation.
"""

from __future__ import annotations

from typing import Literal

from federated_mcts.evaluation.game24_metrics_types import (
    GAME24_METRICS_VERSION,
    CandidateEvent,
    DispositionLiteral,
    ExperimentCandidateMetrics,
    RejectionCounts,
    RejectionReasonLiteral,
    StageLiteral,
    StepCandidateMetrics,
    TaskCandidateMetrics,
)





# ═══════════════════════════════════════════════════════════════════
#  Lineage construction
# ═══════════════════════════════════════════════════════════════════

def build_lineage_id(
    task_id: str,
    depth: int,
    parent_lineage_id: str = "",
    generation_ordinal: int = 0,
) -> str:
    """Return a stable deterministic lineage ID.

    Format: ``{task_id}:d{depth}:{parent or 'root'}:ord{ordinal}``
    """
    if not parent_lineage_id:
        return f"{task_id}:d{depth}:root:ord{generation_ordinal}"
    return f"{task_id}:d{depth}:{parent_lineage_id}:ord{generation_ordinal}"


# ═══════════════════════════════════════════════════════════════════
#  Candidate event construction
# ═══════════════════════════════════════════════════════════════════

def make_candidate_event(
    *,
    task_id: str,
    depth: int,
    parent_lineage_id: str = "",
    generation_ordinal: int = 0,
    stage: StageLiteral = "proposed",
    disposition: DispositionLiteral = "none",
    raw_content: str | None = None,
    canonical_text: str | None = None,
    rejection_reason: RejectionReasonLiteral | None = None,
    evaluator_score: float | None = None,
    evaluator_mode: Literal["vote", "value"] | None = None,
    vote_count: int | None = None,
    is_selected: bool = False,
    is_terminal: bool = False,
    oracle_is_reachable: bool | None = None,
    oracle_exact_pruned: bool | None = None,
) -> CandidateEvent:
    """Build a validated ``CandidateEvent`` with a stable lineage ID.

    Non-finite evaluator scores are rejected.
    """
    lineage_id = build_lineage_id(
        task_id=task_id,
        depth=depth,
        parent_lineage_id=parent_lineage_id,
        generation_ordinal=generation_ordinal,
    )
    if evaluator_score is not None:
        if evaluator_score != evaluator_score or evaluator_score in (
            float("inf"),
            float("-inf"),
        ):
            raise ValueError(
                f"evaluator_score must be finite; got {evaluator_score!r}"
            )

    return CandidateEvent(
        lineage_id=lineage_id,
        task_id=task_id,
        depth=depth,
        parent_lineage_id=parent_lineage_id,
        generation_ordinal=generation_ordinal,
        stage=stage,
        disposition=disposition,
        raw_content=raw_content,
        canonical_text=canonical_text,
        rejection_reason=rejection_reason,
        evaluator_score=evaluator_score,
        evaluator_mode=evaluator_mode,
        vote_count=vote_count,
        is_selected=is_selected,
        is_terminal=is_terminal,
        oracle_is_reachable=oracle_is_reachable,
        oracle_exact_pruned=oracle_exact_pruned,
    )


# ── Internal counter helpers ──────────────────────────────────────

def _empty_rejection_counts() -> RejectionCounts:
    return RejectionCounts(
        empty_content=0,
        malformed_format=0,
        parse_error=0,
        non_integral_division=0,
        divide_by_zero=0,
        number_reuse=0,
        wrong_result=0,
        heuristic_rejected=0,
        exact_prune_unreachable=0,
        vote_parse_failed=0,
        vote_out_of_range=0,
    )


def _merge_rejection_counts(
    a: RejectionCounts, b: RejectionCounts
) -> RejectionCounts:
    out: dict = dict(a)
    for key, value in b.items():
        out[key] = out.get(key, 0) + value
    return RejectionCounts(**out)


# ═══════════════════════════════════════════════════════════════════
#  Step / task / experiment aggregation
# ═══════════════════════════════════════════════════════════════════

_BEYOND_PROPOSED = {"parsed", "locally_valid", "heuristic_rejected",
    "evaluator_scored", "selected", "expanded",
    "terminal_solved", "terminal_unsolved", "terminal_invalid"}
_BEYOND_PARSED = _BEYOND_PROPOSED - {"parsed"}
_EVALUATED_STAGES = {"evaluator_scored", "selected", "expanded",
    "terminal_solved", "terminal_unsolved"}
_SELECTED_STAGES = {"selected", "expanded",
    "terminal_solved", "terminal_unsolved", "terminal_invalid"}

def make_step_metrics(
    *,
    depth: int,
    parent_count: int,
    candidates: list[CandidateEvent],
) -> StepCandidateMetrics:
    """Aggregate one depth's candidate events into step-level metrics."""
    proposed = len(candidates)
    parsed = sum(1 for c in candidates if c["stage"] in _BEYOND_PROPOSED)
    locally_valid = sum(1 for c in candidates if c["stage"] in _BEYOND_PARSED)
    heuristic_rejected = sum(1 for c in candidates
                             if c["stage"] == "heuristic_rejected")
    evaluated = sum(1 for c in candidates
                    if c["stage"] in _EVALUATED_STAGES)
    selected = sum(1 for c in candidates
                   if c["stage"] in _SELECTED_STAGES)

    rejection_counts = _empty_rejection_counts()
    for c in candidates:
        reason = c.get("rejection_reason")
        if reason is not None and reason in rejection_counts:
            rejection_counts[reason] += 1  # type: ignore[literal-required]

    return StepCandidateMetrics(
        version=GAME24_METRICS_VERSION,
        depth=depth,
        parent_count=parent_count,
        proposed_count=proposed,
        parsed_count=parsed,
        locally_valid_count=locally_valid,
        heuristic_rejected_count=heuristic_rejected,
        evaluated_count=evaluated,
        selected_count=selected,
        rejection_counts=rejection_counts,
        candidates=candidates,
    )


def make_task_metrics(
    *,
    task_id: str,
    task_input: str,
    step_details: list[StepCandidateMetrics],
) -> TaskCandidateMetrics:
    """Aggregate step-level metrics into one task-level summary.

    Tasks with at least one ``terminal_solved`` candidate are "solved".
    """
    solved = False
    total_proposed = 0
    total_parsed = 0
    total_selected = 0
    total_evaluated = 0
    total_candidates = 0
    rejection_counts = _empty_rejection_counts()

    for step in step_details:
        total_proposed += step["proposed_count"]
        total_parsed += step["parsed_count"]
        total_selected += step["selected_count"]
        total_evaluated += step["evaluated_count"]
        total_candidates += step["proposed_count"]
        rejection_counts = _merge_rejection_counts(
            rejection_counts, step["rejection_counts"]
        )

    for step in step_details:
        for candidate in step["candidates"]:
            if candidate.get("disposition") == "terminal_solved":
                solved = True
                break

    return TaskCandidateMetrics(
        version=GAME24_METRICS_VERSION,
        task_id=task_id,
        task_input=task_input,
        solved=solved,
        total_steps=len(step_details),
        total_proposed=total_proposed,
        total_parsed=total_parsed,
        total_selected=total_selected,
        total_evaluated=total_evaluated,
        total_candidates=total_candidates,
        rejection_counts=rejection_counts,
        step_details=step_details,
    )


def summarize_experiment_metrics(
    per_task_metrics: list[TaskCandidateMetrics],
) -> ExperimentCandidateMetrics:
    """Aggregate per-task metrics into one experiment-level summary."""
    task_count = len(per_task_metrics)
    solved_task_count = sum(1 for m in per_task_metrics if m["solved"])
    total_proposed = sum(m["total_proposed"] for m in per_task_metrics)
    total_parsed = sum(m["total_parsed"] for m in per_task_metrics)
    total_selected = sum(m["total_selected"] for m in per_task_metrics)
    total_evaluated = sum(m["total_evaluated"] for m in per_task_metrics)
    total_candidates = sum(m["total_candidates"] for m in per_task_metrics)
    rejection_counts: RejectionCounts = _empty_rejection_counts()
    for m in per_task_metrics:
        rejection_counts = _merge_rejection_counts(
            rejection_counts, m["rejection_counts"]
        )

    return ExperimentCandidateMetrics(
        version=GAME24_METRICS_VERSION,
        task_count=task_count,
        solved_task_count=solved_task_count,
        success_rate=(
            solved_task_count / task_count if task_count else 0.0
        ),
        total_proposed=total_proposed,
        total_parsed=total_parsed,
        total_selected=total_selected,
        total_evaluated=total_evaluated,
        total_candidates=total_candidates,
        rejection_counts=rejection_counts,
    )



