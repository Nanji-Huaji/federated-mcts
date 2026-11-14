"""JSON-safe Blocksworld Oracle metrics for the experiment runners.

Pure and offline: computes the exact optimum once via the symbolic oracle and
evaluates every candidate output against it, returning JSON-safe dicts that
both ``merged_run.py`` runners log without importing model or network code and
without touching search, reward or candidate ordering.

The oracle's only non-JSON value is an infinite optimality ratio (a positive
successful detour when the optimum is zero steps); it is serialised as None
plus an explicit ``optimality_ratio_infinite`` flag so that
``json.dumps(..., allow_nan=False)`` always succeeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from federated_mcts.tasks.blocksworld_oracle import TrajectoryMetrics

METRICS_VERSION = 1


class CandidateMetrics(TypedDict):
    """JSON-safe per-candidate projection of a TrajectoryMetrics."""

    output_index: int
    status: str
    submitted_length: int
    legal_count: int
    legal_rate: float
    first_failure_index: int | None
    success: bool
    plan_length: int | None
    optimal_length: int | None
    excess_steps: int | None
    optimality_ratio: float | None
    optimality_ratio_infinite: bool


class TaskOracleMetrics(TypedDict):
    """JSON-safe summary for one Blocksworld instance."""

    version: int
    task_idx: int | None
    task_id: str
    optimal_length: int | None
    candidate_count: int
    success_count: int
    any_success: bool
    mean_legal_action_rate: float
    best_successful_index: int | None
    best_successful: CandidateMetrics | None
    candidates: list[CandidateMetrics]


class ExperimentOracleMetrics(TypedDict):
    """JSON-safe experiment-level aggregate with explicit denominators."""

    version: int
    task_count: int
    solved_task_count: int
    success_rate: float
    candidate_count: int
    successful_candidate_count: int
    legal_rate_task_count: int
    mean_legal_action_rate: float
    finite_ratio_task_count: int
    mean_optimality_ratio: float
    mean_excess_steps: float


def _json_safe_ratio(value: float | None) -> tuple[float | None, bool]:
    """Map a possibly non-finite ratio to (finite-or-None, was_infinite)."""
    if value is None:
        return None, False
    if value != value or value in (float("inf"), float("-inf")):
        return None, True
    return value, False


def _candidate_metrics(
    index: int, metrics: "TrajectoryMetrics"
) -> CandidateMetrics:
    """JSON-safe projection of one trajectory evaluation plus its index."""
    ratio, ratio_infinite = _json_safe_ratio(metrics.optimality_ratio)
    return CandidateMetrics(
        output_index=index,
        status=metrics.status,
        submitted_length=metrics.submitted_length,
        legal_count=metrics.legal_count,
        legal_rate=metrics.legal_rate,
        first_failure_index=metrics.first_failure_index,
        success=metrics.success,
        plan_length=metrics.plan_length,
        optimal_length=metrics.optimal_length,
        excess_steps=metrics.excess_steps,
        optimality_ratio=ratio,
        optimality_ratio_infinite=ratio_infinite,
    )


def evaluate_input_oracle_metrics(
    x: str, outputs: list[str], idx: int | None = None
) -> TaskOracleMetrics:
    """Compute per-task Oracle metrics for a canonical Blocksworld input ``x``.

    The exact optimum is computed once via ``shortest_plan``; every output is
    then evaluated against that same optimum.  The result is a versioned,
    JSON-safe summary with per-candidate metrics and a task-level aggregate
    (best successful candidate chosen by shortest plan then lowest output
    index).
    """
    from federated_mcts.tasks.blocksworld_engine import parse_x
    from federated_mcts.tasks.blocksworld_oracle import (
        SOLVED,
        evaluate_trajectory,
        shortest_plan,
    )

    parsed = parse_x(x)
    result = shortest_plan(parsed)
    optimal_length = len(result.plan) if result.status == SOLVED else None

    candidates = [
        _candidate_metrics(
            index, evaluate_trajectory(parsed, output, optimal=optimal_length)
        )
        for index, output in enumerate(outputs)
    ]
    successful = [candidate for candidate in candidates if candidate["success"]]
    best = (
        min(successful, key=lambda c: (c["plan_length"], c["output_index"]))
        if successful
        else None
    )
    return TaskOracleMetrics(
        version=METRICS_VERSION,
        task_idx=idx,
        task_id=parsed.get("id", ""),
        optimal_length=optimal_length,
        candidate_count=len(candidates),
        success_count=len(successful),
        any_success=bool(successful),
        mean_legal_action_rate=(
            sum(candidate["legal_rate"] for candidate in candidates) / len(candidates)
            if candidates
            else 0.0
        ),
        best_successful_index=best["output_index"] if best is not None else None,
        best_successful=best,
        candidates=candidates,
    )


def task_oracle_metrics(
    task: object, idx: int, outputs: list[str]
) -> TaskOracleMetrics | None:
    """Runner-facing per-task metric hook; ``None`` for non-Blocksworld tasks.

    The oracle (and any trajectory evaluation) is not touched for
    non-Blocksworld tasks; the task module is imported lazily only for the
    type check.
    """
    from federated_mcts.tasks.blocksworld import BlocksworldTask

    if not isinstance(task, BlocksworldTask):
        return None
    return evaluate_input_oracle_metrics(task.get_input(idx), outputs, idx=idx)


def summarize_oracle_tasks(
    per_task_metrics: list[TaskOracleMetrics],
) -> ExperimentOracleMetrics:
    """Aggregate per-task Oracle metrics into one experiment summary.

    Every mean is averaged over the tasks that contribute a value and the
    denominator counts are reported explicitly, so an empty list stays
    JSON-safe and never divides by zero.  Optimality means are computed over
    successful finite-ratio tasks only (each contributing its best successful
    candidate).
    """
    task_count = len(per_task_metrics)
    solved_task_count = sum(1 for m in per_task_metrics if m["any_success"])
    candidate_count = sum(m["candidate_count"] for m in per_task_metrics)
    successful_candidate_count = sum(m["success_count"] for m in per_task_metrics)

    rated_tasks = [m for m in per_task_metrics if m["candidate_count"] > 0]
    legal_rate_task_count = len(rated_tasks)
    mean_legal_action_rate = (
        sum(m["mean_legal_action_rate"] for m in rated_tasks) / legal_rate_task_count
        if rated_tasks
        else 0.0
    )

    finite_best = [
        m["best_successful"]
        for m in per_task_metrics
        if m["best_successful"] is not None
        and m["best_successful"]["optimality_ratio"] is not None
    ]
    finite_ratio_task_count = len(finite_best)
    finite_ratios = [b["optimality_ratio"] for b in finite_best if b["optimality_ratio"] is not None]
    finite_excess = [b["excess_steps"] for b in finite_best if b["excess_steps"] is not None]
    mean_optimality_ratio = (
        sum(finite_ratios) / len(finite_ratios)
        if finite_best
        else 0.0
    )
    mean_excess_steps = (
        sum(finite_excess) / len(finite_excess)
        if finite_best
        else 0.0
    )
    return ExperimentOracleMetrics(
        version=METRICS_VERSION,
        task_count=task_count,
        solved_task_count=solved_task_count,
        success_rate=solved_task_count / task_count if task_count else 0.0,
        candidate_count=candidate_count,
        successful_candidate_count=successful_candidate_count,
        legal_rate_task_count=legal_rate_task_count,
        mean_legal_action_rate=mean_legal_action_rate,
        finite_ratio_task_count=finite_ratio_task_count,
        mean_optimality_ratio=mean_optimality_ratio,
        mean_excess_steps=mean_excess_steps,
    )


def merge_oracle_metrics(
    per_task_metrics: list[TaskOracleMetrics], res_json: dict[str, object]
) -> dict[str, object]:
    """Attach the experiment Oracle aggregate (no-op for non-Blocksworld runs)."""
    if per_task_metrics:
        res_json["blocksworld_oracle"] = summarize_oracle_tasks(per_task_metrics)
    return res_json
