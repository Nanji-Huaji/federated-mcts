"""Reconciliation and schema boundary validation for Game24 metrics.

Verifies internal consistency of task and experiment metric records
and rejects malformed or unsupported schema versions.
"""

from __future__ import annotations

from federated_mcts.evaluation.game24_metrics_types import (
    GAME24_METRICS_VERSION,
    ExperimentCandidateMetrics,
    RejectionCounts,
    TaskCandidateMetrics,
)

__all__ = [
    "reconcile_task_metrics",
    "validate_metrics_schema",
]


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


def reconcile_task_metrics(metrics: TaskCandidateMetrics) -> list[str]:
    """Verify internal consistency of a ``TaskCandidateMetrics`` record.

    Returns a (possibly empty) list of human-readable violation strings.
    Empty list means all aggregate numbers reconcile.
    """
    violations: list[str] = []

    version = metrics.get("version")
    if version is None:
        violations.append("task metrics missing 'version' field")
    elif type(version) is not int:
        violations.append(
            f"task metrics 'version' must be int; "
            f"got {type(version).__name__}"
        )
    elif version != GAME24_METRICS_VERSION:
        violations.append(
            f"task metrics version {version!r} unsupported; "
            f"expected {GAME24_METRICS_VERSION}"
        )

    recomputed_proposed = 0
    recomputed_parsed = 0
    recomputed_selected = 0
    recomputed_evaluated = 0
    recomputed_rejections = _empty_rejection_counts()
    lineage_seen: set[str] = set()
    duplicate_ids: list[str] = []

    for i, step in enumerate(metrics.get("step_details", [])):
        recomputed_proposed += step["proposed_count"]
        recomputed_parsed += step["parsed_count"]
        recomputed_selected += step["selected_count"]
        recomputed_evaluated += step["evaluated_count"]
        recomputed_rejections = _merge_rejection_counts(
            recomputed_rejections, step["rejection_counts"]
        )

        for candidate in step["candidates"]:
            lid = candidate["lineage_id"]
            if lid in lineage_seen:
                duplicate_ids.append(lid)
            lineage_seen.add(lid)

            is_terminal = candidate.get("is_terminal", False)
            disposition = candidate.get("disposition", "none")
            is_terminus = disposition.startswith("terminal_")
            if is_terminal and not is_terminus:
                violations.append(
                    f"step {i} candidate {lid}: is_terminal=True "
                    f"but disposition={disposition!r} (not a terminal "
                    f"disposition)"
                )
            if is_terminus and not is_terminal:
                violations.append(
                    f"step {i} candidate {lid}: "
                    f"disposition={disposition!r} but is_terminal=False"
                )

    if duplicate_ids:
        violations.append(
            f"duplicate lineage IDs: {sorted(set(duplicate_ids))}"
        )

    if metrics["total_proposed"] != recomputed_proposed:
        violations.append(
            f"total_proposed {metrics['total_proposed']} != "
            f"recomputed {recomputed_proposed}"
        )
    if metrics["total_parsed"] != recomputed_parsed:
        violations.append(
            f"total_parsed {metrics['total_parsed']} != "
            f"recomputed {recomputed_parsed}"
        )
    if metrics["total_selected"] != recomputed_selected:
        violations.append(
            f"total_selected {metrics['total_selected']} != "
            f"recomputed {recomputed_selected}"
        )
    if metrics["total_evaluated"] != recomputed_evaluated:
        violations.append(
            f"total_evaluated {metrics['total_evaluated']} != "
            f"recomputed {recomputed_evaluated}"
        )

    declared_rejections = metrics.get("rejection_counts", {})
    for key in recomputed_rejections:
        recomputed_value = recomputed_rejections.get(key, 0)
        declared_value = declared_rejections.get(key, 0)
        if declared_value != recomputed_value:
            violations.append(
                f"rejection_counts['{key}']: declared "
                f"{declared_value} != recomputed {recomputed_value}"
            )

    return violations


def validate_metrics_schema(
    metrics: TaskCandidateMetrics | ExperimentCandidateMetrics,
) -> None:
    """Raise ``ValueError`` if *metrics* has malformed/unsupported schema.

    Must be called before accepting any external record at the boundary.
    Uses ``type(...) is not int`` rather than ``isinstance`` to reject
    ``bool`` (which is a subclass of ``int`` in Python).
    """
    version = metrics.get("version")
    if version is None:
        raise ValueError("metrics record missing 'version' field")
    if type(version) is not int:
        raise ValueError(
            f"metrics 'version' must be int; "
            f"got {type(version).__name__}"
        )
    if version != GAME24_METRICS_VERSION:
        raise ValueError(
            f"unsupported metrics version {version}; "
            f"this runtime supports {GAME24_METRICS_VERSION}"
        )
