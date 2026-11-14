"""Tests for Game24 metrics reconciliation and synthetic tree validation.

Covers:
- Reconcile detects stale aggregates, duplicate IDs, disposition mismatches
- Schema validation rejects malformed/unsupported versions
- Synthetic 3-depth solved tree exercises all stages
"""

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_metrics import (
    make_candidate_event,
    make_step_metrics,
    make_task_metrics,
)
from federated_mcts.evaluation.game24_metrics_reconcile import (
    reconcile_task_metrics,
    validate_metrics_schema,
)
from federated_mcts.evaluation.game24_metrics_types import (
    RejectionCounts,
    TaskCandidateMetrics,
)


def _empty_rc() -> RejectionCounts:
    return RejectionCounts(
        empty_content=0, malformed_format=0, parse_error=0,
        non_integral_division=0, divide_by_zero=0, number_reuse=0,
        wrong_result=0, heuristic_rejected=0,
        exact_prune_unreachable=0, vote_parse_failed=0,
        vote_out_of_range=0,
    )


class TestReconciliation(unittest.TestCase):
    """Given a TaskCandidateMetrics record,
       When reconcile_task_metrics() is called,
       Then it detects stale aggregates, duplicate IDs,
       double dispositions, and inconsistent rejection counts."""

    def test_valid_record_passes(self):
        task = make_task_metrics(
            task_id="t", task_input="1 2 3 4",
            step_details=[make_step_metrics(
                depth=0, parent_count=1, candidates=[
                    make_candidate_event(task_id="t", depth=0,
                                         generation_ordinal=0,
                                         stage="terminal_solved",
                                         disposition="terminal_solved",
                                         is_terminal=True),
                ])])
        violations = reconcile_task_metrics(task)
        self.assertEqual(violations, [],
                         f"Expected no violations, got: {violations}")

    def test_stale_aggregate_detected(self):
        task = make_task_metrics(
            task_id="t", task_input="1 2 3 4",
            step_details=[make_step_metrics(
                depth=0, parent_count=1, candidates=[
                    make_candidate_event(task_id="t", depth=0,
                                         generation_ordinal=0,
                                         stage="terminal_solved",
                                         disposition="terminal_solved",
                                         is_terminal=True),
                ])])
        task["total_proposed"] = 99
        violations = reconcile_task_metrics(task)
        self.assertTrue(
            any("total_proposed" in v for v in violations),
            f"Expected total_proposed violation, got: {violations}",
        )

    def test_duplicate_lineage_ids_detected(self):
        step = make_step_metrics(depth=0, parent_count=1, candidates=[
            make_candidate_event(task_id="t", depth=0, generation_ordinal=0,
                                 stage="terminal_solved",
                                 disposition="terminal_solved",
                                 is_terminal=True),
        ])
        dup = make_candidate_event(task_id="t", depth=0, generation_ordinal=0,
                                   stage="terminal_solved",
                                   disposition="terminal_solved",
                                   is_terminal=True)
        step["candidates"].append(dup)
        step["proposed_count"] = 2
        step["selected_count"] = 2
        task = make_task_metrics(
            task_id="t", task_input="1 2 3 4",
            step_details=[step],
        )
        violations = reconcile_task_metrics(task)
        self.assertTrue(
            any("duplicate lineage" in v for v in violations),
            f"Expected duplicate ID violation, got: {violations}",
        )

    def test_terminal_disposition_without_is_terminal(self):
        task = make_task_metrics(
            task_id="t", task_input="1 2 3 4",
            step_details=[make_step_metrics(
                depth=0, parent_count=1, candidates=[
                    make_candidate_event(task_id="t", depth=0,
                                         generation_ordinal=0,
                                         stage="terminal_unsolved",
                                         disposition="terminal_unsolved",
                                         is_terminal=False),
                ])])
        violations = reconcile_task_metrics(task)
        self.assertTrue(
            any("terminal_unsolved" in v and "is_terminal=False" in v
                for v in violations),
            f"Expected is_terminal/disposition mismatch: {violations}",
        )

    def test_missing_version_detected(self):
        metrics: TaskCandidateMetrics = {
            "version": 999,
            "task_id": "t", "task_input": "1 2 3 4", "solved": False,
            "total_steps": 0, "total_proposed": 0, "total_parsed": 0,
            "total_selected": 0, "total_evaluated": 0, "total_candidates": 0,
            "rejection_counts": _empty_rc(), "step_details": [],
        }  # type: ignore[typeddict-item]
        violations = reconcile_task_metrics(metrics)
        self.assertTrue(
            any("unsupported" in v for v in violations),
            f"Expected unsupported version violation: {violations}",
        )

    def test_reconcile_rejects_bool_true_version(self):
        m: TaskCandidateMetrics = {
            "version": True,  # type: ignore[typeddict-item]
            "task_id": "t", "task_input": "1 2 3 4", "solved": False,
            "total_steps": 0, "total_proposed": 0, "total_parsed": 0,
            "total_selected": 0, "total_evaluated": 0, "total_candidates": 0,
            "rejection_counts": _empty_rc(), "step_details": [],
        }  # type: ignore[typeddict-item]
        violations = reconcile_task_metrics(m)
        self.assertTrue(any("version" in v for v in violations),
                        f"Expected bool version violation: {violations}")

    def test_reconcile_rejects_bool_false_version(self):
        m: TaskCandidateMetrics = {
            "version": False,  # type: ignore[typeddict-item]
            "task_id": "t", "task_input": "1 2 3 4", "solved": False,
            "total_steps": 0, "total_proposed": 0, "total_parsed": 0,
            "total_selected": 0, "total_evaluated": 0, "total_candidates": 0,
            "rejection_counts": _empty_rc(), "step_details": [],
        }  # type: ignore[typeddict-item]
        violations = reconcile_task_metrics(m)
        self.assertTrue(any("version" in v for v in violations),
                        f"Expected bool version violation: {violations}")

    def test_stale_rejection_counts_detected(self):
        task = make_task_metrics(
            task_id="t", task_input="1 2 3 4",
            step_details=[make_step_metrics(
                depth=0, parent_count=1, candidates=[
                    make_candidate_event(
                        task_id="t", depth=0, generation_ordinal=0,
                        stage="proposed",
                        rejection_reason="divide_by_zero"),
                ])])
        rc = task["rejection_counts"]
        rc["divide_by_zero"] = 0
        task["rejection_counts"] = rc
        violations = reconcile_task_metrics(task)
        self.assertTrue(
            any("divide_by_zero" in v for v in violations),
            f"Expected divide_by_zero count violation: {violations}",
        )

    def test_unsolved_task_missing_terminal(self):
        step = make_step_metrics(depth=0, parent_count=1, candidates=[
            make_candidate_event(task_id="t", depth=0, generation_ordinal=0,
                                 stage="terminal_solved",
                                 disposition="terminal_solved",
                                 is_terminal=False),
        ])
        task = make_task_metrics(task_id="t", task_input="1 2 3 4",
                                 step_details=[step])
        violations = reconcile_task_metrics(task)
        self.assertTrue(
            any("terminal_solved" in v and "is_terminal=False" in v
                for v in violations),
            f"Expected terminal_solved/is_terminal mismatch: {violations}",
        )


class TestSchemaValidation(unittest.TestCase):
    """Given a metrics record, When validate_metrics_schema() is called,
       Then it rejects missing version, non-integer version,
       and unsupported version."""

    def test_valid_passes(self):
        task = make_task_metrics(
            task_id="t", task_input="1 2 3 4",
            step_details=[],
        )
        validate_metrics_schema(task)

    def test_missing_version_raises(self):
        metrics: TaskCandidateMetrics = {
            "version": None,  # type: ignore[typeddict-item]
            "task_id": "t", "task_input": "x", "solved": False,
            "total_steps": 0, "total_proposed": 0, "total_parsed": 0,
            "total_selected": 0, "total_evaluated": 0, "total_candidates": 0,
            "rejection_counts": _empty_rc(), "step_details": [],
        }  # type: ignore
        with self.assertRaises(ValueError):
            validate_metrics_schema(metrics)  # type: ignore[arg-type]

    def test_non_int_version_raises(self):
        metrics: TaskCandidateMetrics = {
            "version": "v1",  # type: ignore[typeddict-item]
            "task_id": "t", "task_input": "x", "solved": False,
            "total_steps": 0, "total_proposed": 0, "total_parsed": 0,
            "total_selected": 0, "total_evaluated": 0, "total_candidates": 0,
            "rejection_counts": _empty_rc(), "step_details": [],
        }  # type: ignore
        with self.assertRaises(ValueError):
            validate_metrics_schema(metrics)  # type: ignore[arg-type]

    def test_unsupported_version_raises(self):
        m: TaskCandidateMetrics = {
            "version": 99, "task_id": "t", "task_input": "x",
            "solved": False, "total_steps": 0, "total_proposed": 0,
            "total_parsed": 0, "total_selected": 0, "total_evaluated": 0,
            "total_candidates": 0, "rejection_counts": _empty_rc(),
            "step_details": [],
        }  # type: ignore[typeddict-item]
        with self.assertRaises(ValueError):
            validate_metrics_schema(m)

    def test_bool_true_version_rejected(self):
        """version=True rejected (bool is subclass of int)."""
        m: TaskCandidateMetrics = {
            "version": True,  # type: ignore[typeddict-item]
            "task_id": "t", "task_input": "x", "solved": False,
            "total_steps": 0, "total_proposed": 0, "total_parsed": 0,
            "total_selected": 0, "total_evaluated": 0, "total_candidates": 0,
            "rejection_counts": _empty_rc(), "step_details": [],
        }  # type: ignore[typeddict-item]
        with self.assertRaises(ValueError):
            validate_metrics_schema(m)  # type: ignore[arg-type]

    def test_bool_false_version_rejected(self):
        """version=False rejected (bool is subclass of int)."""
        m: TaskCandidateMetrics = {
            "version": False,  # type: ignore[typeddict-item]
            "task_id": "t", "task_input": "x", "solved": False,
            "total_steps": 0, "total_proposed": 0, "total_parsed": 0,
            "total_selected": 0, "total_evaluated": 0, "total_candidates": 0,
            "rejection_counts": _empty_rc(), "step_details": [],
        }  # type: ignore[typeddict-item]
        with self.assertRaises(ValueError):
            validate_metrics_schema(m)  # type: ignore[arg-type]

    def test_float_version_rejected(self):
        """version=1.0 rejected (not int)."""
        m: TaskCandidateMetrics = {
            "version": 1.0,  # type: ignore[typeddict-item]
            "task_id": "t", "task_input": "x", "solved": False,
            "total_steps": 0, "total_proposed": 0, "total_parsed": 0,
            "total_selected": 0, "total_evaluated": 0, "total_candidates": 0,
            "rejection_counts": _empty_rc(), "step_details": [],
        }  # type: ignore[typeddict-item]
        with self.assertRaises(ValueError):
            validate_metrics_schema(m)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
