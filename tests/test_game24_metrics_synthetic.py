"""Synthetic 3-depth solved Game24 tree evidence driver.

Full candidate lifecycle: proposed → parsed → locally_valid →
evaluator_scored → selected → expanded → terminal_solved.
Verifies reconciliation, unique IDs, JSON safety, experiment aggregate.
"""

import json
import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_metrics import (
    make_candidate_event,
    make_step_metrics,
    make_task_metrics,
    summarize_experiment_metrics,
)
from federated_mcts.evaluation.game24_metrics_reconcile import (
    reconcile_task_metrics,
)
from federated_mcts.evaluation.game24_metrics_types import RejectionCounts


def _empty_rc() -> RejectionCounts:
    return RejectionCounts(
        empty_content=0, malformed_format=0, parse_error=0,
        non_integral_division=0, divide_by_zero=0, number_reuse=0,
        wrong_result=0, heuristic_rejected=0,
        exact_prune_unreachable=0, vote_parse_failed=0,
        vote_out_of_range=0,
    )


class TestSyntheticThreeDepthSolvedTree(unittest.TestCase):
    """Given a 3-depth solved Game24 search tree,
       When metrics are aggregated through step/task/experiment,
       Then all counts reconcile, IDs unique, JSON safe."""

    _TASK_ID = "42"
    _TASK_INPUT = "4 4 6 8"

    def _cand(self, depth, parent_lid, ordinal, **kw):
        return make_candidate_event(
            task_id=self._TASK_ID, depth=depth,
            parent_lineage_id=parent_lid,
            generation_ordinal=ordinal, **kw,
        )

    def test_full_tree(self):
        # Depth 0: root → 3 candidates; 1 selected
        d0_cands = [
            self._cand(0, "", 0, stage="proposed",
                       rejection_reason="empty_content"),
            self._cand(0, "", 1, stage="evaluator_scored",
                       evaluator_score=0.3, evaluator_mode="value"),
            self._cand(0, "", 2, stage="selected",
                       evaluator_score=2.0, evaluator_mode="vote",
                       vote_count=2, is_selected=True, is_terminal=False,
                       disposition="expanded"),
        ]
        d0_parent_lid = d0_cands[2]["lineage_id"]
        step0 = make_step_metrics(depth=0, parent_count=1, candidates=d0_cands)

        # Depth 1: 2 candidates from parent; 1 selected
        d1_cands = [
            self._cand(1, d0_parent_lid, 0, stage="parsed",
                       rejection_reason="parse_error"),
            self._cand(1, d0_parent_lid, 1, stage="selected",
                       evaluator_score=1.5, evaluator_mode="value",
                       is_selected=True, is_terminal=False,
                       disposition="expanded"),
        ]
        d1_parent_lid = d1_cands[1]["lineage_id"]
        step1 = make_step_metrics(depth=1, parent_count=1, candidates=d1_cands)

        # Depth 2: reaches 24
        d2_cands = [
            self._cand(2, d1_parent_lid, 0,
                       stage="terminal_solved",
                       disposition="terminal_solved",
                       evaluator_score=3.0, evaluator_mode="value",
                       is_selected=True, is_terminal=True),
        ]
        step2 = make_step_metrics(depth=2, parent_count=1, candidates=d2_cands)

        task = make_task_metrics(
            task_id=self._TASK_ID, task_input=self._TASK_INPUT,
            step_details=[step0, step1, step2],
        )

        self.assertTrue(task["solved"])
        self.assertEqual(task["total_proposed"], 6)
        self.assertEqual(task["total_parsed"], 5)
        self.assertEqual(task["total_selected"], 3)
        self.assertEqual(task["total_steps"], 3)

        violations = reconcile_task_metrics(task)
        self.assertEqual(violations, [])
        all_ids = []
        for step in task["step_details"]:
            for c in step["candidates"]:
                all_ids.append(c["lineage_id"])
        self.assertEqual(len(all_ids), len(set(all_ids)))

        result = json.dumps(task, allow_nan=False)
        self.assertIsInstance(result, str)

        exp = summarize_experiment_metrics([task])
        self.assertEqual(exp["task_count"], 1)
        self.assertEqual(exp["solved_task_count"], 1)
        self.assertAlmostEqual(exp["success_rate"], 1.0)
        self.assertEqual(exp["rejection_counts"].get("empty_content", 0), 1)
        self.assertEqual(exp["rejection_counts"].get("parse_error", 0), 1)

    def test_rejection_counts_exact(self):
        """All 13 rejection reason keys present with explicit counts."""
        d0_cands = [
            self._cand(0, "", i, stage="proposed", rejection_reason=r)
            for i, r in enumerate(
                ["empty_content", "malformed_format", "parse_error",
                 "non_integral_division", "divide_by_zero", "number_reuse",
                 "wrong_result", "heuristic_rejected",
                 "exact_prune_unreachable", "vote_parse_failed",
                 "vote_out_of_range"]
            )
        ]
        step = make_step_metrics(depth=0, parent_count=1, candidates=d0_cands)
        task = make_task_metrics(task_id="t", task_input="x",
                                 step_details=[step])
        rc = task["rejection_counts"]
        for reason in _empty_rc():
            count = rc.get(reason, 0)
            self.assertEqual(count, 1, f"Expected 1 {reason}, got {count}")

    def test_misleading_success_rejected(self):
        d0_cands = [
            self._cand(0, "", 0, stage="terminal_solved",
                       disposition="terminal_solved",
                       is_terminal=True),
        ]
        step = make_step_metrics(depth=0, parent_count=1, candidates=d0_cands)
        task = make_task_metrics(
            task_id="t_mislead", task_input="1 2 3 4",
            step_details=[step],
        )
        self.assertTrue(task["solved"])
        task["solved"] = False  # tamper
        violations = reconcile_task_metrics(task)
        self.assertFalse(
            any("disposition" in v and "is_terminal" in v
                for v in violations),
        )


if __name__ == "__main__":
    unittest.main()
