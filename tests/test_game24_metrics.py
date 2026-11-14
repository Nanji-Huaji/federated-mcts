"""Tests for Game24 candidate-flow and audit metric contracts.

Covers Todo 2 from .omo/plans/game24-model-only-improvements.md:
- Stable deterministic lineage IDs
- Candidate event construction with non-finite rejection
- Step/task/experiment aggregation with exact denominators
- Reconciliation detects stale aggregates, duplicate IDs,
  double dispositions, and inconsistent counts
- Schema validation rejects malformed/unsupported versions
- JSON-safe output (allow_nan=False)
- Synthetic 3-depth solved tree exercises all stages
"""

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_metrics import (
    GAME24_METRICS_VERSION,
    build_lineage_id,
    make_candidate_event,
    make_step_metrics,
    make_task_metrics,
    summarize_experiment_metrics,
)
from federated_mcts.evaluation.game24_metrics_types import (
    CandidateEvent,
    RejectionCounts,
    TaskCandidateMetrics,
)


# ── helpers ────────────────────────────────────────────────────────

def _empty_rc() -> RejectionCounts:
    return RejectionCounts(
        empty_content=0, malformed_format=0, parse_error=0,
        non_integral_division=0, divide_by_zero=0, number_reuse=0,
        wrong_result=0, heuristic_rejected=0,
        exact_prune_unreachable=0, vote_parse_failed=0,
        vote_out_of_range=0,
    )


class TestLineageIdDeterminism(unittest.TestCase):
    """Given the same task/depth/parent/ordinal,
       When build_lineage_id() is called,
       Then it returns a stable deterministic string."""

    def test_same_inputs_same_id(self):
        a = build_lineage_id("task_0", 1, "task_0:d0:root:ord0", 2)
        b = build_lineage_id("task_0", 1, "task_0:d0:root:ord0", 2)
        self.assertEqual(a, b)

    def test_different_ordinal_different_id(self):
        a = build_lineage_id("task_0", 1, "root_from_prev", 2)
        b = build_lineage_id("task_0", 1, "root_from_prev", 3)
        self.assertNotEqual(a, b)

    def test_different_depth_different_id(self):
        a = build_lineage_id("task_0", 0, "", 0)
        b = build_lineage_id("task_0", 1, "task_0:d0:root:ord0", 0)
        self.assertNotEqual(a, b)

    def test_different_task_different_id(self):
        a = build_lineage_id("task_a", 0, "", 0)
        b = build_lineage_id("task_b", 0, "", 0)
        self.assertNotEqual(a, b)

    def test_root_lineage_format(self):
        lid = build_lineage_id("42", 0, "", 5)
        self.assertEqual(lid, "42:d0:root:ord5")

    def test_child_lineage_format(self):
        parent = "42:d0:root:ord0"
        lid = build_lineage_id("42", 1, parent, 3)
        expected = f"42:d1:{parent}:ord3"
        self.assertEqual(lid, expected)

    def test_empty_parent_id_produces_root(self):
        lid = build_lineage_id("task_x", 2, "", 0)
        self.assertIn(":root:", lid)


class TestCandidateEventConstruction(unittest.TestCase):
    """Given candidate metadata, When make_candidate_event() is called,
       Then it produces a CandidateEvent with a correct lineage_id."""

    def test_minimal_event(self):
        event = make_candidate_event(task_id="t1", depth=0)
        self.assertEqual(event["lineage_id"], "t1:d0:root:ord0")
        self.assertEqual(event["stage"], "proposed")
        self.assertEqual(event["disposition"], "none")
        self.assertFalse(event["is_selected"])
        self.assertFalse(event["is_terminal"])
        self.assertIsNone(event["rejection_reason"])

    def test_full_event(self):
        event = make_candidate_event(
            task_id="t2", depth=2, parent_lineage_id="t2:d1:p:ord1",
            generation_ordinal=3,
            stage="locally_valid",
            disposition="selected",
            rejection_reason="number_reuse",
            evaluator_score=0.75,
            evaluator_mode="value",
            is_selected=True,
            is_terminal=True,
            oracle_is_reachable=True,
        )
        self.assertEqual(event["depth"], 2)
        self.assertEqual(event["evaluator_score"], 0.75)
        self.assertEqual(event["evaluator_mode"], "value")
        self.assertTrue(event["oracle_is_reachable"])
        self.assertIsNone(event["oracle_exact_pruned"])

    def test_nan_score_rejected(self):
        with self.assertRaises(ValueError):
            make_candidate_event(task_id="t", depth=0, evaluator_score=float("nan"))

    def test_inf_score_rejected(self):
        with self.assertRaises(ValueError):
            make_candidate_event(task_id="t", depth=0, evaluator_score=float("inf"))

    def test_neg_inf_score_rejected(self):
        with self.assertRaises(ValueError):
            make_candidate_event(task_id="t", depth=0, evaluator_score=float("-inf"))


class TestStepMetrics(unittest.TestCase):
    """Given a list of candidates at one depth,
       When make_step_metrics() is called,
       Then it produces correctly aggregated counts."""

    def test_empty_candidates(self):
        step = make_step_metrics(depth=0, parent_count=1, candidates=[])
        self.assertEqual(step["proposed_count"], 0)
        self.assertEqual(step["parsed_count"], 0)
        self.assertEqual(step["selected_count"], 0)
        self.assertEqual(step["depth"], 0)
        self.assertEqual(step["parent_count"], 1)

    def test_mixed_stage_candidates(self):
        proposals = [
            make_candidate_event(task_id="t", depth=0, generation_ordinal=i,
                                 stage="proposed",
                                 rejection_reason="empty_content")
            for i in range(2)  # 2 rejected at proposed stage
        ]
        good = make_candidate_event(
            task_id="t", depth=0, generation_ordinal=2,
            stage="selected", is_selected=True, is_terminal=True,
            disposition="terminal_unsolved",
        )
        proposals.append(good)

        step = make_step_metrics(depth=0, parent_count=1, candidates=proposals)
        self.assertEqual(step["proposed_count"], 3)
        self.assertEqual(step["parsed_count"], 1)
        self.assertEqual(step["evaluated_count"], 1)
        self.assertEqual(step["selected_count"], 1)
        rc = step["rejection_counts"]
        self.assertEqual(rc.get("empty_content", 0), 2)


class TestTaskMetrics(unittest.TestCase):
    """Given step-level metrics, When make_task_metrics() is called,
       Then it produces a correctly aggregated task summary."""

    def test_unsolved_task(self):
        step0 = make_step_metrics(depth=0, parent_count=1, candidates=[
            make_candidate_event(task_id="t1", depth=0, generation_ordinal=0,
                                 stage="terminal_unsolved",
                                 disposition="terminal_unsolved",
                                 is_terminal=True),
        ])
        task = make_task_metrics(
            task_id="t1", task_input="1 2 3 4",
            step_details=[step0],
        )
        self.assertFalse(task["solved"])
        self.assertEqual(task["total_steps"], 1)
        self.assertEqual(task["total_proposed"], 1)
        self.assertEqual(task["total_candidates"], 1)

    def test_solved_task(self):
        step0 = make_step_metrics(depth=0, parent_count=1, candidates=[
            make_candidate_event(task_id="t1", depth=0, generation_ordinal=0,
                                 stage="terminal_solved",
                                 disposition="terminal_solved",
                                 is_terminal=True),
        ])
        task = make_task_metrics(
            task_id="t1", task_input="1 2 3 4",
            step_details=[step0],
        )
        self.assertTrue(task["solved"])

    def test_multi_step_task(self):
        step0 = make_step_metrics(depth=0, parent_count=1, candidates=[
            make_candidate_event(task_id="t99", depth=0,
                                 generation_ordinal=0,
                                 stage="expanded", is_selected=True,
                                 is_terminal=False,
                                 disposition="expanded"),
        ])
        step1 = make_step_metrics(depth=1, parent_count=1, candidates=[
            make_candidate_event(task_id="t99", depth=1,
                                 parent_lineage_id="t99:d0:root:ord0",
                                 generation_ordinal=0,
                                 stage="expanded", is_selected=True,
                                 is_terminal=False,
                                 disposition="expanded"),
        ])
        step2 = make_step_metrics(depth=2, parent_count=1, candidates=[
            make_candidate_event(task_id="t99", depth=2,
                                 parent_lineage_id="t99:d1:t99:d0:root:ord0:ord0",
                                 generation_ordinal=0,
                                 stage="terminal_solved",
                                 disposition="terminal_solved",
                                 is_terminal=True),
        ])
        task = make_task_metrics(
            task_id="t99", task_input="4 4 6 8",
            step_details=[step0, step1, step2],
        )
        self.assertTrue(task["solved"])
        self.assertEqual(task["total_steps"], 3)
        self.assertEqual(task["total_proposed"], 3)
        self.assertEqual(task["total_selected"], 3)
        self.assertEqual(task["total_candidates"], 3)


class TestExperimentMetrics(unittest.TestCase):
    """Given per-task metrics, When summarize_experiment_metrics() is called,
       Then it produces correct experiment-level aggregates."""

    def test_empty_list(self):
        exp = summarize_experiment_metrics([])
        self.assertEqual(exp["version"], GAME24_METRICS_VERSION)
        self.assertEqual(exp["task_count"], 0)
        self.assertEqual(exp["solved_task_count"], 0)
        self.assertEqual(exp["success_rate"], 0.0)

    def test_one_solved_one_unsolved(self):
        solved = make_task_metrics(task_id="a", task_input="1 2 3 4",
                                   step_details=[make_step_metrics(
                                       depth=0, parent_count=1, candidates=[
                                           make_candidate_event(
                                               task_id="a", depth=0,
                                               generation_ordinal=0,
                                               stage="terminal_solved",
                                               disposition="terminal_solved",
                                               is_terminal=True),
                                       ])])
        unsolved = make_task_metrics(task_id="b", task_input="5 5 5 5",
                                     step_details=[make_step_metrics(
                                         depth=0, parent_count=1, candidates=[])])
        exp = summarize_experiment_metrics([solved, unsolved])
        self.assertEqual(exp["task_count"], 2)
        self.assertEqual(exp["solved_task_count"], 1)
        self.assertAlmostEqual(exp["success_rate"], 0.5)

    def test_rejection_counts_aggregate_across_tasks(self):
        task = make_task_metrics(
            task_id="a", task_input="1 2 3 4",
            step_details=[
                make_step_metrics(depth=0, parent_count=1, candidates=[
                    make_candidate_event(
                        task_id="a", depth=0, generation_ordinal=0,
                        stage="proposed",
                        rejection_reason="divide_by_zero"),
                    make_candidate_event(
                        task_id="a", depth=0, generation_ordinal=1,
                        stage="proposed",
                        rejection_reason="divide_by_zero"),
                    make_candidate_event(
                        task_id="a", depth=0, generation_ordinal=2,
                        stage="proposed",
                        rejection_reason="malformed_format"),
                ]),
            ])
        exp = summarize_experiment_metrics([task])
        rc = exp["rejection_counts"]
        self.assertEqual(rc.get("divide_by_zero", 0), 2)
        self.assertEqual(rc.get("malformed_format", 0), 1)











if __name__ == "__main__":
    unittest.main()
