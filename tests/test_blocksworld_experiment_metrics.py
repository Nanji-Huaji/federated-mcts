"""Tests for the shared Blocksworld Oracle experiment metrics module.

Covers the pure per-input metric computation, the runner-facing task hook
(Blocksworld vs non-Blocksworld discrimination without importing the Oracle),
the experiment aggregate with explicit denominators, and the JSON boundary:
``json.dumps(..., allow_nan=False)`` must always succeed, including the
infinite-ratio (zero-optimum positive detour) case.
"""

import json
import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.blocksworld_metrics import (
    METRICS_VERSION,
    evaluate_input_oracle_metrics,
    merge_oracle_metrics,
    summarize_oracle_tasks,
    task_oracle_metrics,
)
from federated_mcts.tasks.blocksworld_engine import format_action, parse_x
from federated_mcts.tasks.blocksworld_oracle import SOLVED, shortest_plan

OPTIMAL = "pick-up a\nstack a b"
SUBOPTIMAL4 = "pick-up a\nput-down a\npick-up a\nstack a b"
MALFORMED = "pick-up a\nstack a b\ngoal achieved"


def _x(blocks=("a", "b"), init=None, goal=None, max_steps=4, task_id="fixture-1"):
    init = init or (("handempty",), ("ontable", "a"), ("ontable", "b"),
                    ("clear", "a"), ("clear", "b"))
    goal = goal or (("on", "a", "b"),)
    lines = ["Instance: %s" % task_id, "Blocks: %s" % " ".join(blocks), "Initial:"]
    for pred in init:
        lines.append("- %s" % " ".join(pred))
    lines.append("Goal:")
    for pred in goal:
        lines.append("- %s" % " ".join(pred))
    lines.append("MaxSteps: %d" % max_steps)
    return "\n".join(lines) + "\n"


F_AONB = _x()  # goal "on a b", exact optimum is 2 steps
F_TRIVIAL = _x(goal=(("ontable", "a"),), task_id="fixture-trivial")  # optimum 0


class TestTaskOracleMetricsDiscrimination(unittest.TestCase):
    def test_non_blocksworld_returns_none_without_importing_oracle(self):
        sys.modules.pop("federated_mcts.tasks.blocksworld_oracle", None)

        class _OtherTask:
            def get_input(self, idx):
                return "not blocksworld"

        self.assertIsNone(task_oracle_metrics(_OtherTask(), 0, [""]))
        self.assertNotIn("federated_mcts.tasks.blocksworld_oracle", sys.modules)


class TestEvaluateInputMetrics(unittest.TestCase):
    def test_empty_outputs(self):
        metrics = evaluate_input_oracle_metrics(F_AONB, [])
        self.assertEqual(metrics["version"], METRICS_VERSION)
        self.assertEqual(metrics["candidate_count"], 0)
        self.assertEqual(metrics["success_count"], 0)
        self.assertFalse(metrics["any_success"])
        self.assertEqual(metrics["mean_legal_action_rate"], 0.0)
        self.assertEqual(metrics["optimal_length"], 2)
        self.assertIsNone(metrics["best_successful_index"])
        self.assertIsNone(metrics["best_successful"])
        self.assertEqual(metrics["candidates"], [])
        json.dumps(metrics, allow_nan=False)

    def test_optimal_success(self):
        metrics = evaluate_input_oracle_metrics(F_AONB, [OPTIMAL])
        candidate = metrics["candidates"][0]
        self.assertTrue(candidate["success"])
        self.assertEqual(candidate["status"], "valid")
        self.assertEqual(candidate["submitted_length"], 2)
        self.assertEqual(candidate["legal_count"], 2)
        self.assertEqual(candidate["legal_rate"], 1.0)
        self.assertIsNone(candidate["first_failure_index"])
        self.assertEqual(candidate["plan_length"], 2)
        self.assertEqual(candidate["optimal_length"], 2)
        self.assertEqual(candidate["excess_steps"], 0)
        self.assertEqual(candidate["optimality_ratio"], 1.0)
        self.assertFalse(candidate["optimality_ratio_infinite"])
        self.assertEqual(metrics["success_count"], 1)
        self.assertTrue(metrics["any_success"])
        self.assertEqual(metrics["best_successful_index"], 0)

    def test_four_step_suboptimal_success(self):
        metrics = evaluate_input_oracle_metrics(F_AONB, [SUBOPTIMAL4])
        candidate = metrics["candidates"][0]
        self.assertTrue(candidate["success"])
        self.assertEqual(candidate["plan_length"], 4)
        self.assertEqual(candidate["optimal_length"], 2)
        self.assertEqual(candidate["excess_steps"], 2)
        self.assertEqual(candidate["optimality_ratio"], 2.0)

    def test_malformed_candidate(self):
        metrics = evaluate_input_oracle_metrics(F_AONB, [MALFORMED])
        candidate = metrics["candidates"][0]
        self.assertEqual(candidate["status"], "malformed")
        self.assertEqual(candidate["first_failure_index"], 2)
        self.assertEqual(candidate["legal_count"], 2)
        self.assertIsNone(candidate["plan_length"])
        self.assertFalse(candidate["success"])
        self.assertIsNone(candidate["optimality_ratio"])
        self.assertFalse(candidate["optimality_ratio_infinite"])

    def test_illegal_candidate(self):
        metrics = evaluate_input_oracle_metrics(F_AONB, ["pick-up a\npick-up b"])
        candidate = metrics["candidates"][0]
        self.assertEqual(candidate["status"], "illegal")
        self.assertEqual(candidate["first_failure_index"], 1)
        self.assertEqual(candidate["legal_count"], 1)
        self.assertFalse(candidate["success"])

    def test_best_successful_by_shortest_plan_then_lowest_index(self):
        metrics = evaluate_input_oracle_metrics(F_AONB, [SUBOPTIMAL4, OPTIMAL, OPTIMAL])
        self.assertEqual(metrics["success_count"], 3)
        self.assertEqual(metrics["best_successful_index"], 1)
        self.assertEqual(metrics["best_successful"]["plan_length"], 2)
        self.assertEqual(metrics["best_successful"]["output_index"], 1)
        self.assertEqual(metrics["best_successful"]["optimality_ratio"], 1.0)
        json.dumps(metrics, allow_nan=False)

    def test_no_success(self):
        metrics = evaluate_input_oracle_metrics(F_AONB, ["pick-up a\nput-down a"])
        self.assertEqual(metrics["success_count"], 0)
        self.assertFalse(metrics["any_success"])
        self.assertIsNone(metrics["best_successful_index"])
        self.assertIsNone(metrics["best_successful"])
        self.assertEqual(metrics["candidate_count"], 1)

    def test_infinite_ratio_is_json_safe(self):
        metrics = evaluate_input_oracle_metrics(F_TRIVIAL, ["pick-up a\nput-down a"])
        candidate = metrics["candidates"][0]
        self.assertTrue(candidate["success"])
        self.assertEqual(candidate["optimal_length"], 0)
        self.assertEqual(candidate["excess_steps"], 2)
        self.assertIsNone(candidate["optimality_ratio"])
        self.assertTrue(candidate["optimality_ratio_infinite"])
        json.dumps(metrics, allow_nan=False)

    def test_zero_optimum_zero_step_success_is_finite(self):
        metrics = evaluate_input_oracle_metrics(F_TRIVIAL, [""])
        candidate = metrics["candidates"][0]
        self.assertTrue(candidate["success"])
        self.assertEqual(candidate["optimality_ratio"], 1.0)
        self.assertFalse(candidate["optimality_ratio_infinite"])


class TestSummarizeOracleTasks(unittest.TestCase):
    def test_empty_list_is_json_safe(self):
        aggregate = summarize_oracle_tasks([])
        self.assertEqual(aggregate["version"], METRICS_VERSION)
        self.assertEqual(aggregate["task_count"], 0)
        self.assertEqual(aggregate["solved_task_count"], 0)
        self.assertEqual(aggregate["success_rate"], 0.0)
        self.assertEqual(aggregate["candidate_count"], 0)
        self.assertEqual(aggregate["successful_candidate_count"], 0)
        self.assertEqual(aggregate["legal_rate_task_count"], 0)
        self.assertEqual(aggregate["mean_legal_action_rate"], 0.0)
        self.assertEqual(aggregate["finite_ratio_task_count"], 0)
        self.assertEqual(aggregate["mean_optimality_ratio"], 0.0)
        self.assertEqual(aggregate["mean_excess_steps"], 0.0)
        json.dumps(aggregate, allow_nan=False)

    def test_exact_denominators(self):
        # t1: one optimal success + one malformed candidate (legal rate 2/3)
        t1 = evaluate_input_oracle_metrics(F_AONB, [OPTIMAL, MALFORMED])
        # t2: zero-step success on an already-satisfied goal
        t2 = evaluate_input_oracle_metrics(F_TRIVIAL, [""])
        # t3: valid but no success
        t3 = evaluate_input_oracle_metrics(F_AONB, ["pick-up a"])
        aggregate = summarize_oracle_tasks([t1, t2, t3])
        self.assertEqual(aggregate["task_count"], 3)
        self.assertEqual(aggregate["solved_task_count"], 2)
        self.assertAlmostEqual(aggregate["success_rate"], 2 / 3)
        self.assertEqual(aggregate["candidate_count"], 4)
        self.assertEqual(aggregate["successful_candidate_count"], 2)
        self.assertEqual(aggregate["legal_rate_task_count"], 3)
        self.assertAlmostEqual(aggregate["mean_legal_action_rate"],
                               (5 / 6 + 1.0 + 1.0) / 3)
        self.assertEqual(aggregate["finite_ratio_task_count"], 2)
        self.assertAlmostEqual(aggregate["mean_optimality_ratio"], 1.0)
        self.assertAlmostEqual(aggregate["mean_excess_steps"], 0.0)
        json.dumps(aggregate, allow_nan=False)

    def test_infinite_ratio_tasks_excluded_from_finite_means(self):
        infinite = evaluate_input_oracle_metrics(F_TRIVIAL, ["pick-up a\nput-down a"])
        aggregate = summarize_oracle_tasks([infinite])
        self.assertEqual(aggregate["solved_task_count"], 1)
        self.assertEqual(aggregate["success_rate"], 1.0)
        self.assertEqual(aggregate["finite_ratio_task_count"], 0)
        self.assertEqual(aggregate["mean_optimality_ratio"], 0.0)
        self.assertEqual(aggregate["mean_excess_steps"], 0.0)
        json.dumps(aggregate, allow_nan=False)


class TestMergeOracleMetrics(unittest.TestCase):
    def test_empty_accumulator_leaves_res_json_untouched(self):
        res_json = {"avg_sum": 0.0}
        result = merge_oracle_metrics([], res_json)
        self.assertIs(result, res_json)
        self.assertNotIn("blocksworld_oracle", res_json)

    def test_non_empty_accumulator_attaches_aggregate(self):
        task_metrics = evaluate_input_oracle_metrics(F_AONB, [OPTIMAL])
        res_json = {"avg_sum": 1.0}
        result = merge_oracle_metrics([task_metrics], res_json)
        self.assertIs(result, res_json)
        self.assertIn("blocksworld_oracle", res_json)
        self.assertEqual(res_json["blocksworld_oracle"]["task_count"], 1)
        self.assertEqual(res_json["blocksworld_oracle"]["solved_task_count"], 1)
        self.assertEqual(res_json["blocksworld_oracle"]["success_rate"], 1.0)
        json.dumps(res_json, allow_nan=False)


class TestRunnerInjectionOffline(unittest.TestCase):
    def test_real_blocksworld_task_injection(self):
        from federated_mcts.tasks import get_task

        task = get_task("blocksworld")
        x = task.get_input(0)
        optimum = shortest_plan(parse_x(x))
        self.assertEqual(optimum.status, SOLVED)
        optimal_y = "".join(format_action(action) + "\n" for action in optimum.plan)
        metrics = task_oracle_metrics(task, 0, [optimal_y, "unrelated-token\n"])
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["task_idx"], 0)
        self.assertEqual(metrics["optimal_length"], len(optimum.plan))
        self.assertEqual(metrics["candidate_count"], 2)
        self.assertEqual(metrics["success_count"], 1)
        self.assertEqual(metrics["best_successful_index"], 0)
        self.assertIsNone(metrics["best_successful"]["first_failure_index"])
        json.dumps(metrics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
