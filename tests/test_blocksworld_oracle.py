"""Tests for the pure Blocksworld symbolic oracle: deterministic legal actions
(four families in order), exact BFS shortest plans (solved/unreachable/cutoff,
visited-on-enqueue, explored counts) and trajectory metrics
(valid/malformed/illegal, success, optimality). In-memory parsed fixtures only.
"""

import sys
import unittest
from dataclasses import FrozenInstanceError

sys.path.insert(0, "src")

from federated_mcts.tasks.blocksworld_engine import format_action, is_goal_satisfied, replay_state
from federated_mcts.tasks.blocksworld_oracle import (
    CUTOFF, ILLEGAL, MALFORMED, SOLVED, UNREACHABLE, VALID,
    ShortestPlanResult, TrajectoryMetrics, evaluate_trajectory,
    legal_actions, shortest_plan,
)


def _parsed(blocks, init, goal):
    return {
        "id": "oracle-fixture",
        "blocks": frozenset(blocks),
        "init": frozenset(init),
        "goal": frozenset(goal),
        "max_steps": 4 * (len(blocks) - 1),
    }


TWO_ON_TABLE = (("handempty",), ("ontable", "a"), ("ontable", "b"),
                ("clear", "a"), ("clear", "b"))
F_ON_AB = _parsed(("a", "b"), TWO_ON_TABLE, {("on", "a", "b")})
F_HOLD_B = _parsed(("a", "b"), TWO_ON_TABLE, {("holding", "b")})
F_TRIVIAL = _parsed(("a", "b"), TWO_ON_TABLE, {("ontable", "a")})
F_INCONSISTENT = _parsed(("a", "b"), TWO_ON_TABLE,
                         {("on", "a", "b"), ("on", "b", "a")})
F_AONB = _parsed(
    ("a", "b", "c"),
    (("handempty",), ("on", "c", "a"), ("clear", "c"),
     ("ontable", "a"), ("ontable", "b"), ("clear", "b")),
    {("on", "a", "b")},
)

HOLDING_A = frozenset({("holding", "a"), ("ontable", "b"), ("clear", "b")})
STACKED_AB = frozenset({("on", "a", "b"), ("ontable", "b"),
                        ("clear", "a"), ("handempty",)})


class TestLegalActions(unittest.TestCase):
    def test_pickup_family_from_table(self):
        self.assertEqual(legal_actions(F_ON_AB, F_ON_AB["init"]),
                         (("pick-up", ("a",)), ("pick-up", ("b",))))

    def test_family_order_pick_put_down_stack_unstack(self):
        self.assertEqual(legal_actions(F_ON_AB, HOLDING_A),
                         (("put-down", ("a",)), ("stack", ("a", "b"))))

    def test_blocks_are_lexicographic_within_family(self):
        state = frozenset({("handempty",), ("ontable", "a"), ("ontable", "b"),
                           ("ontable", "c"), ("clear", "a"), ("clear", "b"),
                           ("clear", "c")})
        parsed = _parsed(("a", "b", "c"), state, set())
        self.assertEqual(legal_actions(parsed, state),
                         (("pick-up", ("a",)), ("pick-up", ("b",)),
                          ("pick-up", ("c",))))

    def test_unstack_family_from_stacked_state(self):
        self.assertEqual(legal_actions(F_ON_AB, STACKED_AB),
                         (("unstack", ("a", "b")),))

    def test_same_block_pairs_excluded_by_step_state(self):
        actions = legal_actions(F_ON_AB, HOLDING_A)
        self.assertNotIn(("stack", ("a", "a")), actions)
        self.assertNotIn(("unstack", ("a", "a")), actions)

    def test_stable_order_and_no_duplicates(self):
        init = F_ON_AB["init"]
        self.assertEqual(legal_actions(F_ON_AB, init), legal_actions(F_ON_AB, init))
        self.assertEqual(len(legal_actions(F_ON_AB, init)),
                         len(set(legal_actions(F_ON_AB, init))))

    def test_inputs_are_not_mutated(self):
        before = dict(F_ON_AB)
        init = F_ON_AB["init"]
        actions = legal_actions(F_ON_AB, init)
        self.assertIsInstance(actions, tuple)
        self.assertEqual(F_ON_AB, before)
        self.assertEqual(init, F_ON_AB["init"])

    def test_all_four_operators_in_shortest_plan(self):
        result = shortest_plan(F_AONB)
        self.assertEqual(result.status, SOLVED)
        self.assertEqual(result.plan,
                         (("unstack", ("c", "a")), ("put-down", ("c",)),
                          ("pick-up", ("a",)), ("stack", ("a", "b"))))
        self.assertEqual({name for name, _ in result.plan},
                         {"pick-up", "put-down", "stack", "unstack"})


class TestShortestPlan(unittest.TestCase):
    def test_solved_at_start_returns_empty_plan(self):
        result = shortest_plan(F_TRIVIAL)
        self.assertEqual(result.status, SOLVED)
        self.assertEqual(result.plan, ())
        self.assertEqual(result.explored, 1)

    def test_one_step_plan(self):
        result = shortest_plan(F_HOLD_B)
        self.assertEqual(result.status, SOLVED)
        self.assertEqual(result.plan, (("pick-up", ("b",)),))
        self.assertEqual(result.explored, 3)

    def test_two_step_plan_shortest_and_deterministic(self):
        result = shortest_plan(F_ON_AB)
        self.assertEqual(result.status, SOLVED)
        self.assertEqual(result.plan,
                         (("pick-up", ("a",)), ("stack", ("a", "b"))))
        self.assertEqual(result.explored, 4)
        self.assertEqual(result, shortest_plan(F_ON_AB))

    def test_plan_replays_to_goal_state(self):
        result = shortest_plan(F_AONB)
        y = "\n".join(format_action(a) for a in result.plan) + "\n"
        self.assertTrue(is_goal_satisfied(F_AONB, replay_state(F_AONB, y)))

    def test_unreachable_inconsistent_goal(self):
        result = shortest_plan(F_INCONSISTENT)
        self.assertEqual(result.status, UNREACHABLE)
        self.assertEqual(result.plan, ())
        self.assertEqual(result.explored, 5)

    def test_cutoff_suppresses_unvisited_successor(self):
        # Bound suppresses a frontier state that still has unvisited
        # successors: expansion is genuinely blocked -> cutoff.
        self.assertEqual(shortest_plan(F_ON_AB, max_depth=0).status, CUTOFF)
        self.assertEqual(shortest_plan(F_ON_AB, max_depth=1).status, CUTOFF)
        self.assertEqual(shortest_plan(F_INCONSISTENT, max_depth=1).status, CUTOFF)

    def test_dead_end_frontier_at_bound_is_unreachable_not_cutoff(self):
        # At max_depth=2 the frontier is {S_ab, S_ba}; every legal successor
        # is already visited, so nothing is suppressed and the graph is
        # exhausted: unreachable, not cutoff.
        self.assertEqual(shortest_plan(F_INCONSISTENT, max_depth=2).status, UNREACHABLE)

    def test_unreachable_only_after_graph_exhaustion(self):
        self.assertEqual(shortest_plan(F_INCONSISTENT, max_depth=3).status, UNREACHABLE)
        self.assertEqual(shortest_plan(F_INCONSISTENT).status, UNREACHABLE)

    def test_max_depth_equal_to_plan_length_still_solves(self):
        self.assertEqual(shortest_plan(F_ON_AB, max_depth=2).status, SOLVED)

    def test_results_are_immutable(self):
        result = shortest_plan(F_ON_AB)
        self.assertIsInstance(result.plan, tuple)
        self.assertTrue(ShortestPlanResult.__dataclass_params__.frozen)
        with self.assertRaises(FrozenInstanceError):
            result.explored = 0


class TestEvaluateTrajectory(unittest.TestCase):
    OPTIMAL = "pick-up a\nstack a b"
    SUBOPTIMAL = "pick-up a\nput-down a\npick-up a\nstack a b"

    def test_empty_trajectory_valid_unsuccessful(self):
        m = evaluate_trajectory(F_ON_AB, "")
        self.assertEqual(m.status, VALID)
        self.assertFalse(m.success)
        self.assertEqual(m.submitted_length, 0)
        self.assertEqual(m.legal_count, 0)
        self.assertEqual(m.plan_length, 0)
        self.assertEqual(m.legal_rate, 1.0)
        self.assertIsNone(m.first_failure_index)

    def test_valid_unsuccessful(self):
        # A fully valid trajectory that misses the goal: plan_length is the
        # actual legal action count (the valid prefix); excess/ratio stay None
        # because the trajectory did not succeed.
        m = evaluate_trajectory(F_ON_AB, "pick-up a\nput-down a")
        self.assertEqual(m.status, VALID)
        self.assertFalse(m.success)
        self.assertEqual(m.plan_length, 2)
        self.assertEqual(m.optimal_length, 2)
        self.assertEqual(m.legal_rate, 1.0)
        self.assertIsNone(m.excess_steps)
        self.assertIsNone(m.optimality_ratio)

    def test_successful_optimal(self):
        m = evaluate_trajectory(F_ON_AB, self.OPTIMAL)
        self.assertTrue(m.success)
        self.assertEqual(m.status, VALID)
        self.assertEqual(m.submitted_length, 2)
        self.assertEqual(m.legal_count, 2)
        self.assertEqual(m.plan_length, 2)
        self.assertEqual(m.optimal_length, 2)
        self.assertEqual(m.excess_steps, 0)
        self.assertEqual(m.optimality_ratio, 1.0)

    def test_successful_suboptimal(self):
        # 4 legal actions reaching the goal: actual plan_length is 4, the
        # optimum is 2, so excess is 2 and the ratio is 4/2 = 2.0.
        m = evaluate_trajectory(F_ON_AB, self.SUBOPTIMAL)
        self.assertTrue(m.success)
        self.assertEqual(m.plan_length, 4)
        self.assertEqual(m.optimal_length, 2)
        self.assertEqual(m.excess_steps, 2)
        self.assertEqual(m.optimality_ratio, 2.0)

    def test_malformed_stops_at_parse_failure(self):
        # The 2 legal actions before the junk line are counted, but a
        # malformed trajectory has no actual plan, so plan_length is None;
        # excess/ratio stay None since the trajectory failed.
        m = evaluate_trajectory(F_ON_AB, self.OPTIMAL + "\ngoal achieved")
        self.assertEqual(m.status, MALFORMED)
        self.assertEqual(m.first_failure_index, 2)
        self.assertEqual(m.legal_count, 2)
        self.assertIsNone(m.plan_length)
        self.assertFalse(m.success)
        self.assertIsNone(m.excess_steps)
        self.assertIsNone(m.optimality_ratio)
        self.assertAlmostEqual(m.legal_rate, 2 / 3)

    def test_malformed_without_prior_success(self):
        m = evaluate_trajectory(F_ON_AB, "pick-up a\nput-down a\ngoal achieved")
        self.assertEqual(m.status, MALFORMED)
        self.assertEqual(m.first_failure_index, 2)
        self.assertIsNone(m.plan_length)
        self.assertFalse(m.success)

    def test_illegal_stops_at_transition_failure(self):
        m = evaluate_trajectory(F_ON_AB, "pick-up a\npick-up b")
        self.assertEqual(m.status, ILLEGAL)
        self.assertEqual(m.first_failure_index, 1)
        self.assertEqual(m.legal_count, 1)
        self.assertIsNone(m.plan_length)
        self.assertFalse(m.success)
        self.assertIsNone(m.excess_steps)
        self.assertIsNone(m.optimality_ratio)
        self.assertAlmostEqual(m.legal_rate, 0.5)

    def test_explicit_optimal_overrides_computation(self):
        m = evaluate_trajectory(F_ON_AB, self.SUBOPTIMAL, optimal=4)
        self.assertEqual(m.plan_length, 4)
        self.assertEqual(m.optimal_length, 4)
        self.assertEqual(m.excess_steps, 0)
        self.assertEqual(m.optimality_ratio, 1.0)

    def test_zero_optimum_handling(self):
        empty = evaluate_trajectory(F_TRIVIAL, "")
        self.assertTrue(empty.success)
        self.assertEqual(empty.plan_length, 0)
        self.assertEqual(empty.optimal_length, 0)
        self.assertEqual(empty.excess_steps, 0)
        self.assertEqual(empty.optimality_ratio, 1.0)
        detour = evaluate_trajectory(F_TRIVIAL, "pick-up a\nput-down a")
        self.assertTrue(detour.success)
        self.assertEqual(detour.plan_length, 2)
        self.assertEqual(detour.optimal_length, 0)
        self.assertEqual(detour.excess_steps, 2)
        self.assertEqual(detour.optimality_ratio, float("inf"))

    def test_unreachable_goal_metrics(self):
        # No plan exists, so the optimum is unknown (None); plan_length stays
        # the legal prefix count and excess/ratio are None.
        m = evaluate_trajectory(F_INCONSISTENT, "pick-up a")
        self.assertEqual(m.status, VALID)
        self.assertFalse(m.success)
        self.assertEqual(m.plan_length, 1)
        self.assertIsNone(m.optimal_length)
        self.assertIsNone(m.excess_steps)
        self.assertIsNone(m.optimality_ratio)

    def test_metrics_are_immutable(self):
        m = evaluate_trajectory(F_ON_AB, self.OPTIMAL)
        self.assertTrue(TrajectoryMetrics.__dataclass_params__.frozen)
        with self.assertRaises(FrozenInstanceError):
            m.success = False


if __name__ == "__main__":
    unittest.main()
