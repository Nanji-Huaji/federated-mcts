"""Task-hook tests for the PlanBench Blocksworld task.

- ``is_success_state`` compares the replayed state to the goal; it never trusts
  model declarations.
- ``pre_generate_check`` stops generation on success or the instance depth
  bound.
- ``process_generate_result`` parses exactly one action line, rejects
  commentary / multiple actions / unknown blocks / illegal preconditions, and
  appends the normalized action only when legal.
"""

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.tasks.blocksworld import BlocksworldTask


def _instance_x(task, instance_id):
    idx = next(
        i for i, record in enumerate(task.data) if record["instance_id"] == instance_id
    )
    return task.get_input(idx)


class TestSuccessState(unittest.TestCase):
    def test_goal_achieved_is_success(self):
        task = BlocksworldTask()
        x = _instance_x(task, 1)
        plan = "unstack b c\nput-down b\npick-up c\nstack c b"
        self.assertTrue(task.is_success_state(x, plan))

    def test_partial_or_empty_is_not_success(self):
        task = BlocksworldTask()
        x = _instance_x(task, 1)
        self.assertFalse(task.is_success_state(x, ""))
        self.assertFalse(task.is_success_state(x, "pick-up a"))
        self.assertFalse(task.is_success_state(x, "unstack b c\nput-down b"))

    def test_model_declarations_are_never_trusted(self):
        task = BlocksworldTask()
        x = _instance_x(task, 1)
        self.assertFalse(task.is_success_state(x, "goal achieved"))
        self.assertFalse(task.is_success_state(x, "done\nall goals are met"))

    def test_success_depends_on_instance_goal(self):
        task = BlocksworldTask()
        plan = "unstack b c\nput-down b\npick-up c\nstack c b"
        self.assertTrue(task.is_success_state(_instance_x(task, 1), plan))
        self.assertFalse(task.is_success_state(_instance_x(task, 2), plan))


class TestPreGenerateCheck(unittest.TestCase):
    def test_empty_trajectory_needs_generation(self):
        task = BlocksworldTask()
        task.get_input(0)
        self.assertTrue(task.pre_generate_check(""))

    def test_success_stops_generation(self):
        task = BlocksworldTask()
        idx = next(i for i, r in enumerate(task.data) if r["instance_id"] == 1)
        plan = "unstack b c\nput-down b\npick-up c\nstack c b"
        self.assertTrue(task.is_success_state(task.get_input(idx), plan))
        self.assertFalse(task.pre_generate_check(plan))

    def test_depth_bound_stops_generation(self):
        task = BlocksworldTask()
        idx = next(i for i, r in enumerate(task.data) if r["instance_id"] == 1)
        task.get_input(idx)
        bound = task.data[idx]["max_steps"]
        self.assertFalse(task.pre_generate_check("pick-up a\n" * bound))

    def test_depth_below_bound_still_generates(self):
        task = BlocksworldTask()
        idx = next(i for i, r in enumerate(task.data) if r["instance_id"] == 1)
        task.get_input(idx)
        self.assertTrue(task.pre_generate_check("pick-up a"))


class TestProcessGenerateResult(unittest.TestCase):
    def setUp(self):
        self.task = BlocksworldTask()
        self.idx = next(
            i for i, r in enumerate(self.task.data) if r["instance_id"] == 1
        )
        self.x = self.task.get_input(self.idx)

    def test_empty_and_commentary_are_rejected(self):
        for bad in ("", "   ", "goal achieved", "think carefully"):
            ok, y = self.task.process_generate_result(bad, self.x, "", True)
            self.assertFalse(ok)
            self.assertEqual(y, "")

    def test_multi_action_line_is_rejected(self):
        ok, y = self.task.process_generate_result("stack a b; pick-up c", self.x, "", True)
        self.assertFalse(ok)
        self.assertEqual(y, "")

    def test_unknown_block_is_rejected(self):
        ok, y = self.task.process_generate_result("pick-up z", self.x, "", True)
        self.assertFalse(ok)
        self.assertEqual(y, "")

    def test_illegal_precondition_is_rejected(self):
        prev = "pick-up a"
        ok, y = self.task.process_generate_result("pick-up c", self.x, prev, True)
        self.assertFalse(ok)
        self.assertEqual(y, prev)

    def test_legal_action_appends_one_canonical_line(self):
        ok, y = self.task.process_generate_result("pick up a", self.x, "", True)
        self.assertTrue(ok)
        self.assertEqual(y, "pick-up a\n")
        ok2, y2 = self.task.process_generate_result("put down a", self.x, y, True)
        self.assertTrue(ok2)
        self.assertEqual(y2, "pick-up a\nput-down a\n")

    def test_stack_requires_holding_and_clear_target(self):
        ok, y = self.task.process_generate_result("pick-up a", self.x, "", True)
        ok2, y2 = self.task.process_generate_result("stack a b", self.x, y, True)
        self.assertTrue(ok)
        self.assertTrue(ok2)
        self.assertEqual(y2, "pick-up a\nstack a b\n")

    def test_unstack_requires_on_and_clear(self):
        ok, y = self.task.process_generate_result("unstack b c", self.x, "", True)
        self.assertTrue(ok)
        self.assertEqual(y, "unstack b c\n")


class TestRewardMethods(unittest.TestCase):
    """Runner-facing reward contract: reward 1 iff deterministic replay
    satisfies the goal, 0 for incomplete, malformed or illegal trajectories."""

    def setUp(self):
        self.task = BlocksworldTask()
        self.idx = next(
            i for i, r in enumerate(self.task.data) if r["instance_id"] == 1
        )
        self.x = self.task.get_input(self.idx)
        self.plan = "unstack b c\nput-down b\npick-up c\nstack c b"

    def test_runner_facing_reward_methods_exist(self):
        self.assertTrue(callable(self.task.test_output))
        self.assertTrue(callable(self.task.test_output_modify))

    def test_test_output_rewards_goal_satisfied_plan(self):
        self.assertEqual(self.task.test_output(self.idx, self.plan), {"r": 1})

    def test_test_output_zero_for_incomplete_trajectory(self):
        self.assertEqual(self.task.test_output(self.idx, ""), {"r": 0})
        self.assertEqual(self.task.test_output(self.idx, "unstack b c\nput-down b"), {"r": 0})

    def test_test_output_zero_for_malformed_trajectory(self):
        self.assertEqual(self.task.test_output(self.idx, "goal achieved"), {"r": 0})
        self.assertEqual(self.task.test_output(self.idx, self.plan + "\nall done"), {"r": 0})

    def test_test_output_zero_for_illegal_trajectory(self):
        self.assertEqual(self.task.test_output(self.idx, "pick-up a\npick-up c"), {"r": 0})

    def test_test_output_modify_returns_reward_dict_and_output(self):
        r, out = self.task.test_output_modify(self.idx, self.plan)
        self.assertEqual(r, {"r": 1})
        self.assertIsInstance(out, str)
        self.assertTrue(out)
        r0, out0 = self.task.test_output_modify(self.idx, "goal achieved")
        self.assertEqual(r0, {"r": 0})
        self.assertIsInstance(out0, str)

    def test_test_output_modify_zero_action_on_real_data(self):
        r, out = self.task.test_output_modify(self.idx, "")
        self.assertEqual(r, {"r": 0})
        self.assertIsInstance(out, str)


class TestValueAndRankingHooks(unittest.TestCase):
    def test_value_outputs_unwrap_matches_evaluator_labels(self):
        task = BlocksworldTask()
        self.assertEqual(task.value_outputs_unwrap("x", "y", ["sure"]), 20)
        self.assertEqual(task.value_outputs_unwrap("x", "y", ["likely", "sure"]), 21)
        self.assertEqual(task.value_outputs_unwrap("x", "y", ["unlikely"]), 0.1)
        self.assertEqual(task.value_outputs_unwrap("x", "y", ["impossible"]), 0.001)

    def test_joint_rank_prompt_requests_parseable_json(self):
        task = BlocksworldTask()
        x = _instance_x(task, 1)
        prompt = task.joint_rank_prompt_wrap(
            x, ["pick-up a", "unstack b c", "pick-up a\nput-down a\npick-up a"]
        )
        self.assertIn('"ranking"', prompt)
        self.assertIn("0:", prompt)
        self.assertIn("1:", prompt)
        self.assertIn("2:", prompt)

    def test_value_prompt_exposes_state_and_goal(self):
        task = BlocksworldTask()
        prompt = task.value_prompt_wrap(_instance_x(task, 1), "pick-up a")
        self.assertIn("holding a", prompt)  # replayed authoritative state
        self.assertIn("on c b", prompt)  # goal predicate
        state_section = prompt.split("Current state:", 1)[1]
        self.assertNotIn("handempty", state_section)  # holding a, hand not empty


if __name__ == "__main__":
    unittest.main()
