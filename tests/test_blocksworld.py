"""Tests for the PlanBench Blocksworld task.

Contract under test:
- ``x`` is a canonical, parseable problem encoding (instance identity, initial
  support/holding state, goal, per-instance depth bound).
- ``y`` is an append-only newline-delimited sequence of canonical actions, one
  action per depth.
- ``canonical_state_key`` is the instance id plus the fully replayed symbolic
  environment state, so distinct trajectories reaching the same state
  deduplicate.
"""

import os
import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.tasks.blocksworld import BlocksworldTask
from federated_mcts.tasks.blocksworld_engine import (
    canonical_key,
    format_action,
    initial_state,
    is_goal_satisfied,
    parse_action_line,
    parse_x,
    replay_state,
    step_state,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_COUNT = 501


def _instance_x(task, instance_id):
    idx = next(
        i for i, record in enumerate(task.data) if record["instance_id"] == instance_id
    )
    return task.get_input(idx)


class TestDataLoading(unittest.TestCase):
    def test_all_official_records_are_present(self):
        task = BlocksworldTask()
        self.assertEqual(len(task), DATA_COUNT)
        ids = [record["id"] for record in task.data]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "planbench-blocksworld-1")
        self.assertEqual(ids[-1], "planbench-blocksworld-501")

    def test_records_have_required_fields_and_invariants(self):
        task = BlocksworldTask()
        for record in task.data:
            self.assertEqual(record["domain"], "blocksworld-4ops")
            self.assertEqual(record["num_blocks"], len(record["blocks"]))
            self.assertEqual(record["max_steps"], 4 * (record["num_blocks"] - 1))
            self.assertIn(record["num_blocks"], (4, 5))
            self.assertIn("source", record)
            for pred in record["init"]:
                self.assertIn(pred[0], ("handempty", "ontable", "on", "clear"))
            for pred in record["goal"]:
                self.assertEqual(pred[0], "on")
                self.assertEqual(len(pred), 3)

    def test_global_steps_is_safe_upper_bound(self):
        task = BlocksworldTask()
        self.assertEqual(task.steps, max(r["max_steps"] for r in task.data))
        self.assertGreaterEqual(task.steps, 10)


class TestParseXBoundary(unittest.TestCase):
    def test_get_input_roundtrips_through_parse_x(self):
        task = BlocksworldTask()
        record = task.data[0]
        parsed = parse_x(task.get_input(0))
        self.assertEqual(parsed["id"], record["id"])
        self.assertEqual(parsed["blocks"], frozenset(record["blocks"]))
        self.assertEqual(parsed["init"], initial_state(parsed))
        self.assertEqual(parsed["goal"], frozenset(map(tuple, record["goal"])))
        self.assertEqual(parsed["max_steps"], record["max_steps"])

    def test_parse_x_is_deterministic_across_calls(self):
        task = BlocksworldTask()
        x = task.get_input(250)
        self.assertEqual(parse_x(x), parse_x(x))


class TestActionParsing(unittest.TestCase):
    def setUp(self):
        self.blocks = frozenset(BlocksworldTask().data[0]["blocks"])

    def test_each_action_type_parses_to_canonical_form(self):
        cases = [
            ("pick-up a", ("pick-up", ("a",))),
            ("put-down b", ("put-down", ("b",))),
            ("stack c d", ("stack", ("c", "d"))),
            ("unstack c d", ("unstack", ("c", "d"))),
        ]
        for line, expected in cases:
            self.assertEqual(parse_action_line(line, self.blocks), expected)
            self.assertEqual(format_action(expected), line)

    def test_common_variants_normalize_to_canonical(self):
        cases = [
            ("pick up a", "pick-up a"),
            ("pickup a", "pick-up a"),
            ("put down b", "put-down b"),
            ("putdown b", "put-down b"),
            ("stack c on d", "stack c d"),
            ("unstack c from d", "unstack c d"),
            ("  stack   c   d  ", "stack c d"),
        ]
        for line, expected in cases:
            action = parse_action_line(line, self.blocks)
            self.assertIsNotNone(action)
            self.assertEqual(format_action(action), expected)

    def test_commentary_and_unknown_blocks_are_rejected(self):
        rejected = [
            "",
            "pick up the red block",
            "stack a b and then pick-up c",
            "stack a b; pick-up c",
            "fly a",
            "pick-up z",
            "stack a",
            "stack a b c",
            "the goal is achieved",
        ]
        for line in rejected:
            self.assertIsNone(parse_action_line(line, self.blocks), line)


class TestReplayDeterminism(unittest.TestCase):
    def test_replay_is_deterministic(self):
        x = _instance_x(BlocksworldTask(), 1)
        plan = "unstack b c\nput-down b\npick-up c\nstack c b"
        self.assertEqual(replay_state(parse_x(x), plan), replay_state(parse_x(x), plan))

    def test_known_plan_reaches_expected_state(self):
        task = BlocksworldTask()
        x = _instance_x(task, 1)
        plan = "unstack b c\nput-down b\npick-up c\nstack c b"
        state = replay_state(parse_x(x), plan)
        self.assertIn(("on", "c", "b"), state)
        self.assertTrue(is_goal_satisfied(parse_x(x), state))

    def test_illegal_action_sequence_marks_trajectory_invalid(self):
        x = _instance_x(BlocksworldTask(), 1)
        bad = "pick-up a\npick-up c"  # arm not empty for the second pick-up
        self.assertIsNone(replay_state(parse_x(x), bad))

    def test_malformed_line_invalidates_the_whole_trajectory(self):
        x = _instance_x(BlocksworldTask(), 1)
        plan = "unstack b c\nput-down b\npick-up c\nstack c b"
        malformed = [
            plan + "\ngoal achieved",
            "unstack b c\npick up the red block\nput-down b",
            "some prose",
            "stack a b; pick-up c",
            "done\nall goals are met",
        ]
        for y in malformed:
            self.assertIsNone(replay_state(parse_x(x), y), y)

    def test_step_state_effect_matches_4ops_semantics(self):
        parsed = parse_x(_instance_x(BlocksworldTask(), 1))
        state = step_state(initial_state(parsed), parse_action_line("pick-up a", parsed["blocks"]))
        self.assertIn(("holding", "a"), state)
        self.assertNotIn(("handempty",), state)
        self.assertNotIn(("clear", "a"), state)


class TestCanonicalStateDedup(unittest.TestCase):
    def test_same_state_via_different_trajectories_deduplicates(self):
        task = BlocksworldTask()
        x = _instance_x(task, 1)
        y1 = "pick-up a"
        y2 = "pick-up a\nput-down a\npick-up a"  # same final state via a cycle
        self.assertEqual(task.canonical_state_key(x, y1), task.canonical_state_key(x, y2))

    def test_different_states_are_distinct(self):
        task = BlocksworldTask()
        x = _instance_x(task, 1)
        self.assertNotEqual(
            task.canonical_state_key(x, "pick-up a"),
            task.canonical_state_key(x, "unstack b c"),
        )

    def test_instance_id_is_part_of_the_key(self):
        task = BlocksworldTask()
        self.assertNotEqual(
            task.canonical_state_key(_instance_x(task, 1), "pick-up a"),
            task.canonical_state_key(_instance_x(task, 2), "pick-up a"),
        )

    def test_canonical_key_is_id_plus_symbolic_state(self):
        task = BlocksworldTask()
        x = _instance_x(task, 1)
        key = task.canonical_state_key(x, "pick-up a")
        self.assertEqual(key[0], "planbench-blocksworld-1")
        self.assertIsInstance(key[1], frozenset)
        self.assertIn(("holding", "a"), key[1])

    def test_malformed_trajectories_share_the_invalid_key(self):
        task = BlocksworldTask()
        x = _instance_x(task, 1)
        invalid = task.canonical_state_key(x, "goal achieved")
        self.assertEqual(invalid[0], "planbench-blocksworld-1")
        self.assertIsNone(invalid[1])
        self.assertEqual(invalid, task.canonical_state_key(x, "some prose"))
        self.assertEqual(invalid, task.canonical_state_key(x, "pick-up a\npick-up c"))


class TestZeroActionSuccess(unittest.TestCase):
    def test_zero_action_plan_succeeds_when_initial_satisfies_goal(self):
        task = BlocksworldTask()
        parsed = parse_x(_instance_x(task, 1))
        trivial = dict(parsed, goal=frozenset({("on", "b", "c")}))  # already in init
        self.assertEqual(replay_state(trivial, ""), trivial["init"])
        self.assertTrue(is_goal_satisfied(trivial, replay_state(trivial, "")))


if __name__ == "__main__":
    unittest.main()
