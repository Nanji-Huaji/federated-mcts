"""Regression tests for task assignment strategy sole-root replication.

PRODUCTION DEFECT: All three built-in strategies (RoundRobinStrategy,
DifficultyBasedStrategy, ContextualBanditStrategy) must replicate a sole
root candidate [""] to every available model so each independently
explores from the root.  Multi-candidate routing behaviour must be
preserved.
"""

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.federation.task_assign import (
    RoundRobinStrategy,
    DifficultyBasedStrategy,
    ContextualBanditStrategy,
)


class TestRoundRobinSoleRootReplication(unittest.TestCase):

    def setUp(self):
        self.strategy = RoundRobinStrategy(eval_client="remote")

    def test_sole_root_replicated_to_all_models_two_models(self):
        """Given 2 models and ys=[""], every model should receive [""]."""
        assignments = self.strategy.assign(
            ["local", "remote"], [""], {"step": 0, "total_steps": 3},
        )
        ys_by_client = {a["solve_client"]: a["ys"] for a in assignments}
        self.assertEqual(
            ys_by_client.get("remote"), [""],
            "DEFECT: sole root [\"\"] not replicated to remote model.",
        )

    def test_sole_root_replicated_to_all_models_three_models(self):
        """Given 3 models and ys=[""], every model should receive [""]."""
        assignments = self.strategy.assign(
            ["a", "b", "c"], [""], {"step": 0, "total_steps": 3},
        )
        ys_by_client = {a["solve_client"]: a["ys"] for a in assignments}
        self.assertEqual(ys_by_client.get("a"), [""])
        self.assertEqual(
            ys_by_client.get("b"), [""],
            "DEFECT: sole root not replicated to model b.",
        )
        self.assertEqual(ys_by_client.get("c"), [""])

    def test_multi_candidate_round_robin_deterministic(self):
        """Given 2 models and 4 candidates, round-robin distributes
        deterministically: model[0]=[c0,c2], model[1]=[c1,c3]."""
        ys = ["c0", "c1", "c2", "c3"]
        assignments = self.strategy.assign(
            ["local", "remote"], ys, {"step": 1, "total_steps": 3},
        )
        ys_by_client = {a["solve_client"]: a["ys"] for a in assignments}
        self.assertEqual(ys_by_client["local"], ["c0", "c2"])
        self.assertEqual(ys_by_client["remote"], ["c1", "c3"])

    def test_empty_ys_assigns_empty_buckets(self):
        assignments = self.strategy.assign(
            ["local", "remote"], [], {"step": 0},
        )
        self.assertEqual(len(assignments), 2)
        for a in assignments:
            self.assertEqual(a["ys"], [])


class TestDifficultyBasedSoleRootReplication(unittest.TestCase):

    def setUp(self):
        self.strategy = DifficultyBasedStrategy(
            total_steps=3, local_ratio_at_easy=0.8, local_ratio_at_hard=0.0,
        )

    def test_sole_root_replicated_to_all_models_easy_step(self):
        assignments = self.strategy.assign(
            ["local", "remote"], [""], {"step": 0, "total_steps": 3},
        )
        ys_by_client = {a["solve_client"]: a["ys"] for a in assignments}
        self.assertEqual(ys_by_client.get("local"), [""])
        self.assertEqual(
            ys_by_client.get("remote"), [""],
            "DEFECT: sole root not replicated to remote in DifficultyBasedStrategy.",
        )

    def test_sole_root_replicated_to_all_models_hard_step(self):
        assignments = self.strategy.assign(
            ["local", "remote"], [""], {"step": 2, "total_steps": 3},
        )
        ys_by_client = {a["solve_client"]: a["ys"] for a in assignments}
        self.assertEqual(ys_by_client.get("local"), [""])
        self.assertEqual(ys_by_client.get("remote"), [""])

    def test_multi_candidate_difficulty_routing_preserved(self):
        ys = ["c0", "c1", "c2", "c3", "c4"]
        assignments = self.strategy.assign(
            ["local", "remote"], ys, {"step": 0, "total_steps": 3},
        )
        ys_by_client = {a["solve_client"]: a["ys"] for a in assignments}
        self.assertGreater(len(ys_by_client.get("local", [])), 0)
        self.assertGreater(len(ys_by_client.get("remote", [])), 0)
        combined = ys_by_client.get("local", []) + ys_by_client.get("remote", [])
        self.assertEqual(len(combined), len(ys))


class TestContextualBanditSoleRootReplication(unittest.TestCase):

    def setUp(self):
        self.strategy = ContextualBanditStrategy(total_steps=3)

    def test_sole_root_replicated_to_all_models(self):
        assignments = self.strategy.assign(
            ["local", "remote"], [""], {"step": 0, "total_steps": 3},
        )
        ys_by_client = {a["solve_client"]: a["ys"] for a in assignments}
        self.assertEqual(ys_by_client.get("local"), [""])
        self.assertEqual(
            ys_by_client.get("remote"), [""],
            "DEFECT: sole root not replicated to remote in ContextualBanditStrategy.",
        )

    def test_multi_candidate_bandit_routing_preserved(self):
        ys = ["c0", "c1", "c2", "c3"]
        assignments = self.strategy.assign(
            ["local", "remote"], ys, {"step": 0, "total_steps": 3},
        )
        ys_by_client = {a["solve_client"]: a["ys"] for a in assignments}
        combined = sum((a["ys"] for a in assignments), [])
        self.assertEqual(len(combined), len(ys))


if __name__ == "__main__":
    unittest.main()
