"""Game24 routing/integration tests: exact validation through Game24Task methods.

Covers:
- test_output_modify reward for valid and non-integral trajectories
- is_success_state acceptance and rejection
- process_generate_result acceptance and rejection
- Full trajectory baseline preservation
"""

import sys
import unittest

sys.path.insert(0, "src")

# ── Deterministic-import guard ────────────────────────────────────────
# When a prior import failure leaves partially-loaded modules in
# sys.modules (e.g. federated_mcts.models.api_client survives while
# federated_mcts.models.__init__ failed), a subsequent "from .tasks.game24
# import Game24Task" can silently succeed — producing false-pass tests.
#
# Clean up stale submodule cache so the import below either works
# end-to-end (all deps present) or fails before test discovery.
_MODEL_STALE_SUBMODULES = [
    "federated_mcts.models.api_client",
    "federated_mcts.models.usage",
]
for _m in _MODEL_STALE_SUBMODULES:
    if _m in sys.modules and "federated_mcts.models" not in sys.modules:
        del sys.modules[_m]

from federated_mcts.tasks.game24 import Game24Task  # noqa: E402


class TestRouteThroughGame24Task(unittest.TestCase):
    """Verify exact validation flows through Game24Task methods."""

    def setUp(self):
        self.task = Game24Task()

    def test_is_success_state_rejects_non_integral_trajectory(self):
        """A trajectory containing 3/2=1 must fail is_success_state."""
        bad_traj = ("3 / 2 = 1 (left: 1 4 8)\n"
                    "1 * 4 = 4 (left: 4 8)\n"
                    "4 * 8 = 32 (left: 32)\n")
        self.assertFalse(
            self.task.is_success_state("2 3 4 8", bad_traj),
            "DEFECT: trajectory with 3/2=1 floor must fail is_success_state",
        )

    def test_is_success_state_accepts_integral_trajectory(self):
        """A fully integral valid trajectory must pass is_success_state."""
        good_traj = "1 + 1 = 2 (left: 1 2 8)\n2 + 1 = 3 (left: 3 8)\n3 * 8 = 24 (left: 24)\n"
        self.assertTrue(
            self.task.is_success_state("1 1 1 8", good_traj),
            "DEFECT: valid integral trajectory must pass is_success_state",
        )

    def test_test_output_modify_rejects_non_integral(self):
        """test_output_modify must return r=0 for non-integral division."""
        result, _ = self.task.test_output_modify(
            21,
            "3 / 2 = 1 (left: 1 1 4)\n1 + 1 = 2 (left: 2 4)\n2 * 4 = 8 (left: 8)\n",
        )
        self.assertEqual(result["r"], 0,
                         "DEFECT: non-integral division must yield reward 0")

    def test_process_generate_result_rejects_non_integral(self):
        """process_generate_result with non-integral division must reject."""
        is_correct, _ = self.task.process_generate_result(
            "3 / 2 = 1 (left: 1 4 8)", "2 3 4 8", "", True,
        )
        self.assertFalse(is_correct,
                         "DEFECT: process_generate_result must reject 3/2=1")

    def test_process_generate_result_accepts_integral(self):
        """process_generate_result with integral division must accept."""
        is_correct, updated = self.task.process_generate_result(
            "6 / 3 = 2 (left: 2 4 8)", "4 6 3 8", "", True,
        )
        self.assertTrue(is_correct,
                        "DEFECT: process_generate_result must accept 6/3=2")
        self.assertIn("2 4 8", updated)

    def test_reward_preserved_for_valid_integral_via_test_output_modify(self):
        """Full integral trajectory reaching 24 gets r=1."""
        result3, _ = self.task.test_output_modify(
            3,
            "1 + 1 = 2 (left: 1 2 8)\n2 + 1 = 3 (left: 3 8)\n3 * 8 = 24 (left: 24)\n",
        )
        self.assertEqual(result3["r"], 1,
                         "DEFECT: full trajectory reaching 24 must get r=1")

    def test_full_trajectory_preserves_reward(self):
        """A valid 3-step integral trajectory yields reward 1 via test_output_modify."""
        result, _ = self.task.test_output_modify(
            3,
            "1 + 1 = 2 (left: 1 2 8)\n2 + 1 = 3 (left: 3 8)\n3 * 8 = 24 (left: 24)\n",
        )
        self.assertEqual(result["r"], 1,
                         "DEFECT: valid integral trajectory must get reward 1")


if __name__ == "__main__":
    unittest.main()
