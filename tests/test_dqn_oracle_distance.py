import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.core.dqn.oracle_distance import min_remaining_distance
from federated_mcts.core.dqn.rewards import oracle_distance_reward
from federated_mcts.core.dqn.session import DqnSearchSession
from federated_mcts.tasks.blocksworld import BlocksworldTask


class _Controller:
    pass


class TestOracleDistance(unittest.TestCase):
    def test_remaining_distance_uses_best_valid_trajectory(self):
        task = BlocksworldTask()
        x = task.get_input(0)
        self.assertEqual(min_remaining_distance(x, ["bad", "unstack b c\n"]), 3)

    def test_distance_reward_is_bounded_and_signed(self):
        self.assertEqual(oracle_distance_reward(4, 2), 0.5)
        self.assertEqual(oracle_distance_reward(2, 4), -0.5)
        self.assertEqual(oracle_distance_reward(2, 2), 0.0)
        self.assertEqual(oracle_distance_reward(10, 0), 1.0)
        self.assertEqual(oracle_distance_reward(None, 0), 0.0)

    def test_blocksworld_transition_carries_components(self):
        session = DqnSearchSession(
            _Controller(), oracle_distance_reward_enabled=True
        )
        session._episode.open([0.0], 0, distance_before=4)
        session._episode.pending["distance_after"] = 2
        transition = session._episode.close(next_state=[1.0], done=False, success=False)
        self.assertEqual(transition["distance_delta"], 2)
        self.assertEqual(transition["distance_reward"], 0.5)
        self.assertEqual(transition["reward_components"]["distance"], 0.5)
        self.assertEqual(transition["reward"], 0.5)

    def test_legacy_transition_has_no_oracle_keys(self):
        session = DqnSearchSession(_Controller())
        session._episode.open([0.0], 0, distance_before=None)
        transition = session._episode.close(next_state=[1.0], done=False, success=False)
        self.assertEqual(
            set(transition),
            {"state", "action", "reward", "next_state", "done", "beam", "joint_rank"},
        )


if __name__ == "__main__":
    unittest.main()
