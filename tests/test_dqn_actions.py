"""Action-space contract for the Budget-Aware DQN controller.

The action map is the Cartesian product beam x joint-rank:
beam in {2, 3, 4, 5} x joint-rank in {False, True} -> exactly 8 actions.
The mapping is exact and invertible; out-of-range inputs raise ValueError.
"""

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.core.dqn.actions import (
    ACTION_BEAMS,
    ACTION_COUNT,
    ACTION_JOINT_RANKS,
    action_for,
    beam_and_joint_rank,
)


class TestActionMapping(unittest.TestCase):
    def test_action_space_is_exactly_eight_actions(self):
        """Given the planned beam and joint-rank sets, when the action space
        is described, then there are exactly 8 actions over beam {2,3,4,5}."""
        self.assertEqual(ACTION_COUNT, 8)
        self.assertEqual(ACTION_BEAMS, (2, 3, 4, 5))
        self.assertEqual(ACTION_JOINT_RANKS, (False, True))

    def test_every_beam_joint_pair_maps_to_a_unique_action(self):
        """Given every beam/joint-rank combination, when mapped to an action
        id, then the ids are unique and cover 0..7 exactly once."""
        seen = set()
        for beam in ACTION_BEAMS:
            for joint_rank in ACTION_JOINT_RANKS:
                action = action_for(beam, joint_rank)
                self.assertTrue(0 <= action < ACTION_COUNT)
                seen.add(action)

        self.assertEqual(seen, set(range(ACTION_COUNT)))

    def test_action_inverse_roundtrip_for_all_actions(self):
        """Given every action id, when converted back to beam/joint-rank and
        mapped forward again, then the original action id is recovered."""
        for action in range(ACTION_COUNT):
            beam, joint_rank = beam_and_joint_rank(action)
            self.assertEqual(action_for(beam, joint_rank), action)

    def test_exact_mapping_table(self):
        """Given the documented mapping, when action_for and
        beam_and_joint_rank are called, then they match the table exactly."""
        expected = {
            (2, False): 0, (2, True): 1,
            (3, False): 2, (3, True): 3,
            (4, False): 4, (4, True): 5,
            (5, False): 6, (5, True): 7,
        }
        for (beam, joint_rank), action in expected.items():
            self.assertEqual(action_for(beam, joint_rank), action)
            self.assertEqual(beam_and_joint_rank(action), (beam, joint_rank))

    def test_invalid_beam_and_action_raise_typed_errors(self):
        """Given a beam outside {2,3,4,5} or an action id outside 0..7, when
        mapped, then ValueError is raised."""
        with self.assertRaises(ValueError):
            action_for(6, False)
        with self.assertRaises(ValueError):
            action_for(1, True)
        with self.assertRaises(ValueError):
            beam_and_joint_rank(8)
        with self.assertRaises(ValueError):
            beam_and_joint_rank(-1)


if __name__ == "__main__":
    unittest.main()
