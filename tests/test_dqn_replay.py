"""Replay-buffer contract for the Budget-Aware DQN trainer.

The buffer evicts the oldest transition once capacity is exceeded and
sample() returns typed tensors whose shapes match the requested batch size
and the state dimension.
"""

import sys
import unittest

sys.path.insert(0, "src")

import torch

from federated_mcts.core.dqn.replay_buffer import ReplayBuffer

_STATE_DIM = 12


def _push(buffer, index, done=False):
    buffer.push(
        state=torch.full((_STATE_DIM,), float(index)),
        action=index % 8,
        reward=float(index),
        next_state=torch.full((_STATE_DIM,), float(index + 1)),
        done=done,
    )


class TestReplayBuffer(unittest.TestCase):
    def test_capacity_evicts_oldest_transitions(self):
        """Given capacity 4 and 6 pushes, when sampled exhaustively, then only
        the last 4 transitions remain and the oldest two are gone."""
        buffer = ReplayBuffer(4)
        for index in range(6):
            _push(buffer, index)

        self.assertEqual(len(buffer), 4)
        states, _, _, _, _ = buffer.sample(4)
        present = {float(state[0]) for state in states.tolist()}
        self.assertEqual(present, {2.0, 3.0, 4.0, 5.0})

    def test_sample_shapes_match_batch_and_state_dim(self):
        """Given a buffer holding 16 transitions, when a batch of 8 is
        sampled, then every tensor has the documented shape."""
        buffer = ReplayBuffer(16)
        for index in range(16):
            _push(buffer, index, done=(index == 15))

        states, actions, rewards, next_states, dones = buffer.sample(8)

        self.assertEqual(states.shape, (8, _STATE_DIM))
        self.assertEqual(actions.shape, (8,))
        self.assertEqual(rewards.shape, (8,))
        self.assertEqual(next_states.shape, (8, _STATE_DIM))
        self.assertEqual(dones.shape, (8,))
        self.assertEqual(states.dtype, torch.float32)

    def test_sample_requires_enough_transitions(self):
        """Given only 3 buffered transitions, when a batch of 4 is requested,
        then ValueError is raised."""
        buffer = ReplayBuffer(10)
        for index in range(3):
            _push(buffer, index)

        with self.assertRaises(ValueError):
            buffer.sample(4)

    def test_empty_buffer_has_zero_length(self):
        """Given a fresh buffer, when measured, then its length is zero."""
        self.assertEqual(len(ReplayBuffer(4)), 0)


if __name__ == "__main__":
    unittest.main()
